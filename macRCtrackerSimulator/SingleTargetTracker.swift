//
//  SingleTargetTracker.swift
//  macRCtrackerSimulator
//
//  ★ V7: 分级确认策略 — 让算法在真实场景下更鲁棒
//
//  V6→V7 改动总结:
//    1. selectTarget 改为三阶段分级确认:
//       Phase1 (丢失≤5帧): 纯空间匹配，不做特征确认 — 信任惯性
//       Phase2 (丢失6-15帧): 空间门控 + 轻量评分(IoU+尺寸) — 不提取VNFeaturePrint
//       Phase3 (丢失>15帧): 全特征重捕获 — HS+FP 全上
//    2. incumbent IoU 匹配成功时，模板更新改为惰性(每5帧一次) — 减少计算开销
//    3. HS 颜色确认从 veto(一票否决) 改为 bonus(加分项) — 不再因颜色不匹配直接拒绝
//    4. burn-in 从 15帧 延长到 30帧，且固定模板用 EMA 软更新而非永久冻结
//
//  核心架构(不变):
//    1. 独立 6D 卡尔曼滤波 (cx, cy, w, h, vx, vy)
//    2. 空间门控
//    3. 多特征评分
//    4. Incumbent-Challenger 迟滞
//    5. 三层模板系统
//    6. 全局重捕获
//

import Foundation
import CoreGraphics
import CoreImage
import Vision
import QuartzCore  // CACurrentMediaTime

// MARK: - 单目标追踪统计

struct STStats {
    var stTotalFrames: Int = 0
    var stLockFrames: Int = 0
    var stLockedTrackId: Int = -1
    var stPrevRect: CGRect? = nil

    var stLockRate: Double {
        stTotalFrames > 0 ? Double(stLockFrames) / Double(stTotalFrames) * 100 : 0
    }
}

// MARK: - 单目标追踪器

final class SingleTargetTracker {

    // ═══ 配置 ═══
    // ★ V7: burn-in 延长，给模板更多时间稳定
    private let burnInFrames: Int = 30
    // ★ V7: 模板更新间隔 (incumbent 匹配成功时不用每帧都提取特征)
    private let templateUpdateInterval: Int = 5
    private var framesSinceTemplateUpdate: Int = 0
    // ★ V7: 固定模板 EMA 更新率 (不再永久冻结)
    private let fixedTemplateEMA: Float = 0.02

    // ═══ 分级确认阈值 ═══
    // ★ V7: 三个阶段的丢失帧数边界
    private let phase1MaxLost: Int = 3       // 更快从短时惯性切到恢复态
    private let phase2MaxLost: Int = 8       // 更早放弃局部门控，进入强恢复/全局搜索
    // > phase2MaxLost → Phase3: 全特征重捕获
    private let continuityIoUThreshold: CGFloat = 0.04
    private let phase1GateMultiplier: CGFloat = 4.8
    private let phase2GateMultiplier: CGFloat = 8.5
    private let phase3MinScore: Float = 0.30
    private let earlyGlobalReacquireLost: Int = 5
    private let rawDetectionConfFresh: Float = 0.22
    private let rawDetectionConfRecovering: Float = 0.14
    private let rawDetectionConfLost: Float = 0.08
    private let exploratoryRawConf: Float = 0.05
    private let exploratoryRawTopK: Int = 10
    private let exploratoryDupIoU: CGFloat = 0.25
    private let phase2AcceptScoreBase: Double = 0.27
    private let phase2AcceptScoreRecovery: Double = 0.20
    private let expandedGateRetryBonus: CGFloat = 2.2
    private let sparseCandidateThreshold: Int = 3
    private let lowCandidateThreshold: Int = 5
    private let globalReacquireCandidateFloor: Int = 3
    private let phase3MinScoreFloor: Float = 0.24
    private let phase3MinScoreDecayPerLost: Float = 0.01
    private let phase3MinScoreDecayCap: Float = 0.08
    private let centerScoreNormScale: CGFloat = 5.2
    private let centerScoreNormFloor: CGFloat = 0.08
    private let gateLostFrameBoost: CGFloat = 0.45
    private let gateLostFrameBoostCap: CGFloat = 3.0
    private let gateSparseBoostStrong: CGFloat = 1.2
    private let gateSparseBoostLight: CGFloat = 0.6
    private let relaxedExpandPerLost: CGFloat = 0.10
    private let relaxedExpandCap: CGFloat = 0.60
    private let strictMinCenterScore: CGFloat = 0.08
    private let strictMinSizeScore: CGFloat = 0.16
    private let relaxedMinCenterScore: CGFloat = 0.02
    private let relaxedMinSizeScore: CGFloat = 0.08
    private let relaxedMinCenterScoreFloor: CGFloat = 0.0
    private let relaxedMinSizeScoreFloor: CGFloat = 0.06
    private let phase3MinSizeScoreFloor: CGFloat = 0.08
    private let phase3MinCenterScoreFloor: CGFloat = 0.0
    private let relaxedMinBoundFloor: CGFloat = 0.18
    private let relaxedAspectBoundFloor: CGFloat = 0.20
    private let relaxedAreaBoundFloor: CGFloat = 0.10
    private let relaxedUpperBase: CGFloat = 3.50
    private let relaxedAreaUpperBase: CGFloat = 4.50
    private let relaxedUpperExpandFactor: CGFloat = 2.0
    private let relaxedAreaUpperExpandFactor: CGFloat = 3.0
    private let relaxedAreaLowerExpandFactor: CGFloat = 0.5
    private let phase3WidthLower: CGFloat = 0.25
    private let phase3WidthUpper: CGFloat = 4.00
    private let phase3AreaLower: CGFloat = 0.15
    private let phase3AreaUpper: CGFloat = 6.00
    private let phase3AspectLower: CGFloat = 0.25
    private let phase3AspectUpper: CGFloat = 4.00
    private let phase3SearchSpatialWeight: Float = 0.14
    private let phase3SearchHSWeight: Float = 0.30
    private let phase3SearchFPWeight: Float = 0.36
    private let phase3SearchSizeWeight: Float = 0.20
    private let phase2GlobalEscalationLostFallback: Int = 4
    private let fallbackExploratoryLost: Int = 2

    // ═══ 独立 KF 状态 (6D: cx, cy, w, h, vx, vy) ═══
    private var kfX: [Double] = [0, 0, 0, 0, 0, 0]
    private var kfP: [Double] = [Double](repeating: 0, count: 36)
    private var kfInitialized: Bool = false

    // ═══ 模板系统 ═══
    private var fpFixed: VNFeaturePrintObservation? = nil
    private var fpDynamic: VNFeaturePrintObservation? = nil
    private var fpReacq: VNFeaturePrintObservation? = nil

    private var hsFixed: [Float]? = nil
    private var hsDynamic: [Float]? = nil
    private var hsReacq: [Float]? = nil

    private var burnInCount: Int = 0
    private var burnInHsAccum: [Float]? = nil
    private var burnInHsCount: Int = 0

    // ═══ 追踪状态 ═══
    private var prevCenter: CGPoint? = nil
    private var velocity: CGPoint = .zero
    private var consecutiveLost: Int = 0
    private var reacqUpdateCount: Int = 0
    private var incumbentRect: CGRect? = nil
    private var challengerWinStreak: Int = 0

    // ═══ ByteTracker (可选) ═══
    private var byteTracker: ByteTracker?

    // ═══ 像素缓冲区 ═══
    private var pixelBuffer: CVPixelBuffer? = nil
    private let ci = CIContext(options: [.cacheIntermediates: false])

    // ═══ 检测提供器 (使用真实检测模型推理) ═══
    typealias DetectionProvider = (CVPixelBuffer) -> [(rect: CGRect, conf: Float, label: String)]
    private var detectProvider: DetectionProvider?
    private var lastOwnDetectionTS: CFTimeInterval = 0
    private let ownDetectionMinInterval: CFTimeInterval = 0.033  // 自主推理最高 ~30fps

    // ═══ 协同模式: 增强算法预测 ═══
    private var currentEnhancedPred: CGRect?

    // MARK: - 初始化

    init(useByteTrack: Bool = true) {
        if useByteTrack {
            byteTracker = ByteTracker(config: .rcDefault)
        }
    }

    /// ★ 设置检测提供器 — 使用真实视频训练的检测模型推理
    /// 当外部未传入 yoloDets 时，ST 可自主调用此提供器获取候选框
    func setDetectionProvider(_ provider: @escaping DetectionProvider) {
        self.detectProvider = provider
    }

    // MARK: - 重置

    func reset() {
        kfX = [0, 0, 0, 0, 0, 0]
        kfP = [Double](repeating: 0, count: 36)
        kfInitialized = false

        fpFixed = nil; fpDynamic = nil; fpReacq = nil
        hsFixed = nil; hsDynamic = nil; hsReacq = nil
        burnInCount = 0; burnInHsAccum = nil; burnInHsCount = 0

        prevCenter = nil; velocity = .zero
        consecutiveLost = 0; reacqUpdateCount = 0
        incumbentRect = nil; challengerWinStreak = 0
        framesSinceTemplateUpdate = 0

        byteTracker?.reset()
        pixelBuffer = nil
        lastOwnDetectionTS = 0
        currentEnhancedPred = nil
    }

    // MARK: - 主入口

    func process(
        detRect: CGRect?,
        yoloDets: [(rect: CGRect, conf: Float, label: String)],
        pb: CVPixelBuffer?,
        enhancedPrediction: CGRect? = nil
    ) -> CGRect? {
        self.pixelBuffer = pb
        self.currentEnhancedPred = enhancedPrediction

        // ★ 如果外部未传入检测结果，使用自有检测器推理 (真实模型)
        var effectiveDets = yoloDets
        if effectiveDets.isEmpty, let provider = detectProvider, let pb = pb {
            let now = CACurrentMediaTime()
            if now - lastOwnDetectionTS >= ownDetectionMinInterval {
                effectiveDets = provider(pb)
                lastOwnDetectionTS = now
            }
        }

        // --- 构建候选框列表 ---
        let rawDetConfThreshold: Float
        if consecutiveLost > phase2MaxLost {
            rawDetConfThreshold = rawDetectionConfLost
        } else if consecutiveLost >= 2 {
            rawDetConfThreshold = rawDetectionConfRecovering
        } else {
            rawDetConfThreshold = rawDetectionConfFresh
        }

        var candidateRects: [CGRect]
        if let tracker = byteTracker {
            let btDets = effectiveDets.map { BTDetection(rect: $0.rect, score: Double($0.conf), classId: 0) }
            let tracks = tracker.update(detections: btDets)
            candidateRects = tracks.map { $0.normalizedRect }
            let rawRects = effectiveDets.filter { $0.conf >= rawDetConfThreshold }.map { $0.rect }
            for raw in rawRects {
                let isDuplicate = candidateRects.contains(where: { Self.iou($0, raw) > 0.5 })
                if !isDuplicate { candidateRects.append(raw) }
            }

            if consecutiveLost >= fallbackExploratoryLost || candidateRects.count <= globalReacquireCandidateFloor {
                let exploratoryRects = effectiveDets
                    .filter { $0.conf >= exploratoryRawConf }
                    .sorted { $0.conf > $1.conf }
                    .prefix(exploratoryRawTopK)
                    .map { $0.rect }
                for raw in exploratoryRects {
                    let isDuplicate = candidateRects.contains(where: { Self.iou($0, raw) > exploratoryDupIoU })
                    if !isDuplicate { candidateRects.append(raw) }
                }
            }
        } else {
            candidateRects = effectiveDets.filter { $0.conf >= rawDetConfThreshold }.map { $0.rect }
            if candidateRects.isEmpty, consecutiveLost >= fallbackExploratoryLost {
                candidateRects = effectiveDets
                    .filter { $0.conf >= exploratoryRawConf }
                    .sorted { $0.conf > $1.conf }
                    .prefix(exploratoryRawTopK)
                    .map { $0.rect }
            }
        }

        // --- 1. 首次检测: 初始化锁定 ---
        if !kfInitialized && !candidateRects.isEmpty {
            let best: CGRect
            if let maxDet = effectiveDets.max(by: { $0.conf < $1.conf }) {
                best = maxDet.rect
            } else {
                best = candidateRects.first!
            }

            kfInitWith(rect: best)

            fpFixed = nil
            fpDynamic = extractFeaturePrint(best)
            fpReacq = fpDynamic
            let hs = extractHSHistogram(best)
            hsFixed = nil
            hsDynamic = hs
            hsReacq = hs
            burnInCount = 0
            burnInHsAccum = hs
            burnInHsCount = hs != nil ? 1 : 0

            prevCenter = CGPoint(x: best.midX, y: best.midY)
            velocity = .zero
            consecutiveLost = 0
            reacqUpdateCount = 0
            incumbentRect = best
            challengerWinStreak = 0
            framesSinceTemplateUpdate = 0
        }

        guard kfInitialized else { return nil }

        // --- 2. KF 预测步 ---
        kfPredict()
        let predRect = kfGetPredictedRect()

        // --- 3. ★ V7 分级确认 ---
        let matchResult = selectTarget(candidates: candidateRects, predRect: predRect)

        return matchResult
    }

    // MARK: - ★ V7 分级确认核心逻辑

    private func selectTarget(candidates: [CGRect], predRect: CGRect) -> CGRect? {

        // ★ 协同模式: 融合增强算法的轨迹预测 (二次多项式拟合 + 8D 卡尔曼)
        //   增强预测提供更精确的未来位置估计, 内部 6D KF 提供形状和惯性
        //   未开启增强算法时 currentEnhancedPred 为 nil, 退化为纯内部 KF
        let effectivePred: CGRect
        if let enh = currentEnhancedPred {
            let lostRatio = min(CGFloat(consecutiveLost) / CGFloat(max(phase2MaxLost, 1)), 1)
            let enhWeight = max(0.28, 0.60 - lostRatio * 0.30)
            let cx = enh.midX * enhWeight + predRect.midX * (1 - enhWeight)
            let cy = enh.midY * enhWeight + predRect.midY * (1 - enhWeight)
            let w = max(predRect.width, enh.width * (0.75 + 0.15 * lostRatio))
            let h = max(predRect.height, enh.height * (0.75 + 0.15 * lostRatio))
            effectivePred = CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
        } else {
            effectivePred = predRect
        }

        // --- 情况 A: 无候选框 ---
        guard !candidates.isEmpty else {
            consecutiveLost += 1
            return nil
        }

        // --- 情况 B: 有 incumbent ---
        if let incumbent = incumbentRect {
            let searchRef = searchReferenceRect(predRect: effectivePred, incumbent: incumbent, lostFrames: consecutiveLost)

            // ★ Step 1: IoU 连续性匹配 (所有阶段都先试)，但先做几何合理性过滤
            var bestIoU: CGFloat = 0
            var bestSizeScore: CGFloat = 0
            var bestMatch: CGRect? = nil
            for cand in candidates {
                guard isCandidatePlausible(cand, predRect: searchRef, lostFrames: consecutiveLost, relaxed: consecutiveLost >= 2) else { continue }
                let iou1 = Self.iou(cand, incumbent)
                let iou2 = max(Self.iou(cand, effectivePred), Self.iou(cand, searchRef))
                let iou = max(iou1, iou2)
                let sizeScore = geometrySizeScore(cand, reference: searchRef)
                if iou > bestIoU || (abs(iou - bestIoU) < 0.001 && sizeScore > bestSizeScore) {
                    bestIoU = iou
                    bestSizeScore = sizeScore
                    bestMatch = cand
                }
            }

            if let match = bestMatch,
               bestIoU >= continuityIoUThreshold || (bestIoU >= 0.03 && bestSizeScore >= 0.78) {
                // IoU 匹配成功 — 直接接受，不做特征确认
                return acceptAndUpdate(match, iou: bestIoU)
            }

            // ★ IoU 匹配失败 — 根据丢失时长分级处理

            // ────────────────────────────────────────────
            // Phase 1: 短暂丢失 (≤5帧) — 纯空间匹配
            //   信任 KF 预测 + 距离最近，不提取任何特征
            //   原理: 短暂丢失大概率是检测抖动，目标还在附近
            // ────────────────────────────────────────────
            if consecutiveLost <= phase1MaxLost {
                let gateMultiplier = adaptiveGateMultiplier(base: phase1GateMultiplier, lostFrames: consecutiveLost, candidateCount: candidates.count)
                let nearest = findBestInGate(
                    candidates,
                    predRect: searchRef,
                    multiplier: gateMultiplier,
                    lostFrames: consecutiveLost,
                    relaxed: true
                )
                if let match = nearest {
                    return acceptAndUpdate(match, iou: 0)
                }
                if consecutiveLost >= 2 {
                    let wider = findBestInGate(
                            candidates,
                            predRect: searchRef,
                            multiplier: gateMultiplier + expandedGateRetryBonus,
                            lostFrames: consecutiveLost,
                            relaxed: true

                    )
                    if let match = wider {
                        return acceptAndUpdate(match, iou: 0)
                    }
                }
                let shouldTryEarlyGlobal = consecutiveLost >= 1 || candidates.count <= globalReacquireCandidateFloor
                if shouldTryEarlyGlobal, let reacquired = globalReacquire(candidates: candidates, predRect: searchRef) {
                    return reacquired
                }
                // 没找到也不急，等下一帧
                consecutiveLost += 1
                return nil
            }

            // ────────────────────────────────────────────
            // Phase 2: 中等丢失 (6-15帧) — 空间门控 + 轻量评分
            //   用 IoU + 尺寸一致性，不提取 VNFeaturePrint
            //   HS 颜色作为加分项(bonus)而非一票否决(veto)
            // ────────────────────────────────────────────
            if consecutiveLost <= phase2MaxLost {
                let gateMultiplier = adaptiveGateMultiplier(base: phase2GateMultiplier, lostFrames: consecutiveLost, candidateCount: candidates.count)
                let gateW = max(searchRef.width, 0.02) * gateMultiplier
                let gateH = max(searchRef.height, 0.02) * gateMultiplier
                let gateRect = CGRect(
                    x: searchRef.midX - gateW / 2,
                    y: searchRef.midY - gateH / 2,
                    width: gateW, height: gateH
                )

                var gatedScored: [(rect: CGRect, score: Double)] = []
                for cand in candidates {
                    if gateRect.contains(CGPoint(x: cand.midX, y: cand.midY)) &&
                        isCandidatePlausible(cand, predRect: searchRef, lostFrames: consecutiveLost, relaxed: true) {
                        let score = scoreLightweight(cand: cand, predRect: searchRef)
                        gatedScored.append((cand, score))
                    }
                }

                let acceptScore = consecutiveLost >= earlyGlobalReacquireLost ? phase2AcceptScoreRecovery : phase2AcceptScoreBase
                if let best = gatedScored.max(by: { $0.score < $1.score }), best.score > acceptScore {
                    return acceptAndUpdate(best.rect, iou: 0)
                }

                let shouldEscalateToGlobal =
                    consecutiveLost >= earlyGlobalReacquireLost ||
                    (candidates.count <= globalReacquireCandidateFloor && consecutiveLost >= phase2GlobalEscalationLostFallback)
                if shouldEscalateToGlobal {
                    consecutiveLost += 1
                    if let reacquired = globalReacquire(candidates: candidates, predRect: searchRef) {
                        return reacquired
                    }
                    return nil
                }

                consecutiveLost += 1
                return nil
            }

            // ────────────────────────────────────────────
            // Phase 3: 长期丢失 (>15帧) — 全特征重捕获
            //   HS 颜色 + VNFeaturePrint + 空间距离
            // ────────────────────────────────────────────
            consecutiveLost += 1
            return globalReacquire(candidates: candidates, predRect: effectivePred)

        } else {
            // --- 情况 C: 无 incumbent ---
            var minDist: CGFloat = .greatestFiniteMagnitude
            var nearest: CGRect? = nil
            for cand in candidates {
                let dx = cand.midX - effectivePred.midX
                let dy = cand.midY - effectivePred.midY
                let d = sqrt(dx*dx + dy*dy)
                if d < minDist { minDist = d; nearest = cand }
            }
            if let nr = nearest {
                kfUpdate(rect: nr)
                incumbentRect = nr
                prevCenter = CGPoint(x: nr.midX, y: nr.midY)
                consecutiveLost = 0
                return nr
            }
            consecutiveLost += 1
            return nil
        }
    }

    // MARK: - ★ V7 匹配接受 + 状态更新 (统一入口)

    private func acceptAndUpdate(_ match: CGRect, iou: CGFloat) -> CGRect {
        kfUpdate(rect: match)
        incumbentRect = match
        consecutiveLost = 0
        challengerWinStreak = 0

        let curCenter = CGPoint(x: match.midX, y: match.midY)
        if let prev = prevCenter {
            let rawVel = CGPoint(x: curCenter.x - prev.x, y: curCenter.y - prev.y)
            velocity = CGPoint(
                x: velocity.x * 0.7 + rawVel.x * 0.3,
                y: velocity.y * 0.7 + rawVel.y * 0.3
            )
        }
        prevCenter = curCenter

        // ★ V7: 惰性模板更新 — 不用每帧都提取特征
        framesSinceTemplateUpdate += 1
        if iou > 0.4, framesSinceTemplateUpdate >= templateUpdateInterval {
            framesSinceTemplateUpdate = 0
            let fp = extractFeaturePrint(match)
            let hs = extractHSHistogram(match)
            updateTemplates(rect: match, fp: fp, hs: hs, posterior: 0.8)
        }

        return match
    }

    // MARK: - ★ V7 轻量评分 (Phase 2 用，不提取 VNFeaturePrint)

    /// 只用 IoU + 尺寸 + HS(可选bonus)，不调 Vision 框架
    private func scoreLightweight(cand: CGRect, predRect: CGRect) -> Double {
        var score: Double = 0

        // A. 位置预测 IoU (0.40)
        let motionIoU = Self.iou(cand, predRect)
        score += Double(motionIoU) * 0.40

        // B. 尺寸与长宽比一致性 (0.25)
        score += Double(geometrySizeScore(cand, reference: predRect)) * 0.25

        // C. 中心距离连续性 (0.20)
        score += Double(geometryCenterScore(cand, reference: predRect)) * 0.20

        // D. HS 颜色 (0.15) — 继续作为 bonus
        let hs = extractHSHistogram(cand)
        if let hsSim = computeHSSimilarity(hs) {
            score += Double(hsSim) * 0.15
        }

        return score
    }

    // MARK: - 候选评分 (Phase 3 全特征，与 V6 一致)

    private func scoreCandidate(cand: CGRect, predRect: CGRect) -> Double {
        var score: Double = 0

        let motionIoU = Self.iou(cand, predRect)
        score += Double(motionIoU) * 0.50

        if predRect.width > 0.001, predRect.height > 0.001 {
            let wRatio = cand.width / predRect.width
            let hRatio = cand.height / predRect.height
            let sizeScore = max(0, 1.0 - abs(wRatio - 1.0) - abs(hRatio - 1.0))
            score += Double(sizeScore) * 0.25
        }

        let hs = extractHSHistogram(cand)
        let hsSim = computeHSSimilarity(hs)
        if let hsSim = hsSim {
            score += Double(hsSim) * 0.15
        }

        let fp = extractFeaturePrint(cand)
        let fpSim = computeFeaturePrintSimilarity(fp)
        if let fpSim = fpSim {
            score += Double(fpSim) * 0.10
        }

        return score
    }

    // MARK: - ★ V7 空间最近候选 (Phase 1 用)

    private func geometrySizeScore(_ cand: CGRect, reference ref: CGRect) -> CGFloat {
        guard cand.width > 0.001, cand.height > 0.001, ref.width > 0.001, ref.height > 0.001 else { return 0 }

        let wRatio = cand.width / ref.width
        let hRatio = cand.height / ref.height
        let areaRatio = (cand.width * cand.height) / max(ref.width * ref.height, 0.000001)
        let candAspect = cand.width / cand.height
        let refAspect = ref.width / ref.height
        let aspectRatio = candAspect / max(refAspect, 0.05)

        let widthScore = max(0, 1 - abs(wRatio - 1))
        let heightScore = max(0, 1 - abs(hRatio - 1))
        let areaScore = max(0, 1 - abs(areaRatio - 1) / 2.5)
        let aspectScore = max(0, 1 - abs(aspectRatio - 1) / 1.5)

        let combined = widthScore * 0.30 + heightScore * 0.30 + areaScore * 0.20 + aspectScore * 0.20
        return min(max(combined, 0), 1)
    }

    private func geometryCenterScore(_ cand: CGRect, reference ref: CGRect) -> CGFloat {
        let dx = cand.midX - ref.midX
        let dy = cand.midY - ref.midY
        let dist = sqrt(dx * dx + dy * dy)
        let norm = max(max(ref.width, ref.height) * centerScoreNormScale, centerScoreNormFloor)
        return max(0, 1 - dist / norm)
    }

    private func adaptiveGateMultiplier(base: CGFloat, lostFrames: Int, candidateCount: Int) -> CGFloat {
        let lostBoost = min(CGFloat(max(lostFrames - 1, 0)) * gateLostFrameBoost, gateLostFrameBoostCap)
        let sparseBoost: CGFloat
        if candidateCount <= sparseCandidateThreshold {
            sparseBoost = gateSparseBoostStrong
        } else if candidateCount <= lowCandidateThreshold {
            sparseBoost = gateSparseBoostLight
        } else {
            sparseBoost = 0
        }
        return base + lostBoost + sparseBoost
    }

    private func searchReferenceRect(predRect: CGRect, incumbent: CGRect?, lostFrames: Int) -> CGRect {
        guard let incumbent, lostFrames > 0 else { return predRect }
        let lostRatio = min(CGFloat(lostFrames) / CGFloat(max(phase2MaxLost, 1)), 1)
        let incumbentWeight = min(0.70, 0.25 + lostRatio * 0.45)
        let predWeight = max(0.30, 1 - incumbentWeight)
        let cx = predRect.midX * predWeight + incumbent.midX * incumbentWeight
        let cy = predRect.midY * predWeight + incumbent.midY * incumbentWeight
        let expand = 1 + lostRatio * 0.18
        let w = max(predRect.width, incumbent.width) * expand
        let h = max(predRect.height, incumbent.height) * expand
        return CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
    }

    private func isCandidatePlausible(_ cand: CGRect, predRect: CGRect, lostFrames: Int, relaxed: Bool) -> Bool {
        guard cand.width > 0.003, cand.height > 0.003 else { return false }
        guard predRect.width > 0.003, predRect.height > 0.003 else { return true }

        let wRatio = cand.width / predRect.width
        let hRatio = cand.height / predRect.height
        let areaRatio = (cand.width * cand.height) / max(predRect.width * predRect.height, 0.000001)
        let candAspect = cand.width / cand.height
        let predAspect = predRect.width / predRect.height
        let aspectRatio = candAspect / max(predAspect, 0.05)

        let widthRange: ClosedRange<CGFloat>
        let heightRange: ClosedRange<CGFloat>
        let areaRange: ClosedRange<CGFloat>
        let aspectRange: ClosedRange<CGFloat>
        let minCenterScore: CGFloat
        let minSizeScore: CGFloat

        if lostFrames > phase2MaxLost {
            widthRange = phase3WidthLower...phase3WidthUpper
            heightRange = phase3WidthLower...phase3WidthUpper
            areaRange = phase3AreaLower...phase3AreaUpper
            aspectRange = phase3AspectLower...phase3AspectUpper
            minCenterScore = phase3MinCenterScoreFloor
            minSizeScore = phase3MinSizeScoreFloor
        } else if relaxed {
            let expand = min(CGFloat(max(lostFrames - phase1MaxLost, 0)) * relaxedExpandPerLost, relaxedExpandCap)
            widthRange = max(relaxedMinBoundFloor, 0.30 - expand)...(relaxedUpperBase + expand * relaxedUpperExpandFactor)
            heightRange = max(relaxedMinBoundFloor, 0.30 - expand)...(relaxedUpperBase + expand * relaxedUpperExpandFactor)
            areaRange = max(relaxedAreaBoundFloor, 0.20 - expand * relaxedAreaLowerExpandFactor)...(relaxedAreaUpperBase + expand * relaxedAreaUpperExpandFactor)
            aspectRange = max(relaxedAspectBoundFloor, 0.30 - expand)...(relaxedUpperBase + expand * relaxedUpperExpandFactor)
            minCenterScore = max(relaxedMinCenterScoreFloor, relaxedMinCenterScore - expand * 0.08)
            minSizeScore = max(relaxedMinSizeScoreFloor, relaxedMinSizeScore - expand * 0.10)
        } else {
            widthRange = 0.45...2.50
            heightRange = 0.45...2.50
            areaRange = 0.30...3.00
            aspectRange = 0.40...2.50
            minCenterScore = strictMinCenterScore
            minSizeScore = strictMinSizeScore
        }

        let centerScore = geometryCenterScore(cand, reference: predRect)
        let sizeScore = geometrySizeScore(cand, reference: predRect)

        return widthRange.contains(wRatio) &&
            heightRange.contains(hRatio) &&
            areaRange.contains(areaRatio) &&
            aspectRange.contains(aspectRatio) &&
            centerScore >= minCenterScore &&
            sizeScore >= minSizeScore
    }

    private func findBestInGate(_ candidates: [CGRect], predRect: CGRect, multiplier: CGFloat, lostFrames: Int, relaxed: Bool) -> CGRect? {
        let gateW = max(predRect.width, 0.02) * multiplier
        let gateH = max(predRect.height, 0.02) * multiplier
        let gateRect = CGRect(
            x: predRect.midX - gateW / 2,
            y: predRect.midY - gateH / 2,
            width: gateW, height: gateH
        )

        var bestScore: CGFloat = -.greatestFiniteMagnitude
        var bestRect: CGRect? = nil
        for cand in candidates {
            let cp = CGPoint(x: cand.midX, y: cand.midY)
            guard gateRect.contains(cp),
                  isCandidatePlausible(cand, predRect: predRect, lostFrames: lostFrames, relaxed: relaxed) else { continue }
            let score = geometryCenterScore(cand, reference: predRect) * 0.65 +
                        geometrySizeScore(cand, reference: predRect) * 0.35
            if score > bestScore {
                bestScore = score
                bestRect = cand
            }
        }
        return bestRect
    }

    // MARK: - 全局重捕获

    private func globalReacquire(candidates: [CGRect], predRect: CGRect) -> CGRect? {
        let relaxedCandidates = candidates.filter {
            isCandidatePlausible($0, predRect: predRect, lostFrames: consecutiveLost, relaxed: true)
        }
        let searchCandidates = relaxedCandidates.isEmpty ? candidates : relaxedCandidates

        var scored: [(rect: CGRect, score: Float)] = []
        for cand in searchCandidates {
            let hs = extractHSHistogram(cand)
            let hsSim = computeHSSimilarity(hs) ?? 0
            let spatialScore = Float(geometryCenterScore(cand, reference: predRect))
            let sizeScore = Float(geometrySizeScore(cand, reference: predRect))

            // ★ V7: 加入 VNFeaturePrint (Phase3 才用)
            let fp = extractFeaturePrint(cand)
            let fpSim = computeFeaturePrintSimilarity(fp) ?? 0

            let combined = spatialScore * phase3SearchSpatialWeight + hsSim * phase3SearchHSWeight + fpSim * phase3SearchFPWeight + sizeScore * phase3SearchSizeWeight
            scored.append((cand, combined))
        }

        let phase3MinScoreDynamic = max(
            phase3MinScoreFloor,
            phase3MinScore - min(Float(max(consecutiveLost - phase2MaxLost, 0)) * phase3MinScoreDecayPerLost, phase3MinScoreDecayCap)
        )
        guard let best = scored.max(by: { $0.score < $1.score }),
              best.score > phase3MinScoreDynamic else { return nil }
        let rect = best.rect

        kfInitWith(rect: rect)
        incumbentRect = rect
        prevCenter = CGPoint(x: rect.midX, y: rect.midY)
        velocity = .zero
        consecutiveLost = 0
        challengerWinStreak = 0

        let fp = extractFeaturePrint(rect)
        let hs = extractHSHistogram(rect)
        updateTemplates(rect: rect, fp: fp, hs: hs, posterior: 0.6)

        return rect
    }

    // MARK: - 卡尔曼滤波 (不变)

    private func kfInitWith(rect: CGRect) {
        let cx = Double(rect.midX)
        let cy = Double(rect.midY)
        let w = Double(rect.width)
        let h = Double(rect.height)
        kfX = [cx, cy, w, h, 0, 0]
        kfP = [Double](repeating: 0, count: 36)
        kfP[0]  = 0.01
        kfP[7]  = 0.01
        kfP[14] = 0.005
        kfP[21] = 0.005
        kfP[28] = 0.05
        kfP[35] = 0.05
        kfInitialized = true
    }

    private func kfPredict() {
        guard kfInitialized else { return }
        kfX[0] += kfX[4]
        kfX[1] += kfX[5]
        kfP[0]  += kfP[28] + 0.002
        kfP[7]  += kfP[35] + 0.002
        kfP[14] += 0.001
        kfP[21] += 0.001
        kfP[28] += 0.003
        kfP[35] += 0.003
    }

    private func kfUpdate(rect: CGRect) {
        guard kfInitialized else { return }
        let z = [Double(rect.midX), Double(rect.midY), Double(rect.width), Double(rect.height)]
        let R: [Double] = [0.005, 0.005, 0.003, 0.003]

        for i in 0..<4 {
            let pii = kfP[i * 6 + i]
            let k = pii / (pii + R[i])
            let innovation = z[i] - kfX[i]
            kfX[i] += k * innovation
            kfP[i * 6 + i] = (1.0 - k) * pii
        }
        let velGain = 0.3
        let dx = z[0] - (kfX[0] - kfX[4])
        let dy = z[1] - (kfX[1] - kfX[5])
        kfX[4] += velGain * dx
        kfX[5] += velGain * dy
        kfP[28] *= 0.8
        kfP[35] *= 0.8
    }

    private func kfGetPredictedRect() -> CGRect {
        let cx = CGFloat(kfX[0])
        let cy = CGFloat(kfX[1])
        let w = CGFloat(max(kfX[2], 0.005))
        let h = CGFloat(max(kfX[3], 0.005))
        return CGRect(x: cx - w/2, y: cy - h/2, width: w, height: h)
    }

    // MARK: - VNFeaturePrint

    private func extractFeaturePrint(_ rect: CGRect) -> VNFeaturePrintObservation? {
        guard let pb = pixelBuffer,
              rect.width > 0.005, rect.height > 0.005 else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pb)
        let imgW = ciImage.extent.width
        let imgH = ciImage.extent.height

        let cropRect = CGRect(
            x: rect.minX * imgW,
            y: (1.0 - rect.maxY) * imgH,
            width: rect.width * imgW,
            height: rect.height * imgH
        ).intersection(ciImage.extent)
        guard cropRect.width >= 4, cropRect.height >= 4 else { return nil }

        let cropped = ciImage.cropped(to: cropRect)
        guard let cgImage = ci.createCGImage(cropped, from: cropped.extent) else { return nil }

        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
            return request.results?.first as? VNFeaturePrintObservation
        } catch {
            return nil
        }
    }

    private func computeFeaturePrintSimilarity(_ candidate: VNFeaturePrintObservation?) -> Float? {
        guard let cand = candidate else { return nil }
        let isSearchPhase = consecutiveLost > 5

        var fixedSim: Float = 0
        var dynSim: Float = 0
        var reacqSim: Float = 0
        var hasAny = false

        if let fixed = fpFixed {
            fixedSim = fpSimilarity(fixed, cand)
            hasAny = true
        }
        if let dyn = fpDynamic {
            dynSim = fpSimilarity(dyn, cand)
            hasAny = true
        }
        if let reacq = fpReacq {
            reacqSim = fpSimilarity(reacq, cand)
            hasAny = true
        }

        guard hasAny else { return nil }
        if fpFixed == nil { return dynSim }

        if isSearchPhase {
            return fixedSim * 0.5 + reacqSim * 0.3 + dynSim * 0.2
        } else {
            return fixedSim * 0.4 + dynSim * 0.4 + reacqSim * 0.2
        }
    }

    private func fpSimilarity(_ a: VNFeaturePrintObservation, _ b: VNFeaturePrintObservation) -> Float {
        var distance: Float = 0
        do {
            try a.computeDistance(&distance, to: b)
        } catch {
            return 0
        }
        let sigma: Float = 12.0
        return exp(-distance / sigma)
    }

    // MARK: - HS 直方图

    private func extractHSHistogram(_ rect: CGRect) -> [Float]? {
        guard let pb = pixelBuffer,
              rect.width > 0.005, rect.height > 0.005 else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pb)
        let imgW = ciImage.extent.width
        let imgH = ciImage.extent.height

        let cropRect = CGRect(
            x: rect.minX * imgW,
            y: (1.0 - rect.maxY) * imgH,
            width: rect.width * imgW,
            height: rect.height * imgH
        ).intersection(ciImage.extent)
        guard cropRect.width >= 4, cropRect.height >= 4 else { return nil }

        let cropped = ciImage.cropped(to: cropRect)
        guard let colorBytes = rgbaBytes(ciImage: cropped, size: CGSize(width: 32, height: 32)) else { return nil }

        let hBins = 16, sBins = 8
        var hist = [Float](repeating: 0, count: hBins * sBins)
        for i in stride(from: 0, to: min(colorBytes.count, 32 * 32 * 4), by: 4) {
            let r = Float(colorBytes[i]) / 255.0
            let g = Float(colorBytes[i + 1]) / 255.0
            let b = Float(colorBytes[i + 2]) / 255.0
            let hsv = rgbToHSV(r: r, g: g, b: b)
            let hBin = min(Int(hsv.h * Float(hBins)), hBins - 1)
            let sBin = min(Int(hsv.s * Float(sBins)), sBins - 1)
            hist[hBin * sBins + sBin] += 1.0
        }
        let sumHist = hist.reduce(0, +)
        if sumHist > 0 { hist = hist.map { $0 / sumHist } }
        return hist
    }

    private func computeHSSimilarity(_ candidate: [Float]?) -> Float? {
        guard let cand = candidate else { return nil }
        let isSearchPhase = consecutiveLost > 5

        var fixedSim: Float = 0
        var dynSim: Float = 0
        var reacqSim: Float = 0
        var hasAny = false

        if let fixed = hsFixed {
            fixedSim = histIntersection(a: fixed, b: cand)
            hasAny = true
        }
        if let dyn = hsDynamic {
            dynSim = histIntersection(a: dyn, b: cand)
            hasAny = true
        }
        if let reacq = hsReacq {
            reacqSim = histIntersection(a: reacq, b: cand)
            hasAny = true
        }

        guard hasAny else { return nil }
        if hsFixed == nil { return dynSim }

        if isSearchPhase {
            return fixedSim * 0.5 + reacqSim * 0.3 + dynSim * 0.2
        } else {
            return fixedSim * 0.4 + dynSim * 0.4 + reacqSim * 0.2
        }
    }

    // MARK: - ★ V7 模板更新 (固定模板改为 EMA 软更新)

    private func updateTemplates(
        rect: CGRect,
        fp: VNFeaturePrintObservation?,
        hs: [Float]?,
        posterior: Double
    ) {
        let isHighConf = posterior > 0.6

        // burn-in 阶段
        if burnInCount < burnInFrames {
            burnInCount += 1

            if isHighConf, let fp = fp {
                fpDynamic = fp
            }
            if let hs = hs {
                if burnInHsAccum == nil {
                    burnInHsAccum = hs
                    burnInHsCount = 1
                } else {
                    for i in 0..<min(burnInHsAccum!.count, hs.count) {
                        burnInHsAccum![i] += hs[i]
                    }
                    burnInHsCount += 1
                }
                hsDynamic = hs
            }

            if burnInCount == burnInFrames {
                fpFixed = fpDynamic
                fpReacq = fpDynamic
                if let accum = burnInHsAccum, burnInHsCount > 0 {
                    let avg = accum.map { $0 / Float(burnInHsCount) }
                    let s = avg.reduce(0, +)
                    hsFixed = s > 0 ? avg.map { $0 / s } : avg
                } else {
                    hsFixed = hsDynamic
                }
                hsReacq = hsFixed
            }
            return
        }

        // 正常阶段
        guard isHighConf else { return }

        // VNFeaturePrint 动态模板
        if let fp = fp, let fixed = fpFixed {
            let fixedCheck = fpSimilarity(fixed, fp)
            if fixedCheck > 0.35 {
                fpDynamic = fp
            }
        }

        // ★ V7: 固定 HS 模板也做 EMA 软更新 (不再永久冻结)
        if let hs = hs, let dyn = hsDynamic {
            let alpha: Float = 0.15
            var blended = [Float]()
            blended.reserveCapacity(dyn.count)
            for i in 0..<min(dyn.count, hs.count) {
                blended.append(dyn[i] * (1 - alpha) + hs[i] * alpha)
            }
            let s = blended.reduce(0, +)
            if s > 0 { blended = blended.map { $0 / s } }
            if let fixed = hsFixed {
                let check = histIntersection(a: fixed, b: blended)
                if check > 0.40 {
                    hsDynamic = blended
                }
                // ★ V7: 固定模板慢慢跟随 (EMA 0.02)
                if check > 0.50 {
                    var updatedFixed = [Float]()
                    updatedFixed.reserveCapacity(fixed.count)
                    for i in 0..<min(fixed.count, blended.count) {
                        updatedFixed.append(fixed[i] * (1 - fixedTemplateEMA) + blended[i] * fixedTemplateEMA)
                    }
                    let sf = updatedFixed.reduce(0, +)
                    if sf > 0 { updatedFixed = updatedFixed.map { $0 / sf } }
                    hsFixed = updatedFixed
                }
            } else {
                hsDynamic = blended
            }
        }

        // 重捕获模板: 每 10 帧更新
        reacqUpdateCount += 1
        if reacqUpdateCount >= 10 {
            reacqUpdateCount = 0
            if let fp = fp { fpReacq = fp }
            if let hs = hs { hsReacq = hs }
        }
    }

    // MARK: - 辅助函数 (不变)

    private func rgbaBytes(ciImage: CIImage, size: CGSize) -> [UInt8]? {
        guard size.width > 0, size.height > 0 else { return nil }
        let w = Int(size.width), h = Int(size.height)
        var data = [UInt8](repeating: 0, count: w * h * 4)
        guard let cg = ci.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        guard let ctx = CGContext(
            data: &data, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .low
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return data
    }

    private func rgbToHSV(r: Float, g: Float, b: Float) -> (h: Float, s: Float, v: Float) {
        let maxV = max(r, g, b)
        let minV = min(r, g, b)
        let delta = maxV - minV
        var h: Float = 0
        if delta > 0 {
            if maxV == r { h = (g - b) / delta; if g < b { h += 6 } }
            else if maxV == g { h = (b - r) / delta + 2 }
            else { h = (r - g) / delta + 4 }
            h /= 6
        }
        let s: Float = maxV == 0 ? 0 : (delta / maxV)
        return (h, s, maxV)
    }

    private func histIntersection(a: [Float], b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var sum: Float = 0
        for i in 0..<n { sum += min(a[i], b[i]) }
        return max(0, min(1, sum))
    }

    static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let u = a.width * a.height + b.width * b.height - inter.width * inter.height
        return u > 0 ? inter.width * inter.height / u : 0
    }

    // MARK: - 诊断信息

    var diagnosticInfo: String {
        let locked = kfInitialized
        let lost = consecutiveLost
        let hasFpFixed = fpFixed != nil
        let burnIn = burnInCount < burnInFrames
        let phase = consecutiveLost <= phase1MaxLost ? "P1-spatial" :
                    consecutiveLost <= phase2MaxLost ? "P2-light" : "P3-full"
        let coop = currentEnhancedPred != nil ? " coop=YES" : ""
        let det = detectProvider != nil ? " det=OWN" : ""
        return "[ST-V7] init=\(locked) lost=\(lost) phase=\(phase) fpFixed=\(hasFpFixed) burnIn=\(burnIn)\(coop)\(det) vel=\(String(format: "%.4f,%.4f", velocity.x, velocity.y))"
    }

    var isLocked: Bool { kfInitialized }
    var lostFrames: Int { consecutiveLost }
}
