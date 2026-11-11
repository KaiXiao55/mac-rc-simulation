//
//  Vntrackerlabmodule.swift
//  macRCtrackerSimulator
//
//  Created by Mac on 2026/2/22.
//  Improved: 2026/3/7 — 新增 IoU 贪心匹配跟踪器, 修复已有问题
//

// VNTrackerLabModule.swift
// Vision Tracker 学习实验室 — 独立模块文件
//
// 使用方法:
//   1. 将本文件添加到现有 RCSimApp 项目中
//   2. 修改原 macRCtrackerSimulatorApp.swift 的 @main App 入口 (见文末说明)
//   3. Build & Run — 新增 "🧪 Vision Lab" 标签页
//
// 支持的 Vision API 实验:
//   ① VNTrackObjectRequest         — 通用物体追踪 (点击画面框选目标)
//   ①' 卡尔曼追踪 Kalman            — 块匹配观测 + 卡尔曼滤波 (无 Vision 调用)
//   ★ IoU 贪心匹配 SORT             — YOLO 检测 + IoU 多目标跟踪 (需 mlmodelc)
//   ② VNTrackRectangleRequest      — 矩形追踪 (自动检测矩形并追踪)
//   ③ VNGenerateOpticalFlowRequest — 光流法可视化 (帧间运动场)
//   ④ VNDetectTrajectoriesRequest  — 抛物线/弹道检测 (球类追踪)
//   ⑤ VNDetectHumanBodyPoseRequest — 人体骨骼姿态估计
//   ⑥ VNGeneratePersonSegmentationRequest — 人物分割蒙版
//
// 最低要求: macOS 14.0+, Xcode 15+

import SwiftUI
import Combine
import AVFoundation
import Vision
import CoreML
import CoreImage
import CoreVideo
import UniformTypeIdentifiers
import Accelerate

// ═══════════════════════════════════════════════════════════════════
// MARK: - 算法枚举
// ═══════════════════════════════════════════════════════════════════

enum VNLabAlgorithm: String, CaseIterable, Identifiable {
    case trackObject    = "① 物体追踪 VNTrackObject"
    case kalmanTrack    = "①' 卡尔曼追踪 Kalman"
    case iouTracker     = "★ IoU 贪心匹配 SORT"
    case trackRectangle = "② 矩形追踪 VNTrackRectangle"
    case opticalFlow    = "③ 光流法 OpticalFlow"
    case trajectory     = "④ 抛物线检测 Trajectory"
    case bodyPose       = "⑤ 人体姿态 BodyPose"
    case personSegment  = "⑥ 人物分割 PersonSegment"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .trackObject:    return "scope"
        case .kalmanTrack:    return "waveform.path.ecg"
        case .iouTracker:     return "square.3.layers.3d.down.right"
        case .trackRectangle: return "rectangle.dashed"
        case .opticalFlow:    return "wind"
        case .trajectory:     return "point.topleft.down.to.point.bottomright.curvepath"
        case .bodyPose:       return "figure.walk"
        case .personSegment:  return "person.fill.viewfinder"
        }
    }

    /// 是否需要用户手动框选 ROI（VNTrackObject / 卡尔曼 / IoU 均可选「用检测框」模式）
    var needsROI: Bool { self == .trackObject || self == .kalmanTrack || self == .iouTracker }

    /// 是否需要两帧 (前一帧 + 当前帧)
    var needsPrevFrame: Bool { self == .opticalFlow }

    /// 是否需要 YOLO 模型
    var needsYOLO: Bool { self == .iouTracker }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - 每帧结果快照
// ═══════════════════════════════════════════════════════════════════

struct FlowArrow {
    let origin: CGPoint   // 归一化 TL 坐标
    let dx: Float
    let dy: Float
    var magnitude: Float { sqrt(dx * dx + dy * dy) }
}

struct TrajectoryResult {
    let points: [CGPoint]
    let equation: (a: Float, b: Float, c: Float)?
    let movingAverageRadius: CGFloat
}

struct PoseJoint {
    let name: String
    let position: CGPoint  // 归一化 TL
    let confidence: Float
}

struct PoseBone {
    let from: CGPoint
    let to: CGPoint
    let confidence: Float
}

/// IoU 跟踪器的单个跟踪目标
struct TrackedTarget: Identifiable {
    let id: Int                     // 跟踪 ID
    var rect: CGRect                // 归一化 TL
    var confidence: Float           // YOLO 置信度
    var label: String               // YOLO 类别标签
    var velocity: CGPoint           // 归一化速度 (EMA)
    var age: Int                    // 存活帧数
    var missCount: Int              // 连续未匹配帧数
    var isYoloFrame: Bool           // 本帧是否有检测匹配（YOLO 或块匹配检测框）
}

struct LabFrame {
    let index: Int
    let time: Double
    let image: CGImage?

    var trackRect: CGRect?             // 归一化 TL
    var trackConf: Float = 0
    var trackAlive: Bool = false

    var rectangles: [CGRect] = []      // 归一化 TL (初始检测)
    var trackedRectCorners: [CGPoint]? // 4 角点 (归一化 TL)

    var flowArrows: [FlowArrow] = []
    var flowImage: CGImage?            // HSV 彩色光流图

    var trajectories: [TrajectoryResult] = []

    var joints: [PoseJoint] = []
    var bones: [PoseBone] = []

    var segMask: CGImage?

    /// IoU 跟踪器: 多目标结果
    var trackedTargets: [TrackedTarget] = []
    /// IoU 跟踪器: 本帧 YOLO 原始检测数
    var yoloDetectionCount: Int = 0
    /// IoU 跟踪器: YOLO 推理耗时 (ms)
    var yoloMs: Double = 0

    var processMs: Double = 0
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - IoU 贪心匹配跟踪器 (SORT 风格)
// ═══════════════════════════════════════════════════════════════════

/// 纯 Swift 实现的 SORT 风格多目标跟踪器
/// 每帧: YOLO 检测 → IoU 贪心匹配 → EMA 运动模型预测
final class IoUGreedyTracker {

    struct Track {
        let id: Int
        var rect: CGRect            // 归一化 TL
        var velocity: CGPoint       // EMA 平滑速度
        var confidence: Float
        var label: String
        var age: Int = 1
        var missCount: Int = 0
        var lastMatchedRect: CGRect // 上次匹配到的检测框
    }

    private var tracks: [Track] = []
    private var nextId: Int = 1

    // 可调参数
    var iouThreshold: Float = 0.20      // IoU 匹配阈值
    var maxMiss: Int = 8                // 最大连续丢失帧数
    var velocityAlpha: CGFloat = 0.3    // EMA 速度平滑系数
    var minConfidence: Float = 0.25     // YOLO 最低置信度

    func reset() {
        tracks.removeAll()
        nextId = 1
    }

    /// 输入: YOLO 检测结果 (归一化 TL 坐标)
    /// 输出: 当前帧所有跟踪目标
    func update(detections: [(rect: CGRect, confidence: Float, label: String)]) -> [TrackedTarget] {

        // 1. 用运动模型预测每个 track 的当前位置
        for i in tracks.indices {
            let t = tracks[i]
            let predicted = CGRect(
                x: t.rect.minX + t.velocity.x,
                y: t.rect.minY + t.velocity.y,
                width: t.rect.width,
                height: t.rect.height
            )
            tracks[i].rect = predicted
        }

        // 2. 计算 IoU 矩阵
        var matched = Set<Int>()        // 已匹配的 detection index
        var matchedTracks = Set<Int>()  // 已匹配的 track index

        // 构建 (trackIdx, detIdx, iou) 列表并按 IoU 降序排列
        var pairs: [(trackIdx: Int, detIdx: Int, iou: Float)] = []
        for (ti, track) in tracks.enumerated() {
            for (di, det) in detections.enumerated() {
                let iou = Self.computeIoU(track.rect, det.rect)
                if iou > iouThreshold {
                    pairs.append((ti, di, iou))
                }
            }
        }
        pairs.sort { $0.iou > $1.iou }

        // 3. 贪心匹配
        for pair in pairs {
            if matchedTracks.contains(pair.trackIdx) || matched.contains(pair.detIdx) { continue }
            matchedTracks.insert(pair.trackIdx)
            matched.insert(pair.detIdx)

            let det = detections[pair.detIdx]
            let oldRect = tracks[pair.trackIdx].rect
            let dx = det.rect.midX - oldRect.midX
            let dy = det.rect.midY - oldRect.midY
            let alpha = velocityAlpha
            tracks[pair.trackIdx].velocity = CGPoint(
                x: tracks[pair.trackIdx].velocity.x * (1 - alpha) + dx * alpha,
                y: tracks[pair.trackIdx].velocity.y * (1 - alpha) + dy * alpha
            )
            tracks[pair.trackIdx].rect = det.rect
            tracks[pair.trackIdx].lastMatchedRect = det.rect
            tracks[pair.trackIdx].confidence = det.confidence
            tracks[pair.trackIdx].label = det.label
            tracks[pair.trackIdx].age += 1
            tracks[pair.trackIdx].missCount = 0
        }

        // 4. 未匹配的 track: missCount++
        for ti in tracks.indices {
            if !matchedTracks.contains(ti) {
                tracks[ti].missCount += 1
                tracks[ti].age += 1
            }
        }

        // 5. 删除丢失过久的 track
        tracks.removeAll { $0.missCount > maxMiss }

        // 6. 未匹配的 detection: 创建新 track
        for (di, det) in detections.enumerated() {
            if !matched.contains(di) && det.confidence >= minConfidence {
                tracks.append(Track(
                    id: nextId,
                    rect: det.rect,
                    velocity: .zero,
                    confidence: det.confidence,
                    label: det.label,
                    age: 1,
                    missCount: 0,
                    lastMatchedRect: det.rect
                ))
                nextId += 1
            }
        }

        // 7. 输出
        return tracks.map { t in
            TrackedTarget(
                id: t.id,
                rect: t.rect,
                confidence: t.confidence,
                label: t.label,
                velocity: t.velocity,
                age: t.age,
                missCount: t.missCount,
                isYoloFrame: t.missCount == 0
            )
        }
    }

    static func computeIoU(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull && inter.width > 0 && inter.height > 0 else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return Float(interArea / max(unionArea, 1e-6))
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - 卡尔曼追踪 (轻量块匹配观测，保持与 VNTrackObject 接近帧率)
// ═══════════════════════════════════════════════════════════════════

/// 4 维卡尔曼：状态 [cx, cy, vx, vy]，观测 [cx, cy]。恒定速度模型。
struct KalmanTracker4D {
    private var x: [Double]       // 4
    private var P: [Double]       // 4x4 row-major
    private let F: [Double]      // 4x4 状态转移
    private let H: [Double]      // 2x4 观测矩阵
    private let Q: [Double]      // 4x4 过程噪声
    private let R: [Double]      // 2x2 观测噪声
    private let dt: Double

    /// 连续无观测预测次数
    private(set) var missCount: Int = 0
    /// 位置不确定性
    var uncertainty: Double { sqrt(max(P[0], 0) + max(P[5], 0)) }

    init(cx: Double, cy: Double, dt: Double = 1.0/30.0) {
        self.dt = dt
        x = [cx, cy, 0, 0]
        // ★ FIX: 速度初始方差 0.001→1.0, 让滤波器快速学习速度
        P = [0.01,0,0,0, 0,0.01,0,0, 0,0,1.0,0, 0,0,0,1.0]
        F = [1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1]
        H = [1,0,0,0, 0,1,0,0]
        let q: Double = 0.008
        Q = [q,0,0,0, 0,q,0,0, 0,0,q*0.5,0, 0,0,0,q*0.5]
        let r: Double = 0.015
        R = [r,0, 0,r]
    }

    mutating func predict() {
        var xNew = [Double](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<4 { xNew[i] += F[i*4+j] * x[j] }
        }
        // ★ FIX: P = F×P×Fᵀ + Q  (原代码漏了 ×Fᵀ)
        var temp = [Double](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                for k in 0..<4 { temp[i*4+j] += F[i*4+k] * P[k*4+j] }
            }
        }
        var PNew = [Double](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                for k in 0..<4 { PNew[i*4+j] += temp[i*4+k] * F[j*4+k] }
            }
        }
        for i in 0..<4 {
            for j in 0..<4 { PNew[i*4+j] += Q[i*4+j] }
        }
        x = xNew; P = PNew; missCount += 1
    }

    mutating func update(zCx: Double, zCy: Double) {
        let z = [zCx, zCy]
        var Hx = [Double](repeating: 0, count: 2)
        for i in 0..<2 {
            for j in 0..<4 { Hx[i] += H[i*4+j] * x[j] }
        }
        let y = [z[0]-Hx[0], z[1]-Hx[1]]

        // S = H×P×Hᵀ + R
        var HP = [Double](repeating: 0, count: 8)
        for i in 0..<2 {
            for j in 0..<4 {
                for k in 0..<4 { HP[i*4+j] += H[i*4+k] * P[k*4+j] }
            }
        }
        var S = [Double](repeating: 0, count: 4)
        for i in 0..<2 {
            for j in 0..<2 {
                for k in 0..<4 { S[i*2+j] += HP[i*4+k] * H[j*4+k] }
                S[i*2+j] += R[i*2+j]
            }
        }
        let det = S[0]*S[3] - S[1]*S[2]
        guard abs(det) > 1e-12 else { return }
        let id = 1.0 / det
        let Si = [S[3]*id, -S[1]*id, -S[2]*id, S[0]*id]

        // K = P×Hᵀ×S⁻¹
        var PHt = [Double](repeating: 0, count: 8)
        for i in 0..<4 {
            for j in 0..<2 {
                for k in 0..<4 { PHt[i*2+j] += P[i*4+k] * H[j*4+k] }
            }
        }
        var K = [Double](repeating: 0, count: 8)
        for i in 0..<4 {
            for j in 0..<2 {
                for k in 0..<2 { K[i*2+j] += PHt[i*2+k] * Si[k*2+j] }
            }
        }
        for i in 0..<4 {
            x[i] += K[i*2+0]*y[0] + K[i*2+1]*y[1]
        }
        // ★ FIX: P = (I - K×H) × P  (原代码算的是 P - K×H, 速度列完全不更新)
        var KH = [Double](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                for k in 0..<2 { KH[i*4+j] += K[i*2+k] * H[k*4+j] }
            }
        }
        var IKH = [Double](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 { IKH[i*4+j] = (i==j ? 1.0 : 0.0) - KH[i*4+j] }
        }
        var PNew = [Double](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                for k in 0..<4 { PNew[i*4+j] += IKH[i*4+k] * P[k*4+j] }
            }
        }
        P = PNew; missCount = 0
    }

    var cx: Double { x[0] }
    var cy: Double { x[1] }
    var vx: Double { x[2] }
    var vy: Double { x[3] }
}

/// 轻量块匹配：前一帧中心块在当前帧中的位移（像素）。不调用 Vision，保证高帧率。
func blockMatchDisplacement(prev: CVPixelBuffer, curr: CVPixelBuffer,
                            centerXNorm: CGFloat, centerYNorm: CGFloat,
                            templateHalf: Int, searchRadius: Int, step: Int,
                            maxSADPerPixel: Int = 280) -> (dx: Int, dy: Int)? {
    let w = CVPixelBufferGetWidth(prev), h = CVPixelBufferGetHeight(prev)
    guard CVPixelBufferGetPixelFormatType(prev) == kCVPixelFormatType_32BGRA,
          CVPixelBufferGetWidth(curr) == w, CVPixelBufferGetHeight(curr) == h else { return nil }
    let cx = Int(centerXNorm * CGFloat(w)), cy = Int(centerYNorm * CGFloat(h))
    let t = templateHalf
    // ★ 核心修复：防止 px, py 读取越界导致 EXC_BAD_ACCESS 崩溃！
    guard cx - t >= 0, cx + t < w, cy - t >= 0, cy + t < h else {
        return nil
    }
    let dxMin = -min(searchRadius, cx - t)
    let dxMax = min(searchRadius, w - t - 1 - cx)
    let dyMin = -min(searchRadius, cy - t)
    let dyMax = min(searchRadius, h - t - 1 - cy)
    guard dxMin <= dxMax, dyMin <= dyMax else { return nil }

    CVPixelBufferLockBaseAddress(prev, .readOnly)
    CVPixelBufferLockBaseAddress(curr, .readOnly)
    defer {
        CVPixelBufferUnlockBaseAddress(prev, .readOnly)
        CVPixelBufferUnlockBaseAddress(curr, .readOnly)
    }
    guard let pBase = CVPixelBufferGetBaseAddress(prev),
          let cBase = CVPixelBufferGetBaseAddress(curr) else { return nil }
    let bpr = CVPixelBufferGetBytesPerRow(prev)
    let pp = pBase.assumingMemoryBound(to: UInt8.self)
    let cc = cBase.assumingMemoryBound(to: UInt8.self)

    let templatePixels = (2 * t) * (2 * t)
    var bestSAD = Int.max
    var bestDx = 0, bestDy = 0
    for dy in stride(from: dyMin, through: dyMax, by: step) {
        for dx in stride(from: dxMin, through: dxMax, by: step) {
            var sad = 0
            for oy in -t..<t {
                for ox in -t..<t {
                    let py = cy + oy
                    let px = cx + ox
                    let qy = py + dy
                    let qx = px + dx
                    let pOff = py * bpr + px * 4
                    let qOff = qy * bpr + qx * 4
                    // ★ FIX: 除以 1000 将亮度归回 [0,255] 尺度
                    let lp = (Int(pp[pOff+2]) * 299 + Int(pp[pOff+1]) * 587 + Int(pp[pOff]) * 114) / 1000
                    let lq = (Int(cc[qOff+2]) * 299 + Int(cc[qOff+1]) * 587 + Int(cc[qOff]) * 114) / 1000
                    sad += abs(lp - lq)
                }
            }
            if sad < bestSAD { bestSAD = sad; bestDx = dx; bestDy = dy }
        }
    }
    if bestSAD / templatePixels > maxSADPerPixel { return nil }
    return (bestDx, bestDy)
}


// ═══════════════════════════════════════════════════════════════════
// MARK: - 实验引擎
// ═══════════════════════════════════════════════════════════════════

final class VNLabEngine: ObservableObject {

    // ── UI 可观察状态 ──
    @Published var algorithm: VNLabAlgorithm = .trackObject
    @Published var frame: LabFrame?
    @Published var status: String = "请加载视频"
    @Published var isPlaying: Bool = false
    @Published var progress: Double = 0
    @Published var frameCount: Int = 0
    @Published var totalFrames: Int = 0
    @Published var videoSize: CGSize = .zero

    // ── ROI ──
    @Published var userROI: CGRect? = nil
    @Published var roiReady: Bool = false

    // ── 可调参数 ──
    @Published var flowGridStep: Int = 20
    @Published var flowMagnitudeScale: Float = 8.0
    @Published var flowAccuracy: Int = 0
    @Published var flowDownscale: Int = 1
    @Published var trajLength: Int = 15
    @Published var trajTargetFPS: Int = 30
    @Published var poseConfThreshold: Float = 0.3
    @Published var segQuality: Int = 1
    @Published var trackLevel: Int = 1

    // 卡尔曼参数
    @Published var kalmanTemplateHalf: Int = 14
    @Published var kalmanSearchRadius: Int = 36
    @Published var kalmanStep: Int = 2
    @Published var kalmanSADThreshold: Int = 280

    // IoU 跟踪器参数
    @Published var iouThreshold: Float = 0.20
    @Published var iouMaxMiss: Int = 8
    @Published var iouVelocityAlpha: Float = 0.3
    @Published var iouMinConfidence: Float = 0.25
    @Published var yoloModelLoaded: Bool = false
    @Published var yoloModelName: String = ""

    // ── 内部 ──
    private var videoURL: URL?
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var nominalFPS: Float = 30
    private var workItem: DispatchWorkItem?
    private let ciCtx = CIContext(options: [.useSoftwareRenderer: false])

    // ── Vision ──
    private var seqHandler: VNSequenceRequestHandler?
    private var trackRequest: VNTrackObjectRequest?
    private var rectTrackRequest: VNTrackRectangleRequest?
    private var prevPixelBuffer: CVPixelBuffer?
    private var trajectoryRequest: VNDetectTrajectoriesRequest?
    private var latestTrajectories: [VNTrajectoryObservation] = []

    // ── 卡尔曼追踪 ──
    private var kalmanTracker: KalmanTracker4D?
    private var kalmanBoxSize: (width: CGFloat, height: CGFloat)?

    // ── IoU 跟踪器 ──
    private var iouTracker = IoUGreedyTracker()
    private var yoloRequest: VNCoreMLRequest?
    /// ROI 模式下上一帧跟踪框（用于本帧块匹配中心），与卡尔曼/VNTracker 一样的「检测框」来源
    private var iouRoiLastRect: CGRect?

    private var fIdx: Int = 0
    private var isPaused: Bool = false

    // ═══ YOLO 模型加载 (IoU 跟踪器用) ═══

    /// 自动扫描 Bundle 中的 .mlmodelc 或 .mlpackage（优先 .mlmodelc，避免重复）
    func availableModels() -> [(name: String, url: URL)] {
        var seen = Set<String>()
        var result: [(String, URL)] = []
        let candidateNames = ["yolo11_640imgsz", "Detector", "vsim_yolo11n", "RC_YOLO11n_320", "yolo11n", "yolov8n", "best"]  // 默认优先真实视频用的 YOLO11-640，再回退到 Detector / vsim
        for name in candidateNames {
            if seen.contains(name) { continue }
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                seen.insert(name); result.append((name, url))
            } else if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
                seen.insert(name); result.append((name, url))
            }
        }
        return result
    }

    func loadYOLOModel(_ url: URL) {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let mlModel = try? MLModel(contentsOf: url, configuration: config),
              let vnModel = try? VNCoreMLModel(for: mlModel) else {
            setStatus("YOLO 模型加载失败")
            return
        }
        let req = VNCoreMLRequest(model: vnModel)
        req.imageCropAndScaleOption = .scaleFill
        yoloRequest = req
        DispatchQueue.main.async {
            self.yoloModelLoaded = true
            self.yoloModelName = url.lastPathComponent
        }
        // 预热
        DispatchQueue.global(qos: .utility).async {
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(nil, 320, 320, kCVPixelFormatType_32BGRA, nil, &pb)
            if let buf = pb {
                let handler = VNImageRequestHandler(cvPixelBuffer: buf, orientation: .up, options: [:])
                try? handler.perform([req])
            }
        }
        setStatus("YOLO 模型已加载: \(url.lastPathComponent)")
    }

    func loadYOLOModelByName(_ name: String) {
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            loadYOLOModel(url)
        } else if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            loadYOLOModel(url)
        }
    }

    /// 若当前为 IoU 且未加载 YOLO，则自动加载第一个可用模型（便于直接使用）
    func autoLoadYOLOIfNeeded() {
        guard algorithm.needsYOLO, !yoloModelLoaded else { return }
        let models = availableModels()
        if let first = models.first {
            loadYOLOModel(first.url)
        }
    }

    // ═══ 视频加载 ═══

    func loadVideo(_ url: URL) {
        stop()
        _ = url.startAccessingSecurityScopedResource()
        videoURL = url
        let asset = AVAsset(url: url)
        guard let vTrack = asset.tracks(withMediaType: .video).first else {
            DispatchQueue.main.async { self.status = "无法读取视频轨道" }
            return
        }
        let sz = vTrack.naturalSize.applying(vTrack.preferredTransform)
        let absW = abs(sz.width), absH = abs(sz.height)
        nominalFPS = vTrack.nominalFrameRate
        let total = Int(CMTimeGetSeconds(asset.duration) * Double(nominalFPS))

        DispatchQueue.main.async {
            self.videoSize = CGSize(width: absW, height: absH)
            self.totalFrames = total
            self.frameCount = 0; self.progress = 0
            self.userROI = nil; self.roiReady = false
            self.status = "\(url.lastPathComponent)  \(Int(absW))x\(Int(absH))  \(String(format:"%.0f", self.nominalFPS))fps  \(total)帧"
        }
    }

    // ═══ 播放控制 ═══

    func play() {
        guard videoURL != nil else { setStatus("请先加载视频"); return }
        if algorithm == .iouTracker && !roiReady && !yoloModelLoaded {
            setStatus("IoU 跟踪器：请先框选 ROI（检测框模式）或加载 YOLO 模型")
            return
        }
        if isPaused { isPaused = false; return }
        guard setupReader() else { return }

        isPlaying = true; isPaused = false
        let item = DispatchWorkItem { [weak self] in self?.runLoop() }
        workItem = item
        DispatchQueue.global(qos: .userInitiated).async(execute: item)
    }

    func pause()  { isPaused = true }
    func resume() { isPaused = false }

    func step() {
        if algorithm == .iouTracker && !roiReady && !yoloModelLoaded {
            setStatus("IoU 跟踪器：请先框选 ROI（检测框模式）或加载 YOLO 模型")
            return
        }
        if reader == nil || reader?.status != .reading {
            guard setupReader() else { return }
        }
        processOneFrame()
    }

    func stop() {
        workItem?.cancel(); workItem = nil
        isPlaying = false; isPaused = false
        reader?.cancelReading(); reader = nil; output = nil
        trajectoryRequest = nil; latestTrajectories = []
    }

    func confirmROI(_ rect: CGRect) {
        userROI = rect; roiReady = true
        setStatus("ROI 已设定 (\(String(format:"%.0f%%x%.0f%%", rect.width*100, rect.height*100))) — 点击 Play 开始追踪")
    }

    func clearROI() {
        userROI = nil; roiReady = false; trackRequest = nil
        kalmanTracker = nil; kalmanBoxSize = nil
        iouRoiLastRect = nil
    }

    // ═══ 内部 ═══

    private func setStatus(_ s: String) {
        DispatchQueue.main.async { self.status = s }
    }

    private func setupReader() -> Bool {
        guard let url = videoURL else { return false }
        let asset = AVAsset(url: url)
        guard let vTrack = asset.tracks(withMediaType: .video).first,
              let rd = try? AVAssetReader(asset: asset) else { return false }
        let settings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let out = AVAssetReaderTrackOutput(track: vTrack, outputSettings: settings)
        rd.add(out); rd.startReading()
        reader = rd; output = out
        seqHandler = VNSequenceRequestHandler()
        trackRequest = nil; rectTrackRequest = nil; prevPixelBuffer = nil
        kalmanTracker = nil; kalmanBoxSize = nil
        iouTracker.reset(); iouRoiLastRect = nil
        fIdx = 0; latestTrajectories = []
        if algorithm == .trajectory { setupTrajectoryRequest() }
        return true
    }

    private func runLoop() {
        while !(workItem?.isCancelled ?? true) {
            if isPaused { Thread.sleep(forTimeInterval: 0.05); continue }
            if !processOneFrame() { break }
            Thread.sleep(forTimeInterval: 0.008)
        }
        DispatchQueue.main.async { [weak self] in
            self?.isPlaying = false
            self?.status += "  Done"
        }
    }

    @discardableResult
    private func processOneFrame() -> Bool {
        guard let out = output, let sample = out.copyNextSampleBuffer(),
              let pb = CMSampleBufferGetImageBuffer(sample) else { return false }

        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        let time = CMTimeGetSeconds(pts)
        let t0 = CACurrentMediaTime()

        var result = LabFrame(index: fIdx, time: time, image: makeCGImage(pb))

        switch algorithm {
        case .trackObject:    runTrackObject(pb, &result)
        case .kalmanTrack:    runKalmanTrack(pb, &result)
        case .iouTracker:     runIoUTracker(pb, &result)
        case .trackRectangle: runTrackRectangle(pb, &result)
        case .opticalFlow:    runOpticalFlow(pb, &result)
        case .trajectory:     runTrajectory(pb, time, &result)
        case .bodyPose:       runBodyPose(pb, &result)
        case .personSegment:  runPersonSegment(pb, &result)
        }

        prevPixelBuffer = pb
        result.processMs = (CACurrentMediaTime() - t0) * 1000
        fIdx += 1

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.frame = result
            self.frameCount = self.fIdx
            if self.totalFrames > 0 { self.progress = Double(self.fIdx) / Double(self.totalFrames) }
        }
        return true
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ① VNTrackObjectRequest
    // ═══════════════════════════════════════════════════════════════

    private func runTrackObject(_ pb: CVPixelBuffer, _ r: inout LabFrame) {
        if trackRequest == nil, let roi = userROI {
            let vnRect = CGRect(x: roi.minX, y: 1.0 - roi.maxY, width: roi.width, height: roi.height)
            let obs = VNDetectedObjectObservation(boundingBox: vnRect)
            let req = VNTrackObjectRequest(detectedObjectObservation: obs)
            req.trackingLevel = trackLevel == 0 ? .fast : .accurate
            trackRequest = req
        }
        guard let req = trackRequest else { return }

        try? seqHandler?.perform([req], on: pb, orientation: .up)

        if let res = req.results?.first as? VNDetectedObjectObservation {
            let bb = res.boundingBox
            r.trackRect = CGRect(x: bb.minX, y: 1.0 - bb.maxY, width: bb.width, height: bb.height)
            r.trackConf = res.confidence
            r.trackAlive = res.confidence > 0.3
            // ★ 核心修复：不要重新创建 Request！保留底层 DCF 滤波器的时序记忆
            // ★ 改进：丢失后不立即放弃，允许 VN 框架自行恢复
            if res.confidence < 0.1 {
                // 只有置信度极低时才放弃
                trackRequest = nil
            }
        } else {
            r.trackAlive = false
            trackRequest = nil
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ①' 卡尔曼追踪
    // ═══════════════════════════════════════════════════════════════

    private func runKalmanTrack(_ pb: CVPixelBuffer, _ r: inout LabFrame) {
        let dt = 1.0 / Double(max(nominalFPS, 1))
        let w = CGFloat(CVPixelBufferGetWidth(pb)), h = CGFloat(CVPixelBufferGetHeight(pb))

        if kalmanTracker == nil, let roi = userROI {
            let cx = roi.midX
            let cy = roi.midY
            kalmanTracker = KalmanTracker4D(cx: Double(cx), cy: Double(cy), dt: dt)
            kalmanBoxSize = (roi.width, roi.height)
        }
        guard var tracker = kalmanTracker, let boxSize = kalmanBoxSize else { return }

        let oldCx = tracker.cx
        let oldCy = tracker.cy

        tracker.predict()

        var hasObservation = false
        if let prev = prevPixelBuffer, w > 0, h > 0 {
            let th = max(4, min(24, kalmanTemplateHalf))
            let sr = max(8, min(64, kalmanSearchRadius))
            let st = max(1, min(4, kalmanStep))
            let sadLimit = max(50, min(500, kalmanSADThreshold))
            if let disp = blockMatchDisplacement(prev: prev, curr: pb,
                                                 centerXNorm: CGFloat(oldCx), centerYNorm: CGFloat(oldCy),
                                                 templateHalf: th, searchRadius: sr, step: st,
                                                 maxSADPerPixel: sadLimit) {
                let zCx = oldCx + Double(disp.dx) / Double(w)
                let zCy = oldCy + Double(disp.dy) / Double(h)
                tracker.update(zCx: zCx, zCy: zCy)
                hasObservation = true
            }
        }
        kalmanTracker = tracker

        let cx = CGFloat(tracker.cx), cy = CGFloat(tracker.cy)
        let nw = boxSize.width, nh = boxSize.height
        r.trackRect = CGRect(x: cx - nw/2, y: cy - nh/2, width: nw, height: nh)

        let conf: Float
        if hasObservation {
            conf = 1.0
        } else if tracker.missCount < 5 {
            conf = max(0.3, 1.0 - Float(tracker.missCount) * 0.15)
        } else {
            conf = 0.1
        }
        r.trackConf = conf
        r.trackAlive = conf > 0.2 && tracker.uncertainty < 0.3
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ★ IoU 贪心匹配跟踪器 (SORT)
    // ═══════════════════════════════════════════════════════════════

    private func runIoUTracker(_ pb: CVPixelBuffer, _ r: inout LabFrame) {
        // 同步参数
        iouTracker.iouThreshold = iouThreshold
        iouTracker.maxMiss = iouMaxMiss
        iouTracker.velocityAlpha = CGFloat(iouVelocityAlpha)
        iouTracker.minConfidence = iouMinConfidence

        var detections: [(rect: CGRect, confidence: Float, label: String)] = []
        var yoloMs: Double = 0

        // ── 与 VNTracker/卡尔曼一样：用「检测框」模式（用户框选 ROI，块匹配得到每帧检测框）
        if let roi = userROI {
            let w = roi.width, h = roi.height
            if prevPixelBuffer == nil {
                // 首帧：用户框即为检测框，创建第一个 track
                detections = [(roi, 1.0, "ROI")]
            } else {
                // 后续帧：块匹配得到位移 → 当前帧检测框（与卡尔曼同源）
                let centerRect = iouRoiLastRect ?? roi
                let cxNorm = centerRect.midX
                let cyNorm = centerRect.midY
                let pw = CGFloat(CVPixelBufferGetWidth(pb)), ph = CGFloat(CVPixelBufferGetHeight(pb))
                let th = max(4, min(24, kalmanTemplateHalf))
                let sr = max(8, min(64, kalmanSearchRadius))
                let st = max(1, min(4, kalmanStep))
                let sadLimit = max(50, min(500, kalmanSADThreshold))
                if let disp = blockMatchDisplacement(prev: prevPixelBuffer!, curr: pb,
                                                     centerXNorm: cxNorm, centerYNorm: cyNorm,
                                                     templateHalf: th, searchRadius: sr, step: st,
                                                     maxSADPerPixel: sadLimit) {
                    let newCx = cxNorm + CGFloat(disp.dx) / pw
                    let newCy = cyNorm + CGFloat(disp.dy) / ph
                    let box = CGRect(x: newCx - w/2, y: newCy - h/2, width: w, height: h)
                    detections = [(box, 1.0, "ROI")]
                }
                // 无块匹配结果时仍用预测（IoU 内部会 missCount++）
            }
        } else {
            // 无 ROI：纯 YOLO 检测框
            guard let yoloReq = yoloRequest else { return }
            let t0 = CACurrentMediaTime()
            let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
            try? handler.perform([yoloReq])
            yoloMs = (CACurrentMediaTime() - t0) * 1000
            if let results = yoloReq.results as? [VNRecognizedObjectObservation] {
                for obs in results {
                    guard let lbl = obs.labels.first, lbl.confidence >= iouMinConfidence else { continue }
                    let bb = obs.boundingBox
                    let rectTL = CGRect(x: bb.minX, y: 1.0 - bb.maxY, width: bb.width, height: bb.height)
                    detections.append((rectTL, lbl.confidence, lbl.identifier))
                }
            }
        }

        let tracked = iouTracker.update(detections: detections)
        if userROI != nil {
            iouRoiLastRect = tracked.first?.rect ?? iouRoiLastRect ?? userROI
        }

        r.trackedTargets = tracked
        r.yoloDetectionCount = detections.count
        r.yoloMs = yoloMs

        if let primary = tracked.max(by: { $0.age < $1.age }) {
            r.trackRect = primary.rect
            r.trackConf = primary.confidence
            r.trackAlive = primary.missCount == 0
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ② VNTrackRectangleRequest
    // ═══════════════════════════════════════════════════════════════

    private func runTrackRectangle(_ pb: CVPixelBuffer, _ r: inout LabFrame) {
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)

        if rectTrackRequest == nil {
            // 首帧: 检测矩形
            let detectReq = VNDetectRectanglesRequest()
            detectReq.minimumConfidence = 0.5
            detectReq.maximumObservations = 5
            try? handler.perform([detectReq])
            if let results = detectReq.results {
                r.rectangles = results.map { obs in
                    let bb = obs.boundingBox
                    return CGRect(x: bb.minX, y: 1.0 - bb.maxY, width: bb.width, height: bb.height)
                }
                // 选最大的矩形开始追踪
                if let best = results.max(by: { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }) {
                    let req = VNTrackRectangleRequest(rectangleObservation: best)
                    req.trackingLevel = .accurate
                    rectTrackRequest = req
                }
            }
        } else if let req = rectTrackRequest {
            try? seqHandler?.perform([req], on: pb, orientation: .up)
            if let res = req.results?.first as? VNRectangleObservation {
                let tl = CGPoint(x: res.topLeft.x, y: 1.0 - res.topLeft.y)
                let tr = CGPoint(x: res.topRight.x, y: 1.0 - res.topRight.y)
                let bl = CGPoint(x: res.bottomLeft.x, y: 1.0 - res.bottomLeft.y)
                let br = CGPoint(x: res.bottomRight.x, y: 1.0 - res.bottomRight.y)
                r.trackedRectCorners = [tl, tr, br, bl]
                r.trackConf = res.confidence
                r.trackAlive = res.confidence > 0.3
                if res.confidence < 0.1 {
                    rectTrackRequest = nil
                }
            } else {
                r.trackAlive = false
                rectTrackRequest = nil
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ③ OpticalFlow
    // ═══════════════════════════════════════════════════════════════

    private func runOpticalFlow(_ pb: CVPixelBuffer, _ r: inout LabFrame) {
        guard let prev = prevPixelBuffer else { return }

        let usePrev: CVPixelBuffer
        let useCurr: CVPixelBuffer
        if flowDownscale > 1 {
            usePrev = downsamplePB(prev, factor: flowDownscale) ?? prev
            useCurr = downsamplePB(pb, factor: flowDownscale) ?? pb
        } else {
            usePrev = prev; useCurr = pb
        }

        let req = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: useCurr, options: [:])
        switch flowAccuracy {
        case 0: req.computationAccuracy = .low
        case 2: req.computationAccuracy = .high
        default: req.computationAccuracy = .medium
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: usePrev, orientation: .up)
        try? handler.perform([req])

        guard let obs = req.results?.first as? VNPixelBufferObservation else { return }
        let flowPB = obs.pixelBuffer
        let fw = CVPixelBufferGetWidth(flowPB)
        let fh = CVPixelBufferGetHeight(flowPB)

        CVPixelBufferLockBaseAddress(flowPB, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowPB, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(flowPB) else { return }
        let floatPtr = base.assumingMemoryBound(to: Float32.self)
        let bpr = CVPixelBufferGetBytesPerRow(flowPB)
        let floatsPerRow = bpr / MemoryLayout<Float32>.size

        let step = max(1, flowGridStep)
        var arrows: [FlowArrow] = []
        for y in stride(from: step/2, to: fh, by: step) {
            for x in stride(from: step/2, to: fw, by: step) {
                let offset = y * floatsPerRow + x * 2
                let dx = floatPtr[offset]
                let dy = floatPtr[offset + 1]
                let mag = sqrt(dx * dx + dy * dy)
                if mag > 0.3 {
                    arrows.append(FlowArrow(
                        origin: CGPoint(x: CGFloat(x) / CGFloat(fw), y: CGFloat(y) / CGFloat(fh)),
                        dx: dx * flowMagnitudeScale,
                        dy: -dy * flowMagnitudeScale  // Vision Y 向上, 屏幕 Y 向下
                    ))
                }
            }
        }
        r.flowArrows = arrows
        r.flowImage = makeFlowHSVImage(flowPB, fw: fw, fh: fh)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ④ Trajectory
    // ═══════════════════════════════════════════════════════════════

    private func setupTrajectoryRequest() {
        let fps = max(1, trajTargetFPS)
        let len = max(5, trajLength)
        trajectoryRequest = VNDetectTrajectoriesRequest(frameAnalysisSpacing: CMTime(value: 1, timescale: CMTimeScale(fps)),
                                                         trajectoryLength: len) { [weak self] req, err in
            guard let results = (req as? VNDetectTrajectoriesRequest)?.results else { return }
            self?.latestTrajectories = results
        }
        trajectoryRequest?.objectMinimumNormalizedRadius = 0.01
        trajectoryRequest?.objectMaximumNormalizedRadius = 0.15
    }

    private func runTrajectory(_ pb: CVPixelBuffer, _ time: Double, _ r: inout LabFrame) {
        if trajectoryRequest == nil { setupTrajectoryRequest() }
        guard let req = trajectoryRequest else { return }

        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
        try? handler.perform([req])

        r.trajectories = latestTrajectories.map { obs in
            let pts = obs.detectedPoints.map { p in
                CGPoint(x: p.location.x, y: 1.0 - p.location.y)
            }
            let coeffs = obs.equationCoefficients
            let eq: (a: Float, b: Float, c: Float) = (Float(coeffs.x), Float(coeffs.y), Float(coeffs.z))
            return TrajectoryResult(
                points: pts,
                equation: eq,
                movingAverageRadius: CGFloat(obs.movingAverageRadius)
            )
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ⑤ BodyPose
    // ═══════════════════════════════════════════════════════════════

    private func runBodyPose(_ pb: CVPixelBuffer, _ r: inout LabFrame) {
        let req = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
        try? handler.perform([req])

        guard let obs = req.results?.first else { return }

        let threshold = poseConfThreshold
        var joints: [PoseJoint] = []
        let allKeys = obs.availableJointNames
        for key in allKeys {
            if let pt = try? obs.recognizedPoint(key), pt.confidence > threshold {
                let keyStr = String(describing: key)
                joints.append(PoseJoint(
                    name: keyStr.replacingOccurrences(of: "human_body_pose_", with: ""),
                    position: CGPoint(x: pt.x, y: 1.0 - pt.y),
                    confidence: pt.confidence
                ))
            }
        }
        r.joints = joints

        // 骨骼连线
        let bonePairs: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
            (.nose, .neck), (.neck, .leftShoulder), (.neck, .rightShoulder),
            (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
            (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
            (.neck, .root), (.root, .leftHip), (.root, .rightHip),
            (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
            (.rightHip, .rightKnee), (.rightKnee, .rightAnkle)
        ]
        var bones: [PoseBone] = []
        for (a, b) in bonePairs {
            if let pa = try? obs.recognizedPoint(a), let pb = try? obs.recognizedPoint(b),
               pa.confidence > threshold, pb.confidence > threshold {
                bones.append(PoseBone(
                    from: CGPoint(x: pa.x, y: 1.0 - pa.y),
                    to: CGPoint(x: pb.x, y: 1.0 - pb.y),
                    confidence: min(pa.confidence, pb.confidence)
                ))
            }
        }
        r.bones = bones
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ⑥ PersonSegment
    // ═══════════════════════════════════════════════════════════════

    private func runPersonSegment(_ pb: CVPixelBuffer, _ r: inout LabFrame) {
        let req = VNGeneratePersonSegmentationRequest()
        switch segQuality {
        case 0: req.qualityLevel = .fast
        case 2: req.qualityLevel = .accurate
        default: req.qualityLevel = .balanced
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
        try? handler.perform([req])

        if let obs = req.results?.first {
            let maskPB = obs.pixelBuffer
            r.segMask = makeCGImage(maskPB)
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - 辅助方法
    // ═══════════════════════════════════════════════════════════════

    func makeCGImage(_ pb: CVPixelBuffer) -> CGImage? {
        ciCtx.createCGImage(CIImage(cvPixelBuffer: pb), from: CIImage(cvPixelBuffer: pb).extent)
    }

    private func downsamplePB(_ pb: CVPixelBuffer, factor: Int) -> CVPixelBuffer? {
        let srcW = CVPixelBufferGetWidth(pb)
        let srcH = CVPixelBufferGetHeight(pb)
        let dstW = srcW / factor
        let dstH = srcH / factor
        guard dstW > 0, dstH > 0 else { return nil }

        var dstPB: CVPixelBuffer?
        CVPixelBufferCreate(nil, dstW, dstH, kCVPixelFormatType_32BGRA, nil, &dstPB)
        guard let dst = dstPB else { return nil }

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            CVPixelBufferUnlockBaseAddress(dst, [])
        }

        var srcBuf = vImage_Buffer(
            data: CVPixelBufferGetBaseAddress(pb),
            height: vImagePixelCount(srcH),
            width: vImagePixelCount(srcW),
            rowBytes: CVPixelBufferGetBytesPerRow(pb)
        )
        var dstBuf = vImage_Buffer(
            data: CVPixelBufferGetBaseAddress(dst),
            height: vImagePixelCount(dstH),
            width: vImagePixelCount(dstW),
            rowBytes: CVPixelBufferGetBytesPerRow(dst)
        )
        vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
        return dst
    }

    private func makeFlowHSVImage(_ flowPB: CVPixelBuffer, fw: Int, fh: Int) -> CGImage? {
        CVPixelBufferLockBaseAddress(flowPB, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowPB, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(flowPB) else { return nil }
        let fp = base.assumingMemoryBound(to: Float32.self)
        let bpr = CVPixelBufferGetBytesPerRow(flowPB)
        let fpr = bpr / MemoryLayout<Float32>.size

        var pixels = [UInt8](repeating: 255, count: fw * fh * 4)
        for y in 0..<fh {
            for x in 0..<fw {
                let off = y * fpr + x * 2
                let dx = fp[off], dy = fp[off + 1]
                let mag = sqrt(dx * dx + dy * dy)
                let angle = atan2(dy, dx)
                let h = (angle + .pi) / (2 * .pi)
                let s: Float = 1.0
                let v = min(mag * 4, 1.0)
                let (r, g, b) = hsvToRGB(h: h, s: s, v: v)
                let idx = (y * fw + x) * 4
                pixels[idx] = UInt8(r * 255)
                pixels[idx+1] = UInt8(g * 255)
                pixels[idx+2] = UInt8(b * 255)
                pixels[idx+3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: fw, height: fh, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: fw * 4, space: colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }

    private func hsvToRGB(h: Float, s: Float, v: Float) -> (Float, Float, Float) {
        let i = Int(h * 6) % 6
        let f = h * 6 - Float(Int(h * 6))
        let p = v * (1 - s)
        let q = v * (1 - f * s)
        let t = v * (1 - (1 - f) * s)
        switch i {
        case 0: return (v, t, p)
        case 1: return (q, v, p)
        case 2: return (p, v, t)
        case 3: return (p, q, v)
        case 4: return (t, p, v)
        default: return (v, p, q)
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// MARK: - SwiftUI 主视图
// ═══════════════════════════════════════════════════════════════════

struct VNLabView: View {
    @StateObject private var engine = VNLabEngine()
    @State private var showFileImporter = false
    @State private var showModelImporter = false

    var body: some View {
        HSplitView {
            // ── 左侧: 视频画面 ──
            VStack(spacing: 0) {
                // 标题栏
                HStack {
                    Text("Vision Tracker Lab")
                        .font(.headline)
                    Spacer()
                    Text(engine.status)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                .padding(.horizontal, 12).padding(.vertical, 6)
                .background(Color(nsColor: .controlBackgroundColor))

                // 视频画面 + 叠加层
                ZStack {
                    Color.black
                    if let f = engine.frame, let img = f.image {
                        GeometryReader { geo in
                            let asp = CGFloat(img.width) / CGFloat(img.height)
                            let (fitW, fitH, offX, offY) = aspectFit(sz: geo.size, asp: asp)

                            StaticImageView(cgImage: img)
                                .frame(width: fitW, height: fitH)
                                .offset(x: offX, y: offY)

                            // 叠加绘制层
                            LabOverlayView(frame: f, algorithm: engine.algorithm, fitRect: CGRect(x: offX, y: offY, width: fitW, height: fitH))
                                .frame(width: geo.size.width, height: geo.size.height)
                                .allowsHitTesting(false)

                            // 框选 ROI
                            if engine.algorithm.needsROI && !engine.roiReady {
                                ROISelectionOverlay(fitRect: CGRect(x: offX, y: offY, width: fitW, height: fitH)) { normRect in
                                    engine.confirmROI(normRect)
                                }
                                .frame(width: geo.size.width, height: geo.size.height)
                            }
                        }
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "film")
                                .font(.system(size: 48))
                                .foregroundColor(.gray)
                            Text("拖入视频文件或点击下方加载")
                                .foregroundColor(.gray)
                        }
                    }
                }
                .aspectRatio(16.0/9.0, contentMode: .fit)
                .clipped()

                // 进度条
                ProgressView(value: engine.progress)
                    .progressViewStyle(.linear)
                    .padding(.horizontal, 12)

                // 控制栏
                HStack(spacing: 12) {
                    Button(action: { showFileImporter = true }) {
                        Label("视频", systemImage: "folder.badge.plus")
                    }

                    Divider().frame(height: 20)

                    Button(action: { engine.play() }) {
                        Label("Play", systemImage: "play.fill")
                    }.disabled(engine.isPlaying)

                    Button(action: { engine.pause() }) {
                        Label("Pause", systemImage: "pause.fill")
                    }

                    Button(action: { engine.step() }) {
                        Label("Step", systemImage: "forward.frame.fill")
                    }

                    Button(action: { engine.stop() }) {
                        Label("Stop", systemImage: "stop.fill")
                    }

                    if engine.algorithm.needsROI {
                        Divider().frame(height: 20)
                        Button("清除 ROI") { engine.clearROI() }
                            .disabled(!engine.roiReady)
                    }

                    Spacer()

                    if let f = engine.frame {
                        Text("帧 \(f.index)  \(String(format: "%.1fms", f.processMs))")
                            .font(.caption).monospacedDigit()
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(Color(nsColor: .controlBackgroundColor))
            }
            .frame(minWidth: 500)

            // ── 右侧: 参数面板 ──
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {

                    // 算法选择
                    VStack(alignment: .leading, spacing: 6) {
                        Text("算法").font(.headline)
                        ForEach(VNLabAlgorithm.allCases) { algo in
                            Button(action: {
                                engine.algorithm = algo
                                engine.stop()
                                if !algo.needsROI { engine.clearROI() }
                            }) {
                                HStack {
                                    Image(systemName: algo.icon)
                                        .frame(width: 20)
                                    Text(algo.rawValue)
                                        .font(.system(.body, design: .monospaced))
                                    Spacer()
                                    if engine.algorithm == algo {
                                        Image(systemName: "checkmark")
                                            .foregroundColor(.accentColor)
                                    }
                                }
                                .padding(.vertical, 3)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    Divider()

                    // 算法特定参数
                    algorithmParams

                    Divider()

                    // 指标显示
                    metricsView

                    Divider()

                    // Tips
                    tipsView

                }
                .padding(16)
            }
            .frame(width: 320)
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .fileImporter(isPresented: $showFileImporter,
                      allowedContentTypes: [.movie, .video, .mpeg4Movie, .quickTimeMovie]) { result in
            if case .success(let url) = result {
                _ = url.startAccessingSecurityScopedResource()
                engine.loadVideo(url)
            }
        }
        .fileImporter(isPresented: $showModelImporter,
                      allowedContentTypes: [UTType(filenameExtension: "mlmodelc") ?? .data, UTType(filenameExtension: "mlpackage") ?? .data]) { result in
            if case .success(let url) = result {
                _ = url.startAccessingSecurityScopedResource()
                engine.loadYOLOModel(url)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers {
                p.loadItem(forTypeIdentifier: "public.file-url", options: nil) { data, _ in
                    guard let d = data as? Data, let url = URL(dataRepresentation: d, relativeTo: nil) else { return }
                    _ = url.startAccessingSecurityScopedResource()
                    let ext = url.pathExtension.lowercased()
                    if ["mp4", "mov", "m4v", "avi"].contains(ext) {
                        DispatchQueue.main.async { engine.loadVideo(url) }
                    } else if ext == "mlmodelc" || ext == "mlpackage" {
                        DispatchQueue.main.async { engine.loadYOLOModel(url) }
                    }
                }
            }
            return true
        }
    }

    // ── 算法参数面板 ──
    @ViewBuilder
    private var algorithmParams: some View {
        switch engine.algorithm {
        case .trackObject:
            VStack(alignment: .leading, spacing: 8) {
                Text("VNTrackObject 参数").font(.subheadline).bold()
                Picker("精度", selection: $engine.trackLevel) {
                    Text("Fast").tag(0)
                    Text("Accurate").tag(1)
                }.pickerStyle(.segmented)
                Text("在画面上拖动框选目标区域").font(.caption).foregroundColor(.secondary)
            }

        case .kalmanTrack:
            VStack(alignment: .leading, spacing: 8) {
                Text("卡尔曼追踪参数").font(.subheadline).bold()
                LabeledSlider(label: "模板半宽", value: $engine.kalmanTemplateHalf, range: 4...24, step: 2)
                LabeledSlider(label: "搜索半径", value: $engine.kalmanSearchRadius, range: 8...64, step: 4)
                LabeledSlider(label: "搜索步长", value: $engine.kalmanStep, range: 1...4, step: 1)
                LabeledSlider(label: "SAD 阈值", value: $engine.kalmanSADThreshold, range: 50...500, step: 10)
                Text("在画面上拖动框选目标区域").font(.caption).foregroundColor(.secondary)
            }

        case .iouTracker:
            VStack(alignment: .leading, spacing: 8) {
                Text("IoU 贪心匹配参数").font(.subheadline).bold()

                // YOLO 模型（切换到 IoU 时若未加载会自动尝试加载第一个内置模型）
                HStack {
                    if engine.yoloModelLoaded {
                        Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
                        Text(engine.yoloModelName).font(.caption)
                    } else {
                        Image(systemName: "exclamationmark.triangle").foregroundColor(.orange)
                        Text("需要 YOLO 模型").font(.caption)
                    }
                    Spacer()
                    Button("加载模型") { showModelImporter = true }
                        .font(.caption)
                }
                .onAppear { engine.autoLoadYOLOIfNeeded() }

                // 自动扫描 Bundle 中的模型（.mlmodelc 或 .mlpackage）
                let models = engine.availableModels()
                if !models.isEmpty {
                    HStack {
                        Text("内置模型:").font(.caption).foregroundColor(.secondary)
                        ForEach(models, id: \.name) { m in
                            Button(m.name) { engine.loadYOLOModel(m.url) }
                                .font(.caption)
                        }
                    }
                }

                LabeledFloatSlider(label: "IoU 阈值", value: $engine.iouThreshold, range: 0.05...0.5, step: 0.05)
                LabeledSlider(label: "最大丢失帧", value: $engine.iouMaxMiss, range: 1...30, step: 1)
                LabeledFloatSlider(label: "速度平滑 α", value: $engine.iouVelocityAlpha, range: 0.1...0.9, step: 0.05)
                LabeledFloatSlider(label: "最低置信度", value: $engine.iouMinConfidence, range: 0.1...0.8, step: 0.05)
            }

        case .trackRectangle:
            Text("自动检测并追踪画面中的矩形").font(.caption).foregroundColor(.secondary)

        case .opticalFlow:
            VStack(alignment: .leading, spacing: 8) {
                Text("光流参数").font(.subheadline).bold()
                LabeledSlider(label: "网格步长", value: $engine.flowGridStep, range: 5...50, step: 5)
                LabeledFloatSlider(label: "箭头缩放", value: $engine.flowMagnitudeScale, range: 1...20, step: 1)
                Picker("精度", selection: $engine.flowAccuracy) {
                    Text("Low").tag(0); Text("Medium").tag(1); Text("High").tag(2)
                }.pickerStyle(.segmented)
                Picker("缩放", selection: $engine.flowDownscale) {
                    Text("1x").tag(1); Text("1/2").tag(2); Text("1/4").tag(4)
                }.pickerStyle(.segmented)
            }

        case .trajectory:
            VStack(alignment: .leading, spacing: 8) {
                Text("弹道检测参数").font(.subheadline).bold()
                LabeledSlider(label: "轨迹长度", value: $engine.trajLength, range: 5...30, step: 1)
                LabeledSlider(label: "目标 FPS", value: $engine.trajTargetFPS, range: 10...60, step: 5)
            }

        case .bodyPose:
            VStack(alignment: .leading, spacing: 8) {
                Text("人体姿态参数").font(.subheadline).bold()
                LabeledFloatSlider(label: "置信度阈值", value: $engine.poseConfThreshold, range: 0.1...0.9, step: 0.05)
            }

        case .personSegment:
            VStack(alignment: .leading, spacing: 8) {
                Text("人物分割参数").font(.subheadline).bold()
                Picker("质量", selection: $engine.segQuality) {
                    Text("Fast").tag(0); Text("Balanced").tag(1); Text("Accurate").tag(2)
                }.pickerStyle(.segmented)
            }
        }
    }

    // ── 指标显示 ──
    @ViewBuilder
    private var metricsView: some View {
        if let f = engine.frame {
            VStack(alignment: .leading, spacing: 4) {
                Text("指标").font(.subheadline).bold()

                HStack {
                    MetricBadge(label: "帧", value: "\(f.index)")
                    MetricBadge(label: "耗时", value: String(format: "%.1fms", f.processMs))
                }

                if engine.algorithm == .trackObject || engine.algorithm == .kalmanTrack {
                    HStack {
                        MetricBadge(label: "置信度", value: String(format: "%.0f%%", f.trackConf * 100))
                        MetricBadge(label: "状态", value: f.trackAlive ? "追踪中" : "丢失")
                    }
                }

                if engine.algorithm == .iouTracker {
                    HStack {
                        MetricBadge(label: "YOLO", value: String(format: "%.1fms", f.yoloMs))
                        MetricBadge(label: "检测", value: "\(f.yoloDetectionCount)")
                    }
                    HStack {
                        MetricBadge(label: "跟踪", value: "\(f.trackedTargets.count)")
                        MetricBadge(label: "活跃", value: "\(f.trackedTargets.filter { $0.missCount == 0 }.count)")
                    }
                }

                if engine.algorithm == .opticalFlow {
                    MetricBadge(label: "箭头数", value: "\(f.flowArrows.count)")
                }

                if engine.algorithm == .trajectory {
                    MetricBadge(label: "轨迹数", value: "\(f.trajectories.count)")
                }

                if engine.algorithm == .bodyPose {
                    HStack {
                        MetricBadge(label: "关节", value: "\(f.joints.count)")
                        MetricBadge(label: "骨骼", value: "\(f.bones.count)")
                    }
                }
            }
        }
    }

    // ── Tips ──
    @ViewBuilder
    private var tipsView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Tips").font(.subheadline).bold()
            switch engine.algorithm {
            case .trackObject:
                tipText("VNTrackObjectRequest 使用 DCF (判别相关滤波器) 进行单目标追踪。框选目标后，Vision 框架会在后续帧中自动跟踪该目标。")
                tipText("置信度低于 0.1 时跟踪器会被重置。建议选择纹理丰富、对比度高的目标区域。")
            case .kalmanTrack:
                tipText("卡尔曼滤波器使用恒定速度模型预测目标位置，块匹配提供观测更新。不依赖 Vision 框架，帧率极高。")
                tipText("SAD 阈值控制块匹配的接受条件：过低会频繁丢失观测（纯预测），过高会接受错误匹配。")
            case .iouTracker:
                tipText("SORT 风格跟踪器：可用「检测框」或 YOLO 检测。框选 ROI 时与 VNTrackObject/卡尔曼一样，用块匹配得到每帧检测框，再 IoU 匹配；不框选则用 YOLO 多目标。")
                tipText("IoU 阈值：低→容易匹配（可能错配），高→严格匹配（可能频繁丢失）。")
            case .trackRectangle:
                tipText("自动检测画面中的矩形（如文档、屏幕、标牌），然后追踪其四个角点。")
            case .opticalFlow:
                tipText("光流法计算相邻两帧之间每个像素的运动矢量。箭头方向和长度表示运动方向和速度。")
                tipText("缩放因子可大幅提速：1/4 缩放下速度提升约 16 倍。")
            case .trajectory:
                tipText("VNDetectTrajectoriesRequest 检测画面中的抛物线运动轨迹，适合球类追踪。")
            case .bodyPose:
                tipText("VNDetectHumanBodyPoseRequest 检测人体 19 个关键点，可用于姿态分析。")
            case .personSegment:
                tipText("VNGeneratePersonSegmentationRequest 生成人物分割蒙版，可用于背景替换。")
            }
        }
    }

    private func tipText(_ text: String) -> some View {
        Text(text).font(.caption).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
    }

    private func aspectFit(sz: CGSize, asp: CGFloat) -> (CGFloat, CGFloat, CGFloat, CGFloat) {
        let va = sz.width / sz.height
        if asp > va {
            return (sz.width, sz.width / asp, 0, (sz.height - sz.width / asp) / 2)
        } else {
            return (sz.height * asp, sz.height, (sz.width - sz.height * asp) / 2, 0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - 叠加绘制层
// ═══════════════════════════════════════════════════════════════════

struct LabOverlayView: View {
    let frame: LabFrame
    let algorithm: VNLabAlgorithm
    let fitRect: CGRect

    var body: some View {
        Canvas { ctx, size in
            let r = fitRect

            switch algorithm {
            case .trackObject, .kalmanTrack:
                drawSingleTrack(ctx: ctx, r: r)

            case .iouTracker:
                drawIoUTracks(ctx: ctx, r: r)

            case .trackRectangle:
                drawRectangle(ctx: ctx, r: r)

            case .opticalFlow:
                drawFlow(ctx: ctx, r: r)

            case .trajectory:
                drawTrajectories(ctx: ctx, r: r)

            case .bodyPose:
                drawPose(ctx: ctx, r: r)

            case .personSegment:
                break // 分割蒙版直接叠加在图像上
            }
        }
    }

    private func drawSingleTrack(ctx: GraphicsContext, r: CGRect) {
        guard let rect = frame.trackRect else { return }
        let screenRect = CGRect(
            x: r.minX + rect.minX * r.width,
            y: r.minY + rect.minY * r.height,
            width: rect.width * r.width,
            height: rect.height * r.height
        )
        let color: Color = frame.trackAlive ? .green : .red
        let path = Path(screenRect)
        ctx.stroke(path, with: .color(color), lineWidth: 2)

        // 置信度标签
        let label = String(format: "%.0f%%", frame.trackConf * 100)
        ctx.draw(Text(label).font(.caption).foregroundColor(color),
                 at: CGPoint(x: screenRect.minX + 4, y: screenRect.minY - 10),
                 anchor: .topLeading)
    }

    private func drawIoUTracks(ctx: GraphicsContext, r: CGRect) {
        let colors: [Color] = [.green, .cyan, .yellow, .orange, .pink, .purple, .mint, .teal]

        for target in frame.trackedTargets {
            let rect = target.rect
            let screenRect = CGRect(
                x: r.minX + rect.minX * r.width,
                y: r.minY + rect.minY * r.height,
                width: rect.width * r.width,
                height: rect.height * r.height
            )
            let color = colors[target.id % colors.count]
            let lineWidth: CGFloat = target.isYoloFrame ? 2.5 : 1.5
            let path = Path(screenRect)

            if target.missCount > 0 {
                // 预测帧: 虚线
                ctx.stroke(path, with: .color(color.opacity(0.6)),
                           style: StrokeStyle(lineWidth: lineWidth, dash: [6, 4]))
            } else {
                ctx.stroke(path, with: .color(color), lineWidth: lineWidth)
            }

            // 标签：[D]=本帧有检测匹配（YOLO 或块匹配），[P]=纯预测
            let label = "#\(target.id) \(String(format: "%.0f%%", target.confidence * 100)) \(target.label)"
            let suffix = target.isYoloFrame ? " [D]" : " [P]"
            ctx.draw(Text(label + suffix).font(.system(size: 10, design: .monospaced)).foregroundColor(color),
                     at: CGPoint(x: screenRect.minX, y: screenRect.minY - 2),
                     anchor: .bottomLeading)

            // 速度箭头
            if target.velocity.x != 0 || target.velocity.y != 0 {
                let cx = screenRect.midX
                let cy = screenRect.midY
                let vx = target.velocity.x * r.width * 5
                let vy = target.velocity.y * r.height * 5
                var arrow = Path()
                arrow.move(to: CGPoint(x: cx, y: cy))
                arrow.addLine(to: CGPoint(x: cx + vx, y: cy + vy))
                ctx.stroke(arrow, with: .color(color.opacity(0.7)), lineWidth: 1.5)
            }
        }

        // YOLO 检测数量
        let info = "YOLO: \(frame.yoloDetectionCount) det  \(String(format: "%.1fms", frame.yoloMs))  Tracks: \(frame.trackedTargets.count)"
        ctx.draw(Text(info).font(.system(size: 10, design: .monospaced)).foregroundColor(.white),
                 at: CGPoint(x: r.minX + 4, y: r.maxY - 4),
                 anchor: .bottomLeading)
    }

    private func drawRectangle(ctx: GraphicsContext, r: CGRect) {
        // 初始检测的矩形
        for rect in frame.rectangles {
            let screenRect = CGRect(
                x: r.minX + rect.minX * r.width,
                y: r.minY + rect.minY * r.height,
                width: rect.width * r.width,
                height: rect.height * r.height
            )
            ctx.stroke(Path(screenRect), with: .color(.blue.opacity(0.5)),
                       style: StrokeStyle(lineWidth: 1, dash: [4, 2]))
        }

        // 追踪的四角点
        if let corners = frame.trackedRectCorners, corners.count == 4 {
            var path = Path()
            let pts = corners.map { CGPoint(x: r.minX + $0.x * r.width, y: r.minY + $0.y * r.height) }
            path.move(to: pts[0])
            for i in 1..<4 { path.addLine(to: pts[i]) }
            path.closeSubpath()
            ctx.stroke(path, with: .color(.green), lineWidth: 2)
            for pt in pts {
                ctx.fill(Path(ellipseIn: CGRect(x: pt.x - 4, y: pt.y - 4, width: 8, height: 8)),
                         with: .color(.green))
            }
        }
    }

    private func drawFlow(ctx: GraphicsContext, r: CGRect) {
        for arrow in frame.flowArrows {
            let ox = r.minX + arrow.origin.x * r.width
            let oy = r.minY + arrow.origin.y * r.height
            let ex = ox + CGFloat(arrow.dx) * r.width * 0.01
            let ey = oy + CGFloat(arrow.dy) * r.height * 0.01

            let mag = arrow.magnitude
            let hue = Double(min(mag / 10.0, 1.0)) * 0.7
            let color = Color(hue: hue, saturation: 1, brightness: 1)

            var path = Path()
            path.move(to: CGPoint(x: ox, y: oy))
            path.addLine(to: CGPoint(x: ex, y: ey))
            ctx.stroke(path, with: .color(color), lineWidth: 1)
        }
    }

    private func drawTrajectories(ctx: GraphicsContext, r: CGRect) {
        let colors: [Color] = [.red, .orange, .yellow, .green, .cyan]
        for (i, traj) in frame.trajectories.enumerated() {
            let color = colors[i % colors.count]
            guard traj.points.count > 1 else { continue }
            var path = Path()
            let pts = traj.points.map { CGPoint(x: r.minX + $0.x * r.width, y: r.minY + $0.y * r.height) }
            path.move(to: pts[0])
            for j in 1..<pts.count { path.addLine(to: pts[j]) }
            ctx.stroke(path, with: .color(color), lineWidth: 2)
            // 终点圆
            if let last = pts.last {
                ctx.fill(Path(ellipseIn: CGRect(x: last.x - 5, y: last.y - 5, width: 10, height: 10)),
                         with: .color(color))
            }
        }
    }

    private func drawPose(ctx: GraphicsContext, r: CGRect) {
        // 骨骼
        for bone in frame.bones {
            let from = CGPoint(x: r.minX + bone.from.x * r.width, y: r.minY + bone.from.y * r.height)
            let to = CGPoint(x: r.minX + bone.to.x * r.width, y: r.minY + bone.to.y * r.height)
            var path = Path()
            path.move(to: from); path.addLine(to: to)
            ctx.stroke(path, with: .color(.green.opacity(Double(bone.confidence))), lineWidth: 2)
        }
        // 关节
        for joint in frame.joints {
            let pt = CGPoint(x: r.minX + joint.position.x * r.width, y: r.minY + joint.position.y * r.height)
            let sz: CGFloat = 6
            ctx.fill(Path(ellipseIn: CGRect(x: pt.x - sz/2, y: pt.y - sz/2, width: sz, height: sz)),
                     with: .color(.yellow))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ROI 框选叠加层
// ═══════════════════════════════════════════════════════════════════

struct ROISelectionOverlay: View {
    let fitRect: CGRect
    let onSelected: (CGRect) -> Void

    @State private var dragStart: CGPoint?
    @State private var dragCurrent: CGPoint?

    var body: some View {
        GeometryReader { _ in
            ZStack {
                // 半透明提示
                Color.blue.opacity(0.05)

                // 拖动框
                if let start = dragStart, let current = dragCurrent {
                    let rect = dragRect(start: start, current: current)
                    Rectangle()
                        .stroke(Color.cyan, lineWidth: 2)
                        .background(Color.cyan.opacity(0.15))
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }

                Text("拖动框选目标区域")
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(4)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(4)
                    .position(x: fitRect.midX, y: fitRect.minY + 20)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 5)
                    .onChanged { value in
                        if dragStart == nil { dragStart = value.startLocation }
                        dragCurrent = value.location
                    }
                    .onEnded { value in
                        guard let start = dragStart else { return }
                        let rect = dragRect(start: start, current: value.location)
                        // 转换为归一化坐标
                        let normX = (rect.minX - fitRect.minX) / fitRect.width
                        let normY = (rect.minY - fitRect.minY) / fitRect.height
                        let normW = rect.width / fitRect.width
                        let normH = rect.height / fitRect.height
                        let clamped = CGRect(
                            x: max(0, min(1, normX)),
                            y: max(0, min(1, normY)),
                            width: max(0.02, min(1, normW)),
                            height: max(0.02, min(1, normH))
                        )
                        onSelected(clamped)
                        dragStart = nil; dragCurrent = nil
                    }
            )
        }
    }

    private func dragRect(start: CGPoint, current: CGPoint) -> CGRect {
        CGRect(
            x: min(start.x, current.x),
            y: min(start.y, current.y),
            width: abs(current.x - start.x),
            height: abs(current.y - start.y)
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - 辅助 UI 组件
// ═══════════════════════════════════════════════════════════════════

struct MetricBadge: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(.body, design: .monospaced))
                .bold()
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 4)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(6)
    }
}

struct LabeledSlider: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text("\(value)").font(.caption).monospacedDigit()
            }
            Slider(value: Binding(
                get: { Double(value) },
                set: { value = Int($0) }
            ), in: Double(range.lowerBound)...Double(range.upperBound), step: Double(step))
        }
    }
}

struct LabeledFloatSlider: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let step: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text(String(format: "%.2f", value)).font(.caption).monospacedDigit()
            }
            Slider(value: $value, in: range, step: step)
        }
    }
}
