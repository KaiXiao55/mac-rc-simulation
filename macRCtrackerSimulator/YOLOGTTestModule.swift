//
//  YOLOGTTestModule.swift
//  macRCtrackerSimulator
//
//  YOLO 检测率 GT-vis 测试模块 — 独立 GUI + 命令行接口
//
//  功能:
//    ① 加载主仿真器 YOLO 模型 (Detector.mlpackage / yolo11_640imgsz) 或虚拟赛道模型 (vsim_yolo11n)
//    ② 模式 A — 真实视频 GT 测试: 加载带 SAM3 标注的视频, 逐帧对比 YOLO 检测 vs GT 标注
//    ③ 模式 B — 虚拟赛道 GT 测试: 基于 VirtualRaceEngine 生成 ~300 张合成场景图片 (含精确 GT),
//       覆盖多种赛道/场景/机位, 排除 GT 被遮挡的帧, 对 YOLO 模型进行检测率测试
//    ④ 输出: Precision / Recall / F1 / mAP@0.5 / 检测率曲线 / 逐帧可视化
//    ⑤ 命令行接口: --gt-test [--video <path>] [--annotation <path>] [--model <name>] [--vsim] [--output <dir>]
//
//  v2: 修复 — 自动加载模型 / 自动匹配标注 / 全帧预览 / 置信度显示
//
//  Created by Manus AI on 2026/4/12.
//  macOS 14.0+
//

import SwiftUI
import Combine
import CoreML
import Vision
import AVFoundation
import AppKit
import SceneKit
import CoreImage
import CoreGraphics
import CoreVideo
import UniformTypeIdentifiers

// ═══════════════════════════════════════════════════════════════
// MARK: - GT 测试配置
// ═══════════════════════════════════════════════════════════════

struct GTTestCfg {
    /// IoU 匹配阈值: 检测框与 GT 框的 IoU (交并比) 超过此值才算"命中" (True Positive)
    /// 0.3 表示两个框重叠面积超过并集的 30% 就算匹配成功
    /// RC 车跟踪场景目标较小, 用 0.3 比常规的 0.5 更宽松
    static let iouThreshold: CGFloat = 0.3
    /// PR 曲线采样的置信度阈值列表 (0.05, 0.10, ..., 0.95)
    /// 用于遍历不同置信度截断点, 绘制 Precision-Recall 曲线并计算 mAP
    static let confThresholds: [Float] = stride(from: Float(0.05), through: Float(0.95), by: Float(0.05)).map { $0 }
    /// 虚拟赛道测试图片数量 (约 300 张, 覆盖多种场景/赛道/机位组合)
    static let vsimImageCount: Int = 300
    /// 虚拟赛道渲染图片尺寸 (640×640, 与 YOLO 输入尺寸匹配)
    static let vsimRenderSize: Int = 640
    /// 最小 GT 框面积 (归一化), 小于此值的极小目标不参与评测 (避免对极远处微小目标不合理的惩罚)
    static let minGTArea: CGFloat = 0.0003
    /// 视频采样间隔: 每 N 帧采样一帧, 避免处理太慢 (相邻帧差异很小, 无需每帧都测)
    static var videoSampleInterval: Int = 3
    /// SAM3 标注根目录候选列表
    static let sam3AnnotationRoots: [String] = [
        NSHomeDirectory() + "/Documents/MacRCSimulator/SAM3_Annotations",
        NSHomeDirectory() + "/Desktop/SAM3_Annotations"
    ]
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 单帧检测结果
// ═══════════════════════════════════════════════════════════════

struct GTTestDetection {
    let rect: CGRect       // 归一化 TL 坐标系 (左上角原点, x/y/w/h 均为 0~1)
    /// 检测置信度 (obs.confidence): YOLO 模型对"这里有目标"的把握程度
    /// 值域 0~1, 越高表示模型越确信此处存在 RC 车
    /// 计算方式: objectness(是否有物体) × class_confidence(属于该类别的概率)
    let confidence: Float
    /// 分类置信度 (lbl.confidence): 在已确定有目标的前提下, 属于"RC car"类别的概率
    /// 对于单类别模型 (只检测 RC 车), 此值通常接近 1.0, 参考意义不大
    let classConf: Float
    let label: String
}

struct GTTestFrameResult: Identifiable {
    let id = UUID()
    let frameIndex: Int
    let image: NSImage?                // 可视化图片 (叠加了 GT 框和检测框的合成图)
    let gtBoxes: [CGRect]              // GT 框 (归一化 TL) — 人工标注的真实目标位置
    let detections: [GTTestDetection]  // YOLO 检测结果 — 模型输出的所有候选框
    /// TP (True Positive, 真正例): YOLO 检测框与某个 GT 框 IoU > 阈值, 表示"正确检测到了"
    /// 通俗理解: 模型说"这里有车", 确实有车 → 检测正确
    let truePositives: Int
    /// FP (False Positive, 假正例/误检): YOLO 检测框没有匹配到任何 GT 框
    /// 通俗理解: 模型说"这里有车", 但其实没有 → 虚警/误报
    let falsePositives: Int
    /// FN (False Negative, 假负例/漏检): 某个 GT 框没有被任何检测框匹配到
    /// 通俗理解: 这里确实有车, 但模型没检测到 → 漏掉了
    let falseNegatives: Int
    /// bestIoU (最佳交并比): 该帧中所有匹配对里最高的 IoU 值
    /// IoU = 两个框的交集面积 / 两个框的并集面积, 值域 0~1
    /// IoU 越接近 1 表示检测框和真实框重叠越好, 定位越精准
    let bestIoU: CGFloat
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 汇总统计
// ═══════════════════════════════════════════════════════════════

struct GTTestSummary {
    var totalFrames: Int = 0        // 测试总帧数
    var totalGT: Int = 0            // GT 目标总数 (所有帧的标注框总和)
    var totalDetections: Int = 0    // YOLO 检测总数 (过了置信度阈值的检测框总和)
    var totalTP: Int = 0            // 总 TP 数 (正确检测数)
    var totalFP: Int = 0            // 总 FP 数 (误检数)
    var totalFN: Int = 0            // 总 FN 数 (漏检数)
    
    /// Precision (精确率/准确率): 模型检测出来的结果里, 有多少是对的
    /// 公式: TP / (TP + FP)
    /// 通俗理解: 模型说"检测到 10 辆车", 其中 8 辆确实是车 → Precision = 80%
    /// Precision 低 → 误检多 (把不是车的东西认成了车)
    var precision: Double { totalTP + totalFP > 0 ? Double(totalTP) / Double(totalTP + totalFP) : 0 }
    
    /// Recall (召回率/查全率): 真实存在的目标里, 有多少被模型找到了
    /// 公式: TP / (TP + FN)
    /// 通俗理解: 场景里实际有 10 辆车, 模型找到了其中 7 辆 → Recall = 70%
    /// Recall 低 → 漏检多 (很多车没被检测到)
    /// ★ 对于 RC 车跟踪场景, Recall 是最重要的指标 — 漏检会导致跟丢
    var recall: Double { totalTP + totalFN > 0 ? Double(totalTP) / Double(totalTP + totalFN) : 0 }
    
    /// F1 Score (F1 分数): Precision 和 Recall 的调和平均数, 综合衡量模型表现
    /// 公式: 2 × P × R / (P + R)
    /// 通俗理解: 同时考虑"检测准不准"和"检测全不全"的综合分数
    /// F1 越接近 1 越好, 只有 Precision 和 Recall 都高时 F1 才会高
    var f1: Double {
        let p = precision, r = recall
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0
    }
    
    /// 帧级检测率: 至少检测到一个 GT 目标的帧数 / 有 GT 目标的总帧数
    /// 通俗理解: 每一帧里只要能找到至少一辆车就算该帧"检测成功"
    /// 与 Recall 的区别: Recall 是按目标数算的, 检测率是按帧数算的
    /// 例如: 一帧有 3 辆车, 只检测到 1 辆 → Recall=33%, 但检测率=100%(该帧算检测成功)
    var detectionRate: Double = 0
    
    /// mAP@0.5 (mean Average Precision @ IoU=0.5):
    /// 在 IoU 阈值=0.5 下, PR 曲线下的面积 (AUC)
    /// 通俗理解: 遍历所有可能的置信度阈值, 综合衡量模型在不同置信度下的表现
    /// 值域 0~1, 越接近 1 表示模型在各种置信度下都能保持好的检测效果
    /// 这是目标检测领域最核心的综合指标, COCO 竞赛的主要排名依据
    var mapAt50: Double = 0
    
    /// PR 曲线数据: 在不同置信度阈值下的 (Precision, Recall) 点
    /// 置信度阈值越高 → 只保留模型很确定的检测 → Precision 高但 Recall 低
    /// 置信度阈值越低 → 保留更多检测 (包括不确定的) → Recall 高但 Precision 低
    /// 理想的模型: 无论阈值怎么调, Precision 和 Recall 都很高 (曲线贴近右上角)
    var prCurve: [(conf: Float, precision: Double, recall: Double)] = []
    
    /// 综合评级 (基于 Recall):
    /// Recall ≥ 90% → 优秀 (几乎不漏检)
    /// Recall ≥ 75% → 良好 (偶尔漏检)
    /// Recall ≥ 55% → 一般 (经常漏检)
    /// Recall < 55% → 需优化 (大量漏检, 跟踪会频繁丢失)
    var gradeText: String {
        switch recall * 100 {
        case 90...: return "优秀 ★★★"
        case 75...: return "良好 ★★☆"
        case 55...: return "一般 ★☆☆"
        default:    return "需优化 ☆☆☆"
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - GT 测试引擎
// ═══════════════════════════════════════════════════════════════

final class YOLOGTTestEngine: ObservableObject {
    
    // ── 发布状态 ──
    @Published var status: String = "就绪 — 选择模型和数据源开始测试"
    @Published var progress: Double = 0
    @Published var running: Bool = false
    @Published var summary: GTTestSummary?
    @Published var frameResults: [GTTestFrameResult] = []
    @Published var selectedFrameIdx: Int = 0
    @Published var modelName: String = "未加载"
    @Published var confThreshold: Float = 0.25
    
    // ── 模型 ──
    private var yoloReq: VNCoreMLRequest?
    private(set) var modelLoaded = false
    private let q = DispatchQueue(label: "gt.test", qos: .userInitiated)
    private let ci = CIContext(options: [.useSoftwareRenderer: false])
    private var cancel = false
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 模型加载
    // ═══════════════════════════════════════════════════════════
    
    /// 加载 Bundle 中的模型 (按名称)
    func loadBundledModel(name: String) -> Bool {
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return loadModel(url, displayName: name)
        }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            return loadModel(url, displayName: name)
        }
        DispatchQueue.main.async { self.status = "找不到模型: \(name)" }
        return false
    }
    
    /// 加载指定 URL 的模型
    func loadModel(_ url: URL, displayName: String? = nil) -> Bool {
        let c = MLModelConfiguration()
        c.computeUnits = .all
        do {
            let mlm = try MLModel(contentsOf: url, configuration: c)
            guard let vnm = try? VNCoreMLModel(for: mlm) else {
                DispatchQueue.main.async { self.status = "VNCoreMLModel 创建失败" }
                return false
            }
            // 为带阈值输入的模型设置低阈值, 确保全量检测输出
            let inputNames = Set(mlm.modelDescription.inputDescriptionsByName.keys)
            if inputNames.contains("confidenceThreshold") || inputNames.contains("iouThreshold") {
                vnm.featureProvider = GTTestThresholdProvider(conf: 0.01, iou: 0.45)
            }
            let r = VNCoreMLRequest(model: vnm)
            r.imageCropAndScaleOption = .scaleFill
            yoloReq = r
            modelLoaded = true
            let name = displayName ?? url.deletingPathExtension().lastPathComponent
            DispatchQueue.main.async {
                self.modelName = name
                self.status = "模型已加载: \(name)"
            }
            // GUI 模式预热；CLI 评测直接进入检测，避免与测试任务竞争同一个串行队列。
            if !CommandLine.arguments.contains("--gt-test") {
                q.async { [weak self] in self?.prewarm() }
            }
            return true
        } catch {
            DispatchQueue.main.async { self.status = "模型加载失败: \(error.localizedDescription)" }
            return false
        }
    }
    
    private func prewarm() {
        guard let req = yoloReq else { return }
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 320, 320, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let buf = pb else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: buf, orientation: .up, options: [:])
        try? handler.perform([req])
    }
    
    /// 列出 Bundle 中可用的 YOLO 模型
    static func availableModels() -> [(name: String, display: String)] {
        let candidates = [
            ("yolo11_640imgsz", "YOLO11-640 (主仿真器)"),
            ("Detector", "Detector (主仿真器)"),
            ("RC_YOLO11n_320", "RC_YOLO11n_320"),
            ("vsim_yolo11n", "vsim_yolo11n (虚拟赛道)")
        ]
        var result: [(name: String, display: String)] = []
        for (name, display) in candidates {
            if Bundle.main.url(forResource: name, withExtension: "mlmodelc") != nil ||
               Bundle.main.url(forResource: name, withExtension: "mlpackage") != nil {
                result.append((name: name, display: display))
            }
        }
        return result
    }
    
    /// 根据测试模式返回推荐的默认模型名
    static func defaultModelName(forVSim: Bool) -> String {
        if forVSim {
            return "vsim_yolo11n"
        } else {
            let models = availableModels()
            if models.contains(where: { $0.name == "yolo11_640imgsz" }) { return "yolo11_640imgsz" }
            if models.contains(where: { $0.name == "Detector" }) { return "Detector" }
            return models.first?.name ?? ""
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 自动匹配标注文件
    // ═══════════════════════════════════════════════════════════
    
    /// 根据视频文件名自动在 SAM3_Annotations 目录下查找对应标注
    /// 匹配规则: 视频名(去扩展名) → SAM3_Annotations/<名称>/<名称>_annotations.json
    /// 优先查找 .corrected.annotations.json, 其次 _annotations.json
    static func findAnnotation(forVideo videoURL: URL) -> URL? {
        let videoName = videoURL.deletingPathExtension().lastPathComponent
        let fm = FileManager.default
        
        // 候选标注根目录
        var roots: [String] = GTTestCfg.sam3AnnotationRoots
        
        // 也尝试从 annotate_config.json 读取 output_dir
        if let bundleCfg = Bundle.main.url(forResource: "annotate_config", withExtension: "json"),
           let data = try? Data(contentsOf: bundleCfg),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let outputDir = json["output_dir"] as? String,
           !roots.contains(outputDir) {
            roots.insert(outputDir, at: 0)
        }
        
        for root in roots {
            // 策略 1: <root>/<videoName>/<videoName>.corrected.annotations.json (优先)
            let corrected = URL(fileURLWithPath: root)
                .appendingPathComponent(videoName)
                .appendingPathComponent("\(videoName).corrected.annotations.json")
            if fm.fileExists(atPath: corrected.path) { return corrected }
            
            // 策略 2: <root>/<videoName>/<videoName>_annotations.json
            let standard = URL(fileURLWithPath: root)
                .appendingPathComponent(videoName)
                .appendingPathComponent("\(videoName)_annotations.json")
            if fm.fileExists(atPath: standard.path) { return standard }
            
            // 策略 3: 模糊匹配 — 遍历子目录, 查找名称包含 videoName 的标注文件
            let rootURL = URL(fileURLWithPath: root)
            if let subdirs = try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil) {
                for subdir in subdirs where subdir.hasDirectoryPath {
                    let dirName = subdir.lastPathComponent
                    if dirName.contains(videoName) || videoName.contains(dirName) {
                        if let files = try? fm.contentsOfDirectory(at: subdir, includingPropertiesForKeys: nil) {
                            // 优先 corrected
                            if let corr = files.first(where: { $0.lastPathComponent.contains("corrected") && $0.pathExtension == "json" }) {
                                return corr
                            }
                            if let ann = files.first(where: { $0.lastPathComponent.contains("annotations") && $0.pathExtension == "json" }) {
                                return ann
                            }
                        }
                    }
                }
            }
        }
        
        return nil
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: YOLO 推理
    // ═══════════════════════════════════════════════════════════
    
    /// 对单张图片执行 YOLO 推理, 返回所有检测 (不做置信度过滤)
    /// ★ 修复: confidence 使用 obs.confidence (检测置信度), classConf 使用 lbl.confidence (分类置信度)
    /// 对于单类别 YOLO 模型, lbl.confidence 通常接近 1.0 (分类几乎确定),
    /// 而 obs.confidence 才是真正的检测置信度 (objectness × class_conf)
    private func runYOLO(cgImage: CGImage) -> [GTTestDetection] {
        guard let req = yoloReq else { return [] }
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        try? handler.perform([req])
        guard let results = req.results as? [VNRecognizedObjectObservation] else { return [] }
        return results.compactMap { obs in
            guard let lbl = obs.labels.first else { return nil }
            let r = obs.boundingBox
            // Vision BL → TL 归一化
            let tlRect = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
            return GTTestDetection(
                rect: tlRect,
                confidence: obs.confidence,   // ★ 检测置信度 (不再用 lbl.confidence)
                classConf: lbl.confidence,     // 分类置信度 (仅供参考)
                label: lbl.identifier
            )
        }
    }
    
    private func runYOLO(pixelBuffer: CVPixelBuffer) -> [GTTestDetection] {
        guard let req = yoloReq else { return [] }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try? handler.perform([req])
        guard let results = req.results as? [VNRecognizedObjectObservation] else { return [] }
        return results.compactMap { obs in
            guard let lbl = obs.labels.first else { return nil }
            let r = obs.boundingBox
            let tlRect = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
            return GTTestDetection(
                rect: tlRect,
                confidence: obs.confidence,
                classConf: lbl.confidence,
                label: lbl.identifier
            )
        }
    }
    
     // ═════════════════════════════════════════════════════════
    // MARK: IoU 计算 (Intersection over Union, 交并比)
    // ═════════════════════════════════════════════════════════
    
    /// IoU (Intersection over Union, 交并比): 衡量两个框重叠程度的核心指标
    /// 公式: IoU = 两个框的交集面积 / 两个框的并集面积
    /// 值域: 0 (完全不重叠) ~ 1 (完全重合)
    /// 示例: IoU=0.7 表示两个框有 70% 的面积是重叠的, 定位相当准确
    ///
    /// 图示:
    ///   ┌──────┐
    ///   │  A   ┌─┤────┐   IoU = 斜线区 / (整个A + 整个B - 斜线区)
    ///   │      │//│    │
    ///   └─────┤─┘  B │
    ///          └────┘
    static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 匹配 GT ↔ 检测
    // ═══════════════════════════════════════════════════════════
    
    /// 贪心匹配算法: 将 YOLO 检测框与 GT 框进行一对一匹配, 计算 TP/FP/FN
    ///
    /// 算法流程:
    /// 1. 先过滤掉置信度低于阈值的检测框
    /// 2. 剩余检测框按置信度从高到低排序 (置信度高的优先匹配)
    /// 3. 每个检测框找 IoU 最高的未匹配 GT 框:
    ///    - IoU > 阈值 (0.3) 且该 GT 未被匹配过 → TP (正确检测)
    ///    - 否则 → FP (误检)
    /// 4. 没有被任何检测框匹配到的 GT 框 → FN (漏检)
    private func matchFrame(gtBoxes: [CGRect], detections: [GTTestDetection], confThresh: Float)
        -> (tp: Int, fp: Int, fn: Int, bestIoU: CGFloat)
    {
        let filteredDets = detections.filter { $0.confidence >= confThresh }
        let sortedDets = filteredDets.sorted { $0.confidence > $1.confidence }
        
        var matchedGT = Set<Int>()
        var tp = 0, fp = 0
        var bestIoU: CGFloat = 0
        
        for det in sortedDets {
            var bestIdx = -1
            var bestVal: CGFloat = 0
            for (gi, gt) in gtBoxes.enumerated() {
                if matchedGT.contains(gi) { continue }
                let iou = Self.iou(det.rect, gt)
                if iou > bestVal {
                    bestVal = iou
                    bestIdx = gi
                }
            }
            if bestVal > GTTestCfg.iouThreshold && bestIdx >= 0 {
                tp += 1
                matchedGT.insert(bestIdx)
                bestIoU = max(bestIoU, bestVal)
            } else {
                fp += 1
            }
        }
        let fn = gtBoxes.count - matchedGT.count
        return (tp, fp, fn, bestIoU)
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 计算 PR 曲线 & mAP
    // ═══════════════════════════════════════════════════════════
    
    private func computePRCurveAndMAP(allGT: [[CGRect]], allDets: [[GTTestDetection]]) -> (curve: [(Float, Double, Double)], map: Double) {
        var curve: [(Float, Double, Double)] = []
        var prevRecall: Double = 0
        var auc: Double = 0
        
        // 从高阈值到低阈值遍历
        let thresholds = GTTestCfg.confThresholds.sorted(by: >)
        for thresh in thresholds {
            var totalTP = 0, totalFP = 0, totalFN = 0
            for i in allGT.indices {
                let m = matchFrame(gtBoxes: allGT[i], detections: allDets[i], confThresh: thresh)
                totalTP += m.tp; totalFP += m.fp; totalFN += m.fn
            }
            let p = (totalTP + totalFP) > 0 ? Double(totalTP) / Double(totalTP + totalFP) : 1.0
            let r = (totalTP + totalFN) > 0 ? Double(totalTP) / Double(totalTP + totalFN) : 0
            curve.append((thresh, p, r))
            // 梯形法计算 AUC
            auc += (r - prevRecall) * p
            prevRecall = r
        }
        return (curve.reversed(), auc)
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 可视化 — 在图片上绘制 GT 和检测框
    // ═══════════════════════════════════════════════════════════
    
    /// ★ 修复: 检测框上显示 obs.confidence (检测置信度), 不再显示 lbl.confidence (分类置信度)
    private func drawBoxes(on image: CGImage, gtBoxes: [CGRect], detections: [GTTestDetection], confThresh: Float) -> NSImage {
        let w = CGFloat(image.width)
        let h = CGFloat(image.height)
        let nsImage = NSImage(cgImage: image, size: NSSize(width: w, height: h))
        
        let result = NSImage(size: NSSize(width: w, height: h))
        result.lockFocus()
        nsImage.draw(in: NSRect(x: 0, y: 0, width: w, height: h))
        
        // 绘制 GT 框 (绿色虚线)
        for gt in gtBoxes {
            let pixelRect = NSRect(
                x: gt.minX * w,
                y: (1 - gt.maxY) * h,  // TL → BL (NSImage 坐标)
                width: gt.width * w,
                height: gt.height * h
            )
            let path = NSBezierPath(rect: pixelRect)
            path.lineWidth = 2.5
            let pattern: [CGFloat] = [6, 3]
            path.setLineDash(pattern, count: 2, phase: 0)
            NSColor.green.withAlphaComponent(0.9).setStroke()
            path.stroke()
            
            // GT 标签
            let label = "GT"
            let attrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.boldSystemFont(ofSize: max(12, w * 0.012)),
                .foregroundColor: NSColor.green,
                .backgroundColor: NSColor.black.withAlphaComponent(0.5)
            ]
            label.draw(at: NSPoint(x: pixelRect.minX + 2, y: pixelRect.maxY - max(16, w * 0.016)), withAttributes: attrs)
        }
        
        // 绘制检测框 (命中=青色, 误检=红色)
        let filteredDets = detections.filter { $0.confidence >= confThresh }
        for det in filteredDets {
            let pixelRect = NSRect(
                x: det.rect.minX * w,
                y: (1 - det.rect.maxY) * h,
                width: det.rect.width * w,
                height: det.rect.height * h
            )
            // 判断是否命中 GT
            var hit = false
            for gt in gtBoxes {
                if Self.iou(det.rect, gt) > GTTestCfg.iouThreshold {
                    hit = true; break
                }
            }
            let color: NSColor = hit ? .cyan : .red
            let path = NSBezierPath(rect: pixelRect)
            path.lineWidth = 2.0
            color.withAlphaComponent(0.9).setStroke()
            path.stroke()
            
            // ★ 修复: 显示检测置信度 (obs.confidence), 不再显示分类置信度 (lbl.confidence)
            // det.confidence 现在是 obs.confidence, 对于单类别模型这才是有意义的置信度
            let label = String(format: "%.1f%%", det.confidence * 100)
            let attrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedSystemFont(ofSize: max(11, w * 0.011), weight: .medium),
                .foregroundColor: color,
                .backgroundColor: NSColor.black.withAlphaComponent(0.5)
            ]
            label.draw(at: NSPoint(x: pixelRect.minX + 2, y: pixelRect.maxY - max(15, w * 0.015)), withAttributes: attrs)
        }
        
        result.unlockFocus()
        return result
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 模式 A — 真实视频 GT 测试
    // ═══════════════════════════════════════════════════════════
    
    func runVideoGTTest(videoURL: URL, annotationURL: URL, keepImages: Bool = true) {
        guard modelLoaded else {
            DispatchQueue.main.async { self.status = "请先加载模型" }
            return
        }
        let cliMode = CommandLine.arguments.contains("--gt-test")
        cancel = false
        running = true
        progress = 0
        frameResults = []
        summary = nil
        status = "加载标注数据..."
        DispatchQueue.main.async {
            self.running = true
            self.progress = 0
            self.frameResults = []
            self.summary = nil
            self.status = "加载标注数据..."
        }
        if cliMode {
            print("[GTTest] queueing video detection task")
        }
        q.async { [weak self] in
            guard let self = self else { return }
            if cliMode {
                print("[GTTest] video detection task started")
            }
            
            // ── 1. 加载标注 ──
            guard let data = try? Data(contentsOf: annotationURL),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let framesDict = json["frames"] as? [String: Any] else {
                let fail = {
                    self.status = "标注文件解析失败"
                    self.running = false
                }
                cliMode ? fail() : DispatchQueue.main.async(execute: fail)
                return
            }
            if cliMode {
                print("[GTTest] annotation loaded: frames=\(framesDict.count), bytes=\(data.count)")
            }
            
            // 解析标注: frameIndex → [CGRect] (归一化 TL)
            var annotations: [Int: [CGRect]] = [:]
            for (key, value) in framesDict {
                guard let frameIdx = Int(key),
                      let boxes = value as? [[String: Any]] else { continue }
                var rects: [CGRect] = []
                for box in boxes {
                    guard let cx = box["cx"] as? Double,
                          let cy = box["cy"] as? Double,
                          let w = box["w"] as? Double,
                          let h = box["h"] as? Double else { continue }
                    let rect = CGRect(x: cx - w/2, y: cy - h/2, width: w, height: h)
                    if rect.width * rect.height >= GTTestCfg.minGTArea {
                        rects.append(rect)
                    }
                }
                if !rects.isEmpty {
                    annotations[frameIdx] = rects
                }
            }
            if cliMode {
                let boxCount = annotations.values.reduce(0) { $0 + $1.count }
                print("[GTTest] annotation parsed: annotatedFrames=\(annotations.count), boxes=\(boxCount)")
            }
            
            guard !annotations.isEmpty else {
                let fail = {
                    self.status = "标注数据为空"
                    self.running = false
                }
                cliMode ? fail() : DispatchQueue.main.async(execute: fail)
                return
            }
            
            let setOpeningStatus = { self.status = "打开视频..." }
            cliMode ? setOpeningStatus() : DispatchQueue.main.async(execute: setOpeningStatus)

            // ── 2. 打开视频 ──
            let asset = AVAsset(url: videoURL)
            guard let videoTrack = asset.tracks(withMediaType: .video).first else {
                let fail = {
                    self.status = "无法读取视频轨道"
                    self.running = false
                }
                cliMode ? fail() : DispatchQueue.main.async(execute: fail)
                return
            }

            let fps = videoTrack.nominalFrameRate
            let duration = CMTimeGetSeconds(asset.duration)
            let totalFrames = Int(duration * Double(fps))
            if cliMode {
                print(String(format: "[GTTest] video opened: fps=%.3f, duration=%.3fs, estimatedFrames=%d", fps, duration, totalFrames))
            }

            // ── 2.1 收集真实 PTS (与 VideoAnnotationModule 一致) ──
            // ★ 修复: 不再使用 CMTime(value: frameIdx, timescale: CMTimeScale(fps)) 定位帧
            //   因为 CMTimeScale(fps) 会将非整数 fps (如 29.4, 55.18) 截断为整数,
            //   导致时间累积误差, 后面的帧定位偏差越来越大 (帧号500时可偏差7帧以上)
            //   现在通过 AVAssetReader 收集每帧的真实 PTS, 确保精确定位
            let presentationTimes: [CMTime] = {
                do {
                    if cliMode {
                        print("[GTTest] collecting presentation timestamps...")
                    }
                    let reader = try AVAssetReader(asset: asset)
                    let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: nil)
                    output.alwaysCopiesSampleData = false
                    guard reader.canAdd(output) else { return [] }
                    reader.add(output)
                    guard reader.startReading() else { reader.cancelReading(); return [] }
                    var times: [CMTime] = []
                    while let sb = output.copyNextSampleBuffer() {
                        times.append(CMSampleBufferGetPresentationTimeStamp(sb))
                    }
                    reader.cancelReading()
                    times.sort { CMTimeCompare($0, $1) < 0 }
                    return times
                } catch {
                    if cliMode {
                        print("[GTTest] presentation timestamp collection failed: \(error.localizedDescription)")
                    }
                    return []
                }
            }()
            let hasPTS = !presentationTimes.isEmpty
            if cliMode {
                print("[GTTest] presentation timestamps collected: count=\(presentationTimes.count), hasPTS=\(hasPTS)")
            }

            // 获取有标注的帧索引 (排序)
            let annotatedFrames = annotations.keys.sorted()
            // 采样: 每 N 帧取一帧
            let sampledFrames = annotatedFrames.enumerated().compactMap { idx, frame -> Int? in
                idx % GTTestCfg.videoSampleInterval == 0 ? frame : nil
            }
            if cliMode {
                print("[GTTest] sampled frames: annotated=\(annotatedFrames.count), sampled=\(sampledFrames.count), interval=\(GTTestCfg.videoSampleInterval)")
            }

            guard !sampledFrames.isEmpty else {
                let fail = {
                    self.status = "采样后无可用帧"
                    self.running = false
                }
                cliMode ? fail() : DispatchQueue.main.async(execute: fail)
                return
            }

            let setDetectingStatus = { self.status = "开始检测 \(sampledFrames.count) 帧..." }
            cliMode ? setDetectingStatus() : DispatchQueue.main.async(execute: setDetectingStatus)

            // ── 3. 逐帧读取和检测 ──
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            if hasPTS {
                // 有精确 PTS 时使用零容差, 确保取到的帧与标注完全对齐
                generator.requestedTimeToleranceBefore = .zero
                generator.requestedTimeToleranceAfter = .zero
            } else {
                // 回退: 使用高精度时基的半帧容差
                let halfFrame = CMTimeMakeWithSeconds(0.5 / Double(fps), preferredTimescale: 600)
                generator.requestedTimeToleranceBefore = halfFrame
                generator.requestedTimeToleranceAfter = halfFrame
            }
            
            var allGT: [[CGRect]] = []
            var allDets: [[GTTestDetection]] = []
            var results: [GTTestFrameResult] = []
            var hitFrames = 0
            var totalGTCount = 0
            let confThresh = self.confThreshold
            
            for (idx, frameIdx) in sampledFrames.enumerated() {
                if self.cancel { break }
                if cliMode && (idx == 0 || idx % 10 == 0 || idx == sampledFrames.count - 1) {
                    print("[GTTest] processing frame \(idx + 1)/\(sampledFrames.count) (videoFrame=\(frameIdx))")
                }

                // ★ 修复: 使用真实 PTS 定位帧, 回退时用高精度时基避免截断误差
                let time: CMTime
                if hasPTS, frameIdx < presentationTimes.count {
                    time = presentationTimes[frameIdx]
                } else {
                    time = CMTimeMakeWithSeconds(Double(frameIdx) / Double(fps), preferredTimescale: 600)
                }
                guard let cgImage = try? generator.copyCGImage(at: time, actualTime: nil) else { continue }
                guard let gtBoxes = annotations[frameIdx], !gtBoxes.isEmpty else { continue }
                
                // YOLO 推理
                let dets = self.runYOLO(cgImage: cgImage)
                if cliMode && (idx == 0 || idx % 10 == 0 || idx == sampledFrames.count - 1) {
                    print("[GTTest] frame \(idx + 1): gt=\(gtBoxes.count), rawDets=\(dets.count)")
                }
                let match = self.matchFrame(gtBoxes: gtBoxes, detections: dets, confThresh: confThresh)
                
                allGT.append(gtBoxes)
                allDets.append(dets)
                totalGTCount += gtBoxes.count
                if match.tp > 0 { hitFrames += 1 }
                
                // ★ 修复: 所有帧都生成预览图 (不再跳帧)
                let visImage: NSImage? = keepImages ? self.drawBoxes(on: cgImage, gtBoxes: gtBoxes, detections: dets, confThresh: confThresh) : nil
                
                let frameResult = GTTestFrameResult(
                    frameIndex: frameIdx,
                    image: visImage,
                    gtBoxes: gtBoxes,
                    detections: dets,
                    truePositives: match.tp,
                    falsePositives: match.fp,
                    falseNegatives: match.fn,
                    bestIoU: match.bestIoU
                )
                results.append(frameResult)
                
                let prog = Double(idx + 1) / Double(sampledFrames.count)
                if idx % 3 == 0 || idx == sampledFrames.count - 1 {
                    let updateProgress = {
                        self.progress = prog
                        self.status = "检测中... \(idx + 1)/\(sampledFrames.count) 帧 (TP=\(match.tp) FP=\(match.fp) FN=\(match.fn))"
                    }
                    cliMode ? updateProgress() : DispatchQueue.main.async(execute: updateProgress)
                }
            }
            
            // ── 4. 汇总 ──
            let (prCurve, mapVal) = self.computePRCurveAndMAP(allGT: allGT, allDets: allDets)
            var sumTP = 0, sumFP = 0, sumFN = 0
            for r in results { sumTP += r.truePositives; sumFP += r.falsePositives; sumFN += r.falseNegatives }
            if cliMode {
                print("[GTTest] detection loop done: results=\(results.count), TP=\(sumTP), FP=\(sumFP), FN=\(sumFN)")
            }
            
            let gtFrameCount = allGT.count
            let detRate = gtFrameCount > 0 ? Double(hitFrames) / Double(gtFrameCount) : 0
            
            var summary = GTTestSummary()
            summary.totalFrames = results.count
            summary.totalGT = totalGTCount
            summary.totalDetections = allDets.flatMap { $0 }.filter { $0.confidence >= confThresh }.count
            summary.totalTP = sumTP
            summary.totalFP = sumFP
            summary.totalFN = sumFN
            summary.detectionRate = detRate
            summary.mapAt50 = mapVal
            summary.prCurve = prCurve
            
            let finish = {
                self.frameResults = results
                self.summary = summary
                self.running = false
                self.progress = 1.0
                self.status = String(format: "完成 — Recall=%.1f%% Precision=%.1f%% F1=%.3f mAP@0.5=%.3f 检测率=%.1f%%",
                                     summary.recall * 100, summary.precision * 100, summary.f1, summary.mapAt50, summary.detectionRate * 100)
            }
            cliMode ? finish() : DispatchQueue.main.async(execute: finish)
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 模式 B — 虚拟赛道 GT 测试
    // ═══════════════════════════════════════════════════════════
    
    func runVSimGTTest(keepImages: Bool = true) {
        guard modelLoaded else {
            DispatchQueue.main.async { self.status = "请先加载模型" }
            return
        }
        cancel = false
        DispatchQueue.main.async {
            self.running = true
            self.progress = 0
            self.frameResults = []
            self.summary = nil
            self.status = "生成虚拟测试场景..."
        }
        
        q.async { [weak self] in
            guard let self = self else { return }
            
            let engine = VirtualRaceEngine()
            let totalImages = GTTestCfg.vsimImageCount
            
            // 场景类型: 0=室内, 1=室外草地, 2=室外柏油, 3=娱乐场地, 4=沙地越野, 5=专业赛道, 6=公园小树林, 7=开放草地
            let sceneTypes = [0, 1, 2, 3, 4, 5, 6, 7]
            // 赛道形状: 0~9
            let trackShapes = Array(0...9)
            // 相机预设: 0~8
            let cameraPresets = Array(0...8)
            
            var allGT: [[CGRect]] = []
            var allDets: [[GTTestDetection]] = []
            var results: [GTTestFrameResult] = []
            var hitFrames = 0
            var totalGTCount = 0
            let confThresh = self.confThreshold
            
            var imageIdx = 0
            
            // 均匀分配: 每种场景类型生成 totalImages/sceneTypes.count 张
            let imagesPerScene = max(1, totalImages / sceneTypes.count)
            
            for sceneType in sceneTypes {
                if self.cancel { break }
                
                for imgInScene in 0..<imagesPerScene {
                    if self.cancel { break }
                    if imageIdx >= totalImages { break }
                    
                    // 随机选择赛道形状和相机预设
                    let trackShape = trackShapes[imageIdx % trackShapes.count]
                    let camPreset = cameraPresets[imageIdx % cameraPresets.count]
                    let seed = UInt64(42 + imageIdx * 7)
                    
                    // 配置引擎
                    engine.sceneType = sceneType
                    engine.trackShape = trackShape
                    engine.cameraPreset = camPreset
                    engine.randomSeed = seed
                    engine.carCount = Int.random(in: 3...6)
                    engine.enableOcclusion = true
                    engine.enableLightVariation = true
                    engine.enableDrift = Bool.random()
                    
                    // 构建场景
                    engine.datasetBuildScene()
                    
                    // 随机放置车辆
                    let carCount = engine.datasetCarCount
                    let progValues = (0..<carCount).map { _ in CGFloat.random(in: 0...1) }
                    let latOffsets = (0..<carCount).map { _ in CGFloat.random(in: -0.6...0.6) }
                    let slipAngles = (0..<carCount).map { _ in CGFloat.random(in: -0.15...0.15) }
                    engine.datasetPlaceCars(progressValues: progValues, lateralOffsets: latOffsets, slipAngles: slipAngles)
                    
                    // 相机朝向随机车辆
                    let targetCarIdx = Int.random(in: 0..<carCount)
                    let targetPos = engine.datasetCarPosition(targetCarIdx)
                    engine.datasetSetCameraLookAt(targetPos)
                    
                    // 添加小幅抖动
                    let panJitter = CGFloat.random(in: -0.05...0.05)
                    let tiltJitter = CGFloat.random(in: -0.03...0.03)
                    engine.datasetAddCameraJitter(panJitter: panJitter, tiltJitter: tiltJitter)
                    
                    // 渲染
                    guard let (cgImage, labels) = engine.datasetRenderFrame(imageSize: GTTestCfg.vsimRenderSize) else {
                        imageIdx += 1
                        continue
                    }
                    
                    // 转换 GT 标注为 CGRect (TL 归一化)
                    let gtBoxes: [CGRect] = labels.compactMap { lbl in
                        let rect = CGRect(x: lbl.cx - lbl.w/2, y: lbl.cy - lbl.h/2, width: lbl.w, height: lbl.h)
                        return rect.width * rect.height >= GTTestCfg.minGTArea ? rect : nil
                    }
                    
                    // 跳过无 GT 的帧 (所有车辆被遮挡或不在画面内)
                    guard !gtBoxes.isEmpty else {
                        imageIdx += 1
                        continue
                    }
                    
                    // YOLO 推理
                    let dets = self.runYOLO(cgImage: cgImage)
                    let match = self.matchFrame(gtBoxes: gtBoxes, detections: dets, confThresh: confThresh)
                    
                    allGT.append(gtBoxes)
                    allDets.append(dets)
                    totalGTCount += gtBoxes.count
                    if match.tp > 0 { hitFrames += 1 }
                    
                    // ★ 修复: 所有帧都生成预览图 (不再跳帧)
                    let visImage: NSImage? = keepImages ? self.drawBoxes(on: cgImage, gtBoxes: gtBoxes, detections: dets, confThresh: confThresh) : nil
                    
                    let frameResult = GTTestFrameResult(
                        frameIndex: imageIdx,
                        image: visImage,
                        gtBoxes: gtBoxes,
                        detections: dets,
                        truePositives: match.tp,
                        falsePositives: match.fp,
                        falseNegatives: match.fn,
                        bestIoU: match.bestIoU
                    )
                    results.append(frameResult)
                    
                    imageIdx += 1
                    let prog = Double(imageIdx) / Double(totalImages)
                    if imageIdx % 5 == 0 || imageIdx >= totalImages {
                        DispatchQueue.main.async {
                            self.progress = min(prog, 1.0)
                            self.status = "虚拟场景测试... \(imageIdx)/\(totalImages) (场景\(sceneType) 赛道\(trackShape) 机位\(camPreset))"
                        }
                    }
                }
                if imageIdx >= totalImages { break }
            }
            
            // ── 汇总 ──
            let (prCurve, mapVal) = self.computePRCurveAndMAP(allGT: allGT, allDets: allDets)
            var sumTP = 0, sumFP = 0, sumFN = 0
            for r in results { sumTP += r.truePositives; sumFP += r.falsePositives; sumFN += r.falseNegatives }
            
            let gtFrameCount = allGT.count
            let detRate = gtFrameCount > 0 ? Double(hitFrames) / Double(gtFrameCount) : 0
            
            var summary = GTTestSummary()
            summary.totalFrames = results.count
            summary.totalGT = totalGTCount
            summary.totalDetections = allDets.flatMap { $0 }.filter { $0.confidence >= confThresh }.count
            summary.totalTP = sumTP
            summary.totalFP = sumFP
            summary.totalFN = sumFN
            summary.detectionRate = detRate
            summary.mapAt50 = mapVal
            summary.prCurve = prCurve
            
            DispatchQueue.main.async {
                self.frameResults = results
                self.summary = summary
                self.running = false
                self.progress = 1.0
                self.status = String(format: "虚拟赛道测试完成 — Recall=%.1f%% Precision=%.1f%% F1=%.3f mAP@0.5=%.3f 检测率=%.1f%%",
                                     summary.recall * 100, summary.precision * 100, summary.f1, summary.mapAt50, summary.detectionRate * 100)
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 停止
    // ═══════════════════════════════════════════════════════════
    
    func stop() {
        cancel = true
    }
    
    // ═══════════════════════════════════════════════════════════
    // MARK: 导出报告
    // ═══════════════════════════════════════════════════════════
    
    func exportReport(to url: URL) {
        guard let s = summary else { return }
        var lines: [String] = []
        lines.append("# YOLO GT-vis 检测率测试报告")
        lines.append("")
        lines.append("- 模型: \(modelName)")
        lines.append("- 置信度阈值: \(String(format: "%.2f", confThreshold))")
        lines.append("- 测试帧数: \(s.totalFrames)")
        lines.append("- GT 目标总数: \(s.totalGT)")
        lines.append("- 检测总数: \(s.totalDetections)")
        lines.append("")
        lines.append("## 核心指标")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append("| Precision | \(String(format: "%.3f (%.1f%%)", s.precision, s.precision * 100)) |")
        lines.append("| Recall | \(String(format: "%.3f (%.1f%%)", s.recall, s.recall * 100)) |")
        lines.append("| F1 Score | \(String(format: "%.3f", s.f1)) |")
        lines.append("| mAP@0.5 | \(String(format: "%.3f", s.mapAt50)) |")
        lines.append("| 帧级检测率 | \(String(format: "%.1f%%", s.detectionRate * 100)) |")
        lines.append("| TP | \(s.totalTP) |")
        lines.append("| FP | \(s.totalFP) |")
        lines.append("| FN | \(s.totalFN) |")
        lines.append("| 评级 | \(s.gradeText) |")
        lines.append("")
        lines.append("## PR 曲线数据")
        lines.append("")
        lines.append("| Conf | Precision | Recall |")
        lines.append("|------|-----------|--------|")
        for pt in s.prCurve {
            lines.append("| \(String(format: "%.2f", pt.conf)) | \(String(format: "%.3f", pt.precision)) | \(String(format: "%.3f", pt.recall)) |")
        }
        lines.append("")
        lines.append("## 逐帧详情 (前 50 帧)")
        lines.append("")
        lines.append("| Frame | GT | Det | TP | FP | FN | BestIoU |")
        lines.append("|-------|-----|-----|-----|-----|-----|---------|")
        for r in frameResults.prefix(50) {
            lines.append("| \(r.frameIndex) | \(r.gtBoxes.count) | \(r.detections.filter { $0.confidence >= confThreshold }.count) | \(r.truePositives) | \(r.falsePositives) | \(r.falseNegatives) | \(String(format: "%.3f", r.bestIoU)) |")
        }
        
        let text = lines.joined(separator: "\n")
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 阈值 FeatureProvider
// ═══════════════════════════════════════════════════════════════

private final class GTTestThresholdProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["confidenceThreshold", "iouThreshold"]
    private let conf: Float
    private let iou: Float
    
    init(conf: Float, iou: Float) {
        self.conf = conf
        self.iou = iou
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "confidenceThreshold": return MLFeatureValue(double: Double(conf))
        case "iouThreshold": return MLFeatureValue(double: Double(iou))
        default: return nil
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - GUI 视图
// ═══════════════════════════════════════════════════════════════

struct YOLOGTTestView: View {
    @StateObject private var engine = YOLOGTTestEngine()
    
    // ── 模式选择 ──
    @State private var testMode: Int = 0  // 0=真实视频, 1=虚拟赛道
    
    // ── 模型选择 ──
    @State private var selectedModel: String = ""
    @State private var showModelPicker = false
    @State private var didAutoLoadModel = false
    
    // ── 视频/标注选择 ──
    @State private var videoURL: URL?
    @State private var annotationURL: URL?
    @State private var showVideoPicker = false
    @State private var annotationStatus: String = ""  // 标注自动匹配状态提示
    
    // ── 结果浏览 ──
    @State private var showExportPanel = false
    
    private let availableModels = YOLOGTTestEngine.availableModels()
    
    var body: some View {
        HSplitView {
            // ═══ 左侧: 控制面板 ═══
            VStack(alignment: .leading, spacing: 12) {
                // ── 标题 ──
                HStack {
                    Image(systemName: "scope")
                        .font(.title2)
                        .foregroundColor(.cyan)
                    Text("YOLO GT-vis 检测率测试")
                        .font(.title2.bold())
                }
                .padding(.bottom, 4)
                
                Divider()
                
                // ── 模式选择 ──
                Picker("测试模式", selection: $testMode) {
                    Text("真实视频 GT 测试").tag(0)
                    Text("虚拟赛道 GT 测试").tag(1)
                }
                .pickerStyle(.segmented)
                // ★ 修复: 切换模式时自动加载对应默认模型
                .onChange(of: testMode) { _, newMode in
                    autoLoadModelForMode(newMode)
                }
                
                Divider()
                
                // ── 模型选择 ──
                GroupBox("模型") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("当前: ").foregroundColor(.secondary)
                            Text(engine.modelName).bold()
                            if engine.modelLoaded {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .font(.caption)
                            }
                            Spacer()
                        }
                        
                        if !availableModels.isEmpty {
                            Picker("选择模型", selection: $selectedModel) {
                                ForEach(availableModels, id: \.name) { m in
                                    Text(m.display).tag(m.name)
                                }
                            }
                            .onChange(of: selectedModel) { _, newVal in
                                if !newVal.isEmpty {
                                    _ = engine.loadBundledModel(name: newVal)
                                }
                            }
                        }
                        
                        Button("从文件加载模型...") {
                            showModelPicker = true
                        }
                        .fileImporter(isPresented: $showModelPicker,
                                      allowedContentTypes: [.folder, .data],
                                      allowsMultipleSelection: false) { result in
                            if case .success(let urls) = result, let url = urls.first {
                                _ = engine.loadModel(url)
                            }
                        }
                    }
                }
                
                // ── 置信度阈值 ──
                GroupBox("置信度阈值") {
                    VStack(alignment: .leading) {
                        HStack {
                            Slider(value: Binding(
                                get: { Double(engine.confThreshold) },
                                set: { engine.confThreshold = Float($0) }
                            ), in: 0.01...0.95, step: 0.01)
                            Text(String(format: "%.2f", engine.confThreshold))
                                .font(.caption.monospaced())
                                .frame(width: 40)
                        }
                    }
                }
                
                // ── 模式 A: 视频选择 ──
                if testMode == 0 {
                    GroupBox("真实视频数据") {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("视频:").foregroundColor(.secondary).frame(width: 40, alignment: .leading)
                                Text(videoURL?.lastPathComponent ?? "未选择").lineLimit(1).truncationMode(.middle)
                                Spacer()
                                Button("选择") { showVideoPicker = true }
                            }
                            .fileImporter(isPresented: $showVideoPicker,
                                          allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie],
                                          allowsMultipleSelection: false) { result in
                                if case .success(let urls) = result, let url = urls.first {
                                    videoURL = url
                                    // ★ 修复: 导入视频后自动匹配标注
                                    autoMatchAnnotation(for: url)
                                }
                            }
                            
                            HStack {
                                Text("标注:").foregroundColor(.secondary).frame(width: 40, alignment: .leading)
                                if let aURL = annotationURL {
                                    Text(aURL.lastPathComponent).lineLimit(1).truncationMode(.middle)
                                        .foregroundColor(.green)
                                } else {
                                    Text("未匹配").foregroundColor(.orange)
                                }
                                Spacer()
                                // 仍保留手动选择按钮作为后备
                                Button("手动选择") {
                                    let panel = NSOpenPanel()
                                    panel.allowedContentTypes = [.json]
                                    panel.canChooseFiles = true
                                    panel.canChooseDirectories = false
                                    panel.allowsMultipleSelection = false
                                    if panel.runModal() == .OK, let url = panel.url {
                                        annotationURL = url
                                        annotationStatus = "手动选择: \(url.lastPathComponent)"
                                    }
                                }
                                .font(.caption)
                            }
                            
                            // 标注匹配状态
                            if !annotationStatus.isEmpty {
                                Text(annotationStatus)
                                    .font(.caption2)
                                    .foregroundColor(annotationURL != nil ? .green : .orange)
                            }
                        }
                    }
                }
                
                // ── 模式 B: 虚拟赛道说明 ──
                if testMode == 1 {
                    GroupBox("虚拟赛道测试") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("将自动生成约 \(GTTestCfg.vsimImageCount) 张测试图片")
                            Text("覆盖 8 种场景 × 10 种赛道 × 9 种机位")
                            Text("排除 GT 被遮挡的帧, 仅评测可见目标")
                            Text("推荐使用 vsim_yolo11n 模型测试")
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                    }
                }
                
                Divider()
                
                // ── 操作按钮 ──
                HStack {
                    if engine.running {
                        Button(action: { engine.stop() }) {
                            Label("停止", systemImage: "stop.fill")
                        }
                        .tint(.red)
                    } else {
                        Button(action: { startTest() }) {
                            Label("开始测试", systemImage: "play.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!canStart)
                    }
                    
                    Spacer()
                    
                    if engine.summary != nil {
                        Button(action: { showExportPanel = true }) {
                            Label("导出报告", systemImage: "square.and.arrow.up")
                        }
                        .fileExporter(isPresented: $showExportPanel,
                                      document: GTReportDocument(engine: engine),
                                      contentType: .plainText,
                                      defaultFilename: "gt_test_report.md") { _ in }
                    }
                }
                
                // ── 进度 ──
                if engine.running {
                    ProgressView(value: engine.progress)
                        .progressViewStyle(.linear)
                }
                
                // ── 状态 ──
                Text(engine.status)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                // ── 汇总结果 ──
                if let s = engine.summary {
                    GroupBox("测试结果") {
                        VStack(alignment: .leading, spacing: 6) {
                            // Recall: 真实目标中被检测到的比例, 越高越好, 低则漏检多
                            GTMetricRow(label: "Recall (召回率)", value: s.recall, text: String(format: "%.1f%%", s.recall * 100), good: s.recall > 0.75)
                            Text("实际有车的目标中, 被模型找到的比例; 低=漏检多, 跟踪易丢失")
                                .font(.system(size: 9)).foregroundColor(.secondary).padding(.bottom, 2)
                            
                            // Precision: 检测结果中正确的比例, 越高越好, 低则误检多
                            GTMetricRow(label: "Precision (精确率)", value: s.precision, text: String(format: "%.1f%%", s.precision * 100), good: s.precision > 0.75)
                            Text("模型报告的检测中, 真正是车的比例; 低=误检多, 把背景当成了车")
                                .font(.system(size: 9)).foregroundColor(.secondary).padding(.bottom, 2)
                            
                            // F1: Precision 和 Recall 的综合分数
                            GTMetricRow(label: "F1 Score (综合分数)", value: s.f1, text: String(format: "%.3f", s.f1), good: s.f1 > 0.7)
                            Text("精确率和召回率的调和平均; 两者都高时 F1 才高, 0~1 越大越好")
                                .font(.system(size: 9)).foregroundColor(.secondary).padding(.bottom, 2)
                            
                            // mAP@0.5: 不同置信度下的综合检测能力
                            GTMetricRow(label: "mAP@0.5 (综合检测能力)", value: s.mapAt50, text: String(format: "%.3f", s.mapAt50), good: s.mapAt50 > 0.7)
                            Text("遍历各置信度阈值的综合表现, 目标检测领域最核心的评价指标")
                                .font(.system(size: 9)).foregroundColor(.secondary).padding(.bottom, 2)
                            
                            // 帧级检测率: 每帧至少检测到一个目标的比例
                            GTMetricRow(label: "帧级检测率", value: s.detectionRate, text: String(format: "%.1f%%", s.detectionRate * 100), good: s.detectionRate > 0.8)
                            Text("每帧只要检测到至少一辆车就算成功; 与 Recall 不同, 这里按帧计数")
                                .font(.system(size: 9)).foregroundColor(.secondary).padding(.bottom, 2)
                            
                            Divider()
                            
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    // TP: 正确检测 — 模型说有车, 确实有车
                                    Text("TP(正确检测): \(s.totalTP)").font(.caption.monospaced()).foregroundColor(.green)
                                    // FP: 误检/虚警 — 模型说有车, 其实没有
                                    Text("FP(误检虚警): \(s.totalFP)").font(.caption.monospaced()).foregroundColor(.red)
                                    // FN: 漏检 — 确实有车, 模型没检测到
                                    Text("FN(漏检遗漏): \(s.totalFN)").font(.caption.monospaced()).foregroundColor(.orange)
                                }
                                Spacer()
                                VStack(alignment: .trailing, spacing: 3) {
                                    Text("测试帧: \(s.totalFrames)").font(.caption)
                                    Text("GT 目标: \(s.totalGT)").font(.caption)
                                    Text("评级: \(s.gradeText)").font(.caption.bold())
                                }
                            }
                        }
                    }
                }
                
                Spacer()
            }
            .frame(width: 320)
            .padding()
            
            // ═══ 右侧: 可视化区域 ═══
            VStack(spacing: 0) {
                if let results = engine.frameResults.isEmpty ? nil : engine.frameResults {
                    // ── 图片预览 ──
                    let idx = min(engine.selectedFrameIdx, results.count - 1)
                    let frame = results[max(0, idx)]
                    
                    if let img = frame.image {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .background(Color.black)
                    } else {
                        ZStack {
                            Color.black.opacity(0.8)
                            VStack {
                                Image(systemName: "photo")
                                    .font(.largeTitle)
                                    .foregroundColor(.gray)
                                Text("帧 #\(frame.frameIndex) (CLI 模式无预览)")
                                    .foregroundColor(.gray)
                                Text("TP=\(frame.truePositives) FP=\(frame.falsePositives) FN=\(frame.falseNegatives) IoU=\(String(format: "%.3f", frame.bestIoU))")
                                    .font(.caption.monospaced())
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                    
                    // ── 帧导航 ──
                    HStack {
                        Button(action: { engine.selectedFrameIdx = max(0, engine.selectedFrameIdx - 1) }) {
                            Image(systemName: "chevron.left")
                        }
                        .disabled(engine.selectedFrameIdx <= 0)
                        
                        Text("帧 \(engine.selectedFrameIdx + 1) / \(results.count)")
                            .font(.caption.monospaced())
                        
                        Slider(value: Binding(
                            get: { Double(engine.selectedFrameIdx) },
                            set: { engine.selectedFrameIdx = Int($0) }
                        ), in: 0...Double(max(0, results.count - 1)), step: 1)
                        
                        Button(action: { engine.selectedFrameIdx = min(results.count - 1, engine.selectedFrameIdx + 1) }) {
                            Image(systemName: "chevron.right")
                        }
                        .disabled(engine.selectedFrameIdx >= results.count - 1)
                        
                        // 跳到下一个 FN 帧
                        Button("下一个漏检") {
                            if let nextFN = results.dropFirst(engine.selectedFrameIdx + 1).firstIndex(where: { $0.falseNegatives > 0 }) {
                                engine.selectedFrameIdx = nextFN
                            }
                        }
                        .font(.caption)
                        .disabled(!results.dropFirst(engine.selectedFrameIdx + 1).contains(where: { $0.falseNegatives > 0 }))
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 6)
                    
                    // ── 当前帧详情 ──
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 16) {
                            Text("帧 #\(frame.frameIndex)")
                                .font(.caption.bold())
                            // GT: 该帧人工标注的真实目标数 (绿色框)
                            Text("GT(真实): \(frame.gtBoxes.count)")
                                .font(.caption).foregroundColor(.green)
                            // Det: YOLO 检测到的目标数 (蓝色框)
                            Text("Det(检测): \(frame.detections.filter { $0.confidence >= engine.confThreshold }.count)")
                                .font(.caption).foregroundColor(.cyan)
                            Spacer()
                        }
                        HStack(spacing: 16) {
                            // TP: 正确检测数 (模型找到了, 且确实有车)
                            Text("TP(正确): \(frame.truePositives)")
                                .font(.caption).foregroundColor(.green)
                            // FP: 误检数 (模型说有车, 但实际没有)
                            Text("FP(误检): \(frame.falsePositives)")
                                .font(.caption).foregroundColor(.red)
                            // FN: 漏检数 (实际有车, 但模型没检测到)
                            Text("FN(漏检): \(frame.falseNegatives)")
                                .font(.caption).foregroundColor(.orange)
                            // IoU: 最佳匹配的交并比 (检测框与 GT 框重叠程度, 1.0=完美重合)
                            Text(String(format: "IoU(重叠度): %.3f", frame.bestIoU))
                                .font(.caption.monospaced())
                            Spacer()
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 4)
                    
                } else {
                    // 空状态
                    ZStack {
                        Color(nsColor: .windowBackgroundColor)
                        VStack(spacing: 12) {
                            Image(systemName: "scope")
                                .font(.system(size: 60))
                                .foregroundColor(.cyan.opacity(0.5))
                            Text("YOLO GT-vis 检测率测试")
                                .font(.title3)
                                .foregroundColor(.secondary)
                            Text("选择模型和数据源, 点击「开始测试」")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
        }
        // ★ 修复: 视图首次出现时自动加载当前模式的默认模型
        .onAppear {
            if !didAutoLoadModel {
                didAutoLoadModel = true
                autoLoadModelForMode(testMode)
            }
        }
    }
    
    // ── 辅助: 自动加载对应模式的默认模型 ──
    
    private func autoLoadModelForMode(_ mode: Int) {
        let isVSim = mode == 1
        let defaultName = YOLOGTTestEngine.defaultModelName(forVSim: isVSim)
        guard !defaultName.isEmpty else { return }
        // 更新 Picker 选中项 (会触发 onChange 加载模型)
        selectedModel = defaultName
    }
    
    // ── 辅助: 自动匹配标注文件 ──
    
    private func autoMatchAnnotation(for videoURL: URL) {
        annotationStatus = "正在搜索标注文件..."
        if let annURL = YOLOGTTestEngine.findAnnotation(forVideo: videoURL) {
            annotationURL = annURL
            annotationStatus = "自动匹配: \(annURL.lastPathComponent)"
        } else {
            annotationURL = nil
            annotationStatus = "未找到匹配的标注文件, 请手动选择"
        }
    }
    
    // ── 辅助: 是否可以开始测试 ──
    
    private var canStart: Bool {
        guard engine.modelLoaded || !selectedModel.isEmpty else { return false }
        if testMode == 0 {
            return videoURL != nil && annotationURL != nil
        } else {
            return true
        }
    }
    
    private func startTest() {
        // 确保模型已加载
        if !engine.modelLoaded && !selectedModel.isEmpty {
            _ = engine.loadBundledModel(name: selectedModel)
        }
        
        if testMode == 0 {
            guard let vURL = videoURL, let aURL = annotationURL else { return }
            engine.runVideoGTTest(videoURL: vURL, annotationURL: aURL)
        } else {
            engine.runVSimGTTest()
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 指标行视图
// ═══════════════════════════════════════════════════════════════

struct GTMetricRow: View {
    let label: String
    let value: Double
    let text: String
    let good: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption).foregroundColor(.secondary)
                Spacer()
                Text(text).font(.caption.monospaced())
                    .foregroundColor(good ? .green : .red)
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.gray.opacity(0.2))
                    RoundedRectangle(cornerRadius: 3)
                        .fill(good ? Color.green.opacity(0.7) : Color.red.opacity(0.7))
                        .frame(width: g.size.width * CGFloat(min(value, 1.0)))
                }
            }
            .frame(height: 6)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 报告导出 Document
// ═══════════════════════════════════════════════════════════════

struct GTReportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText] }
    
    let engine: YOLOGTTestEngine
    
    init(engine: YOLOGTTestEngine) {
        self.engine = engine
    }
    
    init(configuration: ReadConfiguration) throws {
        fatalError("Not supported")
    }
    
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("gt_report.md")
        engine.exportReport(to: tmpURL)
        let data = try Data(contentsOf: tmpURL)
        return FileWrapper(regularFileWithContents: data)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 命令行入口
// ═══════════════════════════════════════════════════════════════

/// CLI 模式的占位视图
struct GTTestCLIView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "scope")
                .font(.system(size: 60))
                .foregroundColor(.cyan)
            Text("YOLO GT-vis 检测率测试 (CLI 模式)")
                .font(.title2.bold())
            Text("请查看终端输出了解测试进度和结果")
                .foregroundColor(.secondary)
            ProgressView()
                .scaleEffect(1.5)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

/// 命令行模式执行器
final class GTTestCLIRunner {
    
    static func run() {
        let args = CommandLine.arguments
        
        // 解析参数
        let isVSim = args.contains("--vsim")
        let videoPath = Self.argValue(args, key: "--video")
        var annotationPath = Self.argValue(args, key: "--annotation")
        let modelPath = Self.argValue(args, key: "--model-path")
        let modelName = Self.argValue(args, key: "--model") ?? (isVSim ? "vsim_yolo11n" : "yolo11_640imgsz")
        let outputDir = Self.argValue(args, key: "--output") ?? NSTemporaryDirectory()
        let confStr = Self.argValue(args, key: "--conf")
        let confThreshold: Float = confStr.flatMap { Float($0) } ?? 0.25
        let sampleInterval = Self.argValue(args, key: "--sample-interval").flatMap { Int($0) } ?? GTTestCfg.videoSampleInterval
        GTTestCfg.videoSampleInterval = max(1, sampleInterval)
        
        // ★ 修复: CLI 模式也自动匹配标注
        if !isVSim && annotationPath == nil, let vPath = videoPath {
            let vURL = URL(fileURLWithPath: vPath)
            if let annURL = YOLOGTTestEngine.findAnnotation(forVideo: vURL) {
                annotationPath = annURL.path
                print("  自动匹配标注: \(annURL.path)")
            }
        }
        
        print("═══════════════════════════════════════════════════")
        print("  YOLO GT-vis 检测率测试 (CLI)")
        print("═══════════════════════════════════════════════════")
        print("  模式: \(isVSim ? "虚拟赛道" : "真实视频")")
        print("  模型: \(modelPath ?? modelName)")
        print("  置信度阈值: \(confThreshold)")
        print("  采样间隔: 每 \(GTTestCfg.videoSampleInterval) 帧")
        if !isVSim {
            print("  视频: \(videoPath ?? "未指定")")
            print("  标注: \(annotationPath ?? "未指定")")
        }
        print("  输出: \(outputDir)")
        print("═══════════════════════════════════════════════════")
        
        let engine = YOLOGTTestEngine()
        engine.confThreshold = confThreshold
        
        // 加载模型
        print("\n[1/3] 加载模型...")
        let modelLoaded: Bool
        if let modelPath = modelPath {
            let url = URL(fileURLWithPath: Self.expandHomePath(modelPath))
            modelLoaded = engine.loadModel(url, displayName: url.deletingPathExtension().lastPathComponent)
        } else {
            modelLoaded = engine.loadBundledModel(name: modelName)
        }
        guard modelLoaded else {
            print("错误: 无法加载模型 \(modelName)")
            exit(1)
        }
        print("  模型加载成功: \(modelPath ?? modelName)")
        
        // 等待预热
        Thread.sleep(forTimeInterval: 1.0)
        
        let semaphore = DispatchSemaphore(value: 0)
        
        if isVSim {
            // ── 虚拟赛道模式 ──
            print("\n[2/3] 生成虚拟测试场景 (\(GTTestCfg.vsimImageCount) 张)...")
            
            // 监听进度
            var lastProg = 0
            let progressTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
                let prog = Int(engine.progress * 100)
                if prog > lastProg {
                    lastProg = prog
                    print("  进度: \(prog)%  \(engine.status)")
                }
            }
            RunLoop.current.add(progressTimer, forMode: .common)
            
            engine.runVSimGTTest(keepImages: false)
            
            // 等待完成
            DispatchQueue.global().async {
                var sawRunning = false
                while true {
                    if engine.running { sawRunning = true }
                    if sawRunning && !engine.running { break }
                    Thread.sleep(forTimeInterval: 0.5)
                }
                progressTimer.invalidate()
                semaphore.signal()
            }
            
            // RunLoop 驱动
            while semaphore.wait(timeout: .now()) == .timedOut {
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
            }
            
        } else {
            // ── 真实视频模式 ──
            guard let vPath = videoPath else {
                print("错误: 请指定 --video 参数")
                exit(1)
            }
            guard let aPath = annotationPath else {
                print("错误: 请指定 --annotation 参数 (或确保 SAM3_Annotations 目录下有对应标注)")
                exit(1)
            }
            
            let vURL = URL(fileURLWithPath: vPath)
            let aURL = URL(fileURLWithPath: aPath)
            
            guard FileManager.default.fileExists(atPath: vPath) else {
                print("错误: 视频文件不存在: \(vPath)")
                exit(1)
            }
            guard FileManager.default.fileExists(atPath: aPath) else {
                print("错误: 标注文件不存在: \(aPath)")
                exit(1)
            }
            
            print("\n[2/3] 开始真实视频 GT 测试...")
            
            var lastProg = 0
            let progressTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
                let prog = Int(engine.progress * 100)
                if prog > lastProg {
                    lastProg = prog
                    print("  进度: \(prog)%  \(engine.status)")
                }
            }
            RunLoop.current.add(progressTimer, forMode: .common)
            
            engine.runVideoGTTest(videoURL: vURL, annotationURL: aURL, keepImages: false)
            
            DispatchQueue.global().async {
                var sawRunning = false
                while true {
                    if engine.running { sawRunning = true }
                    if sawRunning && !engine.running { break }
                    Thread.sleep(forTimeInterval: 0.5)
                }
                progressTimer.invalidate()
                semaphore.signal()
            }
            
            while semaphore.wait(timeout: .now()) == .timedOut {
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
            }
        }
        
        // ── 输出结果 ──
        print("\n[3/3] 测试结果:")
        if let s = engine.summary {
            print("  ┌─────────────────────────────────────┐")
            print(String(format: "  │ Recall (召回率)   : %6.1f%%          │", s.recall * 100))
            print(String(format: "  │ Precision (精确率): %6.1f%%          │", s.precision * 100))
            print(String(format: "  │ F1 Score          : %6.3f           │", s.f1))
            print(String(format: "  │ mAP@0.5           : %6.3f           │", s.mapAt50))
            print(String(format: "  │ 帧级检测率        : %6.1f%%          │", s.detectionRate * 100))
            print("  │ TP: \(String(format: "%5d", s.totalTP))  FP: \(String(format: "%5d", s.totalFP))  FN: \(String(format: "%5d", s.totalFN))   │")
            print("  │ 测试帧: \(String(format: "%5d", s.totalFrames))  GT目标: \(String(format: "%5d", s.totalGT))        │")
            print("  │ 评级: \(s.gradeText)                        │")
            print("  └─────────────────────────────────────┘")
            
            // 导出报告
            let reportURL = URL(fileURLWithPath: outputDir).appendingPathComponent("gt_test_report_\(modelName).md")
            engine.exportReport(to: reportURL)
            print("\n  报告已导出: \(reportURL.path)")
        } else {
            print("  测试未产生结果")
        }
        
        exit(0)
    }
    
    private static func argValue(_ args: [String], key: String) -> String? {
        guard let idx = args.firstIndex(of: key), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }

    private static func expandHomePath(_ path: String) -> String {
        guard path == "~" || path.hasPrefix("~/") else { return path }
        return NSHomeDirectory() + String(path.dropFirst())
    }
}
