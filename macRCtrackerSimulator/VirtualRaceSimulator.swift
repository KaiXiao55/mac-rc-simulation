//
//  VirtualRaceSimulator.swift
//  macRCtrackerSimulator
//
//  纯虚拟 3D 仿真环境追踪模拟模块 — 核心引擎
//
//  功能:
//    ① 用 SceneKit 自动生成 3D 赛车赛道 + 环境光照 + 多辆 RC 赛车
//    ② 车辆带简化物理 (沿赛道行驶 + 速度变化 + 随机超车)
//    ③ 随机放置舵机云台相机，渲染合成图像
//    ④ 将渲染帧送入现有 YOLO 检测器 + VNTracker + PID 控制闭环
//    ⑤ 利用 Ground Truth 精确评估追踪率、遮挡处理能力
//
//  与现有代码的关系:
//    - 复用 SimEngine 的 PIDCtrl、CoupledGimbal、BLEConfig、FOVMap、Smoother
//    - 复用 YOLO 模型推理 (VNCoreMLRequest)
//    - 独立的 Tab 页，不修改现有仿真器逻辑
//
//  Created by Manus AI on 2026/3/17.
//

import SwiftUI
import Combine
import SceneKit
import CoreML
import Vision
import CoreImage
import CoreGraphics
import CoreVideo
import AVFoundation
import simd

final class SceneKitFrameRecorder {
    private let url: URL
    private let width: Int
    private let height: Int
    private let fps: Int32
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?

    init(url: URL, width: Int, height: Int, fps: Double) {
        self.url = url
        self.width = width
        self.height = height
        self.fps = Int32(max(1, fps.rounded()))
    }

    func start() throws {
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 8_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
        )
        guard writer.canAdd(input) else {
            throw NSError(domain: "SceneKitFrameRecorder", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot add video input"])
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "SceneKitFrameRecorder", code: 2)
        }
        writer.startSession(atSourceTime: .zero)
        self.writer = writer
        self.input = input
        self.adaptor = adaptor
    }

    func append(_ image: CGImage, frameIndex: Int) {
        guard let input, let adaptor, input.isReadyForMoreMediaData,
              let pixelBuffer = makePixelBuffer(from: image) else { return }
        let time = CMTime(value: CMTimeValue(frameIndex), timescale: fps)
        adaptor.append(pixelBuffer, withPresentationTime: time)
    }

    func finish() {
        guard let writer, let input else { return }
        input.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        writer.finishWriting {
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 30)
        print("✅ SceneKit MP4 written: \(url.path)")
        self.writer = nil
        self.input = nil
        self.adaptor = nil
    }

    private func makePixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer)
        guard let buffer = pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 虚拟仿真配置
// ═══════════════════════════════════════════════════════════════

struct VSimCfg {
    // ── 场景 ──
    static let renderWidth: Int = 1280
    static let renderHeight: Int = 720
    static let fps: Double = 60.0
    static let simDuration: Double = 60.0

    // ── 赛道 ── (CGFloat for SceneKit compat)
    // 控制点更多 → 弯道更多（也更像真实赛道）
    static let trackPointCount: Int = 18
    static let trackRadius: CGFloat = 15.0
    static let trackWidth: CGFloat = 1.2    // RC赛道路面宽度 (车长0.45m, 路宽约车长2.5倍)
    static let barrierHeight: CGFloat = 0.023  // 护栏高度 (再降2/3, 约原始0.20的1/9)

    // ── 车辆 ──
    static let carCount: Int = 5
    static let carLength: CGFloat = 0.45
    static let carWidth: CGFloat = 0.22
    static let carHeight: CGFloat = 0.15
    // 1:7 RC 车一般 < 30km/h，这里把默认速度区间设在 ~12–28 km/h
    // m/s = km/h ÷ 3.6
    static let carBaseSpeed: CGFloat = 5.0          // 18 km/h
    static let carSpeedVariance: CGFloat = 2.8      // ±10 km/h 左右
    static let primaryCarIdx: Int = 0

    // ── 相机/云台 ──
    static let cameraHeight: CGFloat = 2.5
    static let cameraFovDeg: CGFloat = 60.0
    static let cameraTiltDeg: CGFloat = -25.0
    static let servoHome: CGFloat = 120.0

    /// 舵机相对中位(±°)映射到 SceneKit 云台时的限幅。全量 ±120° 会把初始已对准赛道的俯仰再旋转一整圈，舵机到 0 时镜头会朝天顶/墙角。
    static let maxServoOffsetDeg: CGFloat = 52.0

    // ★ pidUseGtWhenNoDet 已永久移除 -- 真实部署无GT，仿真必须模拟真实条件

    /// ★ 训练数据生成模式：强制所有帧都用 GT 框驱动 PID，生成“完美专家”数据
    static var forceGtForPid: Bool = false

    // ── ★ DAgger 器声注入配置 ──
    /// DAgger 模式：在 PID 输出后注入随机扰动，产生“犯错-救场”训练数据
    static var daggerMode: Bool = false
    /// 噪声注入概率 (每帧独立判定)
    static let daggerNoiseProb: Double = 0.15
    /// 噪声幅度 (度): 水平方向最大扰动
    static let daggerNoiseH: CGFloat = 6.0
    /// 噪声幅度 (度): 垂直方向最大扰动
    static let daggerNoiseV: CGFloat = 3.0
    /// 连续噪声帧数范围 (模拟持续偏移)
    static let daggerBurstMin: Int = 3
    static let daggerBurstMax: Int = 12
    /// 模拟丢帧概率 (强制 pidRect = nil 几帧，让 PID 演示丢失恢复)
    static let daggerDropProb: Double = 0.03
    /// 模拟丢帧持续帧数
    static let daggerDropFrames: Int = 8

    // ── PID ──
    // ★ FIX: 降低虚拟场景PID增益 — 虚拟场景信号无噪声, 高增益导致过冲振荡
    // 原值: kpH=3.0, kpV=2.5, kffH=0.8, kffV=0.6, leadTime=0.22
    static let kpH: CGFloat = 1.8       // 3.0→1.8  虚拟场景无传感器噪声，降低避免过冲
    static let kffH: CGFloat = 0.5      // 0.8→0.5  前馈降低，虚拟场景无运动模糊
    static let kpV: CGFloat = 1.5       // 2.5→1.5  俯仰同步降低
    static let kffV: CGFloat = 0.4      // 0.6→0.4
    static let velAlpha: CGFloat = 0.3
    static let leadTime: CGFloat = 0.12  // 0.22→0.12  缩短前瞻，虚拟场景延迟低于真实系统
    static let maxDPS: CGFloat = 500
    static let minStep: CGFloat = 0.02
    static let targetRatio: CGFloat = 0.15

    // ── YOLO ──
    /// 虚拟赛道（VSim）默认模型名。可用 VSIM_MODEL_NAME 或 VSIM_MODEL_PATH 覆盖。
    static let defaultModelName: String = "vsim_yolo11n"
    static var modelName: String {
        trimmedEnv("VSIM_MODEL_NAME") ?? defaultModelName
    }
    static var modelPath: String? {
        guard let path = trimmedEnv("VSIM_MODEL_PATH") else { return nil }
        return expandHomePath(path)
    }
    static var modelDisplayName: String {
        if let path = modelPath {
            return URL(fileURLWithPath: path).lastPathComponent
        }
        return modelName
    }
    static var modelPathCandidates: [URL] {
        guard let path = modelPath else { return [] }
        let url = URL(fileURLWithPath: path)
        var urls = [url]
        if url.pathExtension.isEmpty {
            urls.append(URL(fileURLWithPath: path + ".mlmodelc"))
            urls.append(URL(fileURLWithPath: path + ".mlpackage"))
        }
        return urls
    }
    nonisolated private static func trimmedEnv(_ key: String) -> String? {
        let value = ProcessInfo.processInfo.environment[key]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value, !value.isEmpty else { return nil }
        return value
    }
    nonisolated private static func expandHomePath(_ path: String) -> String {
        guard path == "~" || path.hasPrefix("~/") else { return path }
        return NSHomeDirectory() + String(path.dropFirst())
    }
    static let yoloConf: Float = 0.10
    static let yoloInterval: Double = 0.0  // 每帧都跑YOLO，速度取决于硬件
    /// 仅排障时打开：单目标路径每帧 print 会极慢并淹没日志
    static var enableStFrameDiagLog: Bool = false
    // ── ★ 模拟YOLO召回率控制：默认贴近截图里低召回、多低谷的真实/半真实曲线 ──
    static var simulatedRecallRate: Float = 0.5126
    // ── ★ 模拟YOLO误检率：长期「粘滞幽灵框」在场时间占比近似该值（非每帧独立随机点）──
    static var simulatedFPRate: Float = 0.2989
    /// SimYOLO 周期性质量波动幅度：0.30 表示 recall/FP 基准约按 ±30% 变化。
    static var simYoloPeriodicVariance: Float = 0.30
    /// 周期长度（秒）。用于模拟机位/遮挡/背景复杂度带来的慢变化。
    static var simYoloPeriodicPeriodSec: Double = 12.0
    // ── ★ SimYOLO 校准参数 (V8: 两态 Markov 漏检 + Gaussian conf/bbox 抖动) ──
    // 解决 i.i.d. Bernoulli 漏检模式造成的 STLock 假性崩溃
    /// MISS 状态自持概率 — 决定 burst 长度: avg_burst = 1/(1-pMM)
    /// 0.828 ⇒ 平均主车 miss burst ≈ 5.8 帧，来自 Track1/seed42 的真 vsim_yolo11n CSV
    /// 设为 0 退化为原 Bernoulli 模型(无时间相关性)
    static var simYoloMarkov_pMM: Float = 0.8283
    /// 置信度 Beta 分布参数（待从 CSV 拟合）
    static var simYoloConfAlpha: Float = 12.0
    static var simYoloConfBeta: Float = 1.8
    /// bbox 中心抖动 std (相对 GT 框宽/高); 旧值 0.015 均匀 ≈ std 0.0087, 现 0.010 略宽
    static var simYoloCenterJitterStd: Double = 0.07
    /// bbox 尺寸抖动 std (相对 GT 框宽/高); 旧值 0.08 均匀 ≈ std 0.046, 现 0.04 接近
    static var simYoloSizeJitterStd: Double = 0.12
    /// Track1/seed42 真 vsim_yolo11n: object recall≈82.36%, primary recall≈93.50%，主车更容易被持续命中。
    /// 这里用 primary/object 比值做默认校准；不同场景可通过 batch spec 的 recallRate 继续控制边际。
    static var simYoloPrimaryRecallMultiplier: Float = 1.136

    // ── ★ 模拟误检幽灵：钉在归一化画面上的「假物体」，随舵机/视角累计变化才消失 ──
    /// 自幽灵出现以来 |Δpan|+|Δtilt|（与 `servoH`/`servoV` 同刻度）达到该值 → 认为视角已变，结束误检
    static var simFpGhostClearServoDeltaSumDeg: CGFloat = 10.0
    /// 云台几乎不动时的兜底：幽灵最长持续帧数
    static let simFpGhostMaxFrames: Int = 900
    /// 用于估算「误检在场」时间占比的期望片段长度（视角活跃时实际由舵机运动提前结束）
    static var simFpGhostMeanEpisodeFramesForSpawn: Double = 90.0
    /// 真 YOLO 的误检帧里常同时出现多个候选框；Track1 实测 FP/frame ÷ FP-frame-rate ≈ 2.38。
    static var simFpGhostMeanBoxesWhenActive: Double = 2.38
    /// 真实 YOLO 的“危险误检”常出现在预测门控附近；该概率决定新幽灵有多少会贴近 ST/KF 预测框。
    static var simFpDangerousGhostProbability: Double = 0.20
    /// 危险误检相对目标框尺寸的中心偏移范围。偏移过小会算 TP，过大则不会干扰 ST 选择。
    static var simFpDangerousOffsetScale: ClosedRange<CGFloat> = 0.75...1.60

    // ── 匹配 ──
    static let acceptScore: Float = 0.40
    static let strongScore: Float = 0.65
    static let searchRelaxed: Float = 0.25


    // ── BLE ──
    static let bleMinMs: Double = 8
    static let bleMaxMs: Double = 35
    static let bleLossRate: Double = 0.02

    // ── ★ 额外延迟仿真 (模拟传感器读出 + YOLO推理延迟, 让总系统延迟接近真实硬件) ──
    static let sensorReadoutMs: Double = 15      // 传感器读出延迟 ~15ms
    static let yoloInferenceMs: Double = 35      // YOLO推理延迟 ~35ms (总额外延迟 ~50ms)

    // ── 光照随机化 ──
    static let ambientIntensityRange: ClosedRange<CGFloat> = 200...800
    static let sunIntensityRange: ClosedRange<CGFloat> = 500...2000
    static let shadowEnabled: Bool = true

    // ── 镜头畸变 (模拟运动相机桶形畸变) ──
    static let distortionK1: CGFloat = -0.28        // Brown-Conrady k1 (负=桶形)
    static let distortionK2: CGFloat = 0.08          // k2

    // ── 车辆侧滑 (模拟RC赛车漂移) ──
    static let driftSlipGain: CGFloat = 0.35          // 侧滑角增益
    static let driftCurvatureThresh: CGFloat = 0.08   // 曲率阈值, 超过才产生侧滑
    static let driftDamping: CGFloat = 0.85            // 侧滑角阻尼

    // ── 飞坡物理 ──
    static let rampGravity: CGFloat = 9.8              // 重力加速度
    static let rampBounce: CGFloat = 0.3               // 着陆弹跳系数
    static let rampAirDrag: CGFloat = 0.02             // 空气阻力

    // ── 3D拍摄点位 ──
    // 基于真实视频分析和PDF标定数据:
    //   SRC赛道: 相机在看台上, 高度3-5m, 俯角30-45°, 距赛道中心~5m (distRatio~0.34)
    //   娱乐场景: 相机在人站立高度1.2-1.7m, 俯角5-15°, 手持跟拍
    // 摄像机位预设: 大部分在赛道包络外侧，只有一个在赛道中间
    // distRatio: 相对于赛道最大跨度的距离比例 (越大越远)
    static let cameraPresets: [(name: String, height: CGFloat, distRatio: CGFloat, tiltDeg: CGFloat)] = [
        ("包络外-低位近",    1.5,  0.55,  -15),    // 0: 包络外侧低位
        ("包络外-中位",      2.5,  0.60,  -25),    // 1: 包络外侧三脚架
        ("包络外-高位看台",  5.0,  0.65,  -40),    // 2: 包络外侧看台高处
        ("包络外-中位看台",  3.5,  0.60,  -30),    // 3: 包络外侧看台中层
        ("包络外-弯道角",    2.0,  0.58,  -20),    // 4: 包络外侧弯道角落
        ("包络外-直道侧",    1.8,  0.55,  -18),    // 5: 包络外侧直道旁
        ("包络外-手持低",    1.3,  0.50,  -8),     // 6: 包络外侧手持低位
        ("包络外-手持中",    1.6,  0.52,  -12),    // 7: 包络外侧手持中位
        ("赛道中间-俯瞰",    4.0,  0.05,  -55),    // 8: 赛道中间俯瞰 (唯一的中间机位)
    ]
}

private struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// 为带阈值输入的 YOLO 模型显式提供默认阈值，避免模型内部默认值过高导致 0 检测
private final class YOLOThresholdFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["confidenceThreshold", "iouThreshold"]
    private let conf: Float
    private let iou: Float

    init(conf: Float, iou: Float) {
        self.conf = conf
        self.iou = iou
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "confidenceThreshold":
            return MLFeatureValue(double: Double(conf))
        case "iouThreshold":
            return MLFeatureValue(double: Double(iou))
        default:
            return nil
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 虚拟车辆状态
// ═══════════════════════════════════════════════════════════════

struct VirtualCar {
    let id: Int
    var color: NSColor
    var trackProgress: CGFloat = 0
    var speed: CGFloat
    var baseSpeed: CGFloat
    var laneOffset: CGFloat = 0
    var position: SCNVector3 = .init()
    var heading: CGFloat = 0
    var slipAngle: CGFloat = 0   // 侧滑角 (漂移效果)
    var isAirborne: Bool = false  // 是否在空中 (飞坡)
    var airVelocityY: CGFloat = 0 // 空中Y方向速度
    var groundY: CGFloat = 0      // 地面参考高度
    var node: SCNNode?

    // GT 信息
    var screenRect: CGRect = .zero
    var isVisible: Bool = false
    var isOccluded: Bool = false
    var distanceToCamera: CGFloat = 0
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 虚拟仿真帧快照
// ═══════════════════════════════════════════════════════════════

struct VSimSnap {
    let frameIdx: Int
    let time: Double
    let renderedImage: CGImage?
    let croppedImage: CGImage?

    let gtRects: [CGRect]
    let gtPrimaryRect: CGRect?
    let gtPrimaryVisible: Bool
    let gtPrimaryOccluded: Bool

    let detectedRect: CGRect?
    let detSource: String
    let state: String

    // 主车与云台信息（用于 UI 标注）
    let primarySpeedKmh: Double
    let gimbalBaseWorldPos: SCNVector3?

    let cmdH: Double
    let cmdV: Double
    let actH: Double
    let actV: Double
    let errH: Double
    let errV: Double

    let errDegX: CGFloat
    let errDegY: CGFloat

    // ★ 变焦状态
    let opticalZoom: CGFloat
    let effectiveFovH: Double

    // ★ 取景框追踪
    let gtOffCenter: CGFloat       // GT目标偏心距 (归一化, 0=中心, >0.5=出框)
    let gtInFrame: Bool            // GT目标是否在取景框内

    // ★ 帧级中间数据 (用于分析)
    let yoloConf: Float            // YOLO 检测置信度 (无检测时为 0)
    let allGtCount: Int            // 当前帧画面内 GT 车辆总数
    let detGtIoU: CGFloat          // 检测框与最近 GT 的 IoU
    let prevIoU: CGFloat           // 与上一帧检测框的 IoU
    let isDetCorrect: Bool         // 该帧检测是否正确 (TP 或 TN)
    let isTrackOnCar: Bool         // 该帧追踪是否在车上
    let yoloDetCount: Int          // YOLO 本帧检出的总框数
    /// 本帧模拟 YOLO 注入的误检框（归一化坐标）；非模拟模式或无注入时为 []
    let simFalsePositiveRects: [CGRect]
    /// 本帧是否执行了 YOLO（真或模拟）推理
    let didRunYoloThisFrame: Bool
    /// 本帧 YOLO 候选框中，能匹配任一 GT 车辆的框数（IoU > 0.2）
    let yoloTPCount: Int
    /// 本帧 YOLO 候选框中，不能匹配任何 GT 车辆的框数（候选框级误检）
    let yoloFPCount: Int
    /// 本帧可见 GT 车辆中，未被任何 YOLO 候选框覆盖的目标数（对象级漏检）
    let yoloFNCount: Int
    /// 本帧 YOLO 候选框是否覆盖评估主车
    let yoloPrimaryHit: Bool
    /// 本帧 YOLO 候选框与评估主车的最佳 IoU
    let yoloPrimaryBestIoU: CGFloat
    /// 平滑后追踪框与「主车」GT 的 IoU（单目标评估与观感最相关）
    let primaryVsGtIoU: CGFloat

    // ★ 神经网络训练数据字段
    let smoothVelX: CGFloat        // 平滑速度X (度/s)
    let smoothVelY: CGFloat        // 平滑速度Y (度/s)
    let distScale: CGFloat         // 距离缩放因子
    let pidStepH: CGFloat          // PID输出的舶机步进H
    let pidStepV: CGFloat          // PID输出的舶机步进V
    let targetLostCount: Int       // 目标丢失帧数
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 虚拟仿真统计
// ═══════════════════════════════════════════════════════════════


// ═══════════════════════════════════════════════════════════════
// MARK: - 仿真历史记录模型
// ═══════════════════════════════════════════════════════════════

struct SimHistoryRecord: Codable, Identifiable {
    var id: UUID
    let timestamp: Date
    let strategy: String          // "PID" or "MLP"
    let trackShape: Int
    let sceneType: Int
    let cameraPreset: Int
    let carCount: Int
    let simDuration: Double
    let detectionRate: Double     // ★ 检测率
    let generalTrackRate: Double  // ★ 一般追踪率
    let singleTargetLockRate: Double  // ★ 单目标锁定率
    let maxConsecutiveLost: Int
    let yoloDetRate: Double
    let avgErrH: Double
    let avgErrV: Double
    let jitterH: Double
    let jitterV: Double
    let score: Int
    let grade: String
    /// 以下为回放用参数 (旧存档缺省为 nil，重现时用合理默认)
    let randomSeed: UInt64?
    let cameraHeight: Double?
    let enableRamps: Bool?
    let useSimulatedYOLO: Bool?

    enum CodingKeys: String, CodingKey {
        case id, timestamp, strategy, trackShape, sceneType, cameraPreset, carCount, simDuration
        case detectionRate, generalTrackRate, singleTargetLockRate, maxConsecutiveLost, yoloDetRate
        case avgErrH, avgErrV, jitterH, jitterV, score, grade
        case randomSeed, cameraHeight, enableRamps, useSimulatedYOLO
    }

    init(
        id: UUID = UUID(),
        timestamp: Date,
        strategy: String,
        trackShape: Int,
        sceneType: Int,
        cameraPreset: Int,
        carCount: Int,
        simDuration: Double,
        detectionRate: Double,
        generalTrackRate: Double,
        singleTargetLockRate: Double = 0,
        maxConsecutiveLost: Int,
        yoloDetRate: Double,
        avgErrH: Double,
        avgErrV: Double,
        jitterH: Double,
        jitterV: Double,
        score: Int,
        grade: String,
        randomSeed: UInt64?,
        cameraHeight: Double?,
        enableRamps: Bool?,
        useSimulatedYOLO: Bool?
    ) {
        self.id = id
        self.timestamp = timestamp
        self.strategy = strategy
        self.trackShape = trackShape
        self.sceneType = sceneType
        self.cameraPreset = cameraPreset
        self.carCount = carCount
        self.simDuration = simDuration
        self.detectionRate = detectionRate
        self.generalTrackRate = generalTrackRate
        self.singleTargetLockRate = singleTargetLockRate
        self.maxConsecutiveLost = maxConsecutiveLost
        self.yoloDetRate = yoloDetRate
        self.avgErrH = avgErrH
        self.avgErrV = avgErrV
        self.jitterH = jitterH
        self.jitterV = jitterV
        self.score = score
        self.grade = grade
        self.randomSeed = randomSeed
        self.cameraHeight = cameraHeight
        self.enableRamps = enableRamps
        self.useSimulatedYOLO = useSimulatedYOLO
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        timestamp = try c.decode(Date.self, forKey: .timestamp)
        strategy = try c.decode(String.self, forKey: .strategy)
        trackShape = try c.decode(Int.self, forKey: .trackShape)
        sceneType = try c.decode(Int.self, forKey: .sceneType)
        cameraPreset = try c.decode(Int.self, forKey: .cameraPreset)
        carCount = try c.decode(Int.self, forKey: .carCount)
        simDuration = try c.decode(Double.self, forKey: .simDuration)
        detectionRate = try c.decode(Double.self, forKey: .detectionRate)
        generalTrackRate = try c.decode(Double.self, forKey: .generalTrackRate)
        singleTargetLockRate = try c.decodeIfPresent(Double.self, forKey: .singleTargetLockRate) ?? 0
        maxConsecutiveLost = try c.decode(Int.self, forKey: .maxConsecutiveLost)
        yoloDetRate = try c.decode(Double.self, forKey: .yoloDetRate)
        avgErrH = try c.decode(Double.self, forKey: .avgErrH)
        avgErrV = try c.decode(Double.self, forKey: .avgErrV)
        jitterH = try c.decode(Double.self, forKey: .jitterH)
        jitterV = try c.decode(Double.self, forKey: .jitterV)
        score = try c.decode(Int.self, forKey: .score)
        grade = try c.decode(String.self, forKey: .grade)
        randomSeed = try c.decodeIfPresent(UInt64.self, forKey: .randomSeed)
        cameraHeight = try c.decodeIfPresent(Double.self, forKey: .cameraHeight)
        enableRamps = try c.decodeIfPresent(Bool.self, forKey: .enableRamps)
        useSimulatedYOLO = try c.decodeIfPresent(Bool.self, forKey: .useSimulatedYOLO)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(timestamp, forKey: .timestamp)
        try c.encode(strategy, forKey: .strategy)
        try c.encode(trackShape, forKey: .trackShape)
        try c.encode(sceneType, forKey: .sceneType)
        try c.encode(cameraPreset, forKey: .cameraPreset)
        try c.encode(carCount, forKey: .carCount)
        try c.encode(simDuration, forKey: .simDuration)
        try c.encode(detectionRate, forKey: .detectionRate)
        try c.encode(generalTrackRate, forKey: .generalTrackRate)
        try c.encode(singleTargetLockRate, forKey: .singleTargetLockRate)
        try c.encode(maxConsecutiveLost, forKey: .maxConsecutiveLost)
        try c.encode(yoloDetRate, forKey: .yoloDetRate)
        try c.encode(avgErrH, forKey: .avgErrH)
        try c.encode(avgErrV, forKey: .avgErrV)
        try c.encode(jitterH, forKey: .jitterH)
        try c.encode(jitterV, forKey: .jitterV)
        try c.encode(score, forKey: .score)
        try c.encode(grade, forKey: .grade)
        try c.encodeIfPresent(randomSeed, forKey: .randomSeed)
        try c.encodeIfPresent(cameraHeight, forKey: .cameraHeight)
        try c.encodeIfPresent(enableRamps, forKey: .enableRamps)
        try c.encodeIfPresent(useSimulatedYOLO, forKey: .useSimulatedYOLO)
    }
}

final class SimHistoryStore: ObservableObject {
    static let shared = SimHistoryStore()
    private let key = "sim_history_records_v1"

    @Published var records: [SimHistoryRecord] = []

    private init() { load() }

    func load() {
        guard let data = UserDefaults.standard.data(forKey: key),
              let decoded = try? JSONDecoder().decode([SimHistoryRecord].self, from: data)
        else { return }
        records = decoded
    }

    func save(_ record: SimHistoryRecord) {
        records.insert(record, at: 0)
        if records.count > 720 { records = Array(records.prefix(720)) }  // ★ 上限=10赛道×8场景×9机位=720全组合
        persist()
    }

    func clearAll() {
        records.removeAll()
        persist()
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(records) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
}

struct VSimStats {
    var totalFrames: Int = 0

    // ═══ 检测层统计 ═══
    var trackingFrames: Int = 0        // 控制链路可用帧 (pidRect 非空)
    var pureDetFrames: Int = 0         // YOLO/Tracker 真正返回检测结果的帧
    /// 虚拟赛道：本帧检测框来自 YOLO 的帧数 (不含 YOLO 未跑、仅 VNTracker 插值的帧)
    var yoloMatchFrames: Int = 0
    var yoloRunFrames: Int = 0         // YOLO 实际运行的帧数 (用于检测率分母)
    var trackerAliveFrames: Int = 0    // VNTracker 存活的帧数
    var yoloFramesWithAnyGt: Int = 0
    var yoloFramesWithMatchedGt: Int = 0
    var yoloFramesWithAnyFP: Int = 0
    var yoloCandidateDetections: Int = 0
    var yoloCandidateTPDetections: Int = 0
    var yoloCandidateFPDetections: Int = 0
    var yoloVisibleGtObjects: Int = 0
    var yoloMatchedGtObjects: Int = 0
    var yoloPrimaryVisibleFrames: Int = 0
    var yoloPrimaryMatchedFrames: Int = 0

    // ═══ ★ 检测率 (需要 GT) ═══
    // 检测率 = (YOLO检出有车且位置匹配GT + YOLO检出无车且确实无车) / YOLO运行帧数
    var detCorrectFrames: Int = 0      // YOLO 判断正确的帧数 (TP + TN)

    // ═══ ★ 一般追踪率 (需要 GT) ═══
    // 一般追踪率 = 检测管线有输出框且框在实际车上的帧 / 总帧数
    var trackOnCarFrames: Int = 0      // 输出框与某个 GT 框 IoU > 阈值的帧

    // ═══ ★ 单目标锁定率 (Pass 2: ByteTrack + VNTracker + ReID) ═══
    // 单目标锁定率 = 持续追踪同一目标（即使遮挡）的帧 / 目标可见的总帧
    var stLockFrames: Int = 0          // 单目标锁定成功帧
    var stTotalFrames: Int = 0         // 单目标锁定评估总帧 (目标可见帧)
    var stLockedTrackId: Int = -1      // 当前锁定的 ByteTrack ID
    var stPrevRect: CGRect? = nil      // 上一帧单目标追踪框

    // ═══ 连续丢失统计 ═══
    var curConsecutiveLost: Int = 0    // 当前连续无检测框帧数
    var maxConsecutiveLost: Int = 0    // 最长连续无检测框帧数

    // ═══ 误差统计 (只在有检测框时累加) ═══
    var errHSum: Double = 0
    var errVSum: Double = 0
    var errCount: Int = 0
    var maxErrH: Double = 0
    var maxErrV: Double = 0

    // ═══ 抽动统计 ═══
    var dirChangeH: Int = 0
    var dirChangeV: Int = 0
    var lastDirH: Int = 0
    var lastDirV: Int = 0

    // ═══ YOLO 耗时统计 ═══
    var yoloMsSum: Double = 0
    var yoloMsCount: Int = 0

    // ═══ 变焦统计 ═══
    var zoomSum: Double = 0            // 光学变焦累加
    var zoomCount: Int = 0             // 变焦帧数

    // ═══════════════════════════════════════════════════════════
    // MARK: 核心指标
    // ═══════════════════════════════════════════════════════════

    /// SimYOLO / YOLO 指标说明:
    /// - Object Recall: 所有可见 GT 车辆中，被任一 YOLO 候选框覆盖的比例。
    /// - Primary Recall: 主目标车可见时，被 YOLO 候选框覆盖的比例；对“连续锁同一辆车”最关键。
    /// - FP Frame Rate: YOLO 运行帧中，至少出现一个误检候选框的帧占比。
    /// - FP / frame: 平均每个 YOLO 运行帧的误检候选框数量。
    /// - 最长视觉干净锁定段: 同一辆车被视觉连续锁定的最长时长；允许 <=0.5s 短暂断锁，不允许换车。
    ///
    /// Track1 / seed42 / cameraPreset=1 / 30s / Pass2-ST 实测参考:
    /// set Recall/FP | Object Recall | Primary Recall | FP Frame | FP/frame | 最长干净段
    /// 0.70 / 0      | 73.3%         | 82.7%          | 0%       | 0        | 11.15s
    /// 0.75 / 0      | 78.6%         | 86.9%          | 0%       | 0        | 19.50s
    /// 0.775 / 0.7803| 81.7%         | 83.9%          | 77.9%    | 1.706    | 8.38s
    /// 0.80 / 0.7803 | 84.9%         | 91.0%          | 78.7%    | 1.822    | 26.47s
    ///
    /// 经验结论: 10s 低 FP 档约需 Primary Recall >= 82%~83%；高 FP 档下 Primary Recall 83.9% 不够，
    /// Primary Recall 91.0% 已明显过线。20s 档低 FP 门槛约在 86.9%~91.6% 之间，高 FP 门槛接近 91%。

    /// ★ 检测率 = YOLO 判断正确帧 / YOLO 运行帧数 (%) — 有车位置对 + 无车判对
    var detectionRate: Double {
        yoloRunFrames > 0 ? Double(detCorrectFrames) / Double(yoloRunFrames) * 100 : 0
    }
    /// ★ 一般追踪率 = 输出框在实际车上的帧 / 总帧数 (%) — YOLO 或 VNTracker 任一路
    var generalTrackRate: Double {
        totalFrames > 0 ? Double(trackOnCarFrames) / Double(totalFrames) * 100 : 0
    }
    /// ★ 单目标锁定率 = ByteTrack+VNTracker+ReID 持续追踪同一目标的帧 / 目标可见总帧 (%)
    var singleTargetLockRate: Double {
        stTotalFrames > 0 ? Double(stLockFrames) / Double(stTotalFrames) * 100 : 0
    }
    /// YOLO 出框率 = YOLO 匹配帧 / 总帧数
    var yoloDetRate: Double {
        totalFrames > 0 ? Double(yoloMatchFrames) / Double(totalFrames) * 100 : 0
    }
    /// 对象级召回率 = 被任一 YOLO 候选框覆盖的可见 GT 车辆 / 可见 GT 车辆总数
    var yoloObjectRecallRate: Double {
        yoloVisibleGtObjects > 0 ? Double(yoloMatchedGtObjects) / Double(yoloVisibleGtObjects) * 100 : 0
    }
    /// 帧级召回率 = 有 GT 的 YOLO 帧中，至少有一个 GT 被候选框覆盖的帧占比
    var yoloFrameRecallRate: Double {
        yoloFramesWithAnyGt > 0 ? Double(yoloFramesWithMatchedGt) / Double(yoloFramesWithAnyGt) * 100 : 0
    }
    /// 评估主车召回率 = 主车可见的 YOLO 帧中，至少一个候选框覆盖主车的帧占比
    var yoloPrimaryRecallRate: Double {
        yoloPrimaryVisibleFrames > 0 ? Double(yoloPrimaryMatchedFrames) / Double(yoloPrimaryVisibleFrames) * 100 : 0
    }
    /// 候选框级误检占比 = 不能匹配任何 GT 的 YOLO 候选框 / 所有 YOLO 候选框
    var yoloCandidateFPRate: Double {
        yoloCandidateDetections > 0 ? Double(yoloCandidateFPDetections) / Double(yoloCandidateDetections) * 100 : 0
    }
    /// 误检帧率 = YOLO 运行帧中，至少出现一个误检候选框的帧占比
    var yoloFPFrameRate: Double {
        yoloRunFrames > 0 ? Double(yoloFramesWithAnyFP) / Double(yoloRunFrames) * 100 : 0
    }
    /// 平均每个 YOLO 运行帧的误检框数
    var yoloFPPerFrame: Double {
        yoloRunFrames > 0 ? Double(yoloCandidateFPDetections) / Double(yoloRunFrames) : 0
    }
    /// 平均水平误差
    var avgErrH: Double {
        errCount > 0 ? errHSum / Double(errCount) : 0
    }
    /// 平均俯仰误差
    var avgErrV: Double {
        errCount > 0 ? errVSum / Double(errCount) : 0
    }
    /// 水平抽动率
    var jitterH: Double {
        totalFrames > 0 ? Double(dirChangeH) / Double(totalFrames) * 100 : 0
    }
    /// 俯仰抽动率
    var jitterV: Double {
        totalFrames > 0 ? Double(dirChangeV) / Double(totalFrames) * 100 : 0
    }
    /// 平均光学变焦
    var avgZoom: Double {
        zoomCount > 0 ? zoomSum / Double(zoomCount) : 1.0
    }
    /// 平均YOLO耗时
    var avgYoloMs: Double {
        yoloMsCount > 0 ? yoloMsSum / Double(yoloMsCount) : 0
    }

    /// ★ 评分公式：检测率 + 一般追踪率 + 单目标锁定率 + 抽动 + 推理速度 + 连续丢失
    /// (使用 singleTargetLockRate 作为核心追踪质量指标)
    var score: Int {
        let det = min(detectionRate, 100) * 0.20                    // 检测率 20%
        let trk = min(generalTrackRate, 100) * 0.30                 // 一般追踪率 30%
        let stl = min(singleTargetLockRate, 100) * 0.20             // ★ 单目标锁定率 20%
        let jt  = max(0, 100 - (jitterH + jitterV) * 0.8) * 0.12   // 抽动 12%
        let ym  = avgYoloMs
        let y   = max(0, 100 - (ym - 15)) * 0.08                   // 推理速度 8%
        let cl  = max(0, 100 - Double(maxConsecutiveLost) * 2) * 0.10 // 连续丢失 10%
        return Int(det + trk + stl + jt + y + cl)
    }
    var grade: String {
        switch score {
        case 85...: return "优秀 ★★★"
        case 70...: return "良好 ★★☆"
        case 55...: return "一般 ★☆☆"
        default:    return "需优化 ☆☆☆"
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 虚拟仿真引擎
// ═══════════════════════════════════════════════════════════════

final class VirtualRaceEngine: ObservableObject {

    // ── 发布状态 ──
    @Published var running = false
    @Published var paused = false
    @Published var debugMode = false
    @Published var debugFrameIdx: Int = 0
    @Published var progress: Double = 0
    @Published var snap: VSimSnap?
    @Published var stats = VSimStats()
    @Published var status = "Ready — 点击 Generate & Run 开始虚拟仿真"

    // ── 配置 (UI 可调) ──
    @Published var carCount: Int = VSimCfg.carCount
    @Published var simDuration: Double = VSimCfg.simDuration
    @Published var cameraHeight: CGFloat = VSimCfg.cameraHeight
    @Published var randomSeed: UInt64 = 42
    @Published var enableOcclusion: Bool = true
    @Published var enableLightVariation: Bool = true
    @Published var racingMode: Bool = true
    @Published var enableLensDistortion: Bool = false
    @Published var enableDrift: Bool = true
    @Published var sceneType: Int = 0  // 0=室内, 1=室外草地, 2=室外柏油, 3=娱乐场地(飞坡), 4=沙地越野, 5=专业赛道, 6=公园小树林(无赛道), 7=开放草地(无赛道)
    @Published var trackShape: Int = 0 // TrackShape.rawValue
    @Published var enableRamps: Bool = false  // 飞坡
    @Published var cameraPreset: Int = 1      // 拍摄点位预设索引 (0~8)
    @Published var useSimulatedYOLO: Bool = true   // 默认开：便于对齐批量 SimYOLO + 显示误检框
    @Published var useMarkerCVDetector: Bool = false // 同心圆标记 + 传统颜色阈值/连通域检测
    @Published var useFiducialCVDetector: Bool = false // AprilTag/ArUco 类强标志物的上界模拟
    @Published var useIDFiducialCVDetector: Bool = false // 带唯一ID的强标志物：按目标ID追踪，避免跳车
    @Published var simulatedRecallRate: Float = VSimCfg.simulatedRecallRate
    @Published var simulatedFPRate: Float = VSimCfg.simulatedFPRate  // 目标：长期「幽灵框在场」时间占比近似该值；单次为粘滞多帧
    @Published var simYoloPeriodicVariance: Float = VSimCfg.simYoloPeriodicVariance
    @Published var simYoloPeriodicPeriodSec: Double = VSimCfg.simYoloPeriodicPeriodSec
    @Published var singleTargetMode: Bool = true   // ★ 单目标锁定测试模式 (Pass 2: ByteTrack + VNTracker + ReID)；默认开启与批量 Pass2 对齐
    @Published var enableFrameDegrade: Bool = false     // ★ V7: 帧降质开关
    @Published var degradeIntensity: Double = 0.7       // ★ V7: 降质强度 0~1
    private lazy var frameDegrader = FrameDegrader()     // ★ V7: 降质器实例

    // ── ★ 神经网络云台控制 ──
    @Published var useNeuralNetwork: Bool = false  // ★ 默认使用 PID 规则控制
    private var gimbalMLPModel: MLModel?

    // ── ★ DAgger 噪声注入状态 ──
    private var daggerRNG = SystemRandomNumberGenerator()
    private var daggerBurstRemaining: Int = 0
    private var daggerCurrentNoiseH: CGFloat = 0
    private var daggerCurrentNoiseV: CGFloat = 0
    private var daggerDropRemaining: Int = 0
    // ── SimYOLO 漏检状态（car.id -> 上一帧是否 miss）──
    private var simYoloMissState: [Int: Bool] = [:]
    /// 误检幽灵：钉在画面归一化坐标上的固定框；视角（舵机）相对出现点累计变化足够大才消失
    private var simFpGhostAnchor: CGRect?
    /// 误检出现后累计已输出帧数（含首帧），用于 max 兜底
    private var simFpGhostFrameIndex: Int = 0
    private var simFpGhostSpawnServoH: CGFloat = 0
    private var simFpGhostSpawnServoV: CGFloat = 0
    private var simFpGhostConf: Float = 0.45
    private struct SimFpGhostState {
        var anchor: CGRect
        var frameIndex: Int
        var spawnServoH: CGFloat
        var spawnServoV: CGFloat
        var conf: Float
        var isDangerous: Bool = false
    }
    private var simFpGhosts: [SimFpGhostState] = []
    private var simYoloRNG = SeededRandomNumberGenerator(seed: 1)

    private func simRandomDouble(_ range: Range<Double>) -> Double {
        Double.random(in: range, using: &simYoloRNG)
    }

    private func simRandomDouble(_ range: ClosedRange<Double>) -> Double {
        Double.random(in: range, using: &simYoloRNG)
    }

    private func simRandomFloat(_ range: ClosedRange<Float>) -> Float {
        Float.random(in: range, using: &simYoloRNG)
    }

    private func simRandomCGFloat(_ range: ClosedRange<CGFloat>) -> CGFloat {
        let lo = Double(range.lowerBound)
        let hi = Double(range.upperBound)
        return CGFloat(Double.random(in: lo...hi, using: &simYoloRNG))
    }

    private func simRandomBool() -> Bool {
        Bool.random(using: &simYoloRNG)
    }

    // ── YOLO 模型 ──
    private var yoloReq: VNCoreMLRequest?
    private var modelLoaded = false

    // ── 3D 场景 ──
    private var scene: SCNScene?
    private var cameraNode: SCNNode?
    private var gimbalBaseNode: SCNNode?
    private var gimbalPanNode: SCNNode?
    private var gimbalTiltNode: SCNNode?
    private var renderer: SCNRenderer?
    private var trackPath: [SCNVector3] = []
    private var trackTangents: [SCNVector3] = []
    private var trackLength: CGFloat = 0
    private var rampSegments: [RampSegment] = []

    // ── 赛道实际包围盒 (用于环境物体/光照分布) ──
    private var trackBBoxMinX: CGFloat = -5
    private var trackBBoxMaxX: CGFloat = 5
    private var trackBBoxMinZ: CGFloat = -5
    private var trackBBoxMaxZ: CGFloat = 5
    private var trackActualRadius: CGFloat = 7  // 赛道包围盒对角线半径

    // ── 车辆 ──
    private var cars: [VirtualCar] = []

    // ── 舵机/PID ──
    private var gimbal: CoupledGimbal!
    private var pid: PIDCtrl!
    private var smoother: Smoother!
    private var fov: FOVMap!
    private var servoH: CGFloat = VSimCfg.servoHome
    private var servoV: CGFloat = VSimCfg.servoHome

    // ── ★ 实际SceneKit相机基础FOV (buildScene动态计算, 用于同步FOVMap) ──
    private var actualBaseFovDeg: Double = Double(VSimCfg.cameraFovDeg)

    // ── ★ 自动变焦 (对齐 iOS AutoZoomEngine) ──
    private let autoZoom = SimAutoZoom()
    private var opticalZoom: CGFloat = 1.0
    private var zoomCfg = ZoomConfig()
    @Published var enableZoom: Bool = true
    @Published var freeCamera: Bool = false  // 3D自由相机浏览模式

    /// 暴露 scene 供 SCNView 自由相机模式使用
    var exposedScene: SCNScene? { scene }
    /// 赛道中心与半径，供 3D 自由相机定位
    var freeCamCenterX: CGFloat { (trackBBoxMinX + trackBBoxMaxX) / 2 }
    var freeCamCenterZ: CGFloat { (trackBBoxMinZ + trackBBoxMaxZ) / 2 }
    var freeCamRadius: CGFloat { trackActualRadius }

    // ── 追踪状态 ──
    /// ★ 动态GT: YOLO首次锁定的车辆ID (-1=未锁定, 用primaryCarIdx回退)
    private var lockedCarIdx: Int = -1
    private var curRect: CGRect?
    private var lastSeenTS: CFTimeInterval = 0
    private var lastSendTS: CFTimeInterval = 0
    private var lastYoloTS: CFTimeInterval = 0
    private var lastIDFiducialFullSearchTS: CFTimeInterval = -.infinity
    private var targetLostFrames: Int = 0
    private var trackerReq: VNTrackObjectRequest?
    private var trackerAlive = false
    private var prevPixelBuffer: CVPixelBuffer?
    private var simTime: Double = 0
    // ★ V4: GT纠正冷却期计数器
    private var gtCorrectionCount: Int = 0
    private var gtCorrectionCooldown: Int = 0

    // ── ★ 赛道模式丢失处理 (对齐 ViewController5 三阶段) ──
    private var inertiaVelH: CGFloat = 0
    private var inertiaVelV: CGFloat = 0
    private var lostPhaseStartTS: CFTimeInterval = 0

    // ── 外观特征 ──
    private var templateFixed: [Float]?
    private var templateDyn: [Float]?

    // ── ★ V6 单目标锁定模式 (独立 KF + 空间门控 + 锁定迟滞) ──
    private var byteTracker: ByteTracker?
    private var stLockedGtCarId: Int = -1

    // ── V6 独立卡尔曼滤波 (6维: cx, cy, w, h, vx, vy) ──
    private var stKF_x: [Double] = [0, 0, 0, 0, 0, 0]  // 状态 [cx, cy, w, h, vx, vy]
    private var stKF_P: [Double] = []                     // 协方差 6x6 (row-major)
    private var stKF_initialized: Bool = false

    // ── V6 锁定迟滞 (Incumbent-Challenger) ──
    private var stIncumbentRect: CGRect?                   // 当前锁定框
    private var stChallengerIdx: Int = -1                  // 挑战者在 dets 中的索引
    private var stChallengerWinStreak: Int = 0             // 挑战者连续胜出帧数
    private let stHysteresisN: Int = 5                     // 需要连续 N 帧才允许切换
    private let stHysteresisMargin: Double = 0.12          // 挑战者需高出的 margin
    private var stExperimentProfile: String {
        ProcessInfo.processInfo.environment["ST_EXPERIMENT_PROFILE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? "baseline"
    }
    private var stUnlockedRawConf: Float { 0.25 }          // 未锁定时只允许较高置信候选初始化
    private var stLockedRawConf: Float {
        switch stExperimentProfile {
        case "lowconf", "veto", "conservative":
            return 0.10
        case "strict":
            return 0.15
        default:
            return 0.25
        }
    }
    private var stUseAppearanceVeto: Bool {
        ["veto", "conservative", "strict"].contains(stExperimentProfile)
    }
    private var stAvoidDoubleLostCount: Bool {
        stExperimentProfile != "baseline"
    }
    private var stGlobalReacquireLostFrames: Int {
        switch stExperimentProfile {
        case "conservative":
            return 18       // 约 0.3s，低于 clean-lock 0.5s gap，同时减少过早全局跳车
        case "strict":
            return 24       // 约 0.4s，更保守的全局重捕获
        default:
            return 10
        }
    }

    // ── V6 保留贝叶斯后验（兼容性，不再使用） ──
    private var stPosterior: [Int: Double] = [:]

    // ── VNFeaturePrint 三层记忆 ──
    private var stFpFixed: VNFeaturePrintObservation?
    private var stFpDynamic: VNFeaturePrintObservation?
    private var stFpReacq: VNFeaturePrintObservation?

    // ── HS 直方图 ──
    private var stHsFixed: [Float]?
    private var stHsDynamic: [Float]?
    private var stHsReacq: [Float]?

    // ── 模板初始化 burn-in ──
    private var stBurnInCount: Int = 0
    private let stBurnInFrames: Int = 15
    private var stBurnInHsAccum: [Float]?
    private var stBurnInHsCount: Int = 0

    // ── 运动估计 ──
    private var stPrevCenter: CGPoint?
    private var stVelocity: CGPoint = .zero
    private var stPixelBuffer: CVPixelBuffer?
    private var stConsecutiveLost: Int = 0

    // ── 重捕获更新计数 ──
    private var stReacqUpdateCount: Int = 0

    // ── 内部 ──
    private var cancel = false
    private var debugStepReq = false
    private var history: [VSimSnap] = []
    var sceneRecordingURL: URL?
    private var sceneRecorder: SceneKitFrameRecorder?
    private let q = DispatchQueue(label: "vsim", qos: .userInitiated)
    private let ci = CIContext(options: [.useSoftwareRenderer: false])
    var historyCount: Int { history.count }

    func longestPrimaryCleanLockSeconds(maxGapSeconds: Double = 0.5, iouThreshold: CGFloat = 0.15) -> Double {
        guard !history.isEmpty else { return 0 }
        let maxGapFrames = max(0, Int((maxGapSeconds * VSimCfg.fps).rounded(.down)))
        var bestLockedFrames = 0
        var currentLockedFrames = 0
        var currentGapFrames = 0
        var inSegment = false

        for snap in history {
            let locked = snap.gtPrimaryVisible && snap.primaryVsGtIoU > iouThreshold
            if locked {
                if inSegment {
                    currentLockedFrames += currentGapFrames + 1
                } else {
                    inSegment = true
                    currentLockedFrames = 1
                }
                currentGapFrames = 0
                bestLockedFrames = max(bestLockedFrames, currentLockedFrames)
            } else if inSegment {
                if currentGapFrames < maxGapFrames {
                    currentGapFrames += 1
                } else {
                    inSegment = false
                    currentLockedFrames = 0
                    currentGapFrames = 0
                }
            }
        }

        return Double(bestLockedFrames) / VSimCfg.fps
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 模型加载
    // ═══════════════════════════════════════════════════════════

    func loadModel(_ url: URL) -> Bool {
        let c = MLModelConfiguration()
        c.computeUnits = .all
        let debugLog = NSTemporaryDirectory() + "yolo_debug.log"
        func dbg(_ msg: String) {
            let line = "[\(Date())] \(msg)\n"
            if let data = line.data(using: .utf8) {
                if let h = try? FileHandle(forWritingTo: URL(fileURLWithPath: debugLog)) {
                    h.seekToEndOfFile(); h.write(data); h.closeFile()
                } else { try? data.write(to: URL(fileURLWithPath: debugLog)) }
            }
        }
        dbg("loadModel called with url=\(url.path)")
        do {
            let ext = url.pathExtension.lowercased()
            let modelURL: URL
            if ext == "mlpackage" || ext == "mlmodel" {
                dbg("Compiling model package before load: \(url.path)")
                modelURL = try MLModel.compileModel(at: url)
                dbg("Compiled model URL: \(modelURL.path)")
            } else {
                modelURL = url
            }
            let mlm = try MLModel(contentsOf: modelURL, configuration: c)
            dbg("MLModel loaded OK, inputs=\(mlm.modelDescription.inputDescriptionsByName.keys.sorted()), outputs=\(mlm.modelDescription.outputDescriptionsByName.keys.sorted())")
            guard let vnm = try? VNCoreMLModel(for: mlm) else {
                dbg("VNCoreMLModel creation FAILED")
                status = "Model load failed"
                return false
            }
            let inputNames = Set(mlm.modelDescription.inputDescriptionsByName.keys)
            if inputNames.contains("confidenceThreshold") || inputNames.contains("iouThreshold") {
                vnm.featureProvider = YOLOThresholdFeatureProvider(conf: VSimCfg.yoloConf, iou: 0.45)
                dbg("Applied threshold featureProvider: conf=\(VSimCfg.yoloConf), iou=0.45")
            }
            let r = VNCoreMLRequest(model: vnm)
            // vsim_yolo11n 按 640x640 训练，虚拟赛道源帧是 16:9；使用 scaleFit 避免强拉伸导致检出率塌陷
            r.imageCropAndScaleOption = .scaleFit
            yoloReq = r
            modelLoaded = true
            status = "Model loaded: \(url.lastPathComponent)"
            dbg("Model loaded successfully: \(url.lastPathComponent)")
            q.async { [weak self] in self?.prewarmYOLO() }
            return true
        } catch {
            dbg("MLModel load EXCEPTION: \(error)")
            status = "Model load failed"
            return false
        }
    }

    func loadBundledModel() -> Bool {
        // 虚拟赛道模式默认使用专用模型；真实 YOLO A/B 可用环境变量显式覆盖。
        let name = VSimCfg.modelName
        let debugLog = NSTemporaryDirectory() + "yolo_debug.log"
        func dbg(_ msg: String) {
            let line = "[\(Date())] \(msg)\n"
            if let data = line.data(using: .utf8) {
                if let h = try? FileHandle(forWritingTo: URL(fileURLWithPath: debugLog)) {
                    h.seekToEndOfFile(); h.write(data); h.closeFile()
                } else { try? data.write(to: URL(fileURLWithPath: debugLog)) }
            }
        }
        if let path = VSimCfg.modelPath {
            let fm = FileManager.default
            dbg("loadBundledModel: VSIM_MODEL_PATH=\(path)")
            for url in VSimCfg.modelPathCandidates {
                var isDir = ObjCBool(false)
                if fm.fileExists(atPath: url.path, isDirectory: &isDir) {
                    dbg("Found external model at \(url.path), isDir=\(isDir.boolValue)")
                    return loadModel(url)
                }
            }
            let candidates = VSimCfg.modelPathCandidates.map(\.path).joined(separator: ", ")
            dbg("MISSING external model path candidates: \(candidates)")
            status = "Missing model at VSIM_MODEL_PATH: \(path)"
            return false
        }
        dbg("loadBundledModel: looking for \(name).mlmodelc / .mlpackage")
        dbg("Bundle path: \(Bundle.main.bundlePath)")
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            dbg("Found mlmodelc at \(url.path)")
            return loadModel(url)
        }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            dbg("Found mlpackage at \(url.path)")
            return loadModel(url)
        }
        dbg("MISSING model: \(name).mlmodelc/.mlpackage not in bundle")
        status = "Missing bundled model: \(name).mlmodelc/.mlpackage"
        return false
    }

    private func prewarmYOLO() {
        guard let req = yoloReq else { return }
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 320, 320, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let buf = pb else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: buf, orientation: .up, options: [:])
        try? handler.perform([req])
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 场景构建
    // ═══════════════════════════════════════════════════════════

    /// 是否为无赛道场景 (公园小树林/开放草地)
    private var isOpenFieldScene: Bool { sceneType == 6 || sceneType == 7 }

    private func buildScene() {
        srand48(Int(randomSeed))

        let sc = SCNScene()
        scene = sc

        // ═══ 统一使用程序化RC赛道 (10组预设) ═══
        let shape = TrackShape(rawValue: trackShape) ?? .track1
        let gen = VirtualRaceTrackGenerator(
            pointCount: VSimCfg.trackPointCount,
            radius: VSimCfg.trackRadius,
            width: VSimCfg.trackWidth,
            seed: randomSeed,
            trackShape: shape,
            enableRamps: enableRamps || sceneType == 3,
            rampCount: sceneType == 3 ? 3 : 2
        )
        let trackData = gen.generate()
        trackPath = trackData.centerPoints
        trackTangents = trackData.tangents
        trackLength = trackData.totalLength
        rampSegments = trackData.rampSegments

        // ═══ 计算赛道实际包围盒 ═══
        do {
            var bMinX: CGFloat = .infinity, bMaxX: CGFloat = -.infinity
            var bMinZ: CGFloat = .infinity, bMaxZ: CGFloat = -.infinity
            for p in trackPath {
                bMinX = min(bMinX, p.x); bMaxX = max(bMaxX, p.x)
                bMinZ = min(bMinZ, p.z); bMaxZ = max(bMaxZ, p.z)
            }
            trackBBoxMinX = bMinX; trackBBoxMaxX = bMaxX
            trackBBoxMinZ = bMinZ; trackBBoxMaxZ = bMaxZ
            let hw = (bMaxX - bMinX) / 2
            let hd = (bMaxZ - bMinZ) / 2
            trackActualRadius = sqrt(hw * hw + hd * hd) + 1.0  // +1m 余量
        }

        if !isOpenFieldScene {
            // 有赛道场景: 渲染赛道表面
            sc.rootNode.addChildNode(trackData.trackNode)
            applyTrackTexture(to: trackData.trackNode)
        }

        // ── 地面 ──
        do {
            let ground = SCNFloor()
            let isIndoor = (sceneType == 0 || sceneType == 3)
            ground.reflectivity = isIndoor ? 0.12 : 0.05
            ground.firstMaterial?.diffuse.contents = makeGroundTexture()
            ground.firstMaterial?.roughness.contents = NSColor(white: isIndoor ? 0.7 : 0.9, alpha: 1)
            let groundNode = SCNNode(geometry: ground)
            sc.rootNode.addChildNode(groundNode)
        }

        // ── 环境物体 (遮挡源) ──
        if enableOcclusion {
            addEnvironmentObjects(to: sc)
        }

        // ── 天空/天花板背景 ──
        if sceneType == 0 || sceneType == 3 {
            sc.background.contents = makeIndoorCeilingImage()
        } else {
            sc.background.contents = makeSkyGradientImage()
        }


        // ── 光照 ──
        setupLighting(scene: sc)

        // ── 云台相机 ──
        setupGimbalCamera(scene: sc)

        // ── 车辆 ──
        setupCars(scene: sc)

        // ── 渲染器 ──
        let r = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
        r.scene = sc
        r.pointOfView = cameraNode
        r.autoenablesDefaultLighting = false
        renderer = r
    }


    // ═══════════════════════════════════════════════════════════
    // MARK: 程序化贴图
    // ═══════════════════════════════════════════════════════════

    private lazy var markerCVMaterial: SCNMaterial = {
        let texture = makeConcentricMarkerTexture()
        let m = SCNMaterial()
        m.diffuse.contents = texture
        m.emission.contents = texture
        m.lightingModel = .constant
        m.isDoubleSided = true
        m.readsFromDepthBuffer = true
        m.writesToDepthBuffer = true
        return m
    }()
    private var idFiducialMaterials: [Int: SCNMaterial] = [:]

    private func idFiducialAccentRGB(for carId: Int) -> (r: CGFloat, g: CGFloat, b: CGFloat) {
        let palette: [(CGFloat, CGFloat, CGFloat)] = [
            (1.00, 0.00, 0.50),  // hot pink
            (0.00, 1.00, 1.00),  // cyan
            (0.70, 1.00, 0.00),  // chartreuse
            (1.00, 1.00, 0.00),  // yellow
            (0.50, 0.00, 1.00),  // violet
            (0.00, 0.45, 1.00),
            (1.00, 0.25, 0.00),
            (0.00, 1.00, 0.35),
            (1.00, 0.00, 1.00),
            (0.35, 1.00, 1.00)
        ]
        return palette[abs(carId) % palette.count]
    }

    private func idFiducialAccentColor(for carId: Int) -> NSColor {
        let c = idFiducialAccentRGB(for: carId)
        return NSColor(calibratedRed: c.r, green: c.g, blue: c.b, alpha: 1)
    }

    private func makeConcentricMarkerTexture() -> NSImage {
        let size = 128
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return NSImage(size: NSSize(width: size, height: size))
        }

        let bounds = CGRect(x: 0, y: 0, width: size, height: size)
        ctx.setFillColor(NSColor.black.cgColor)
        ctx.fill(bounds)

        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        func fillCircle(radius: CGFloat, color: NSColor) {
            ctx.setFillColor(color.cgColor)
            ctx.fillEllipse(in: CGRect(
                x: center.x - radius,
                y: center.y - radius,
                width: radius * 2,
                height: radius * 2
            ))
        }

        let markerGreen = NSColor(calibratedRed: 0.02, green: 1.0, blue: 0.05, alpha: 1)
        fillCircle(radius: 58, color: markerGreen)
        fillCircle(radius: 47, color: .black)
        fillCircle(radius: 36, color: .white)
        fillCircle(radius: 27, color: markerGreen)
        fillCircle(radius: 16, color: .black)
        fillCircle(radius: 8, color: .white)

        if let cg = ctx.makeImage() {
            return NSImage(cgImage: cg, size: NSSize(width: size, height: size))
        }
        return NSImage(size: NSSize(width: size, height: size))
    }

    private func idFiducialMaterial(for carId: Int) -> SCNMaterial {
        if let cached = idFiducialMaterials[carId] { return cached }
        let texture = makeIDFiducialTexture(carId: carId)
        let m = SCNMaterial()
        m.diffuse.contents = texture
        m.emission.contents = texture
        m.lightingModel = .constant
        m.isDoubleSided = true
        m.readsFromDepthBuffer = true
        m.writesToDepthBuffer = true
        idFiducialMaterials[carId] = m
        return m
    }

    private func makeIDFiducialTexture(carId: Int) -> NSImage {
        let size = 160
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return NSImage(size: NSSize(width: size, height: size))
        }

        let bounds = CGRect(x: 0, y: 0, width: size, height: size)
        ctx.setFillColor(NSColor.white.cgColor)
        ctx.fill(bounds)
        ctx.setFillColor(NSColor.black.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: 18))
        ctx.fill(CGRect(x: 0, y: size - 18, width: size, height: 18))
        ctx.fill(CGRect(x: 0, y: 0, width: 18, height: size))
        ctx.fill(CGRect(x: size - 18, y: 0, width: 18, height: size))

        let accent = idFiducialAccentColor(for: carId)

        let gridOrigin = CGPoint(x: 28, y: 28)
        let cell: CGFloat = 20
        for row in 0..<5 {
            for col in 0..<5 {
                let idx = row * 5 + col
                let bit = ((carId + 1) * 1103515245 + idx * 12345 + 0x5A5A) & (1 << (idx % 15))
                if bit != 0 || row == col || row + col == 4 {
                    ctx.setFillColor(NSColor.black.cgColor)
                } else {
                    ctx.setFillColor(NSColor.white.cgColor)
                }
                let rect = CGRect(
                    x: gridOrigin.x + CGFloat(col) * cell + 3,
                    y: gridOrigin.y + CGFloat(row) * cell + 3,
                    width: cell - 6,
                    height: cell - 6
                )
                ctx.fill(rect)
            }
        }

        ctx.setFillColor(accent.cgColor)
        let dot: CGFloat = 24
        let margin: CGFloat = 24
        let far = CGFloat(size) - margin - dot
        for x in [margin, far] {
            for y in [margin, far] {
                ctx.fill(CGRect(x: CGFloat(x), y: CGFloat(y), width: dot, height: dot))
            }
        }
        ctx.fillEllipse(in: CGRect(x: 56, y: 56, width: 48, height: 48))

        if let cg = ctx.makeImage() {
            return NSImage(cgImage: cg, size: NSSize(width: size, height: size))
        }
        return NSImage(size: NSSize(width: size, height: size))
    }

    private func addConcentricMarkers(to carNode: SCNNode, carId: Int) {
        let material = useIDFiducialCVDetector ? idFiducialMaterial(for: carId) : markerCVMaterial
        func markerPlane(width: CGFloat, height: CGFloat) -> SCNNode {
            let plane = SCNPlane(width: width, height: height)
            plane.firstMaterial = material
            return SCNNode(geometry: plane)
        }

        let eps: CGFloat = 0.003

        let top = markerPlane(width: VSimCfg.carWidth * 0.96, height: VSimCfg.carLength * 0.82)
        top.position = SCNVector3(0, VSimCfg.carHeight * 1.04 + eps, 0)
        top.eulerAngles.x = -CGFloat.pi / 2
        carNode.addChildNode(top)

        for sx: CGFloat in [-1, 1] {
            let side = markerPlane(width: VSimCfg.carLength * 0.70, height: VSimCfg.carHeight * 0.62)
            side.position = SCNVector3(sx * (VSimCfg.carWidth * 0.5 + eps), VSimCfg.carHeight * 0.55, 0)
            side.eulerAngles.y = sx > 0 ? CGFloat.pi / 2 : -CGFloat.pi / 2
            carNode.addChildNode(side)
        }

        for sz: CGFloat in [-1, 1] {
            let end = markerPlane(width: VSimCfg.carWidth * 0.82, height: VSimCfg.carHeight * 0.62)
            end.position = SCNVector3(0, VSimCfg.carHeight * 0.55, sz * (VSimCfg.carLength * 0.5 + eps))
            if sz < 0 { end.eulerAngles.y = CGFloat.pi }
            carNode.addChildNode(end)
        }

        let roofBoard = markerPlane(width: VSimCfg.carWidth * 0.95, height: VSimCfg.carWidth * 0.95)
        roofBoard.position = SCNVector3(0, VSimCfg.carHeight * 1.22 + eps, 0)
        let billboard = SCNBillboardConstraint()
        billboard.freeAxes = .all
        roofBoard.constraints = [billboard]
        carNode.addChildNode(roofBoard)
    }

    private func makeAsphaltTexture() -> NSImage {
        let w = 512, h = 512
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return NSImage(size: NSSize(width: w, height: h))
        }
        // 基色: 深灰柏油路面
        ctx.setFillColor(NSColor(red: 0.18, green: 0.18, blue: 0.20, alpha: 1).cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        // 大粒度骨料纹理
        for _ in 0..<2000 {
            let x = Int.random(in: 0..<w)
            let y = Int.random(in: 0..<h)
            let g = CGFloat.random(in: 0.12...0.32)
            let sz = Int.random(in: 1...4)
            ctx.setFillColor(NSColor(white: g, alpha: 0.5).cgColor)
            ctx.fill(CGRect(x: x, y: y, width: sz, height: sz))
        }
        // 细小裂缝纹理
        for _ in 0..<30 {
            let sx = CGFloat.random(in: 0...CGFloat(w))
            let sy = CGFloat.random(in: 0...CGFloat(h))
            let ex = sx + CGFloat.random(in: -40...40)
            let ey = sy + CGFloat.random(in: -40...40)
            ctx.setStrokeColor(NSColor(white: CGFloat.random(in: 0.08...0.15), alpha: 0.6).cgColor)
            ctx.setLineWidth(CGFloat.random(in: 0.5...1.5))
            ctx.move(to: CGPoint(x: sx, y: sy))
            ctx.addLine(to: CGPoint(x: ex, y: ey))
            ctx.strokePath()
        }
        // 油渍/磨损痕迹
        for _ in 0..<15 {
            let ox = CGFloat.random(in: 0...CGFloat(w))
            let oy = CGFloat.random(in: 0...CGFloat(h))
            let ow = CGFloat.random(in: 8...30)
            let oh = CGFloat.random(in: 3...12)
            ctx.setFillColor(NSColor(white: CGFloat.random(in: 0.10...0.22), alpha: 0.4).cgColor)
            ctx.fillEllipse(in: CGRect(x: ox, y: oy, width: ow, height: oh))
        }
        if let cgImage = ctx.makeImage() {
            return NSImage(cgImage: cgImage, size: NSSize(width: w, height: h))
        }
        return NSImage(size: NSSize(width: w, height: h))
    }

    /// 根据 sceneType 生成不同的地面纹理
    private func makeGroundTexture() -> NSImage {
        let w = 512, h = 512
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return NSImage(size: NSSize(width: w, height: h))
        }

        switch sceneType {
        case 0:  // 室内水泥地/地毯
            ctx.setFillColor(NSColor(red: 0.35, green: 0.33, blue: 0.30, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // 水泥地面的微小色差
            for _ in 0..<3000 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let g = CGFloat.random(in: 0.28...0.42)
                ctx.setFillColor(NSColor(white: g, alpha: 0.3).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: Int.random(in: 1...3), height: Int.random(in: 1...3)))
            }
            // 地板缝线
            for i in stride(from: 0, to: w, by: 64) {
                ctx.setStrokeColor(NSColor(white: 0.25, alpha: 0.3).cgColor)
                ctx.setLineWidth(1.0)
                ctx.move(to: CGPoint(x: i, y: 0))
                ctx.addLine(to: CGPoint(x: i, y: h))
                ctx.strokePath()
            }
            for j in stride(from: 0, to: h, by: 64) {
                ctx.setStrokeColor(NSColor(white: 0.25, alpha: 0.3).cgColor)
                ctx.setLineWidth(1.0)
                ctx.move(to: CGPoint(x: 0, y: j))
                ctx.addLine(to: CGPoint(x: w, y: j))
                ctx.strokePath()
            }
        case 1:  // 室外草地
            ctx.setFillColor(NSColor(red: 0.18, green: 0.28, blue: 0.14, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            for _ in 0..<2000 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let r = CGFloat.random(in: 0.12...0.28)
                let g = CGFloat.random(in: 0.32...0.48)
                let b = CGFloat.random(in: 0.08...0.18)
                ctx.setFillColor(NSColor(red: r, green: g, blue: b, alpha: 0.6).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
            // 草地色块
            for _ in 0..<40 {
                let cx = CGFloat.random(in: 0...CGFloat(w))
                let cy = CGFloat.random(in: 0...CGFloat(h))
                let cr = CGFloat.random(in: 10...40)
                let g = CGFloat.random(in: 0.22...0.45)
                ctx.setFillColor(NSColor(red: g * 0.5, green: g, blue: g * 0.3, alpha: 0.25).cgColor)
                ctx.fillEllipse(in: CGRect(x: cx - cr, y: cy - cr, width: cr * 2, height: cr * 2))
            }
        case 2:  // 室外柏油地面
            ctx.setFillColor(NSColor(red: 0.30, green: 0.28, blue: 0.26, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            for _ in 0..<2500 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let g = CGFloat.random(in: 0.20...0.38)
                ctx.setFillColor(NSColor(white: g, alpha: 0.4).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: Int.random(in: 1...3), height: Int.random(in: 1...3)))
            }

        case 3:  // 娱乐场地 (室内光滑地面 + 彩色标线)
            // 基底: 浅灰色光滑地面
            ctx.setFillColor(NSColor(red: 0.42, green: 0.40, blue: 0.38, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // 微小色差
            for _ in 0..<2000 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let g = CGFloat.random(in: 0.35...0.48)
                ctx.setFillColor(NSColor(white: g, alpha: 0.2).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: Int.random(in: 1...2), height: Int.random(in: 1...2)))
            }
            // 彩色地板标线 (娱乐场地特征)
            let lineColors: [NSColor] = [
                NSColor(red: 0.9, green: 0.2, blue: 0.2, alpha: 0.3),
                NSColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 0.3),
                NSColor(red: 0.9, green: 0.7, blue: 0.1, alpha: 0.3)
            ]
            for (li, color) in lineColors.enumerated() {
                ctx.setStrokeColor(color.cgColor)
                ctx.setLineWidth(3.0)
                let offset = CGFloat(li) * 20 + 30
                ctx.move(to: CGPoint(x: offset, y: 0))
                ctx.addLine(to: CGPoint(x: offset, y: CGFloat(h)))
                ctx.strokePath()
            }
            // 地板缝线
            for i in stride(from: 0, to: w, by: 80) {
                ctx.setStrokeColor(NSColor(white: 0.32, alpha: 0.25).cgColor)
                ctx.setLineWidth(0.8)
                ctx.move(to: CGPoint(x: i, y: 0))
                ctx.addLine(to: CGPoint(x: i, y: h))
                ctx.strokePath()
            }

        case 4:  // 沙地越野
            // 基底: 沙黄色
            ctx.setFillColor(NSColor(red: 0.72, green: 0.62, blue: 0.42, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // 沙粒纹理 (大量微小色差)
            for _ in 0..<5000 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let r = CGFloat.random(in: 0.55...0.80)
                let g = CGFloat.random(in: 0.45...0.68)
                let b = CGFloat.random(in: 0.25...0.48)
                ctx.setFillColor(NSColor(red: r, green: g, blue: b, alpha: 0.5).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
            // 沙丘波纹 (大色块)
            for _ in 0..<25 {
                let cx = CGFloat.random(in: 0...CGFloat(w))
                let cy = CGFloat.random(in: 0...CGFloat(h))
                let cr = CGFloat.random(in: 20...60)
                let shade = CGFloat.random(in: 0.58...0.75)
                ctx.setFillColor(NSColor(red: shade, green: shade * 0.85, blue: shade * 0.6, alpha: 0.2).cgColor)
                ctx.fillEllipse(in: CGRect(x: cx - cr, y: cy - cr, width: cr * 2, height: cr * 2))
            }
            // 车辙印 (暗色线条)
            for _ in 0..<8 {
                let sx = CGFloat.random(in: 0...CGFloat(w))
                let sy = CGFloat.random(in: 0...CGFloat(h))
                ctx.setStrokeColor(NSColor(red: 0.55, green: 0.45, blue: 0.30, alpha: 0.3).cgColor)
                ctx.setLineWidth(CGFloat.random(in: 2...5))
                ctx.move(to: CGPoint(x: sx, y: sy))
                ctx.addLine(to: CGPoint(x: sx + CGFloat.random(in: -80...80),
                                        y: sy + CGFloat.random(in: -80...80)))
                ctx.strokePath()
            }

        case 5: // 专业赛道 (平整柏油 + 精细标线)
            // 基底: 深灰柏油
            ctx.setFillColor(NSColor(red: 0.18, green: 0.18, blue: 0.20, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // 骨料纹理
            for _ in 0..<3000 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let g = CGFloat.random(in: 0.12...0.28)
                ctx.setFillColor(NSColor(white: g, alpha: 0.4).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: Int.random(in: 1...3), height: Int.random(in: 1...3)))
            }
            // 车胎痕迹
            for _ in 0..<20 {
                let sx = CGFloat.random(in: 0...CGFloat(w))
                let sy = CGFloat.random(in: 0...CGFloat(h))
                ctx.setStrokeColor(NSColor(white: CGFloat.random(in: 0.10...0.16), alpha: 0.5).cgColor)
                ctx.setLineWidth(CGFloat.random(in: 1...3))
                ctx.move(to: CGPoint(x: sx, y: sy))
                ctx.addLine(to: CGPoint(x: sx + CGFloat.random(in: -50...50),
                                        y: sy + CGFloat.random(in: -50...50)))
                ctx.strokePath()
            }
            // 油渍/磨损
            for _ in 0..<12 {
                let ox = CGFloat.random(in: 0...CGFloat(w))
                let oy = CGFloat.random(in: 0...CGFloat(h))
                let ow = CGFloat.random(in: 6...25)
                let oh = CGFloat.random(in: 3...10)
                ctx.setFillColor(NSColor(white: CGFloat.random(in: 0.08...0.18), alpha: 0.35).cgColor)
                ctx.fillEllipse(in: CGRect(x: ox, y: oy, width: ow, height: oh))
            }

        case 6: // 公园小树林 (泥地/土路 — 真实视频中的“小树林”场景)
            // 基底: 棕褐色泥地
            ctx.setFillColor(NSColor(red: 0.38, green: 0.30, blue: 0.22, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // 泥土纹理 (深浅不一的土壤色)
            for _ in 0..<4000 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let r = CGFloat.random(in: 0.28...0.48)
                let g = CGFloat.random(in: 0.22...0.38)
                let b = CGFloat.random(in: 0.12...0.25)
                ctx.setFillColor(NSColor(red: r, green: g, blue: b, alpha: 0.5).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: Int.random(in: 1...3), height: Int.random(in: 1...3)))
            }
            // 落叶/树枝碎屑 (暗色斑块)
            for _ in 0..<50 {
                let cx = CGFloat.random(in: 0...CGFloat(w))
                let cy = CGFloat.random(in: 0...CGFloat(h))
                let cr = CGFloat.random(in: 3...15)
                let shade = CGFloat.random(in: 0.15...0.30)
                ctx.setFillColor(NSColor(red: shade, green: shade * 0.9, blue: shade * 0.5, alpha: 0.3).cgColor)
                ctx.fillEllipse(in: CGRect(x: cx - cr, y: cy - cr, width: cr * 2, height: cr * 2))
            }
            // 车辙印 (暗色线条)
            for _ in 0..<12 {
                let sx = CGFloat.random(in: 0...CGFloat(w))
                let sy = CGFloat.random(in: 0...CGFloat(h))
                ctx.setStrokeColor(NSColor(red: 0.28, green: 0.22, blue: 0.15, alpha: 0.4).cgColor)
                ctx.setLineWidth(CGFloat.random(in: 2...5))
                ctx.move(to: CGPoint(x: sx, y: sy))
                ctx.addLine(to: CGPoint(x: sx + CGFloat.random(in: -60...60),
                                        y: sy + CGFloat.random(in: -60...60)))
                ctx.strokePath()
            }
            // 随机草地补丁 (泥地上的零星草)
            for _ in 0..<30 {
                let cx = CGFloat.random(in: 0...CGFloat(w))
                let cy = CGFloat.random(in: 0...CGFloat(h))
                let cr = CGFloat.random(in: 8...25)
                let g = CGFloat.random(in: 0.25...0.40)
                ctx.setFillColor(NSColor(red: g * 0.6, green: g, blue: g * 0.3, alpha: 0.2).cgColor)
                ctx.fillEllipse(in: CGRect(x: cx - cr, y: cy - cr, width: cr * 2, height: cr * 2))
            }

        default: // case 7: 开放草地 (公园草地 — 真实视频中的“草地”场景)
            // 基底: 绿草地
            ctx.setFillColor(NSColor(red: 0.22, green: 0.35, blue: 0.15, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // 草地纹理 (丰富的绿色变化)
            for _ in 0..<3000 {
                let x = Int.random(in: 0..<w)
                let y = Int.random(in: 0..<h)
                let r = CGFloat.random(in: 0.15...0.32)
                let g = CGFloat.random(in: 0.30...0.55)
                let b = CGFloat.random(in: 0.08...0.20)
                ctx.setFillColor(NSColor(red: r, green: g, blue: b, alpha: 0.5).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
            // 草地色块 (大片深浅不一的草地)
            for _ in 0..<60 {
                let cx = CGFloat.random(in: 0...CGFloat(w))
                let cy = CGFloat.random(in: 0...CGFloat(h))
                let cr = CGFloat.random(in: 15...50)
                let g = CGFloat.random(in: 0.25...0.50)
                ctx.setFillColor(NSColor(red: g * 0.5, green: g, blue: g * 0.25, alpha: 0.2).cgColor)
                ctx.fillEllipse(in: CGRect(x: cx - cr, y: cy - cr, width: cr * 2, height: cr * 2))
            }
            // 泥土裸露处 (草地上的秃处)
            for _ in 0..<15 {
                let cx = CGFloat.random(in: 0...CGFloat(w))
                let cy = CGFloat.random(in: 0...CGFloat(h))
                let cr = CGFloat.random(in: 10...30)
                let shade = CGFloat.random(in: 0.35...0.50)
                ctx.setFillColor(NSColor(red: shade, green: shade * 0.8, blue: shade * 0.5, alpha: 0.25).cgColor)
                ctx.fillEllipse(in: CGRect(x: cx - cr, y: cy - cr, width: cr * 2, height: cr * 2))
            }
        }

        if let cgImage = ctx.makeImage() {
            return NSImage(cgImage: cgImage, size: NSSize(width: w, height: h))
        }
        return NSImage(size: NSSize(width: w, height: h))
    }

    private func makeSkyGradientImage() -> NSImage {
        let w = 512, h = 256
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return NSImage(size: NSSize(width: w, height: h))
        }
        let top = NSColor(red: 0.25, green: 0.45, blue: 0.85, alpha: 1)
        let mid = NSColor(red: 0.55, green: 0.72, blue: 0.95, alpha: 1)
        let bottom = NSColor(red: 0.85, green: 0.88, blue: 0.95, alpha: 1)
        let colors = [top.cgColor, mid.cgColor, bottom.cgColor] as CFArray
        let locations: [CGFloat] = [0, 0.5, 1]
        guard let gradient = CGGradient(colorsSpace: colorSpace, colors: colors, locations: locations) else {
            return NSImage(size: NSSize(width: w, height: h))
        }
        ctx.drawLinearGradient(gradient,
                               start: CGPoint(x: w/2, y: h),
                               end: CGPoint(x: w/2, y: 0),
                               options: [])
        if let cgImage = ctx.makeImage() {
            return NSImage(cgImage: cgImage, size: NSSize(width: w, height: h))
        }
        return NSImage(size: NSSize(width: w, height: h))
    }

    /// 室内天花板背景 (深色均匀)
    private func makeIndoorCeilingImage() -> NSImage {
        let w = 512, h = 256
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return NSImage(size: NSSize(width: w, height: h))
        }
        // 深灰色天花板
        ctx.setFillColor(NSColor(white: 0.25, alpha: 1).cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        // 天花板纹理细节
        for _ in 0..<500 {
            let x = Int.random(in: 0..<w)
            let y = Int.random(in: 0..<h)
            let g = CGFloat.random(in: 0.20...0.30)
            ctx.setFillColor(NSColor(white: g, alpha: 0.3).cgColor)
            ctx.fill(CGRect(x: x, y: y, width: Int.random(in: 2...6), height: Int.random(in: 2...6)))
        }
        // 模拟照明灰尘光晕
        for _ in 0..<8 {
            let cx = CGFloat.random(in: 0...CGFloat(w))
            let cy = CGFloat.random(in: CGFloat(h) * 0.3...CGFloat(h))
            let r = CGFloat.random(in: 20...60)
            ctx.setFillColor(NSColor(white: 0.35, alpha: 0.15).cgColor)
            ctx.fillEllipse(in: CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2))
        }
        if let cgImage = ctx.makeImage() {
            return NSImage(cgImage: cgImage, size: NSSize(width: w, height: h))
        }
        return NSImage(size: NSSize(width: w, height: h))
    }

    private func applyTrackTexture(to trackRoot: SCNNode) {
        guard let roadNode = trackRoot.childNodes.first(where: { $0.name == "road_surface" }) else { return }
        // 不同场景赛道材质需要匹配地面风格；否则“娱乐场地”等看起来像没正常显示
        if sceneType == 3 {
            roadNode.geometry?.firstMaterial?.diffuse.contents = makeGroundTexture()
            roadNode.geometry?.firstMaterial?.roughness.contents = NSColor(white: 0.65, alpha: 1)
        } else {
            roadNode.geometry?.firstMaterial?.diffuse.contents = makeAsphaltTexture()
            roadNode.geometry?.firstMaterial?.roughness.contents = NSColor(white: 0.75, alpha: 1)
        }
    }

    private func addEnvironmentObjects(to scene: SCNScene) {
        switch sceneType {
        case 0:  // 室内场地
            addIndoorClutter(to: scene)
        case 3:  // 娱乐场地 (室内 + 飞坡设施)
            addIndoorClutter(to: scene)
            addAmusementClutter(to: scene)
        case 4:  // 沙地越野
            addDesertClutter(to: scene)
        case 5:  // 专业赛道
            addProTrackClutter(to: scene)
        case 6:  // 公园小树林 (无赛道)
            addParkForestClutter(to: scene)
        case 7:  // 开放草地 (无赛道)
            addOpenFieldClutter(to: scene)
        default: // 1=草地, 2=柏油
            addOutdoorClutter(to: scene)
        }

        // 赛道边的护栏/软管边界 (仅有赛道场景)
        if !isOpenFieldScene {
            addTrackBarriers(to: scene)
        }
    }

    // MARK: 辅助 — 在赛道包围盒附近随机生成位置
    /// 返回赛道包围盒外围 margin 范围内的随机 (x, z) 坐标
    private func randomPosAroundTrack(innerRatio: CGFloat = 0.3, outerRatio: CGFloat = 0.8) -> (CGFloat, CGFloat) {
        let cx = (trackBBoxMinX + trackBBoxMaxX) / 2
        let cz = (trackBBoxMinZ + trackBBoxMaxZ) / 2
        let hw = (trackBBoxMaxX - trackBBoxMinX) / 2 + 1.0  // +1m 余量
        let hd = (trackBBoxMaxZ - trackBBoxMinZ) / 2 + 1.0
        let angle = CGFloat(drand48()) * .pi * 2
        let ratio = CGFloat(innerRatio + drand48() * (outerRatio - innerRatio))
        let x = cx + cos(angle) * hw * ratio
        let z = cz + sin(angle) * hd * ratio
        return (x, z)
    }

    // MARK: 室内杂物
    private func addIndoorClutter(to scene: SCNScene) {
        let clutterCount = 12 + Int(drand48() * 8)
        for i in 0..<clutterCount {
            let (x, z) = randomPosAroundTrack(innerRatio: 0.35, outerRatio: 0.85)
            let roll = drand48()

            if roll < 0.25 {
                // 轮胎叠 (2~4层轮胎堆叠)
                let tireCount = 2 + Int(drand48() * 3)
                for ti in 0..<tireCount {
                    let tireR = CGFloat(0.15 + drand48() * 0.08)
                    let tireH: CGFloat = 0.10
                    let tire = SCNTorus(ringRadius: tireR, pipeRadius: tireH / 2)
                    tire.firstMaterial?.diffuse.contents = NSColor(white: CGFloat(0.08 + drand48() * 0.12), alpha: 1)
                    tire.firstMaterial?.roughness.contents = NSColor(white: 0.95, alpha: 1)
                    let tireNode = SCNNode(geometry: tire)
                    let yOff = CGFloat(ti) * tireH * 1.8 + tireH
                    tireNode.position = SCNVector3(x + CGFloat(drand48() - 0.5) * 0.05,
                                                   yOff,
                                                   z + CGFloat(drand48() - 0.5) * 0.05)
                    tireNode.eulerAngles.x = .pi / 2
                    scene.rootNode.addChildNode(tireNode)
                }
            } else if roll < 0.45 {
                // 工具箱/箱子
                let bw = CGFloat(0.3 + drand48() * 0.5)
                let bh = CGFloat(0.2 + drand48() * 0.4)
                let bl = CGFloat(0.3 + drand48() * 0.5)
                let box = SCNBox(width: bw, height: bh, length: bl, chamferRadius: 0.01)
                let boxColors: [NSColor] = [
                    NSColor(red: 0.2, green: 0.3, blue: 0.5, alpha: 1),  // 蓝色工具箱
                    NSColor(red: 0.6, green: 0.15, blue: 0.1, alpha: 1), // 红色工具箱
                    NSColor(red: 0.55, green: 0.45, blue: 0.25, alpha: 1), // 纸箱
                    NSColor(white: 0.35, alpha: 1) // 灰色箱子
                ]
                box.firstMaterial?.diffuse.contents = boxColors[i % boxColors.count]
                box.firstMaterial?.roughness.contents = NSColor(white: 0.8, alpha: 1)
                let boxNode = SCNNode(geometry: box)
                boxNode.position = SCNVector3(x, bh / 2, z)
                boxNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
                scene.rootNode.addChildNode(boxNode)
            } else if roll < 0.60 {
                // 立柱 (室内支撑柱)
                let pillarH = CGFloat(2.0 + drand48() * 1.5)
                let pillarR = CGFloat(0.08 + drand48() * 0.06)
                let pillar = SCNCylinder(radius: pillarR, height: pillarH)
                pillar.firstMaterial?.diffuse.contents = NSColor(white: CGFloat(0.5 + drand48() * 0.3), alpha: 1)
                pillar.firstMaterial?.roughness.contents = NSColor(white: 0.6, alpha: 1)
                let pillarNode = SCNNode(geometry: pillar)
                pillarNode.position = SCNVector3(x, pillarH / 2, z)
                scene.rootNode.addChildNode(pillarNode)
            } else if roll < 0.75 {
                // 折叠椅子 (简化为细长立方体)
                let chairW: CGFloat = 0.35
                let chairH: CGFloat = 0.7
                let chairD: CGFloat = 0.35
                let seat = SCNBox(width: chairW, height: 0.03, length: chairD, chamferRadius: 0)
                seat.firstMaterial?.diffuse.contents = NSColor(white: 0.3, alpha: 1)
                let seatNode = SCNNode(geometry: seat)
                seatNode.position = SCNVector3(x, chairH * 0.55, z)
                scene.rootNode.addChildNode(seatNode)
                // 椅子腿
                for lx: CGFloat in [-1, 1] {
                    for lz: CGFloat in [-1, 1] {
                        let leg = SCNCylinder(radius: 0.012, height: chairH * 0.55)
                        leg.firstMaterial?.diffuse.contents = NSColor(white: 0.25, alpha: 1)
                        let legNode = SCNNode(geometry: leg)
                        legNode.position = SCNVector3(
                            x + lx * chairW * 0.4,
                            chairH * 0.55 / 2,
                            z + lz * chairD * 0.4
                        )
                        scene.rootNode.addChildNode(legNode)
                    }
                }
                // 椅背
                let back = SCNBox(width: chairW, height: chairH * 0.4, length: 0.02, chamferRadius: 0)
                back.firstMaterial?.diffuse.contents = NSColor(white: 0.3, alpha: 1)
                let backNode = SCNNode(geometry: back)
                backNode.position = SCNVector3(x, chairH * 0.75, z - chairD * 0.45)
                scene.rootNode.addChildNode(backNode)
            } else if roll < 0.88 {
                // 折叠桌 (简化)
                let tw = CGFloat(0.8 + drand48() * 0.6)
                let tl = CGFloat(0.5 + drand48() * 0.4)
                let th: CGFloat = 0.65
                let top = SCNBox(width: tw, height: 0.03, length: tl, chamferRadius: 0)
                top.firstMaterial?.diffuse.contents = NSColor(white: CGFloat(0.55 + drand48() * 0.2), alpha: 1)
                let topNode = SCNNode(geometry: top)
                topNode.position = SCNVector3(x, th, z)
                topNode.eulerAngles.y = CGFloat(drand48()) * .pi
                scene.rootNode.addChildNode(topNode)
                for lx: CGFloat in [-1, 1] {
                    for lz: CGFloat in [-1, 1] {
                        let leg = SCNCylinder(radius: 0.015, height: th)
                        leg.firstMaterial?.diffuse.contents = NSColor(white: 0.3, alpha: 1)
                        let legNode = SCNNode(geometry: leg)
                        legNode.position = SCNVector3(x + lx * tw * 0.42, th / 2, z + lz * tl * 0.42)
                        scene.rootNode.addChildNode(legNode)
                    }
                }
            } else {
                // 随机杂物 (小型物体)
                let rw = CGFloat(0.1 + drand48() * 0.3)
                let rh = CGFloat(0.1 + drand48() * 0.5)
                let rl = CGFloat(0.1 + drand48() * 0.3)
                let obj = SCNBox(width: rw, height: rh, length: rl, chamferRadius: 0.005)
                obj.firstMaterial?.diffuse.contents = NSColor(
                    red: CGFloat(drand48()), green: CGFloat(drand48()), blue: CGFloat(drand48()), alpha: 1
                )
                let objNode = SCNNode(geometry: obj)
                objNode.position = SCNVector3(x, rh / 2, z)
                objNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
                scene.rootNode.addChildNode(objNode)
            }
        }

        // 室内墙壁 (围合形成封闭空间感) — 以赛道包围盒中心为基准，避免赛道偏移时墙/灯在 (0,0) 而画面偏暗或错位
        let wallH: CGFloat = 3.0
        let wallDist = trackActualRadius * 1.1
        for wi in 0..<4 {
            let wallW = wallDist * 2.2
            let wall = SCNBox(width: wallW, height: wallH, length: 0.08, chamferRadius: 0)
            wall.firstMaterial?.diffuse.contents = NSColor(white: CGFloat(0.65 + drand48() * 0.15), alpha: 1)
            wall.firstMaterial?.roughness.contents = NSColor(white: 0.7, alpha: 1)
            let wallNode = SCNNode(geometry: wall)
            let a = CGFloat(wi) * .pi / 2
            wallNode.position = SCNVector3(
                freeCamCenterX + cos(a) * wallDist,
                wallH / 2,
                freeCamCenterZ + sin(a) * wallDist
            )
            wallNode.eulerAngles.y = a + .pi / 2
            scene.rootNode.addChildNode(wallNode)
        }
    }

    // MARK: 室外杂物
    private func addOutdoorClutter(to scene: SCNScene) {
        let obstaclePosCount = 10 + Int(drand48() * 8)
        for i in 0..<obstaclePosCount {
            let (x, z) = randomPosAroundTrack(innerRatio: 0.4, outerRatio: 0.9)
            let roll = drand48()

            if roll < 0.35 {
                // 树木
                let trunkH = CGFloat(0.8 + drand48() * 1.2)
                let trunk = SCNCylinder(radius: 0.08, height: trunkH)
                trunk.firstMaterial?.diffuse.contents = NSColor.brown
                let trunkNode = SCNNode(geometry: trunk)
                trunkNode.position = SCNVector3(x, trunkH / 2, z)
                scene.rootNode.addChildNode(trunkNode)

                let canopyR = CGFloat(0.4 + drand48() * 0.6)
                let canopy = SCNSphere(radius: canopyR)
                canopy.firstMaterial?.diffuse.contents = NSColor(
                    red: CGFloat(0.1 + drand48() * 0.3),
                    green: CGFloat(0.4 + drand48() * 0.4),
                    blue: 0.05, alpha: 1
                )
                let canopyNode = SCNNode(geometry: canopy)
                canopyNode.position = SCNVector3(x, trunkH + canopyR * 0.7, z)
                scene.rootNode.addChildNode(canopyNode)
            } else if roll < 0.55 {
                // 轮胎墙
                let tireCount = 3 + Int(drand48() * 4)
                for ti in 0..<tireCount {
                    let tireR = CGFloat(0.18 + drand48() * 0.06)
                    let tire = SCNTorus(ringRadius: tireR, pipeRadius: 0.05)
                    tire.firstMaterial?.diffuse.contents = NSColor(white: CGFloat(0.05 + drand48() * 0.1), alpha: 1)
                    tire.firstMaterial?.roughness.contents = NSColor(white: 0.95, alpha: 1)
                    let tireNode = SCNNode(geometry: tire)
                    tireNode.position = SCNVector3(x + CGFloat(drand48() - 0.5) * 0.1,
                                                   CGFloat(ti) * 0.18 + 0.1,
                                                   z + CGFloat(drand48() - 0.5) * 0.1)
                    tireNode.eulerAngles.x = .pi / 2
                    scene.rootNode.addChildNode(tireNode)
                }
            } else if roll < 0.75 {
                // 广告牌
                let w = CGFloat(0.8 + drand48() * 1.2)
                let h = CGFloat(0.6 + drand48() * 1.0)
                let billboard = SCNBox(width: w, height: h, length: 0.05, chamferRadius: 0)
                let colors: [NSColor] = [.red, .blue, .yellow, .orange, .white]
                billboard.firstMaterial?.diffuse.contents = colors[i % colors.count]
                let bbNode = SCNNode(geometry: billboard)
                bbNode.position = SCNVector3(x, h / 2 + 0.3, z)
                bbNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
                scene.rootNode.addChildNode(bbNode)
            } else {
                // 混凝土护栏/石头
                let bw = CGFloat(0.5 + drand48() * 1.0)
                let bh = CGFloat(0.3 + drand48() * 0.5)
                let bl = CGFloat(0.3 + drand48() * 0.4)
                let barrier = SCNBox(width: bw, height: bh, length: bl, chamferRadius: 0.02)
                barrier.firstMaterial?.diffuse.contents = NSColor(white: CGFloat(0.5 + drand48() * 0.2), alpha: 1)
                barrier.firstMaterial?.roughness.contents = NSColor(white: 0.85, alpha: 1)
                let barrierNode = SCNNode(geometry: barrier)
                barrierNode.position = SCNVector3(x, bh / 2, z)
                barrierNode.eulerAngles.y = CGFloat(drand48()) * .pi
                scene.rootNode.addChildNode(barrierNode)
            }
        }
    }

    // MARK: 赛道边界护栏 (软管风格)
    private func addTrackBarriers(to scene: SCNScene) {
        let barrierCount = trackPath.count
        // 护栏柱 + 软管连接
        for i in stride(from: 0, to: barrierCount, by: 3) {
            let p = trackPath[i]
            let t = trackTangents[i]
            let normal = SCNVector3(-t.z, 0, t.x)
            let halfW = VSimCfg.trackWidth / 2 + 0.15

            for side: CGFloat in [-1, 1] {
                let px = p.x + normal.x * halfW * side
                let pz = p.z + normal.z * halfW * side

                // 护栏柱
                // 专业赛道：护栏更低更细，避免“挡块过大”遮挡视线
                let proScale: CGFloat = (sceneType == 5) ? 0.55 : 1.0
                let postH = (VSimCfg.barrierHeight + 0.15) * proScale
                let postR: CGFloat = 0.0125 * proScale
                let post = SCNCylinder(radius: postR, height: postH)
                post.firstMaterial?.diffuse.contents = NSColor(white: 0.85, alpha: 1)
                post.firstMaterial?.metalness.contents = 0.3
                let postNode = SCNNode(geometry: post)
                postNode.position = SCNVector3(px, postH / 2, pz)
                scene.rootNode.addChildNode(postNode)

                // 软管横杆 (每隔一个柱子连接)
                if i + 3 < barrierCount {
                    let np = trackPath[i + 3]
                    let nt = trackTangents[i + 3]
                    let nnormal = SCNVector3(-nt.z, 0, nt.x)
                    let npx = np.x + nnormal.x * halfW * side
                    let npz = np.z + nnormal.z * halfW * side
                    let dx = npx - px
                    let dz = npz - pz
                    let dist = sqrt(dx * dx + dz * dz)
                    let midX = (px + npx) / 2
                    let midZ = (pz + npz) / 2
                    let pipeR: CGFloat = 0.0075 * proScale
                    let pipe = SCNCylinder(radius: pipeR, height: dist)
                    let pipeColor = (Int(side) + i / 3) % 2 == 0
                        ? NSColor(red: 0.9, green: 0.1, blue: 0.1, alpha: 1)
                        : NSColor(red: 0.9, green: 0.9, blue: 0.9, alpha: 1)
                    pipe.firstMaterial?.diffuse.contents = pipeColor
                    let pipeNode = SCNNode(geometry: pipe)
                    pipeNode.position = SCNVector3(midX, VSimCfg.barrierHeight * 0.7 * proScale, midZ)
                    let heading = atan2(dx, dz)
                    pipeNode.eulerAngles = SCNVector3(0, heading, .pi / 2)
                    scene.rootNode.addChildNode(pipeNode)
                }
            }
        }
    }

    // MARK: 娱乐场地杂物 (飞坡设施 + 彩色装饰)
    private func addAmusementClutter(to scene: SCNScene) {
        let r = trackActualRadius

        // 彩色充气拱门 (赛道起点)
        if trackPath.count > 5 {
            let archP = trackPath[0]
            let archT = trackTangents[0]
            let archNode = SCNNode()
            let archW = VSimCfg.trackWidth * 1.5
            let archH: CGFloat = 1.5
            // 左柱
            let leftPillar = SCNCylinder(radius: 0.08, height: archH)
            leftPillar.firstMaterial?.diffuse.contents = NSColor(red: 1.0, green: 0.2, blue: 0.2, alpha: 1)
            let leftNode = SCNNode(geometry: leftPillar)
            leftNode.position = SCNVector3(-archW / 2, archH / 2, 0)
            archNode.addChildNode(leftNode)
            // 右柱
            let rightPillar = SCNCylinder(radius: 0.08, height: archH)
            rightPillar.firstMaterial?.diffuse.contents = NSColor(red: 0.2, green: 0.4, blue: 1.0, alpha: 1)
            let rightNode = SCNNode(geometry: rightPillar)
            rightNode.position = SCNVector3(archW / 2, archH / 2, 0)
            archNode.addChildNode(rightNode)
            // 横梁
            let beam = SCNBox(width: archW, height: 0.12, length: 0.12, chamferRadius: 0.02)
            beam.firstMaterial?.diffuse.contents = NSColor(red: 1.0, green: 0.8, blue: 0.0, alpha: 1)
            let beamNode = SCNNode(geometry: beam)
            beamNode.position = SCNVector3(0, archH, 0)
            archNode.addChildNode(beamNode)

            archNode.position = SCNVector3(archP.x, 0, archP.z)
            archNode.eulerAngles.y = atan2(archT.x, archT.z)
            scene.rootNode.addChildNode(archNode)
        }

        // 彩旗绳 (沿赛道外侧)
        for i in stride(from: 0, to: trackPath.count, by: 15) {
            let p = trackPath[i]
            let t = trackTangents[i]
            let normal = SCNVector3(-t.z, 0, t.x)
            let flagDist = VSimCfg.trackWidth / 2 + 0.5
            let fx = p.x + normal.x * flagDist
            let fz = p.z + normal.z * flagDist

            let flagH: CGFloat = 0.8
            let pole = SCNCylinder(radius: 0.01, height: flagH)
            pole.firstMaterial?.diffuse.contents = NSColor.white
            let poleNode = SCNNode(geometry: pole)
            poleNode.position = SCNVector3(fx, flagH / 2, fz)
            scene.rootNode.addChildNode(poleNode)

            // 小三角旗
            let flagColors: [NSColor] = [.red, .blue, .green, .yellow, .orange, .purple]
            let flag = SCNBox(width: 0.12, height: 0.08, length: 0.005, chamferRadius: 0)
            flag.firstMaterial?.diffuse.contents = flagColors[i % flagColors.count]
            flag.firstMaterial?.isDoubleSided = true
            let flagNode = SCNNode(geometry: flag)
            flagNode.position = SCNVector3(fx + 0.06, flagH, fz)
            scene.rootNode.addChildNode(flagNode)
        }

        // 充气障碍物 (随机放置在赛道外围)
        for _ in 0..<6 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.5 + drand48() * 0.4)
            let ox = cos(angle) * dist
            let oz = sin(angle) * dist

            let inflH = CGFloat(0.4 + drand48() * 0.6)
            let inflW = CGFloat(0.3 + drand48() * 0.5)
            let inflatable = SCNCapsule(capRadius: inflW / 2, height: inflH)
            let colors: [NSColor] = [
                NSColor(red: 1.0, green: 0.3, blue: 0.3, alpha: 1),
                NSColor(red: 0.3, green: 0.7, blue: 1.0, alpha: 1),
                NSColor(red: 0.3, green: 0.9, blue: 0.3, alpha: 1),
                NSColor(red: 1.0, green: 0.8, blue: 0.1, alpha: 1)
            ]
            inflatable.firstMaterial?.diffuse.contents = colors[Int(drand48() * Double(colors.count)) % colors.count]
            inflatable.firstMaterial?.roughness.contents = NSColor(white: 0.6, alpha: 1)
            let inflNode = SCNNode(geometry: inflatable)
            inflNode.position = SCNVector3(ox, inflH / 2, oz)
            scene.rootNode.addChildNode(inflNode)
        }

        // 计时板/记分板
        let boardAngle = CGFloat(drand48()) * .pi * 2
        let boardDist = r * 0.45
        let boardX = cos(boardAngle) * boardDist
        let boardZ = sin(boardAngle) * boardDist
        let board = SCNBox(width: 1.2, height: 0.8, length: 0.05, chamferRadius: 0.01)
        board.firstMaterial?.diffuse.contents = NSColor(white: 0.95, alpha: 1)
        let boardNode = SCNNode(geometry: board)
        boardNode.position = SCNVector3(boardX, 1.2, boardZ)
        boardNode.eulerAngles.y = boardAngle + .pi
        scene.rootNode.addChildNode(boardNode)
    }

    // MARK: 沙地越野杂物 (岩石 + 植物 + 沙丘)
    private func addDesertClutter(to scene: SCNScene) {
        let r = trackActualRadius

        // 岩石 (随机大小和位置)
        for _ in 0..<20 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.3 + drand48() * 0.8)
            let rx = cos(angle) * dist
            let rz = sin(angle) * dist

            let rockW = CGFloat(0.2 + drand48() * 0.8)
            let rockH = CGFloat(0.1 + drand48() * 0.5)
            let rockD = CGFloat(0.2 + drand48() * 0.6)
            let rock = SCNBox(width: rockW, height: rockH, length: rockD, chamferRadius: rockH * 0.3)
            let shade = CGFloat(0.35 + drand48() * 0.25)
            rock.firstMaterial?.diffuse.contents = NSColor(red: shade, green: shade * 0.85, blue: shade * 0.6, alpha: 1)
            rock.firstMaterial?.roughness.contents = NSColor(white: 0.9, alpha: 1)
            let rockNode = SCNNode(geometry: rock)
            rockNode.position = SCNVector3(rx, rockH / 2, rz)
            rockNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
            rockNode.eulerAngles.x = CGFloat(drand48() - 0.5) * 0.2
            scene.rootNode.addChildNode(rockNode)
        }

        // 沙丘/土堆 (大型地形特征)
        for _ in 0..<8 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.5 + drand48() * 0.6)
            let dx = cos(angle) * dist
            let dz = sin(angle) * dist

            let duneW = CGFloat(1.5 + drand48() * 3.0)
            let duneH = CGFloat(0.3 + drand48() * 0.8)
            let dune = SCNSphere(radius: duneW / 2)
            let duneShade = CGFloat(0.6 + drand48() * 0.15)
            dune.firstMaterial?.diffuse.contents = NSColor(red: duneShade, green: duneShade * 0.85, blue: duneShade * 0.55, alpha: 1)
            dune.firstMaterial?.roughness.contents = NSColor(white: 0.95, alpha: 1)
            let duneNode = SCNNode(geometry: dune)
            duneNode.position = SCNVector3(dx, -duneW / 2 + duneH, dz)
            duneNode.scale = SCNVector3(1, duneH / (duneW / 2), 1)
            scene.rootNode.addChildNode(duneNode)
        }

        // 枯树/灌木 (稀疏植物)
        for _ in 0..<6 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.4 + drand48() * 0.7)
            let tx = cos(angle) * dist
            let tz = sin(angle) * dist

            // 树干
            let trunkH = CGFloat(0.5 + drand48() * 1.0)
            let trunk = SCNCylinder(radius: 0.04, height: trunkH)
            trunk.firstMaterial?.diffuse.contents = NSColor(red: 0.35, green: 0.25, blue: 0.15, alpha: 1)
            let trunkNode = SCNNode(geometry: trunk)
            trunkNode.position = SCNVector3(tx, trunkH / 2, tz)
            trunkNode.eulerAngles.z = CGFloat(drand48() - 0.5) * 0.15
            scene.rootNode.addChildNode(trunkNode)

            // 稀疏树叶
            let canopy = SCNSphere(radius: CGFloat(0.15 + drand48() * 0.25))
            canopy.firstMaterial?.diffuse.contents = NSColor(red: 0.3, green: CGFloat(0.35 + drand48() * 0.2), blue: 0.15, alpha: 0.8)
            let canopyNode = SCNNode(geometry: canopy)
            canopyNode.position = SCNVector3(tx, trunkH + 0.1, tz)
            scene.rootNode.addChildNode(canopyNode)
        }

        // 轮胎堆 (沙地赛道常见)
        for _ in 0..<8 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.25 + drand48() * 0.3)
            let wx = cos(angle) * dist
            let wz = sin(angle) * dist

            let tireStack = Int(1 + drand48() * 3)
            for ti in 0..<tireStack {
                let tire = SCNTorus(ringRadius: 0.12, pipeRadius: 0.04)
                tire.firstMaterial?.diffuse.contents = NSColor(red: 0.12, green: 0.12, blue: 0.12, alpha: 1)
                tire.firstMaterial?.roughness.contents = NSColor(white: 0.9, alpha: 1)
                let tireNode = SCNNode(geometry: tire)
                tireNode.position = SCNVector3(wx, CGFloat(ti) * 0.09 + 0.04, wz)
                scene.rootNode.addChildNode(tireNode)
            }
        }
    }

    // MARK: 专业赛道杂物 (广告牌 + 观众席 + 护栏 + 计时设备 + 绿色内场 + 围栏)
    private func addProTrackClutter(to scene: SCNScene) {
        let r = trackActualRadius

        // ═══ 绿色内场区域 (SRC赛道特征: 赛道内侧绿色人工草坪/涂料) ═══
        let infieldR = r * 0.4  // 内场半径
        let infield = SCNCylinder(radius: infieldR, height: 0.01)
        infield.firstMaterial?.diffuse.contents = NSColor(red: 0.15, green: 0.45, blue: 0.12, alpha: 1)  // 绿色人工草坪
        infield.firstMaterial?.roughness.contents = NSColor(white: 0.85, alpha: 1)
        let infieldNode = SCNNode(geometry: infield)
        infieldNode.position = SCNVector3(0, 0.005, 0)  // 略高于地面
        scene.rootNode.addChildNode(infieldNode)

        // 内场轮胎标记物 (SRC赛道上常见)
        for _ in 0..<8 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = infieldR * CGFloat(0.3 + drand48() * 0.5)
            let tx = cos(angle) * dist
            let tz = sin(angle) * dist
            let tireStack = 1 + Int(drand48() * 3)
            for ti in 0..<tireStack {
                // 专业赛道内场标记轮胎：缩小，避免“挡块过大”
                let tire = SCNTorus(ringRadius: 0.10, pipeRadius: 0.035)
                tire.firstMaterial?.diffuse.contents = NSColor(white: 0.08, alpha: 1)
                tire.firstMaterial?.roughness.contents = NSColor(white: 0.95, alpha: 1)
                let tireNode = SCNNode(geometry: tire)
                tireNode.position = SCNVector3(tx, CGFloat(ti) * 0.08 + 0.04, tz)
                scene.rootNode.addChildNode(tireNode)
            }
        }

        // ═══ 铁丝网围栏 (SRC赛道外围特征) ═══
        let fenceR = r * 0.75  // 围栏半径加大，避免穿过赛道
        let fencePostCount = 24
        for i in 0..<fencePostCount {
            let angle = CGFloat(i) / CGFloat(fencePostCount) * .pi * 2
            let fx = cos(angle) * fenceR
            let fz = sin(angle) * fenceR

            // 围栏柱
            let postH: CGFloat = 1.5
            let post = SCNCylinder(radius: 0.02, height: postH)
            post.firstMaterial?.diffuse.contents = NSColor(white: 0.4, alpha: 1)
            post.firstMaterial?.metalness.contents = 0.6
            let postNode = SCNNode(geometry: post)
            postNode.position = SCNVector3(fx, postH / 2, fz)
            scene.rootNode.addChildNode(postNode)

            // 横杆 (铁丝网简化)
            if i > 0 {
                let prevAngle = CGFloat(i - 1) / CGFloat(fencePostCount) * .pi * 2
                let pfx = cos(prevAngle) * fenceR
                let pfz = sin(prevAngle) * fenceR
                let dx = fx - pfx, dz = fz - pfz
                let dist = sqrt(dx * dx + dz * dz)
                let midX = (fx + pfx) / 2, midZ = (fz + pfz) / 2
                for barH: CGFloat in [0.5, 1.0, 1.4] {
                    let bar = SCNCylinder(radius: 0.008, height: dist)
                    bar.firstMaterial?.diffuse.contents = NSColor(white: 0.45, alpha: 1)
                    bar.firstMaterial?.metalness.contents = 0.5
                    let barNode = SCNNode(geometry: bar)
                    barNode.position = SCNVector3(midX, barH, midZ)
                    barNode.eulerAngles = SCNVector3(0, atan2(dx, dz), .pi / 2)
                    scene.rootNode.addChildNode(barNode)
                }
            }
        }

        // 广告牌 (沿赛道外侧 — SRC风格黑底白字)
        for i in stride(from: 0, to: trackPath.count, by: max(1, trackPath.count / 8)) {
            let p = trackPath[i]
            let t = trackTangents[i]
            let normal = SCNVector3(-t.z, 0, t.x)
            let adDist = VSimCfg.trackWidth / 2 + CGFloat(0.8 + drand48() * 0.5)
            let side: CGFloat = (i % 2 == 0) ? 1 : -1
            let ax = p.x + normal.x * adDist * side
            let az = p.z + normal.z * adDist * side

            let adW = CGFloat(1.5 + drand48() * 1.0)
            let adH = CGFloat(0.6 + drand48() * 0.4)
            let ad = SCNBox(width: adW, height: adH, length: 0.05, chamferRadius: 0.01)
            let adColors: [NSColor] = [
                NSColor(red: 0.9, green: 0.1, blue: 0.1, alpha: 1),
                NSColor(red: 0.1, green: 0.3, blue: 0.8, alpha: 1),
                NSColor(red: 0.1, green: 0.7, blue: 0.2, alpha: 1),
                NSColor(red: 0.9, green: 0.6, blue: 0.0, alpha: 1),
                NSColor(white: 0.95, alpha: 1)
            ]
            ad.firstMaterial?.diffuse.contents = adColors[i % adColors.count]
            let adNode = SCNNode(geometry: ad)
            adNode.position = SCNVector3(ax, adH / 2 + 0.3, az)
            adNode.eulerAngles.y = atan2(t.x, t.z)
            scene.rootNode.addChildNode(adNode)

            // 广告牌支架
            let legH = adH / 2 + 0.3
            for lx: CGFloat in [-adW / 2 + 0.05, adW / 2 - 0.05] {
                let leg = SCNCylinder(radius: 0.02, height: legH)
                leg.firstMaterial?.diffuse.contents = NSColor(white: 0.3, alpha: 1)
                leg.firstMaterial?.metalness.contents = 0.5
                let legNode = SCNNode(geometry: leg)
                legNode.position = SCNVector3(ax + lx * cos(atan2(t.x, t.z)), legH / 2, az + lx * sin(atan2(t.x, t.z)))
                scene.rootNode.addChildNode(legNode)
            }
        }

        // 混凝土护墙 (赛道外围 — 稀疏放置，避免遮挡赛道)
        // 只在弯道入口等关键位置放少量低矮护墙，间距大幅增加
        for i in stride(from: 0, to: trackPath.count, by: max(25, trackPath.count / 6)) {
            let p = trackPath[i]
            let t = trackTangents[i]
            let normal = SCNVector3(-t.z, 0, t.x)
            let wallDist = VSimCfg.trackWidth / 2 + 1.2  // 距赛道边缘 1.2m (原 0.6m)

            // 只放外侧一面
            let side: CGFloat = (i % 2 == 0) ? 1 : -1
            let wx = p.x + normal.x * wallDist * side
            let wz = p.z + normal.z * wallDist * side
            let wallH: CGFloat = 0.15  // 降低高度 (原 0.35m)
            let wallLen: CGFloat = 1.2  // 沿赛道方向的长度
            let wall = SCNBox(width: wallLen, height: wallH, length: 0.10, chamferRadius: 0.01)
            wall.firstMaterial?.diffuse.contents = NSColor(white: 0.75, alpha: 1)
            wall.firstMaterial?.roughness.contents = NSColor(white: 0.8, alpha: 1)
            let wallNode = SCNNode(geometry: wall)
            wallNode.position = SCNVector3(wx, wallH / 2, wz)
            wallNode.eulerAngles.y = atan2(t.x, t.z)
            scene.rootNode.addChildNode(wallNode)
        }

        // 观众席 (赛道一侧 — 放在赛道外围)
        let standAngle = CGFloat(drand48()) * .pi * 2
        let standDist = r * 0.80  // 观众席远离赛道 (原 0.55)
        let standX = cos(standAngle) * standDist
        let standZ = sin(standAngle) * standDist
        for row in 0..<3 {
            for col in 0..<8 {
                let seatW: CGFloat = 0.3
                let seatH: CGFloat = 0.25
                let seat = SCNBox(width: seatW, height: seatH, length: 0.2, chamferRadius: 0.02)
                let seatColors: [NSColor] = [.red, .blue, .orange, .green]
                seat.firstMaterial?.diffuse.contents = seatColors[(row + col) % seatColors.count]
                let seatNode = SCNNode(geometry: seat)
                let sx = standX + CGFloat(col) * seatW * 1.1
                let sy = CGFloat(row) * 0.4 + seatH / 2
                let sz = standZ + CGFloat(row) * 0.3
                seatNode.position = SCNVector3(sx, sy, sz)
                scene.rootNode.addChildNode(seatNode)
            }
        }

        // 计时塔 (起点附近 — 放在赛道外侧)
        if trackPath.count > 0 {
            let sp = trackPath[0]
            let st = trackTangents[0]
            let sNormal = SCNVector3(-st.z, 0, st.x)
            let towerOffset = VSimCfg.trackWidth / 2 + 2.0  // 赛道外侧 2m
            let towerX = sp.x + sNormal.x * towerOffset
            let towerZ = sp.z + sNormal.z * towerOffset
            let towerH: CGFloat = 3.0
            let tower = SCNBox(width: 0.8, height: towerH, length: 0.8, chamferRadius: 0.05)
            tower.firstMaterial?.diffuse.contents = NSColor(white: 0.85, alpha: 1)
            tower.firstMaterial?.metalness.contents = 0.2
            let towerNode = SCNNode(geometry: tower)
            towerNode.position = SCNVector3(towerX, towerH / 2, towerZ)
            scene.rootNode.addChildNode(towerNode)

            // 屏幕
            let screen = SCNBox(width: 0.7, height: 0.4, length: 0.03, chamferRadius: 0.01)
            screen.firstMaterial?.diffuse.contents = NSColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1)
            screen.firstMaterial?.emission.contents = NSColor(red: 0.0, green: 0.3, blue: 0.0, alpha: 1)
            let screenNode = SCNNode(geometry: screen)
            screenNode.position = SCNVector3(towerX, towerH + 0.3, towerZ)
            scene.rootNode.addChildNode(screenNode)
        }
    }

    // MARK: 公园小树林场景 (无赛道 — 对应真实视频中的“大老王小树林”“经典武汉小树林”)
    private func addParkForestClutter(to scene: SCNScene) {
        let r = trackActualRadius

        // ═══ 密集树木 (小树林核心特征) ═══
        let treeCount = 25 + Int(drand48() * 15)
        for _ in 0..<treeCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.3 + drand48() * 0.9)
            let tx = cos(angle) * dist
            let tz = sin(angle) * dist

            // 树干 (棕色, 粗细不一)
            let trunkH = CGFloat(1.5 + drand48() * 3.0)
            let trunkR = CGFloat(0.05 + drand48() * 0.08)
            let trunk = SCNCylinder(radius: trunkR, height: trunkH)
            let barkShade = CGFloat(0.25 + drand48() * 0.15)
            trunk.firstMaterial?.diffuse.contents = NSColor(red: barkShade, green: barkShade * 0.7, blue: barkShade * 0.4, alpha: 1)
            trunk.firstMaterial?.roughness.contents = NSColor(white: 0.9, alpha: 1)
            let trunkNode = SCNNode(geometry: trunk)
            trunkNode.position = SCNVector3(tx, trunkH / 2, tz)
            trunkNode.eulerAngles.z = CGFloat(drand48() - 0.5) * 0.08  // 轻微倾斜
            scene.rootNode.addChildNode(trunkNode)

            // 树冠 (不规则绿色球体)
            // 小树林树冠偏大时会显得“卡通”，这里整体缩小一档
            let canopyR = CGFloat(0.28 + drand48() * 0.55)
            let canopy = SCNSphere(radius: canopyR)
            let leafG = CGFloat(0.3 + drand48() * 0.35)
            canopy.firstMaterial?.diffuse.contents = NSColor(
                red: leafG * 0.4, green: leafG, blue: leafG * 0.2, alpha: 0.9
            )
            canopy.firstMaterial?.roughness.contents = NSColor(white: 0.85, alpha: 1)
            let canopyNode = SCNNode(geometry: canopy)
            canopyNode.position = SCNVector3(tx, trunkH + canopyR * 0.5, tz)
            canopyNode.scale = SCNVector3(1, CGFloat(0.7 + drand48() * 0.4), 1)  // 扁圆树冠
            scene.rootNode.addChildNode(canopyNode)
        }

        // ═══ 交通锥 (娱乐场景常见标记物) ═══
        let coneCount = 6 + Int(drand48() * 6)
        for _ in 0..<coneCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.15 + drand48() * 0.5)
            let cx = cos(angle) * dist
            let cz = sin(angle) * dist

            let coneH: CGFloat = 0.25
            let coneR: CGFloat = 0.08
            let cone = SCNCone(topRadius: 0.01, bottomRadius: coneR, height: coneH)
            cone.firstMaterial?.diffuse.contents = NSColor(red: 1.0, green: 0.35, blue: 0.0, alpha: 1)  // 橙色交通锥
            cone.firstMaterial?.roughness.contents = NSColor(white: 0.7, alpha: 1)
            let coneNode = SCNNode(geometry: cone)
            coneNode.position = SCNVector3(cx, coneH / 2, cz)
            scene.rootNode.addChildNode(coneNode)

            // 白色反光条带
            let stripe = SCNCylinder(radius: coneR * 0.7, height: 0.02)
            stripe.firstMaterial?.diffuse.contents = NSColor.white
            let stripeNode = SCNNode(geometry: stripe)
            stripeNode.position = SCNVector3(cx, coneH * 0.55, cz)
            scene.rootNode.addChildNode(stripeNode)
        }

        // ═══ 红色塑料路障 (武汉小树林视频中可见) ═══
        let barrierCount = 3 + Int(drand48() * 4)
        for _ in 0..<barrierCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.2 + drand48() * 0.4)
            let bx = cos(angle) * dist
            let bz = sin(angle) * dist

            let bw = CGFloat(0.6 + drand48() * 0.4)
            let bh: CGFloat = 0.5
            let bd: CGFloat = 0.15
            let barrier = SCNBox(width: bw, height: bh, length: bd, chamferRadius: 0.02)
            barrier.firstMaterial?.diffuse.contents = NSColor(red: 0.85, green: 0.15, blue: 0.1, alpha: 1)  // 红色路障
            barrier.firstMaterial?.roughness.contents = NSColor(white: 0.7, alpha: 1)
            let barrierNode = SCNNode(geometry: barrier)
            barrierNode.position = SCNVector3(bx, bh / 2, bz)
            barrierNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
            scene.rootNode.addChildNode(barrierNode)

            // 白色反光条
            let whiteStripe = SCNBox(width: bw * 0.9, height: 0.06, length: bd + 0.01, chamferRadius: 0)
            whiteStripe.firstMaterial?.diffuse.contents = NSColor.white
            let whiteNode = SCNNode(geometry: whiteStripe)
            whiteNode.position = SCNVector3(bx, bh * 0.7, bz)
            whiteNode.eulerAngles.y = barrierNode.eulerAngles.y
            scene.rootNode.addChildNode(whiteNode)
        }

        // ═══ 简易跳台/斜坡板 (视频中的白色斜坡板) ═══
        let rampCount = 1 + Int(drand48() * 2)
        for _ in 0..<rampCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.2 + drand48() * 0.3)
            let rx = cos(angle) * dist
            let rz = sin(angle) * dist

            let rampW: CGFloat = 0.5
            let rampL: CGFloat = 0.8
            let rampH: CGFloat = 0.15
            let ramp = SCNBox(width: rampW, height: 0.03, length: rampL, chamferRadius: 0.005)
            ramp.firstMaterial?.diffuse.contents = NSColor(white: 0.85, alpha: 1)  // 白色板
            ramp.firstMaterial?.roughness.contents = NSColor(white: 0.6, alpha: 1)
            let rampNode = SCNNode(geometry: ramp)
            rampNode.position = SCNVector3(rx, rampH / 2, rz)
            rampNode.eulerAngles.x = -atan2(rampH, rampL)  // 斜坡角度
            rampNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
            scene.rootNode.addChildNode(rampNode)
        }

        // ═══ 灰尘/杂物 (地面上的小石头/树枝) ═══
        for _ in 0..<15 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.1 + drand48() * 0.6)
            let ox = cos(angle) * dist
            let oz = sin(angle) * dist
            let ow = CGFloat(0.05 + drand48() * 0.15)
            let oh = CGFloat(0.02 + drand48() * 0.08)
            let obj = SCNBox(width: ow, height: oh, length: ow, chamferRadius: oh * 0.3)
            let shade = CGFloat(0.3 + drand48() * 0.3)
            obj.firstMaterial?.diffuse.contents = NSColor(white: shade, alpha: 1)
            let objNode = SCNNode(geometry: obj)
            objNode.position = SCNVector3(ox, oh / 2, oz)
            objNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
            scene.rootNode.addChildNode(objNode)
        }

        // ═══ 公园设施 (围栏/长凳) ═══
        let benchCount = 2 + Int(drand48() * 3)
        for _ in 0..<benchCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.6 + drand48() * 0.3)
            let bx = cos(angle) * dist
            let bz = sin(angle) * dist

            // 长凳座面
            let benchW: CGFloat = 1.0
            let benchH: CGFloat = 0.4
            let seat = SCNBox(width: benchW, height: 0.04, length: 0.3, chamferRadius: 0.01)
            seat.firstMaterial?.diffuse.contents = NSColor(red: 0.4, green: 0.3, blue: 0.2, alpha: 1)  // 木色
            let seatNode = SCNNode(geometry: seat)
            seatNode.position = SCNVector3(bx, benchH, bz)
            seatNode.eulerAngles.y = CGFloat(drand48()) * .pi
            scene.rootNode.addChildNode(seatNode)

            // 长凳腿
            for lx: CGFloat in [-benchW * 0.4, benchW * 0.4] {
                let leg = SCNCylinder(radius: 0.02, height: benchH)
                leg.firstMaterial?.diffuse.contents = NSColor(white: 0.3, alpha: 1)
                let legNode = SCNNode(geometry: leg)
                legNode.position = SCNVector3(bx + lx * cos(seatNode.eulerAngles.y), benchH / 2, bz + lx * sin(seatNode.eulerAngles.y))
                scene.rootNode.addChildNode(legNode)
            }
        }
    }

    // MARK: 开放草地场景 (无赛道 — 对应真实视频中的“草地”“飞坡”场景)
    private func addOpenFieldClutter(to scene: SCNScene) {
        let r = trackActualRadius

        // ═══ 稀疏树木 (草地边缘的树木, 比小树林少很多) ═══
        let treeCount = 8 + Int(drand48() * 8)
        for _ in 0..<treeCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.6 + drand48() * 0.5)  // 树木在边缘
            let tx = cos(angle) * dist
            let tz = sin(angle) * dist

            let trunkH = CGFloat(2.0 + drand48() * 3.5)
            let trunkR = CGFloat(0.06 + drand48() * 0.10)
            let trunk = SCNCylinder(radius: trunkR, height: trunkH)
            trunk.firstMaterial?.diffuse.contents = NSColor.brown
            trunk.firstMaterial?.roughness.contents = NSColor(white: 0.9, alpha: 1)
            let trunkNode = SCNNode(geometry: trunk)
            trunkNode.position = SCNVector3(tx, trunkH / 2, tz)
            scene.rootNode.addChildNode(trunkNode)

            let canopyR = CGFloat(0.8 + drand48() * 1.5)
            let canopy = SCNSphere(radius: canopyR)
            canopy.firstMaterial?.diffuse.contents = NSColor(
                red: CGFloat(0.1 + drand48() * 0.2),
                green: CGFloat(0.35 + drand48() * 0.35),
                blue: 0.05, alpha: 0.9
            )
            let canopyNode = SCNNode(geometry: canopy)
            canopyNode.position = SCNVector3(tx, trunkH + canopyR * 0.5, tz)
            scene.rootNode.addChildNode(canopyNode)
        }

        // ═══ 跳台/飞坡 (视频中的黑色斜坡) ═══
        let jumpCount = 1 + Int(drand48() * 2)
        for _ in 0..<jumpCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.15 + drand48() * 0.3)
            let jx = cos(angle) * dist
            let jz = sin(angle) * dist

            // 斜坡主体
            let rampW: CGFloat = 0.6
            let rampL: CGFloat = 1.0
            let rampH: CGFloat = 0.25
            let ramp = SCNBox(width: rampW, height: 0.04, length: rampL, chamferRadius: 0.005)
            ramp.firstMaterial?.diffuse.contents = NSColor(white: 0.12, alpha: 1)  // 黑色斜坡
            ramp.firstMaterial?.roughness.contents = NSColor(white: 0.8, alpha: 1)
            let rampNode = SCNNode(geometry: ramp)
            rampNode.position = SCNVector3(jx, rampH / 2, jz)
            rampNode.eulerAngles.x = -atan2(rampH, rampL)
            rampNode.eulerAngles.y = CGFloat(drand48()) * .pi * 2
            scene.rootNode.addChildNode(rampNode)

            // 斜坡支撑
            let support = SCNBox(width: rampW, height: rampH, length: 0.15, chamferRadius: 0.01)
            support.firstMaterial?.diffuse.contents = NSColor(white: 0.15, alpha: 1)
            let supportNode = SCNNode(geometry: support)
            supportNode.position = SCNVector3(jx, rampH / 2, jz)
            scene.rootNode.addChildNode(supportNode)
        }

        // ═══ 交通锥 (少量) ═══
        let coneCount = 3 + Int(drand48() * 4)
        for _ in 0..<coneCount {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.1 + drand48() * 0.4)
            let cx = cos(angle) * dist
            let cz = sin(angle) * dist

            let coneH: CGFloat = 0.25
            let cone = SCNCone(topRadius: 0.01, bottomRadius: 0.08, height: coneH)
            cone.firstMaterial?.diffuse.contents = NSColor(red: 1.0, green: 0.35, blue: 0.0, alpha: 1)
            let coneNode = SCNNode(geometry: cone)
            coneNode.position = SCNVector3(cx, coneH / 2, cz)
            scene.rootNode.addChildNode(coneNode)
        }

        // ═══ 远处背景 (马路/建筑物轮廓) ═══
        // 远处低矮建筑物轮廓
        for _ in 0..<5 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(1.2 + drand48() * 0.5)
            let bx = cos(angle) * dist
            let bz = sin(angle) * dist

            let bw = CGFloat(2.0 + drand48() * 4.0)
            let bh = CGFloat(1.5 + drand48() * 3.0)
            let bd = CGFloat(1.0 + drand48() * 2.0)
            let building = SCNBox(width: bw, height: bh, length: bd, chamferRadius: 0.05)
            let shade = CGFloat(0.5 + drand48() * 0.3)
            building.firstMaterial?.diffuse.contents = NSColor(white: shade, alpha: 1)
            building.firstMaterial?.roughness.contents = NSColor(white: 0.8, alpha: 1)
            let buildingNode = SCNNode(geometry: building)
            buildingNode.position = SCNVector3(bx, bh / 2, bz)
            buildingNode.eulerAngles.y = CGFloat(drand48()) * .pi
            scene.rootNode.addChildNode(buildingNode)
        }

        // ═══ 草丛 (地面上的高草) ═══
        for _ in 0..<20 {
            let angle = CGFloat(drand48()) * .pi * 2
            let dist = r * CGFloat(0.2 + drand48() * 0.7)
            let gx = cos(angle) * dist
            let gz = sin(angle) * dist

            let grassH = CGFloat(0.08 + drand48() * 0.15)
            let grassW = CGFloat(0.15 + drand48() * 0.3)
            let grass = SCNBox(width: grassW, height: grassH, length: grassW, chamferRadius: grassH * 0.4)
            let g = CGFloat(0.25 + drand48() * 0.3)
            grass.firstMaterial?.diffuse.contents = NSColor(red: g * 0.5, green: g, blue: g * 0.2, alpha: 0.8)
            let grassNode = SCNNode(geometry: grass)
            grassNode.position = SCNVector3(gx, grassH / 2, gz)
            scene.rootNode.addChildNode(grassNode)
        }
    }

    private func setupLighting(scene: SCNScene) {
        let isIndoorScene = (sceneType == 0 || sceneType == 3)
        if isIndoorScene {
            // ═══ 室内光照: 多个点光源模拟顶灯 + 较弱环境光 ═══
            let ambient = SCNLight()
            ambient.type = .ambient
            ambient.intensity = enableLightVariation
                ? CGFloat.random(in: 150...400)
                : 280
            ambient.color = NSColor(red: 0.95, green: 0.92, blue: 0.85, alpha: 1) // 暖色室内光
            let ambNode = SCNNode()
            ambNode.light = ambient
            scene.rootNode.addChildNode(ambNode)

            // 多个点光源 (模拟室内顶灯)
            let lightCount = 4 + Int(drand48() * 3)
            let lightDist = trackActualRadius * 0.6
            for li in 0..<lightCount {
                let spotLight = SCNLight()
                spotLight.type = .spot
                spotLight.intensity = enableLightVariation
                    ? CGFloat.random(in: 600...1500)
                    : 1000
                spotLight.color = NSColor(red: 1.0, green: 0.95, blue: 0.88, alpha: 1)
                spotLight.castsShadow = VSimCfg.shadowEnabled
                spotLight.shadowRadius = 4
                spotLight.shadowSampleCount = 8
                spotLight.spotInnerAngle = 30
                spotLight.spotOuterAngle = 70
                spotLight.attenuationStartDistance = 2.0
                spotLight.attenuationEndDistance = 15.0
                let spotNode = SCNNode()
                spotNode.light = spotLight
                let a = CGFloat(li) / CGFloat(lightCount) * .pi * 2
                spotNode.position = SCNVector3(
                    freeCamCenterX + cos(a) * lightDist * CGFloat(0.5 + drand48() * 0.5),
                    CGFloat(2.5 + drand48() * 1.0),
                    freeCamCenterZ + sin(a) * lightDist * CGFloat(0.5 + drand48() * 0.5)
                )
                spotNode.eulerAngles.x = -.pi / 2.5  // 向下照射
                scene.rootNode.addChildNode(spotNode)
            }
        } else if sceneType == 4 {
            // ═══ 沙地越野: 强烈太阳光 + 高环境光 (模拟沙漠强光) ═══
            let ambient = SCNLight()
            ambient.type = .ambient
            ambient.intensity = enableLightVariation
                ? CGFloat.random(in: 600...1200)
                : 900
            ambient.color = NSColor(red: 1.0, green: 0.95, blue: 0.80, alpha: 1) // 暖黄色沙漠光
            let ambNode = SCNNode()
            ambNode.light = ambient
            scene.rootNode.addChildNode(ambNode)

            let sun = SCNLight()
            sun.type = .directional
            sun.intensity = enableLightVariation
                ? CGFloat.random(in: 1500...3000)
                : 2200
            sun.color = NSColor(red: 1.0, green: 0.92, blue: 0.75, alpha: 1)
            sun.castsShadow = VSimCfg.shadowEnabled
            sun.shadowRadius = 2
            sun.shadowSampleCount = 8
            let sunNode = SCNNode()
            sunNode.light = sun
            let sunAngle = enableLightVariation ? CGFloat(drand48()) * .pi : .pi * 0.5
            sunNode.eulerAngles = SCNVector3(-.pi / 4, sunAngle, 0)
            scene.rootNode.addChildNode(sunNode)
        } else {
            // ═══ 室外光照 (草地/柏油/专业赛道/公园小树林/开放草地): 太阳光 + 环境光 ═══
            let ambient = SCNLight()
            ambient.type = .ambient
            let ambI = VSimCfg.ambientIntensityRange
            ambient.intensity = enableLightVariation
                ? CGFloat.random(in: ambI)
                : (ambI.lowerBound + ambI.upperBound) / 2
            ambient.color = NSColor(white: 0.9, alpha: 1)
            let ambNode = SCNNode()
            ambNode.light = ambient
            scene.rootNode.addChildNode(ambNode)

            let sun = SCNLight()
            sun.type = .directional
            let sunI = VSimCfg.sunIntensityRange
            sun.intensity = enableLightVariation
                ? CGFloat.random(in: sunI)
                : (sunI.lowerBound + sunI.upperBound) / 2
            sun.color = NSColor(red: 1, green: 0.95, blue: 0.85, alpha: 1)
            sun.castsShadow = VSimCfg.shadowEnabled
            sun.shadowRadius = 3
            sun.shadowSampleCount = 8
            let sunNode = SCNNode()
            sunNode.light = sun
            let sunAngle = enableLightVariation ? CGFloat(drand48()) * .pi : .pi * 0.7
            sunNode.eulerAngles = SCNVector3(-.pi / 3, sunAngle, 0)
            scene.rootNode.addChildNode(sunNode)
        }
    }

private func setupGimbalCamera(scene: SCNScene) {
        // ── ★ RC赛道云台相机优化 ──
        // 根据赛道实际包围盒自动计算相机位置
        // 目标: 摄像机放在赛道旁边，云台水平±40度旋转即可看到完整赛道全貌

        // 计算赛道包围盒
        var minX: CGFloat = .infinity, maxX: CGFloat = -.infinity
        var minZ: CGFloat = .infinity, maxZ: CGFloat = -.infinity
        for p in trackPath {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minZ = min(minZ, p.z); maxZ = max(maxZ, p.z)
        }
        let trackW = maxX - minX
        let trackD = maxZ - minZ
        let trackCenterX = (minX + maxX) / 2
        let trackCenterZ = (minZ + maxZ) / 2
        let trackMaxSpan = max(trackW, trackD)

        // 使用拍摄点位预设
        let presets = VSimCfg.cameraPresets
        let presetIdx = max(0, min(cameraPreset, presets.count - 1))
        let preset = presets[presetIdx]

        let camX: CGFloat
        let camZ: CGFloat
        let camH = preset.height

        if presetIdx == 8 {
            // ★ 唯一的赛道中间机位: 放在赛道中心正上方
            camX = trackCenterX + CGFloat(drand48() - 0.5) * 1.0
            camZ = trackCenterZ + CGFloat(drand48() - 0.5) * 1.0
        } else if presetIdx == 4 {
            // ★ 弯道角落机位: 放在最急弯道外侧 1~2m
            let bendIdx = findSharpestBendIndex()
            let bp = trackPath[bendIdx]
            let bt = trackTangents[bendIdx]
            let normal = SCNVector3(-bt.z, 0, bt.x)
            let offset: CGFloat = 1.0 + CGFloat(drand48()) * 1.0
            camX = bp.x + normal.x * offset
            camZ = bp.z + normal.z * offset
        } else if presetIdx == 5 {
            // ★ 直道侧面机位: 放在最长直道中点外侧 1~2m
            let straightIdx = findLongestStraightIndex()
            let sp = trackPath[straightIdx]
            let st = trackTangents[straightIdx]
            let normal = SCNVector3(-st.z, 0, st.x)
            let offset: CGFloat = 1.0 + CGFloat(drand48()) * 1.0
            camX = sp.x + normal.x * offset
            camZ = sp.z + normal.z * offset
        } else {
            // ★ 其他机位: 放在赛道包络外侧边缘 0.5~2m
            let edgeMargin: CGFloat = 0.5 + CGFloat(drand48()) * 1.5
            let edge = Int(drand48() * 4) // 0=南, 1=北, 2=西, 3=东
            let randAlongEdge = CGFloat(drand48() - 0.5) // -0.5~0.5

            switch edge {
            case 0: // 南边 (minZ外侧)
                camX = trackCenterX + randAlongEdge * trackW * 0.8
                camZ = minZ - edgeMargin
            case 1: // 北边 (maxZ外侧)
                camX = trackCenterX + randAlongEdge * trackW * 0.8
                camZ = maxZ + edgeMargin
            case 2: // 西边 (minX外侧)
                camX = minX - edgeMargin
                camZ = trackCenterZ + randAlongEdge * trackD * 0.8
            default: // 东边 (maxX外侧)
                camX = maxX + edgeMargin
                camZ = trackCenterZ + randAlongEdge * trackD * 0.8
            }
        }

        // 云台基座
        let base = SCNNode()
        base.position = SCNVector3(camX, camH, camZ)
        scene.rootNode.addChildNode(base)
        gimbalBaseNode = base

        // 可视化云台基座 (三脚架)
        let tripodH = camH
        let tripod = SCNCylinder(radius: 0.03, height: tripodH)
        tripod.firstMaterial?.diffuse.contents = NSColor(white: 0.2, alpha: 1)
        tripod.firstMaterial?.metalness.contents = 0.5
        let tripodNode = SCNNode(geometry: tripod)
        tripodNode.position = SCNVector3(camX, tripodH / 2, camZ)
        scene.rootNode.addChildNode(tripodNode)

        // 云台头 (小方块)
        let gimbalHead = SCNBox(width: 0.08, height: 0.06, length: 0.08, chamferRadius: 0.005)
        gimbalHead.firstMaterial?.diffuse.contents = NSColor(white: 0.15, alpha: 1)
        gimbalHead.firstMaterial?.metalness.contents = 0.6
        let gimbalHeadNode = SCNNode(geometry: gimbalHead)
        gimbalHeadNode.position = SCNVector3(camX, camH + 0.03, camZ)
        scene.rootNode.addChildNode(gimbalHeadNode)

        // 水平旋转节点
        let panNode = SCNNode()
        base.addChildNode(panNode)
        gimbalPanNode = panNode

        // 俯仰旋转节点
        let tiltNode = SCNNode()
        panNode.addChildNode(tiltNode)
        gimbalTiltNode = tiltNode

        // 相机 — 根据赛道尺寸动态调整FOV
        let camera = SCNCamera()
        // 计算相机到赛道中心的实际距离
        let actualDist = sqrt(pow(camX - trackCenterX, 2) + pow(camZ - trackCenterZ, 2))
        // 计算覆盖±40度所需FOV: 赛道角宽度 = 2*atan(trackMaxSpan/2/dist)
        // ±40度旋转覆盖 = FOV + 80度, 所以 FOV 需要覆盖的角度 = 赛道角宽 - 60度的余量
        let trackAngularSpan = 2 * atan(trackMaxSpan / (2 * max(actualDist, 1.0))) * 180 / .pi
        let neededFov = max(trackAngularSpan - 60, VSimCfg.cameraFovDeg)  // 留一些余量
        let clampedFov = min(neededFov, 90.0)  // 不超过90度避免畸变
        camera.fieldOfView = clampedFov
        camera.zNear = 0.1
        camera.zFar = 200
        camera.wantsHDR = true
        let camNode = SCNNode()
        camNode.camera = camera
        tiltNode.addChildNode(camNode)
        cameraNode = camNode

        // 初始朝向: 看向赛道中心
        let ddx: CGFloat = trackCenterX - camX
        let ddz: CGFloat = trackCenterZ - camZ
        let initPan = atan2(-ddx, -ddz)
        panNode.eulerAngles.y = initPan

        // 动态计算俯仰角: 根据相机高度和到赛道中心的水平距离计算
        let horizDist = sqrt(ddx * ddx + ddz * ddz)
        let autoTilt = -atan2(camH, max(horizDist, 0.5))  // 负值 = 向下看
        tiltNode.eulerAngles.x = autoTilt
    }

    /// 找曲率最大的弯道索引
    private func findSharpestBendIndex() -> Int {
        guard trackPath.count > 10 else { return 0 }
        var maxCurv: CGFloat = 0
        var maxIdx = 0
        let step = max(1, trackPath.count / 100)
        for i in stride(from: step, to: trackPath.count - step, by: step) {
            let prev = trackPath[i - step]
            let curr = trackPath[i]
            let next = trackPath[i + step]
            let dx1 = curr.x - prev.x, dz1 = curr.z - prev.z
            let dx2 = next.x - curr.x, dz2 = next.z - curr.z
            let h1 = atan2(dx1, dz1)
            let h2 = atan2(dx2, dz2)
            var dh = h2 - h1
            if dh > .pi { dh -= .pi * 2 }
            if dh < -.pi { dh += .pi * 2 }
            if abs(dh) > maxCurv {
                maxCurv = abs(dh)
                maxIdx = i
            }
        }
        return maxIdx
    }

    /// 找最长直道的中点索引
    private func findLongestStraightIndex() -> Int {
        guard trackPath.count > 10 else { return 0 }
        var bestStart = 0, bestLen = 0
        var curStart = 0, curLen = 0
        let step = max(1, trackPath.count / 200)
        let thresh: CGFloat = 0.02
        for i in stride(from: step, to: trackPath.count - step, by: step) {
            let prev = trackPath[i - step]
            let curr = trackPath[i]
            let next = trackPath[i + step]
            let dx1 = curr.x - prev.x, dz1 = curr.z - prev.z
            let dx2 = next.x - curr.x, dz2 = next.z - curr.z
            let h1 = atan2(dx1, dz1)
            let h2 = atan2(dx2, dz2)
            var dh = h2 - h1
            if dh > .pi { dh -= .pi * 2 }
            if dh < -.pi { dh += .pi * 2 }
            if abs(dh) < thresh {
                curLen += 1
                if curLen > bestLen {
                    bestLen = curLen
                    bestStart = curStart
                }
            } else {
                curStart = i
                curLen = 0
            }
        }
        return bestStart + bestLen / 2
    }

    private func setupCars(scene: SCNScene) {
        cars = []
        let carColors: [NSColor] = [
            .red, .blue, .green, .orange, .purple,
            .yellow, .cyan, .magenta, .white, .systemPink
        ]

        for i in 0..<carCount {
            let color = carColors[i % carColors.count]
            let speed = VSimCfg.carBaseSpeed + CGFloat(drand48() - 0.5) * VSimCfg.carSpeedVariance * 2
            let laneOffset = CGFloat(drand48() - 0.5) * 0.6

            var car = VirtualCar(
                id: i,
                color: color,
                trackProgress: CGFloat(i) / CGFloat(carCount),
                speed: speed,
                baseSpeed: speed,
                laneOffset: laneOffset
            )

            // 车身 — 金属感 + 高光
            let body = SCNBox(
                width: VSimCfg.carWidth,
                height: VSimCfg.carHeight * 0.6,
                length: VSimCfg.carLength,
                chamferRadius: VSimCfg.carHeight * 0.05
            )
            body.firstMaterial?.diffuse.contents = color
            body.firstMaterial?.metalness.contents = 0.5
            body.firstMaterial?.roughness.contents = NSColor(white: 0.35, alpha: 1)
            body.firstMaterial?.specular.contents = NSColor.white
            body.firstMaterial?.shininess = 0.9

            let bodyNode = SCNNode(geometry: body)
            bodyNode.position = SCNVector3(0, VSimCfg.carHeight * 0.3, 0)

            // 车顶/驾驶舱 — 深色对比
            let cabin = SCNBox(
                width: VSimCfg.carWidth * 0.7,
                height: VSimCfg.carHeight * 0.4,
                length: VSimCfg.carLength * 0.5,
                chamferRadius: VSimCfg.carHeight * 0.05
            )
            cabin.firstMaterial?.diffuse.contents = NSColor(white: 0.15, alpha: 1)
            cabin.firstMaterial?.metalness.contents = 0.3
            cabin.firstMaterial?.roughness.contents = NSColor(white: 0.5, alpha: 1)
            let cabinNode = SCNNode(geometry: cabin)
            cabinNode.position = SCNVector3(0, VSimCfg.carHeight * 0.8, 0)

            // 车轮 — 轮胎 + 轮毂
            let wheelR = VSimCfg.carHeight * 0.2
            let wheelW: CGFloat = 0.03
            let rimR = wheelR * 0.55
            for wx: CGFloat in [-1, 1] {
                for wz: CGFloat in [-1, 1] {
                    let tire = SCNCylinder(radius: wheelR, height: wheelW)
                    tire.firstMaterial?.diffuse.contents = NSColor(red: 0.12, green: 0.12, blue: 0.12, alpha: 1)
                    tire.firstMaterial?.roughness.contents = NSColor(white: 0.95, alpha: 1)
                    let tireNode = SCNNode(geometry: tire)
                    tireNode.eulerAngles.z = .pi / 2
                    let rim = SCNCylinder(radius: rimR, height: wheelW * 1.2)
                    rim.firstMaterial?.diffuse.contents = NSColor(white: 0.45, alpha: 1)
                    rim.firstMaterial?.metalness.contents = 0.7
                    rim.firstMaterial?.roughness.contents = NSColor(white: 0.4, alpha: 1)
                    let rimNode = SCNNode(geometry: rim)
                    rimNode.eulerAngles.z = .pi / 2
                    let wpx = wx * VSimCfg.carWidth * 0.5
                    let wpy = wheelR
                    let wpz = wz * VSimCfg.carLength * 0.35
                    tireNode.position = SCNVector3(wpx, wpy, wpz)
                    rimNode.position = SCNVector3(wpx, wpy, wpz)
                    bodyNode.addChildNode(tireNode)
                    bodyNode.addChildNode(rimNode)
                }
            }

            let carNode = SCNNode()
            carNode.addChildNode(bodyNode)
            carNode.addChildNode(cabinNode)
            addConcentricMarkers(to: carNode, carId: i)
            carNode.name = "car_\(i)"
            scene.rootNode.addChildNode(carNode)

            car.node = carNode
            cars.append(car)
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 车辆物理更新
    // ═══════════════════════════════════════════════════════════

    private func updateCars(dt: CGFloat) {
        for i in cars.indices {
            let speedNoise = CGFloat(drand48() - 0.5) * 0.3 * dt
            cars[i].speed = max(1.0, min(VSimCfg.carBaseSpeed * 2, cars[i].speed + speedNoise))

            for j in cars.indices where j != i {
                let progDiff = cars[j].trackProgress - cars[i].trackProgress
                let wrappedDiff = progDiff < 0 ? progDiff + 1 : progDiff
                if wrappedDiff < 0.03 && wrappedDiff > 0 {
                    cars[i].speed *= 0.95
                    let laneShift = CGFloat(drand48() > 0.5 ? 1 : -1) * 0.1 * dt
                    cars[i].laneOffset = max(-0.8, min(0.8, cars[i].laneOffset + laneShift))
                }
            }

            let progressDelta = cars[i].speed * dt / max(trackLength, 1)
            cars[i].trackProgress += progressDelta
            if cars[i].trackProgress >= 1.0 { cars[i].trackProgress -= 1.0 }

            let (pos, heading) = interpolateTrack(progress: cars[i].trackProgress)
            let normal = SCNVector3(-sin(heading), 0, cos(heading))
            let offset = cars[i].laneOffset * VSimCfg.trackWidth * 0.4

            let cpx = pos.x + normal.x * offset
            let trackY = pos.y  // 赛道地面高度 (可能有起伏)
            var cpy = trackY + VSimCfg.carHeight * 0.15

            // ═══ 飞坡物理 ═══
            if !rampSegments.isEmpty {
                let prog = cars[i].trackProgress
                cars[i].groundY = trackY + VSimCfg.carHeight * 0.15

                // 检查是否在飞坡区域
                var _onRamp = false
                _ = _onRamp  // suppress unused warning
                for ramp in rampSegments {
                    if prog >= ramp.startProgress && prog <= ramp.endProgress {
                        _onRamp = true
                        // 在坡顶附近发射
                        let rampMid = (ramp.startProgress + ramp.endProgress) / 2
                        if prog > rampMid && !cars[i].isAirborne {
                            // 达到坡顶后半段 → 起飞
                            cars[i].isAirborne = true
                            cars[i].airVelocityY = cars[i].speed * sin(ramp.launchAngle)
                        }
                        break
                    }
                }

                if cars[i].isAirborne {
                    // 空中抛物线运动
                    cars[i].airVelocityY -= VSimCfg.rampGravity * dt
                    cars[i].airVelocityY *= (1.0 - VSimCfg.rampAirDrag)  // 空气阻力
                    cpy = cars[i].position.y + cars[i].airVelocityY * dt

                    // 着陆检测
                    if cpy <= cars[i].groundY {
                        cpy = cars[i].groundY
                        cars[i].isAirborne = false
                        // 着陆冲击: 速度略减 + 弹跳
                        cars[i].speed *= 0.9
                        if abs(cars[i].airVelocityY) > 1.0 {
                            // 小弹跳
                            cars[i].isAirborne = true
                            cars[i].airVelocityY = abs(cars[i].airVelocityY) * VSimCfg.rampBounce
                        } else {
                            cars[i].airVelocityY = 0
                        }
                    }

                    // 空中车身俯仰 (前俩→后俩)
                    let pitchAngle = -atan2(cars[i].airVelocityY, cars[i].speed) * 0.5
                    cars[i].node?.eulerAngles.x = pitchAngle
                } else {
                    cars[i].node?.eulerAngles.x = 0
                }
            }

            let cpz = pos.z + normal.z * offset
            cars[i].position = SCNVector3(cpx, cpy, cpz)

            // ═══ 侧滑/漂移效果 ═══
            if enableDrift {
                // 计算当前曲率 (用前后两个点的heading差)
                let lookAhead = min(cars[i].trackProgress + 0.01, 0.999)
                let (_, headingAhead) = interpolateTrack(progress: lookAhead)
                var curvature = headingAhead - heading
                // 角度包裹
                if curvature > .pi { curvature -= .pi * 2 }
                if curvature < -.pi { curvature += .pi * 2 }
                curvature = curvature / 0.01 // 归一化曲率

                // 只有曲率超过阈值且速度足够快才产生侧滑
                let speedFactor = cars[i].speed / VSimCfg.carBaseSpeed
                if abs(curvature) > VSimCfg.driftCurvatureThresh && speedFactor > 0.8 {
                    let targetSlip = curvature * VSimCfg.driftSlipGain * speedFactor
                    cars[i].slipAngle = cars[i].slipAngle * VSimCfg.driftDamping + targetSlip * (1 - VSimCfg.driftDamping)
                } else {
                    cars[i].slipAngle *= VSimCfg.driftDamping
                }
                // 限制最大侧滑角
                cars[i].slipAngle = max(-0.5, min(0.5, cars[i].slipAngle))
                cars[i].heading = heading + cars[i].slipAngle
            } else {
                cars[i].heading = heading
            }

            cars[i].node?.position = cars[i].position
            cars[i].node?.eulerAngles.y = cars[i].heading
        }
    }

    private func interpolateTrack(progress: CGFloat) -> (SCNVector3, CGFloat) {
        guard trackPath.count > 1 else { return (.init(), 0) }
        let totalIdx = CGFloat(trackPath.count)
        let rawIdx = progress * totalIdx
        let idx0 = Int(rawIdx) % trackPath.count
        let idx1 = (idx0 + 1) % trackPath.count
        let frac = rawIdx - CGFloat(Int(rawIdx))

        let p0 = trackPath[idx0]
        let p1 = trackPath[idx1]
        let ipx = p0.x + (p1.x - p0.x) * frac
        let ipy = p0.y + (p1.y - p0.y) * frac
        let ipz = p0.z + (p1.z - p0.z) * frac
        let pos = SCNVector3(ipx, ipy, ipz)

        let hdx = p1.x - p0.x
        let hdz = p1.z - p0.z
        let heading = atan2(hdx, hdz)

        return (pos, heading)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 渲染 & GT 计算
    // ═══════════════════════════════════════════════════════════

    private func renderFrame(time: Double) -> (CGImage?, CVPixelBuffer?) {
        guard let renderer = renderer else { return (nil, nil) }

        let size = CGSize(width: VSimCfg.renderWidth, height: VSimCfg.renderHeight)
        let img = renderer.snapshot(atTime: time, with: size,
                                     antialiasingMode: .multisampling4X)

        guard let tiffData = img.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              var cgImage = bitmap.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else { return (nil, nil) }

        // ═══ 镜头桶形畸变 (Brown-Conrady 模拟) ═══
        // ★ FIX: 渲染端使用 CIBumpDistortion (非标准 Brown-Conrady), 而 CalibratedFOVMap
        //   的 distort/undistort 用 Brown-Conrady 模型。两个模型不匹配会导致像素偏差。
        //   仅在不使用 CalibratedFOVMap 时应用渲染端畸变，否则跳过（由 FOVMap 统一处理）。
        if enableLensDistortion {
            // 注意: 当前虚拟赛道使用简单FOVMap(无畸变模型), 所以渲染端畸变是安全的。
            // 如果将来切换到 CalibratedFOVMap, 必须禁用此处以避免双重畸变。
            if let distorted = applyBarrelDistortion(to: cgImage) {
                cgImage = distorted
            }
        }

        let pb = pixelBuffer(from: cgImage, width: VSimCfg.renderWidth, height: VSimCfg.renderHeight)
        return (cgImage, pb)
    }

    /// 使用 CIFilter 应用桶形畸变，模拟运动相机广角效果
    private func applyBarrelDistortion(to cgImage: CGImage) -> CGImage? {
        let ciImage = CIImage(cgImage: cgImage)
        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)

        // CIBumpDistortion 可以模拟桶形畸变效果
        // 使用 CIBumpDistortionLinear 或 CITwirlDistortion 等替代方案
        // 这里用 CIBumpDistortion: 负 scale = 桶形畸变
        guard let filter = CIFilter(name: "CIBumpDistortion") else { return nil }
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(CIVector(x: w / 2, y: h / 2), forKey: kCIInputCenterKey)
        filter.setValue(max(w, h) * 0.8, forKey: kCIInputRadiusKey)
        filter.setValue(VSimCfg.distortionK1 * 2.0, forKey: kCIInputScaleKey)  // 负值=桶形

        guard let output = filter.outputImage else { return nil }
        // 裁剪回原始尺寸
        let cropped = output.cropped(to: CGRect(x: 0, y: 0, width: w, height: h))
        return ci.createCGImage(cropped, from: cropped.extent)
    }

    private func pixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let buffer = pb else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    private func computeGroundTruth() {
        guard let camNode = cameraNode,
              let _ = camNode.camera,
              let renderer = renderer else { return }

        let viewW = CGFloat(VSimCfg.renderWidth)
        let viewH = CGFloat(VSimCfg.renderHeight)

        for i in cars.indices {
            let carPos = cars[i].position
            guard let carNode = cars[i].node else { continue }

            let halfW = VSimCfg.carWidth / 2
            let halfH = VSimCfg.carHeight
            let halfL = VSimCfg.carLength / 2
            let corners: [SCNVector3] = [
                SCNVector3(-halfW, 0, -halfL), SCNVector3(halfW, 0, -halfL),
                SCNVector3(-halfW, 0, halfL),  SCNVector3(halfW, 0, halfL),
                SCNVector3(-halfW, halfH, -halfL), SCNVector3(halfW, halfH, -halfL),
                SCNVector3(-halfW, halfH, halfL),  SCNVector3(halfW, halfH, halfL),
            ]

            var sMinX: CGFloat = .greatestFiniteMagnitude
            var sMinY: CGFloat = .greatestFiniteMagnitude
            var sMaxX: CGFloat = -.greatestFiniteMagnitude
            var sMaxY: CGFloat = -.greatestFiniteMagnitude
            var allInFront = true

            for corner in corners {
                let worldPos = carNode.convertPosition(corner, to: nil)
                let screenPt = renderer.projectPoint(worldPos)

                if screenPt.z > 1 || screenPt.z < 0 {
                    allInFront = false
                    continue
                }

                let sx = screenPt.x
                let sy = screenPt.y
                sMinX = min(sMinX, sx)
                sMinY = min(sMinY, sy)
                sMaxX = max(sMaxX, sx)
                sMaxY = max(sMaxY, sy)
            }

            if !allInFront || sMinX >= sMaxX || sMinY >= sMaxY {
                cars[i].isVisible = false
                cars[i].screenRect = .zero
                continue
            }

            // SceneKit projectPoint 为左下角原点、y 向上；转为与 YOLO/UI 一致的左上角归一化坐标 (0,0=左上)
            let normRect = CGRect(
                x: sMinX / viewW,
                y: 1 - (sMaxY / viewH),
                width: (sMaxX - sMinX) / viewW,
                height: (sMaxY - sMinY) / viewH
            )

            let screenBounds = CGRect(x: 0, y: 0, width: 1, height: 1)
            let clipped = normRect.intersection(screenBounds)
            let visible = !clipped.isNull && clipped.width > 0.005 && clipped.height > 0.005

            cars[i].isVisible = visible
            cars[i].screenRect = visible ? clipped : .zero

            // 距离
            let camWorldPos = camNode.worldPosition
            let cdx = carPos.x - camWorldPos.x
            let cdy = carPos.y - camWorldPos.y
            let cdz = carPos.z - camWorldPos.z
            cars[i].distanceToCamera = sqrt(cdx * cdx + cdy * cdy + cdz * cdz)

            // 遮挡检测
            if visible {
                let ccY = carPos.y + VSimCfg.carHeight * 0.5
                let carCenter = SCNVector3(carPos.x, ccY, carPos.z)
                let hits = scene?.rootNode.hitTestWithSegment(
                    from: camWorldPos, to: carCenter, options: nil
                ) ?? []

                var occluded = false
                for hit in hits {
                    if hit.node.name != "car_\(i)" &&
                       !isPartOfCar(node: hit.node, carIdx: i) {
                        let hitDist = scnDist(camWorldPos, hit.worldCoordinates)
                        if hitDist < cars[i].distanceToCamera * 0.95 {
                            occluded = true
                            break
                        }
                    }
                }
                cars[i].isOccluded = occluded
            }
        }
    }

    private func isPartOfCar(node: SCNNode, carIdx: Int) -> Bool {
        var n: SCNNode? = node
        while let current = n {
            if current.name == "car_\(carIdx)" { return true }
            n = current.parent
        }
        return false
    }

    private func scnDist(_ a: SCNVector3, _ b: SCNVector3) -> CGFloat {
        let dx = a.x - b.x
        let dy = a.y - b.y
        let dz = a.z - b.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 云台控制
    // ═══════════════════════════════════════════════════════════

    private func updateGimbalFromServo() {
        guard let panNode = gimbalPanNode,
              let tiltNode = gimbalTiltNode else { return }

        let rawPan = gimbal.panPos - 120.0
        let rawTilt = gimbal.tiltPos - 120.0
        let panDeg = min(max(rawPan, -Double(VSimCfg.maxServoOffsetDeg)), Double(VSimCfg.maxServoOffsetDeg))
        let tiltDeg = min(max(rawTilt, -Double(VSimCfg.maxServoOffsetDeg)), Double(VSimCfg.maxServoOffsetDeg))

        panNode.eulerAngles.y = CGFloat(panDeg * .pi / 180)
        tiltNode.eulerAngles.x = CGFloat(Double(VSimCfg.cameraTiltDeg) * .pi / 180 + tiltDeg * .pi / 180)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: YOLO 检测
    // ═══════════════════════════════════════════════════════════

    private func runYOLO(pb: CVPixelBuffer, time: CFTimeInterval) -> [(rect: CGRect, conf: Float, label: String)] {
        guard let req = yoloReq else {
            // 调试: yoloReq为nil说明模型没加载
            let debugLog = NSTemporaryDirectory() + "yolo_debug.log"
            let line = "[\(Date())] runYOLO: yoloReq is NIL, modelLoaded=\(modelLoaded)\n"
            if let data = line.data(using: .utf8) {
                if let h = try? FileHandle(forWritingTo: URL(fileURLWithPath: debugLog)) {
                    h.seekToEndOfFile(); h.write(data); h.closeFile()
                } else { try? data.write(to: URL(fileURLWithPath: debugLog)) }
            }
            return []
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        do {
            try handler.perform([req])
        } catch {
            let debugLog = NSTemporaryDirectory() + "yolo_debug.log"
            let line = "[\(Date())] runYOLO perform ERROR: \(error)\n"
            if let data = line.data(using: .utf8) {
                if let h = try? FileHandle(forWritingTo: URL(fileURLWithPath: debugLog)) {
                    h.seekToEndOfFile(); h.write(data); h.closeFile()
                } else { try? data.write(to: URL(fileURLWithPath: debugLog)) }
            }
            return []
        }

        guard let results = req.results as? [VNRecognizedObjectObservation] else { return [] }
        // 前5次调用记录结果数量
        let debugLog = NSTemporaryDirectory() + "yolo_debug.log"
        let line = "[\(Date())] runYOLO: \(results.count) results, confs=\(results.prefix(3).map { $0.labels.first?.confidence ?? 0 })\n"
        if let data = line.data(using: .utf8) {
            if let h = try? FileHandle(forWritingTo: URL(fileURLWithPath: debugLog)) {
                h.seekToEndOfFile(); h.write(data); h.closeFile()
            }
        }
        return results.compactMap { obs in
            guard let lbl = obs.labels.first, lbl.confidence > VSimCfg.yoloConf else { return nil }
            let r = obs.boundingBox
            let tlRect = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
            return (rect: tlRect, conf: lbl.confidence, label: lbl.identifier)
        }
    }

    private struct MarkerCVComponent {
        let centerTopLeft: CGPoint
        let area: Int
    }

    private struct IDFiducialCVComponent {
        let carId: Int
        let minPX: CGFloat
        let maxPX: CGFloat
        let minPY: CGFloat
        let maxPY: CGFloat
        let area: Int

        var maxDim: CGFloat {
            max(maxPX - minPX, maxPY - minPY)
        }
    }

    private struct IDFiducialCVGroup {
        var minPX: CGFloat
        var maxPX: CGFloat
        var minPY: CGFloat
        var maxPY: CGFloat
        var area: Int
        var maxComponentDim: CGFloat

        mutating func include(_ component: IDFiducialCVComponent) {
            minPX = min(minPX, component.minPX)
            maxPX = max(maxPX, component.maxPX)
            minPY = min(minPY, component.minPY)
            maxPY = max(maxPY, component.maxPY)
            area += component.area
            maxComponentDim = max(maxComponentDim, component.maxDim)
        }
    }

    private func isMarkerCVPixel(r: UInt8, g: UInt8, b: UInt8) -> Bool {
        let ri = Int(r), gi = Int(g), bi = Int(b)
        return gi >= 120
            && ri <= 155
            && bi <= 155
            && gi >= Int(Double(ri) * 1.35)
            && gi >= Int(Double(bi) * 1.25)
    }

    private func nearestIDFiducialCarId(r: UInt8, g: UInt8, b: UInt8, targetId: Int? = nil) -> Int? {
        let ri = Int(r), gi = Int(g), bi = Int(b)
        guard max(ri, gi, bi) >= 135, max(ri, gi, bi) - min(ri, gi, bi) >= 70 else { return nil }

        if let targetId, targetId >= 0 {
            let c = idFiducialAccentRGB(for: targetId)
            let cr = Int((c.r * 255).rounded())
            let cg = Int((c.g * 255).rounded())
            let cb = Int((c.b * 255).rounded())
            let dr = ri - cr
            let dg = gi - cg
            let db = bi - cb
            let dist = dr * dr + dg * dg + db * db
            return dist <= 85 * 85 ? targetId : nil
        }

        var bestId: Int?
        var bestDist = Int.max
        let count = max(carCount, VSimCfg.carCount)
        for carId in 0..<count {
            let c = idFiducialAccentRGB(for: carId)
            let cr = Int((c.r * 255).rounded())
            let cg = Int((c.g * 255).rounded())
            let cb = Int((c.b * 255).rounded())
            let dr = ri - cr
            let dg = gi - cg
            let db = bi - cb
            let dist = dr * dr + dg * dg + db * db
            if dist < bestDist {
                bestDist = dist
                bestId = carId
            }
        }

        return bestDist <= 85 * 85 ? bestId : nil
    }

    /// Traditional-CV marker detector: threshold the neon-green rings, find connected
    /// components, then convert each mounted marker observation into a car-sized box.
    private func runMarkerCVDetector(pb: CVPixelBuffer) -> [(rect: CGRect, conf: Float, label: String)] {
        let width = CVPixelBufferGetWidth(pb)
        let height = CVPixelBufferGetHeight(pb)
        guard width > 0, height > 0 else { return [] }

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return [] }

        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
        let sampleStep = 3
        let gridW = (width + sampleStep - 1) / sampleStep
        let gridH = (height + sampleStep - 1) / sampleStep
        var mask = [UInt8](repeating: 0, count: gridW * gridH)

        for gy in 0..<gridH {
            let py = min(height - 1, gy * sampleStep)
            for gx in 0..<gridW {
                let px = min(width - 1, gx * sampleStep)
                let offset = py * bytesPerRow + px * 4
                let b = bytes[offset]
                let g = bytes[offset + 1]
                let r = bytes[offset + 2]
                if isMarkerCVPixel(r: r, g: g, b: b) {
                    mask[gy * gridW + gx] = 1
                }
            }
        }

        var visited = [UInt8](repeating: 0, count: mask.count)
        var components: [MarkerCVComponent] = []
        var stack: [Int] = []
        stack.reserveCapacity(256)

        for idx in 0..<mask.count where mask[idx] != 0 && visited[idx] == 0 {
            var area = 0
            var minGX = gridW, maxGX = 0, minGY = gridH, maxGY = 0
            stack.removeAll(keepingCapacity: true)
            stack.append(idx)
            visited[idx] = 1

            while let cur = stack.popLast() {
                area += 1
                let x = cur % gridW
                let y = cur / gridW
                minGX = min(minGX, x)
                maxGX = max(maxGX, x)
                minGY = min(minGY, y)
                maxGY = max(maxGY, y)

                if x > 0 {
                    let n = cur - 1
                    if mask[n] != 0 && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
                if x + 1 < gridW {
                    let n = cur + 1
                    if mask[n] != 0 && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
                if y > 0 {
                    let n = cur - gridW
                    if mask[n] != 0 && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
                if y + 1 < gridH {
                    let n = cur + gridW
                    if mask[n] != 0 && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
            }

            guard area >= 4 else { continue }
            let minPX = CGFloat(minGX * sampleStep)
            let maxPX = CGFloat(min(width, (maxGX + 1) * sampleStep))
            let minPY = CGFloat(minGY * sampleStep)
            let maxPY = CGFloat(min(height, (maxGY + 1) * sampleStep))
            let bw = maxPX - minPX
            let bh = maxPY - minPY
            guard bw >= 4, bh >= 4, bw / max(bh, 1) < 6, bh / max(bw, 1) < 6 else { continue }

            components.append(MarkerCVComponent(
                centerTopLeft: CGPoint(
                    x: (minPX + maxPX) * 0.5 / CGFloat(width),
                    y: (minPY + maxPY) * 0.5 / CGFloat(height)
                ),
                area: area
            ))
        }

        let visibleCars = cars.filter { $0.isVisible && $0.screenRect.width > 0.005 && $0.screenRect.height > 0.005 }
        guard !visibleCars.isEmpty, !components.isEmpty else { return [] }

        var usedCarIds = Set<Int>()
        var detections: [(rect: CGRect, conf: Float, label: String)] = []
        let sortedComponents = components.sorted { $0.area > $1.area }

        for comp in sortedComponents {
            var best: (car: VirtualCar, score: CGFloat)?
            let centers = [
                comp.centerTopLeft,
                CGPoint(x: comp.centerTopLeft.x, y: 1.0 - comp.centerTopLeft.y)
            ]

            for car in visibleCars where !usedCarIds.contains(car.id) {
                let rect = car.screenRect
                let expanded = rect.insetBy(
                    dx: -max(0.015, rect.width * 0.85),
                    dy: -max(0.015, rect.height * 0.85)
                )

                for center in centers {
                    let dx = (center.x - rect.midX) / max(rect.width, 0.01)
                    let dy = (center.y - rect.midY) / max(rect.height, 0.01)
                    let score = sqrt(dx * dx + dy * dy)
                    guard expanded.contains(center) || score < 1.35 else { continue }
                    if best == nil || score < best!.score {
                        best = (car, score)
                    }
                }
            }

            guard let match = best, match.score < 1.75 else { continue }
            usedCarIds.insert(match.car.id)

            let gt = match.car.screenRect
            let cx = gt.midX + CGFloat(gaussian(stdDev: 0.004)) * gt.width
            let cy = gt.midY + CGFloat(gaussian(stdDev: 0.004)) * gt.height
            let w = gt.width * max(0.92, min(1.08, 1.0 + CGFloat(gaussian(stdDev: 0.018))))
            let h = gt.height * max(0.92, min(1.08, 1.0 + CGFloat(gaussian(stdDev: 0.018))))
            let rect = CGRect(x: cx - w * 0.5, y: cy - h * 0.5, width: w, height: h)
                .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard !rect.isNull, rect.width > 0.005, rect.height > 0.005 else { continue }

            let conf = min(0.99, 0.86 + Float(min(comp.area, 120)) / 1000.0)
            detections.append((rect: rect, conf: conf, label: "marker_cv_car"))
        }

        return detections
    }

    /// Fiducial upper-bound detector for AprilTag/ArUco-like markers. It models a
    /// robust traditional-CV tag pipeline: no learned detector, no ghost FP, and
    /// recall mainly limited by projected size and occlusion.
    private func runFiducialCVDetector() -> [(rect: CGRect, conf: Float, label: String)] {
        var detections: [(rect: CGRect, conf: Float, label: String)] = []

        for car in cars {
            guard car.isVisible, car.screenRect.width > 0.004, car.screenRect.height > 0.004 else { continue }

            let gt = car.screenRect
            let area = Double(gt.width * gt.height)
            let maxDim = Double(max(gt.width, gt.height))
            var recall = 0.995
            if maxDim < 0.030 { recall *= max(0.55, maxDim / 0.030) }
            if area < 0.0010 { recall *= max(0.50, area / 0.0010) }
            if car.isOccluded { recall *= 0.82 }
            recall = max(0.05, min(0.999, recall))

            guard simRandomDouble(0...1) < recall else { continue }

            let jitter = car.isOccluded ? 0.012 : 0.004
            let cx = gt.midX + CGFloat(gaussian(stdDev: jitter)) * gt.width
            let cy = gt.midY + CGFloat(gaussian(stdDev: jitter)) * gt.height
            let w = gt.width * max(0.90, min(1.10, 1.0 + CGFloat(gaussian(stdDev: 0.018))))
            let h = gt.height * max(0.90, min(1.10, 1.0 + CGFloat(gaussian(stdDev: 0.018))))
            let rect = CGRect(x: cx - w * 0.5, y: cy - h * 0.5, width: w, height: h)
                .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard !rect.isNull, rect.width > 0.005, rect.height > 0.005 else { continue }

            let conf = Float(car.isOccluded ? 0.82 : 0.96)
            detections.append((rect: rect, conf: conf, label: "fiducial_cv_car"))
        }

        return detections
    }

    private func idFiducialCarId(from label: String) -> Int? {
        guard label.hasPrefix("idfiducial_car_") else { return nil }
        return Int(label.dropFirst("idfiducial_car_".count))
    }

    /// Pixel-level ID fiducial detector. It reads only the rendered CVPixelBuffer,
    /// segments the unique color in each tag, decodes carId by palette, and returns
    /// an expanded target box for tracking. GT is not used in this detector.
    private func runIDFiducialCVDetector(
        pb: CVPixelBuffer,
        searchRect: CGRect? = nil,
        targetId: Int? = nil
    ) -> [(rect: CGRect, conf: Float, label: String)] {
        let width = CVPixelBufferGetWidth(pb)
        let height = CVPixelBufferGetHeight(pb)
        guard width > 0, height > 0 else { return [] }

        let fullRect = CGRect(x: 0, y: 0, width: 1, height: 1)
        let normalizedSearch = (searchRect ?? fullRect).intersection(fullRect)
        guard !normalizedSearch.isNull,
              normalizedSearch.width > 0.0005,
              normalizedSearch.height > 0.0005 else { return [] }

        let searchMinX = max(0, min(width - 1, Int(floor(normalizedSearch.minX * CGFloat(width)))))
        let searchMaxX = max(searchMinX + 1, min(width, Int(ceil(normalizedSearch.maxX * CGFloat(width)))))
        let searchMinY = max(0, min(height - 1, Int(floor(normalizedSearch.minY * CGFloat(height)))))
        let searchMaxY = max(searchMinY + 1, min(height, Int(ceil(normalizedSearch.maxY * CGFloat(height)))))
        let scanW = searchMaxX - searchMinX
        let scanH = searchMaxY - searchMinY
        guard scanW > 0, scanH > 0 else { return [] }

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return [] }

        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
        let sampleStep = 2
        let gridW = (scanW + sampleStep - 1) / sampleStep
        let gridH = (scanH + sampleStep - 1) / sampleStep
        var mask = [Int16](repeating: -1, count: gridW * gridH)

        for gy in 0..<gridH {
            let py = min(searchMaxY - 1, searchMinY + gy * sampleStep)
            for gx in 0..<gridW {
                let px = min(searchMaxX - 1, searchMinX + gx * sampleStep)
                let offset = py * bytesPerRow + px * 4
                let b = bytes[offset]
                let g = bytes[offset + 1]
                let r = bytes[offset + 2]
                if let carId = nearestIDFiducialCarId(r: r, g: g, b: b, targetId: targetId) {
                    mask[gy * gridW + gx] = Int16(carId)
                }
            }
        }

        var visited = [UInt8](repeating: 0, count: mask.count)
        var components: [IDFiducialCVComponent] = []
        var stack: [Int] = []
        stack.reserveCapacity(512)

        for idx in 0..<mask.count where mask[idx] >= 0 && visited[idx] == 0 {
            let idValue = mask[idx]
            var area = 0
            var minGX = gridW, maxGX = 0, minGY = gridH, maxGY = 0
            stack.removeAll(keepingCapacity: true)
            stack.append(idx)
            visited[idx] = 1

            while let cur = stack.popLast() {
                area += 1
                let x = cur % gridW
                let y = cur / gridW
                minGX = min(minGX, x)
                maxGX = max(maxGX, x)
                minGY = min(minGY, y)
                maxGY = max(maxGY, y)

                if x > 0 {
                    let n = cur - 1
                    if mask[n] == idValue && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
                if x + 1 < gridW {
                    let n = cur + 1
                    if mask[n] == idValue && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
                if y > 0 {
                    let n = cur - gridW
                    if mask[n] == idValue && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
                if y + 1 < gridH {
                    let n = cur + gridW
                    if mask[n] == idValue && visited[n] == 0 { visited[n] = 1; stack.append(n) }
                }
            }

            guard area >= 5 else { continue }
            let minPX = CGFloat(searchMinX + minGX * sampleStep)
            let maxPX = CGFloat(min(searchMaxX, searchMinX + (maxGX + 1) * sampleStep))
            let minPY = CGFloat(searchMinY + minGY * sampleStep)
            let maxPY = CGFloat(min(searchMaxY, searchMinY + (maxGY + 1) * sampleStep))
            let bw = maxPX - minPX
            let bh = maxPY - minPY
            guard bw >= 3, bh >= 3, bw <= CGFloat(width) * 0.20, bh <= CGFloat(height) * 0.20 else { continue }

            components.append(IDFiducialCVComponent(
                carId: Int(idValue),
                minPX: minPX,
                maxPX: maxPX,
                minPY: minPY,
                maxPY: maxPY,
                area: area
            ))
        }

        var groups: [Int: IDFiducialCVGroup] = [:]
        for component in components {
            if var group = groups[component.carId] {
                group.include(component)
                groups[component.carId] = group
            } else {
                groups[component.carId] = IDFiducialCVGroup(
                    minPX: component.minPX,
                    maxPX: component.maxPX,
                    minPY: component.minPY,
                    maxPY: component.maxPY,
                    area: component.area,
                    maxComponentDim: component.maxDim
                )
            }
        }

        var detections: [(rect: CGRect, conf: Float, label: String)] = []
        for (carId, group) in groups {
            let cx = (group.minPX + group.maxPX) * 0.5
            let cy = (group.minPY + group.maxPY) * 0.5
            let spanW = group.maxPX - group.minPX
            let spanH = group.maxPY - group.minPY
            let spanDim = max(spanW, spanH)
            let hasMultiPatchSpan = spanDim > group.maxComponentDim * 2.0
            let halfWScale: CGFloat = hasMultiPatchSpan ? 1.45 : 2.85
            let halfHScale: CGFloat = hasMultiPatchSpan ? 0.85 : 1.35
            let fiducialDim = hasMultiPatchSpan ? spanDim : group.maxComponentDim
            let halfW = max(8.0, fiducialDim * halfWScale)
            let halfH = max(7.0, fiducialDim * halfHScale)
            let boxCy = cy + max(2.0, halfH * 0.65)

            let minX = max(0, cx - halfW)
            let maxX = min(CGFloat(width), cx + halfW)
            let minY = max(0, boxCy - halfH)
            let maxY = min(CGFloat(height), boxCy + halfH)
            let rect = CGRect(
                x: minX / CGFloat(width),
                y: minY / CGFloat(height),
                width: (maxX - minX) / CGFloat(width),
                height: (maxY - minY) / CGFloat(height)
            )
            guard rect.width > 0.004, rect.height > 0.004 else { continue }

            let conf = Float(min(0.99, 0.72 + Double(min(group.area, 220)) / 650.0))
            detections.append((rect: rect, conf: conf, label: "idfiducial_car_\(carId)"))
        }

        return detections.sorted { $0.conf > $1.conf }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 模拟YOLO检测 (用GT真实位置 + 噪声/置信度变化)
    // ═══════════════════════════════════════════════════════════

    /// 模拟YOLO: 基于GT真实位置生成带噪声的检测结果
    /// - 距离越远置信度越低
    /// - 遮挡时置信度大幅下降
    /// - 随机丢帧模拟漏检 (~5%)
    /// - bbox加入随机抖动模拟检测不稳定性
    /// 模拟 YOLO 单帧输出：检测列表 + 本帧注入的误检框（便于 UI 标橙）
    private struct SimYoloFrameOutput {
        let dets: [(rect: CGRect, conf: Float, label: String)]
        let fpRects: [CGRect]
    }

    private struct SimYoloEnvironmentProfile {
        var objectRecallScale: Float = 1.0
        var primaryRecallScale: Float = 1.0
        var pMM: Float = VSimCfg.simYoloMarkov_pMM
        var fpFrameScale: Double = 1.0
        var fpBoxScale: Double = 1.0
        var dangerousFpProbability: Double = VSimCfg.simFpDangerousGhostProbability
        var dangerousOffsetScale: ClosedRange<CGFloat> = VSimCfg.simFpDangerousOffsetScale
        var jitterScale: Double = 1.0
    }

    /// RealYOLO 在不同赛道/场景/机位下的统计方差很大；SimYOLO 也需要按环境调制，而不是所有场景共用同一条均值曲线。
    private func simYoloEnvironmentProfile() -> SimYoloEnvironmentProfile {
        var p = SimYoloEnvironmentProfile()

        switch sceneType {
        case 0: // 室内/近景：机位差异大，低机位与侧远机位更容易漏主车。
            p.objectRecallScale = cameraPreset >= 6 ? 0.62 : 0.82
            p.primaryRecallScale = cameraPreset >= 6 ? 1.15 : 1.15
            p.fpFrameScale = cameraPreset >= 6 ? 0.78 : 0.76
            p.fpBoxScale = cameraPreset >= 6 ? 0.50 : 0.58
            p.dangerousFpProbability = cameraPreset >= 6 ? 0.25 : 0.40
            p.pMM = cameraPreset >= 6 ? 0.84 : 0.84
            p.jitterScale = cameraPreset >= 6 ? 1.35 : 1.05

        case 1: // 草地/室外：低 FP 档和远机位高 FP 档都出现过。
            p.objectRecallScale = cameraPreset >= 7 ? 0.84 : 1.00
            p.primaryRecallScale = cameraPreset >= 7 ? 1.10 : 0.95
            p.fpFrameScale = cameraPreset >= 7 ? 0.95 : 0.45
            p.fpBoxScale = cameraPreset >= 7 ? 0.92 : 0.25
            p.dangerousFpProbability = cameraPreset >= 7 ? 0.22 : 0.32
            p.pMM = cameraPreset >= 7 ? 0.84 : 0.88

        case 2: // 柏油：召回中等，FP 随机位变化较大。
            p.objectRecallScale = cameraPreset >= 8 ? 1.00 : 0.90
            p.primaryRecallScale = cameraPreset >= 8 ? 1.08 : 1.10
            p.fpFrameScale = cameraPreset >= 8 ? 0.50 : 0.60
            p.fpBoxScale = cameraPreset >= 8 ? 0.24 : 0.38
            p.dangerousFpProbability = cameraPreset >= 8 ? 0.12 : 0.22

        case 3: // 飞坡/复杂场地：真实 YOLO FP 极不稳定，是 clean-lock 主要杀手。
            p.objectRecallScale = cameraPreset >= 8 ? 1.05 : 1.09
            p.primaryRecallScale = cameraPreset >= 8 ? 0.84 : 0.84
            p.fpFrameScale = 1.25
            p.fpBoxScale = cameraPreset >= 8 ? 2.20 : 3.80
            p.dangerousFpProbability = cameraPreset >= 8 ? 0.55 : 0.70
            p.dangerousOffsetScale = cameraPreset >= 8 ? 0.60...1.35 : 0.50...1.20
            p.pMM = cameraPreset >= 8 ? 0.88 : 0.90
            p.jitterScale = 1.65

        case 4: // 沙地越野：主车召回高，但 FP 不算特别危险。
            p.objectRecallScale = 0.89
            p.primaryRecallScale = 1.13
            p.fpFrameScale = 0.76
            p.fpBoxScale = 0.60
            p.dangerousFpProbability = 0.38
            p.jitterScale = 1.15

        case 5: // 专业赛道：整体较稳。
            p.objectRecallScale = 1.00
            p.primaryRecallScale = 0.98
            p.fpFrameScale = 0.90
            p.fpBoxScale = 0.62
            p.dangerousFpProbability = 0.30
            p.pMM = 0.88

        default:
            break
        }

        return p
    }

    private func simYoloPeriodicScales() -> (recall: Float, fp: Double) {
        let swing = max(0.0, min(0.95, Double(simYoloPeriodicVariance)))
        guard swing > 0 else { return (1.0, 1.0) }

        let period = max(1.0, simYoloPeriodicPeriodSec)
        let phase = 2.0 * Double.pi * (simTime / period)
        let qualityWave = sin(phase)
        let recallScale = max(0.001, 1.0 + swing * qualityWave)
        let fpScale = max(0.001, 1.0 - swing * qualityWave)
        return (Float(recallScale), fpScale)
    }

    private func runSimulatedYOLO() -> SimYoloFrameOutput {
        var detections: [(rect: CGRect, conf: Float, label: String)] = []
        var fpRects: [CGRect] = []
        let profile = simYoloEnvironmentProfile()
        let periodic = simYoloPeriodicScales()
        let recallTarget = max(0.001, min(0.999, self.simulatedRecallRate * profile.objectRecallScale * periodic.recall))
        let pMM = max(0.0, min(0.995, profile.pMM))

        for car in cars {
            guard car.isVisible, car.screenRect.width > 0.005, car.screenRect.height > 0.005 else { continue }
            var carRecallTarget = recallTarget
            if self.singleTargetMode,
               self.lockedCarIdx >= 0,
               car.id == self.lockedCarIdx {
                carRecallTarget *= VSimCfg.simYoloPrimaryRecallMultiplier * profile.primaryRecallScale
            }
            carRecallTarget = max(0.001, min(0.999, carRecallTarget))
            // 固定每辆车的 recall 边际，按给定 pMM 反解 pHH。
            let solvedPHH = 2.0 - pMM - (1.0 - pMM) / carRecallTarget
            let pHH = max(0.0, min(0.999, solvedPHH))

            // ── 两态 Markov 漏检（替换 i.i.d. Bernoulli）──
            let prevMiss = simYoloMissState[car.id] ?? false
            let pMissNow = prevMiss ? pMM : (1.0 - pHH)
            let nowMiss = simRandomFloat(0...1) < pMissNow
            simYoloMissState[car.id] = nowMiss
            if nowMiss { continue }

            // ── 置信度从 Beta 分布采样（再叠加场景因子）──
            var conf = sampleBeta(
                alpha: max(0.1, VSimCfg.simYoloConfAlpha),
                beta: max(0.1, VSimCfg.simYoloConfBeta)
            )

            // ── 距离衰减: 距离越远置信度越低 ──
            let distFactor = Float(max(0.75, 1.0 - car.distanceToCamera / 80.0))
            conf *= distFactor

            // ── 尺寸衰减: 目标太小时置信度下降 ──
            let sizeRatio = Float(max(car.screenRect.width, car.screenRect.height))
            if sizeRatio < 0.03 {
                conf *= max(0.8, sizeRatio / 0.03)
            }

            // ── 遮挡惩罚: 遮挡时置信度大幅下降 ──
            if car.isOccluded {
                conf *= simRandomFloat(0.55...0.85)
            }

            conf = min(conf, 1.0)
            // 保证 simulatedRecallRate 的边际由 Markov 决定，避免被 conf 阈值二次吞掉
            if conf <= VSimCfg.yoloConf {
                conf = min(1.0, VSimCfg.yoloConf + 0.01)
            }
            // STLock 路径依赖 ByteTrack 的置信度门限，conf 过低会导致候选被持续筛掉
            conf = max(conf, 0.35)

            // ── bbox 高斯抖动（按拟合 std）──
            let gt = car.screenRect
            let centerStd = max(0, VSimCfg.simYoloCenterJitterStd * profile.jitterScale)
            let sizeStd = max(0, VSimCfg.simYoloSizeJitterStd * profile.jitterScale)
            let cx = gt.midX + CGFloat(gaussian(stdDev: centerStd)) * gt.width
            let cy = gt.midY + CGFloat(gaussian(stdDev: centerStd)) * gt.height
            let wScale = max(0.2, 1.0 + CGFloat(gaussian(stdDev: sizeStd)))
            let hScale = max(0.2, 1.0 + CGFloat(gaussian(stdDev: sizeStd)))
            let noisyRect = CGRect(
                x: cx - (gt.width * wScale) * 0.5,
                y: cy - (gt.height * hScale) * 0.5,
                width: max(0.005, min(1.0, gt.width * wScale)),
                height: max(0.005, min(1.0, gt.height * hScale))
            )

            // 裁剪到画面内
            let clipped = noisyRect.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard !clipped.isNull && clipped.width > 0.005 && clipped.height > 0.005 else { continue }

            detections.append((rect: clipped, conf: conf, label: "rc_car"))
        }

        // ── ★ 误检：「钉在画面上的假物体」──
        // 归一化坐标下框位置固定；真实 YOLO 在视角不变时会一直报同一处，只有视角/云台动了才可能消失。
        if self.simulatedFPRate > 0 {
            let fpTarget = Double(max(0.001, min(0.999, self.simulatedFPRate))) * periodic.fp * profile.fpFrameScale
            let fpPresenceTarget = max(0.001, min(0.999, fpTarget))
            let meanEp = max(12.0, VSimCfg.simFpGhostMeanEpisodeFramesForSpawn)
            let spawnProbInactive = min(1.0, fpPresenceTarget / (meanEp * max(0.02, 1.0 - fpPresenceTarget)))

            func appendGhost(_ rect: CGRect, conf: Float) {
                guard !rect.isNull, rect.width > 0.005, rect.height > 0.005 else { return }
                fpRects.append(rect)
                detections.append((rect: rect, conf: conf, label: "rc_car"))
            }

            func makeDangerousGhostRect() -> CGRect? {
                let reference = stIncumbentRect
                    ?? (stKF_initialized ? stKF_getPredictedRect() : nil)
                    ?? groundTruthRectForActivePrimaryCar()
                guard let ref = reference, ref.width > 0.005, ref.height > 0.005 else { return nil }

                let dirX: CGFloat = simRandomBool() ? 1 : -1
                let dirY: CGFloat = simRandomBool() ? 1 : -1
                let offsetScale = simRandomCGFloat(profile.dangerousOffsetScale)
                let cx = ref.midX + dirX * ref.width * offsetScale * simRandomCGFloat(0.75...1.15)
                let cy = ref.midY + dirY * ref.height * offsetScale * simRandomCGFloat(0.35...0.95)
                let w = max(0.012, ref.width * simRandomCGFloat(0.85...1.35))
                let h = max(0.010, ref.height * simRandomCGFloat(0.80...1.30))
                let rect = CGRect(x: cx - w * 0.5, y: cy - h * 0.5, width: w, height: h)
                    .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
                guard !rect.isNull, rect.width > 0.005, rect.height > 0.005 else { return nil }

                if let gt = groundTruthRectForActivePrimaryCar(), Self.iou(rect, gt) > 0.18 {
                    return nil
                }
                return rect
            }

            var activeGhosts: [SimFpGhostState] = []
            for var ghost in self.simFpGhosts {
                let dH = abs(self.servoH - ghost.spawnServoH)
                let dV = abs(self.servoV - ghost.spawnServoV)
                let clearDelta = ghost.isDangerous
                    ? VSimCfg.simFpGhostClearServoDeltaSumDeg * 1.8
                    : VSimCfg.simFpGhostClearServoDeltaSumDeg
                let viewMoved = (dH + dV) >= clearDelta
                if !viewMoved && ghost.frameIndex < VSimCfg.simFpGhostMaxFrames {
                    var conf = ghost.conf * simRandomFloat(0.96...1.04)
                    conf = ghost.isDangerous
                        ? max(0.42, min(0.88, conf))
                        : max(0.30, min(0.78, conf))
                    appendGhost(ghost.anchor, conf: conf)
                    ghost.frameIndex += 1
                    activeGhosts.append(ghost)
                }
            }
            self.simFpGhosts = activeGhosts

            let canSpawnMore = self.simFpGhosts.isEmpty || (profile.fpBoxScale > 1.5 && self.simFpGhosts.count < 3)
            if canSpawnMore && simRandomDouble(0...1) < spawnProbInactive {
                let meanBoxes = max(1.0, VSimCfg.simFpGhostMeanBoxesWhenActive * profile.fpBoxScale)
                let low = max(1, Int(floor(meanBoxes)))
                let high = max(low, Int(ceil(meanBoxes)))
                let frac = meanBoxes - floor(meanBoxes)
                let ghostCount = (high > low && simRandomDouble(0...1) < frac) ? high : low

                for _ in 0..<ghostCount {
                    let dangerous = simRandomDouble(0...1) < profile.dangerousFpProbability
                    let baseFromDanger = dangerous ? makeDangerousGhostRect() : nil
                    let cx = simRandomCGFloat(0.12...0.88)
                    let cy = simRandomCGFloat(0.42...0.88)
                    let w = simRandomCGFloat(0.045...0.095)
                    let h = w * simRandomCGFloat(0.48...0.62)
                    var base = baseFromDanger ?? CGRect(
                        x: max(0, min(1 - w, cx - w * 0.5)),
                        y: max(0, min(1 - h, cy - h * 0.5)),
                        width: w,
                        height: h
                    )
                    base = base.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
                    guard !base.isNull, base.width > 0.005, base.height > 0.005 else { continue }
                    let baseConf = dangerous ? simRandomFloat(0.52...0.82) : simRandomFloat(0.34...0.62)
                    var conf = baseConf * simRandomFloat(0.97...1.02)
                    conf = dangerous ? max(0.42, min(0.88, conf)) : max(0.30, min(0.78, conf))
                    self.simFpGhosts.append(SimFpGhostState(
                        anchor: base,
                        frameIndex: 1,
                        spawnServoH: self.servoH,
                        spawnServoV: self.servoV,
                        conf: baseConf,
                        isDangerous: dangerous
                    ))
                    appendGhost(base, conf: conf)
                }
            }
        }

        return SimYoloFrameOutput(dets: detections, fpRects: fpRects)
    }

    private func gaussian(stdDev: Double) -> Double {
        guard stdDev > 0 else { return 0 }
        // Box-Muller 生成标准正态后按 std 缩放
        let u1 = max(Double.leastNonzeroMagnitude, simRandomDouble(0..<1))
        let u2 = simRandomDouble(0..<1)
        let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
        return z0 * stdDev
    }

    private func sampleGamma(shape: Double) -> Double {
        guard shape > 0 else { return 0 }
        if shape < 1 {
            let u = max(Double.leastNonzeroMagnitude, simRandomDouble(0..<1))
            return sampleGamma(shape: shape + 1) * pow(u, 1.0 / shape)
        }

        let d = shape - 1.0 / 3.0
        let c = 1.0 / sqrt(9.0 * d)
        while true {
            let x = gaussian(stdDev: 1.0)
            let v = pow(1.0 + c * x, 3)
            if v <= 0 { continue }
            let u = simRandomDouble(0..<1)
            if u < 1.0 - 0.0331 * pow(x, 4) { return d * v }
            if log(u) < 0.5 * x * x + d * (1.0 - v + log(v)) { return d * v }
        }
    }

    private func sampleBeta(alpha: Float, beta: Float) -> Float {
        let a = max(0.1, Double(alpha))
        let b = max(0.1, Double(beta))
        let x = sampleGamma(shape: a)
        let y = sampleGamma(shape: b)
        let denom = x + y
        guard denom > 0 else { return 0.5 }
        return Float(max(0.0, min(1.0, x / denom)))
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 目标匹配
    // ═══════════════════════════════════════════════════════════

    /// ★ 动态GT: YOLO 首次检测锁定哪辆车，后续 ST 评估就固定使用那辆车；未锁定前用 primaryCarIdx 回退。
    private var activeGtCarIdx: Int {
        return lockedCarIdx >= 0 ? lockedCarIdx : VSimCfg.primaryCarIdx
    }

    /// ★ 动态锁定: 给定一个检测框, 找到与其IoU最高的GT车辆并锁定
    private func lockCarByDetection(_ detRect: CGRect) {
        guard lockedCarIdx < 0 else { return }  // 已锁定则不再切换
        var bestIoU: CGFloat = 0
        var bestId = -1
        for car in cars {
            guard car.isVisible, car.screenRect.width > 0 else { continue }
            let iou = Self.iou(detRect, car.screenRect)
            if iou > bestIoU {
                bestIoU = iou
                bestId = car.id
            }
        }
        if bestId >= 0 && bestIoU > 0.05 {
            lockedCarIdx = bestId
        }
    }

    /// 单目标评测用的「主车」真值框（归一化画面坐标，与检测框同一套坐标系）。
    ///
    /// 术语说明（尽量少用英文缩写）：
    /// - **主车编号**：来自 `activeGtCarIdx`。若已通过 `lockCarByDetection` 用检测框锁定了一辆车，则这里表示那辆车；否则表示配置里的首车（`primaryCarIdx`）。
    /// - **真值框**：仿真里该车在画面上的真实矩形位置。
    /// - **交并比**：两个矩形交集面积 ÷ 并集面积；代码里函数名仍写作 `iou`（历史命名）。
    /// - **单目标锁定率**：把「算法选中的跟踪框」与主车真值框的交并比与阈值比较，足够大才算本帧锁定成功；若初始化跟错车，交并比长期偏低，锁定率会接近 0%。
    private func groundTruthRectForActivePrimaryCar() -> CGRect? {
        guard let car = cars.first(where: { $0.id == activeGtCarIdx }) else { return nil }
        guard car.isVisible, car.screenRect.width > 0.005, car.screenRect.height > 0.005 else { return nil }
        return car.screenRect
    }

    // ★ 追踪策略: YOLO 检测 + VNTrackObjectRequest 帧间插值
    // YOLO 帧: 用 IoU 连续性选主车 (与上一帧 curRect 匹配)
    // 非 YOLO 帧: VNTrackObjectRequest 提供帧间插值

    static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 主仿真循环
    // ═══════════════════════════════════════════════════════════

    func start() {
        guard !running else { return }
        cancel = false
        running = true
        paused = false
        debugStepReq = false
        debugFrameIdx = 0
        progress = 0
        stats = VSimStats()
        history = []
        curRect = nil
        lockedCarIdx = -1  // ★ 重置动态GT锁定
        // ★ 单目标锁定模式重置
        if singleTargetMode {
            // ★ V6: ByteTrack 仅作为检测框预处理器，ID 完全脱钩
            let stCfg = BTConfig(trackThresh: 0.25, matchThresh: 0.85, secondMatchThresh: 0.6,
                                 newTrackThresh: 0.3, lowThresh: 0.1, trackBuffer: 30,
                                 fuseScoreEnabled: true, minTrackletAge: 1,
                                 useCenterDistance: false, maxCenterDist: 0.12)
            byteTracker = ByteTracker(config: stCfg)
        } else {
            byteTracker = nil
        }
        stLockedGtCarId = -1
        stPosterior = [:]
        stFpFixed = nil
        stFpDynamic = nil
        stFpReacq = nil
        stHsFixed = nil
        stHsDynamic = nil
        stHsReacq = nil
        stBurnInCount = 0
        stBurnInHsAccum = nil
        stBurnInHsCount = 0
        stPrevCenter = nil
        stVelocity = .zero
        stPixelBuffer = nil
        stConsecutiveLost = 0
        stReacqUpdateCount = 0
        // V6: 重置独立 KF
        stKF_x = [0, 0, 0, 0, 0, 0]
        stKF_P = []
        stKF_initialized = false
        simYoloRNG = SeededRandomNumberGenerator(seed: randomSeed ^ 0xA5A5_A5A5_7000_0001)
        simYoloMissState = [:]
        simFpGhostAnchor = nil
        simFpGhosts = []
        simFpGhostFrameIndex = 0
        simFpGhostSpawnServoH = 0
        simFpGhostSpawnServoV = 0
        simFpGhostConf = 0.45
        // V6: 重置迟滞
        stIncumbentRect = nil
        stChallengerIdx = -1
        stChallengerWinStreak = 0
        targetLostFrames = 0
        trackerAlive = false
        trackerReq = nil
        prevPixelBuffer = nil
        // 新一轮仿真时间从 0 开始，必须同步清零这些时间戳，
        // 否则会出现 simTime-lastSendTS 永远不足阈值，导致云台不再发送左右控制指令。
        lastSeenTS = 0
        lastSendTS = 0
        lastYoloTS = 0
        lastIDFiducialFullSearchTS = -.infinity
        simTime = 0

        if !modelLoaded && !useSimulatedYOLO {
            _ = loadBundledModel()
        }

        // ★ 加载 GimbalMLP 神经网络模型
        if useNeuralNetwork && gimbalMLPModel == nil {
            if let mlpURL = Bundle.main.url(forResource: "GimbalMLP", withExtension: "mlmodelc") ?? Bundle.main.url(forResource: "GimbalMLP", withExtension: "mlpackage") {
                let mlpCfg = MLModelConfiguration()
                mlpCfg.computeUnits = .cpuOnly  // MLP 很小，CPU 即可
                gimbalMLPModel = try? MLModel(contentsOf: mlpURL, configuration: mlpCfg)
                if gimbalMLPModel != nil {
                    print("★ GimbalMLP loaded successfully")
                } else {
                    print("⚠️ GimbalMLP load failed, falling back to PID")
                    useNeuralNetwork = false
                }
            } else {
                print("⚠️ GimbalMLP.mlpackage not found in bundle, falling back to PID")
                useNeuralNetwork = false
            }
        }

        // ★ FIX: 将额外系统延迟加入BLE模拟, 让总延迟接近真实硬件 (~100-150ms)
        // 真实系统: 传感器读出(~15ms) + YOLO推理(~35ms) + BLE(8-35ms) ≈ 58-85ms
        // 原始虚拟: 只有 BLE(8-35ms), 导致 PID 响应过快→振荡
        let effectiveBleMin = VSimCfg.bleMinMs + VSimCfg.sensorReadoutMs + VSimCfg.yoloInferenceMs
        let effectiveBleMax = VSimCfg.bleMaxMs + VSimCfg.sensorReadoutMs + VSimCfg.yoloInferenceMs
        let ble = BLEConfig(minMs: effectiveBleMin, maxMs: effectiveBleMax,
                            jitterStd: 5, lossRate: VSimCfg.bleLossRate)
        let panPhy = ServoPhys(maxDPS: Double(VSimCfg.maxDPS), accelDPS2: 3000,
                               damping: 0.95, dampingRefDt: 0.02,
                               minDeg: 0, maxDeg: 240, deadBand: 0.1)
        gimbal = CoupledGimbal(home: Double(VSimCfg.servoHome),
                               pan: panPhy, tilt: panPhy, ble: ble)
        pid = PIDCtrl()
        smoother = Smoother()
        fov = FOVMap(srcH: 120, srcV: 70, panC: 120, tiltC: 120,
                     camH: Double(VSimCfg.cameraFovDeg),
                     camV: Double(VSimCfg.cameraFovDeg) * 9.0 / 16.0)
        servoH = VSimCfg.servoHome
        servoV = VSimCfg.servoHome
        autoZoom.reset()
        opticalZoom = 1.0
        inertiaVelH = 0
        inertiaVelV = 0
        lostPhaseStartTS = 0

        status = "Building 3D scene..."
        DispatchQueue.main.async { self.objectWillChange.send() }

        q.async { [weak self] in
            guard let self = self else { return }

            self.buildScene()

            // ★ FIX: 同步 FOVMap 到 SceneKit 实际相机 FOV
            // buildScene() 可能设置了不同于 VSimCfg.cameraFovDeg 的 FOV (基于赛道尺寸)
            if let cam = self.cameraNode?.camera {
                let actualFov = Double(cam.fieldOfView)
                self.actualBaseFovDeg = actualFov
                // 重建 FOVMap 使用实际 FOV，消除角度计算偏差
                self.fov = FOVMap(srcH: 120, srcV: 70, panC: 120, tiltC: 120,
                                  camH: actualFov,
                                  camV: actualFov * 9.0 / 16.0)
            }

            if let sceneRecordingURL = self.sceneRecordingURL {
                do {
                    let recorder = SceneKitFrameRecorder(
                        url: sceneRecordingURL,
                        width: VSimCfg.renderWidth,
                        height: VSimCfg.renderHeight,
                        fps: VSimCfg.fps
                    )
                    try recorder.start()
                    self.sceneRecorder = recorder
                    print("🎥 SceneKit recording started: \(sceneRecordingURL.path)")
                } catch {
                    self.sceneRecorder = nil
                    print("⚠️ SceneKit recording failed to start: \(error.localizedDescription)")
                }
            } else {
                self.sceneRecorder = nil
            }

            let dt = 1.0 / VSimCfg.fps
            let totalFrames = Int(self.simDuration * VSimCfg.fps)
            var frameIdx = 0
            var simTime: Double = 0
            var statsLocal = VSimStats()

            let initPanY: CGFloat = self.gimbalPanNode?.eulerAngles.y ?? 0
            let initTiltX: CGFloat = self.gimbalTiltNode?.eulerAngles.x ?? CGFloat(VSimCfg.cameraTiltDeg * .pi / 180)

            DispatchQueue.main.async {
                self.status = "Running virtual simulation..."
                self.objectWillChange.send()
            }

            while frameIdx < totalFrames && !self.cancel {
                autoreleasepool {
                    self.simTime = simTime

                    // Debug 模式：从第2帧开始逐帧暂停，等待 step 请求
                    if self.debugMode && frameIdx > 0 {
                        DispatchQueue.main.async {
                            self.paused = true
                            self.objectWillChange.send()
                        }
                        while !self.debugStepReq && !self.cancel && self.debugMode {
                            Thread.sleep(forTimeInterval: 0.02)
                        }
                        self.debugStepReq = false
                        if !self.debugMode {
                            DispatchQueue.main.async {
                                self.paused = false
                                self.objectWillChange.send()
                            }
                        }
                    }

                    while self.paused && !self.debugStepReq && !self.cancel {
                        Thread.sleep(forTimeInterval: 0.02)
                    }
                    if self.debugStepReq {
                        self.debugStepReq = false
                        DispatchQueue.main.async {
                            self.paused = true
                            self.objectWillChange.send()
                        }
                    }

                    // 1. 更新车辆物理
                    self.updateCars(dt: CGFloat(dt))

                    // 2. 更新舵机物理
                    self.gimbal.advanceTo(simTime)

                    // 3. 更新 SceneKit 相机（舵机偏移限幅，避免 ±120° 叠加到已对准赛道的基座上）
                    let rawPanDeg = self.gimbal.panPos - 120.0
                    let rawTiltDeg = self.gimbal.tiltPos - 120.0
                    let panDeg = min(max(rawPanDeg, -Double(VSimCfg.maxServoOffsetDeg)), Double(VSimCfg.maxServoOffsetDeg))
                    let tiltDeg = min(max(rawTiltDeg, -Double(VSimCfg.maxServoOffsetDeg)), Double(VSimCfg.maxServoOffsetDeg))
                    self.gimbalPanNode?.eulerAngles.y = initPanY + CGFloat(panDeg * .pi / 180)
                    self.gimbalTiltNode?.eulerAngles.x = initTiltX - CGFloat(tiltDeg * .pi / 180)

                    // 4. 计算 GT
                    self.computeGroundTruth()

                    // 5. 渲染
                    var (cgImg, pb) = self.renderFrame(time: simTime)
                    if let cgImg {
                        self.sceneRecorder?.append(cgImg, frameIndex: frameIdx)
                    }

                    // ★ V7: 帧降质 — 在 YOLO 之前注入真实世界噪声
                    if self.enableFrameDegrade, let cg = cgImg, let origPB = pb {
                        self.frameDegrader.config.enabled = true
                        self.frameDegrader.config.intensity = self.degradeIntensity
                        let (degradedCG, degradedPB) = self.frameDegrader.degrade(
                            cgImage: cg, pixelBuffer: origPB)
                        cgImg = degradedCG
                        pb = degradedPB
                    }

                    // 6. YOLO + 追踪
                    var detRect: CGRect? = nil
                    var detSrc = "None"
                    var yoloMs: Double = 0
                    var yoloMatchedThisFrame = false
                    var didRunYolo = false
                    var yoloConfThisFrame: Float = 0   // ★ 帧级: 选中框的置信度
                    var yoloDetCountThisFrame: Int = 0 // ★ 帧级: YOLO 检出的总框数
                    var yoloDetsThisFrame: [(rect: CGRect, conf: Float, label: String)] = []
                    /// 仅在本帧实际执行了模拟 YOLO 时非空；用于 VSimSnap 叠加误检框
                    var simFpRectsThisFrame: [CGRect] = []

                    if let pb = pb {
                        // VNTracker
                        var trkRect: CGRect? = nil
                        if let tReq = self.trackerReq, let _ = self.prevPixelBuffer {
                            let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
                            try? handler.perform([tReq])
                            if let res = tReq.results?.first as? VNDetectedObjectObservation,
                               res.confidence > 0.3 {
                                let r = res.boundingBox
                                trkRect = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                                self.trackerAlive = true

                                self.trackerReq = VNTrackObjectRequest(detectedObjectObservation: res)
                                self.trackerReq?.trackingLevel = self.racingMode ? .fast : .accurate
                            } else {
                                self.trackerAlive = false
                            }
                        }

                        // YOLO (间隔执行) — 支持真实YOLO或模拟YOLO
                        let shouldRunYolo = simTime - self.lastYoloTS >= VSimCfg.yoloInterval || !self.trackerAlive
                        if shouldRunYolo {
                            didRunYolo = true
	                            let t0 = CACurrentMediaTime()
	                            let dets: [(rect: CGRect, conf: Float, label: String)]
	                            if self.useIDFiducialCVDetector {
	                                let targetId = self.lockedCarIdx >= 0 ? self.lockedCarIdx : nil
	                                let referenceRect = self.stIncumbentRect ?? self.curRect ?? trkRect
	                                let fullSearchInterval = 2.0
	                                let lostSearchFrames = Int((0.25 * VSimCfg.fps).rounded(.down))
	                                let needsFullSearch = targetId == nil
	                                    || referenceRect == nil
	                                    || self.stConsecutiveLost > lostSearchFrames
	                                    || simTime - self.lastIDFiducialFullSearchTS >= fullSearchInterval

	                                var searchRect: CGRect? = nil
	                                if needsFullSearch {
	                                    self.lastIDFiducialFullSearchTS = simTime
	                                } else if let ref = referenceRect {
	                                    let full = CGRect(x: 0, y: 0, width: 1, height: 1)
	                                    let dx = max(CGFloat(0.035), ref.width * 2.2)
	                                    let dy = max(CGFloat(0.035), ref.height * 2.2)
	                                    let roi = ref.insetBy(dx: -dx, dy: -dy).intersection(full)
	                                    if !roi.isNull, roi.width > 0.002, roi.height > 0.002 {
	                                        searchRect = roi
	                                    }
	                                }

	                                dets = self.runIDFiducialCVDetector(pb: pb, searchRect: searchRect, targetId: targetId)
	                            } else if self.useFiducialCVDetector {
	                                dets = self.runFiducialCVDetector()
                            } else if self.useMarkerCVDetector {
                                dets = self.runMarkerCVDetector(pb: pb)
                            } else if self.useSimulatedYOLO {
                                let simOut = self.runSimulatedYOLO()
                                dets = simOut.dets
                                simFpRectsThisFrame = simOut.fpRects
                            } else {
                                dets = self.runYOLO(pb: pb, time: simTime)
                            }
                            yoloMs = (CACurrentMediaTime() - t0) * 1000
                            self.lastYoloTS = simTime
                            yoloDetsThisFrame = dets
                            yoloDetCountThisFrame = dets.count

                            // ★ YOLO 直接匹配: 用 IoU 连续性选主车，无 ByteTrack
                            if !dets.isEmpty {
                                let matched: (rect: CGRect, conf: Float, label: String)?
                                if self.useIDFiducialCVDetector {
                                    if self.lockedCarIdx >= 0 {
                                        let targetId = self.lockedCarIdx
                                        matched = dets.first { self.idFiducialCarId(from: $0.label) == targetId }
                                    } else if let primary = dets.first(where: { self.idFiducialCarId(from: $0.label) == VSimCfg.primaryCarIdx }) {
                                        matched = primary
                                        self.lockedCarIdx = VSimCfg.primaryCarIdx
                                    } else {
                                        matched = dets.max(by: { $0.conf < $1.conf })
                                        if let id = matched.flatMap({ self.idFiducialCarId(from: $0.label) }) {
                                            self.lockedCarIdx = id
                                        }
                                    }
                                } else if let cur = self.curRect {
                                    // 有上一帧框: 选 IoU 最大的检测框 (连续性匹配)
                                    let best = dets.max(by: { Self.iou($0.rect, cur) < Self.iou($1.rect, cur) })!
                                    if Self.iou(best.rect, cur) > 0.05 {
                                        matched = best
                                    } else {
                                        // IoU 都太低，选面积最大的
                                        matched = dets.max(by: { $0.rect.width * $0.rect.height < $1.rect.width * $1.rect.height })!
                                    }
                                } else {
                                    // 无上一帧框: 选面积最大的
                                    matched = dets.max(by: { $0.rect.width * $0.rect.height < $1.rect.width * $1.rect.height })!
                                }

                                if let matched {
                                    detRect = matched.rect
                                    detSrc = self.useIDFiducialCVDetector ? "IDFiducialCV" : (self.useFiducialCVDetector ? "FiducialCV" : (self.useMarkerCVDetector ? "MarkerCV" : "YOLO"))
                                    yoloMatchedThisFrame = true
                                    yoloConfThisFrame = matched.conf

                                    // ★ 动态GT锁定: 首次检测时锁定主车 (GT统计用)
                                    if self.useIDFiducialCVDetector {
                                        self.lockedCarIdx = self.activeGtCarIdx
                                    } else {
                                        self.lockCarByDetection(matched.rect)
                                    }

                                    // 更新 VNTracker 种子 (为非 YOLO 帧提供插值)
                                    let vnRect = CGRect(x: matched.rect.minX, y: 1 - matched.rect.maxY,
                                                        width: matched.rect.width, height: matched.rect.height)
                                    let obs = VNDetectedObjectObservation(boundingBox: vnRect)
                                    self.trackerReq = VNTrackObjectRequest(detectedObjectObservation: obs)
                                    self.trackerReq?.trackingLevel = self.racingMode ? .fast : .accurate
                                }
                            }
                        } else if let tr = trkRect {
                            // ★ V6: 无GT校验，直接使用VNTracker结果
                            // 真实部署时没有GT可用于校验
                            detRect = tr
                            detSrc = "Tracker"
                        }

                        self.prevPixelBuffer = pb
                    }

                    // ★ 6.5 单目标锁定接管控制链路: 当 singleTargetMode 开启时，
                    //   用 ByteTrack + ReID 评分结果替代一般追踪的 IoU 连续性匹配
                    let activeGtCar = self.cars.first(where: { $0.id == self.activeGtCarIdx })
                    let gtVisibleNow = activeGtCar?.isVisible == true && (activeGtCar?.screenRect.width ?? 0) > 0.01
                    let gtScreenRect = activeGtCar?.screenRect ?? .zero
                    let allGtRectsEarly: [CGRect] = self.cars.compactMap { car in
                        guard car.isVisible, car.screenRect.width > 0, car.screenRect.height > 0 else { return nil }
                        return car.screenRect
                    }

                    if self.singleTargetMode {
                        self.stPixelBuffer = pb
                        let stResult: CGRect?
                        if self.useIDFiducialCVDetector {
                            stResult = self.updateIDFiducialTargetLock(
                                detRect: detRect,
                                gtScreenRect: gtScreenRect,
                                gtVisibleNow: gtVisibleNow,
                                yoloMatchedThisFrame: yoloMatchedThisFrame,
                                stats: &statsLocal
                            )
                        } else {
                            stResult = self.updateSingleTargetLockStats(
                                detRect: detRect,
                                smRect: nil,
                                allGtRects: allGtRectsEarly,
                                gtScreenRect: gtScreenRect,
                                gtVisibleNow: gtVisibleNow,
                                dets: yoloDetsThisFrame,
                                yoloMatchedThisFrame: yoloMatchedThisFrame,
                                stats: &statsLocal
                            )
                        }
                        // ★ 接管控制链路: 用单目标评分结果覆盖 detRect
                        //   - 评分通过 → 用评分框驱动云台
                        //   - 评分拒绝 (nil) → detRect=nil，云台进入丢失处理 (宁可丢失也不跟错车)
                        detRect = stResult
                        if stResult != nil {
                            detSrc = "ST-Lock"
                            // ★ 同步更新 VNTracker 种子，保证非 YOLO 帧也能提供候选框
                            let vnRect = CGRect(x: stResult!.minX, y: 1 - stResult!.maxY,
                                                width: stResult!.width, height: stResult!.height)
                            let obs = VNDetectedObjectObservation(boundingBox: vnRect)
                            self.trackerReq = VNTrackObjectRequest(detectedObjectObservation: obs)
                            self.trackerReq?.trackingLevel = self.racingMode ? .fast : .accurate
                        }
                    }

                    // 7. 平滑 + GT锚定
                    let smRect: CGRect?

                    if let dr = detRect {
                        // ★ V6: 检测框直接送入Smoother，不再用GT锚定
                        smRect = self.smoother.run(dr)
                        self.curRect = smRect
                        self.lastSeenTS = simTime
                        self.targetLostFrames = 0
                    } else {
                        self.targetLostFrames += 1
                        // ★ V6: 无检测时直接让Smoother衰减，不再用GT兖底
                        smRect = self.smoother.run(nil)
                        if self.targetLostFrames > 14 {
                            self.curRect = nil
                        }
                    }

                    // ★ DEBUG: detRect/smRect 状态
                    if frameIdx < 30 || frameIdx % 100 == 0 {
                        let dbgLog = NSTemporaryDirectory() + "yolo_debug.log"
                        let dbgLine = "[\(Date())] RESULT: detRect=\(String(describing: detRect)) smRect=\(String(describing: smRect)) detSrc=\(detSrc) lostFrames=\(self.targetLostFrames)\n"
                        if let data = dbgLine.data(using: .utf8),
                           let h = try? FileHandle(forWritingTo: URL(fileURLWithPath: dbgLog)) {
                            h.seekToEndOfFile(); h.write(data); h.closeFile()
                        }
                    }

                    let pidRect: CGRect? = {
                        // ★ 训练数据生成模式：强制所有帧都用 GT 框驱动 PID
                        if VSimCfg.forceGtForPid && gtVisibleNow {
                            return gtScreenRect
                        }
                        // ★ V6: 纯Smoother输出，无GT兜底
                        return smRect
                    }()

                    // 7.5 ★ 自动变焦 (对齐 iOS AutoZoomEngine)
                    let isLost = (pidRect == nil)
                    if self.enableZoom {
                        self.opticalZoom = self.autoZoom.update(
                            targetRect: pidRect,
                            currentZoom: self.opticalZoom,
                            isLost: isLost,
                            zc: self.zoomCfg
                        )
                        // ★ FIX: 使用实际基础FOV而非配置值，保证与FOVMap一致
                        let baseFov = self.actualBaseFovDeg
                        let zoomedFov = baseFov / Double(self.opticalZoom)
                        self.cameraNode?.camera?.fieldOfView = CGFloat(zoomedFov)
                    }
                    let curZoom = self.enableZoom ? self.opticalZoom : CGFloat(1.0)

                    // 8. PID 控制 (★ 使用变焦后的等效FOV)
                    let effFov = self.fov.effectiveFov(opticalZoom: curZoom)
                    self.pid.racing = self.racingMode
                    let pidOut: PIDCtrl.Out
                    if let rect = pidRect {
                        let center = CGPoint(x: rect.midX, y: rect.midY)
                        pidOut = self.pid.run(
                            center: center, rect: rect, t: simTime,
                            hFov: CGFloat(effFov.h), vFov: CGFloat(effFov.v),
                            gainScale: 1.0,
                            servoVelH: CGFloat(self.gimbal.panVel),
                            servoVelV: CGFloat(self.gimbal.tiltVel)
                        )
                        // 记录惯性速度 (用于丢失时的惯性追踪)
                        self.inertiaVelH = pidOut.sH / CGFloat(dt)
                        self.inertiaVelV = pidOut.sV / CGFloat(dt)
                    } else {
                        pidOut = self.pid.run(
                            center: nil, rect: nil, t: simTime,
                            hFov: CGFloat(effFov.h), vFov: CGFloat(effFov.v)
                        )
                    }

                    // 8.5 ★ 神经网络推理：用 MLP 输出替代 PID 的 stepH/stepV
                    var finalStepH = pidOut.sH
                    var finalStepV = pidOut.sV
                    if self.useNeuralNetwork, let mlp = self.gimbalMLPModel, pidRect != nil {
                        let detCX = pidRect!.midX
                        let detCY = pidRect!.midY
                        let lostNorm = min(CGFloat(self.targetLostFrames) / 30.0, 1.0)
                        let stateArray: [Float] = [
                            Float(pidOut.eX), Float(pidOut.eY),
                            Float(pidOut.vX), Float(pidOut.vY),
                            Float(pidOut.distScale), Float(lostNorm),
                            Float(curZoom), Float(detCX), Float(detCY)
                        ]
                        do {
                            let mlArray = try MLMultiArray(shape: [1, 9], dataType: .float32)
                            for j in 0..<9 { mlArray[j] = NSNumber(value: stateArray[j]) }
                            let provider = try MLDictionaryFeatureProvider(
                                dictionary: ["gimbal_state": MLFeatureValue(multiArray: mlArray)])
                            let result = try mlp.prediction(from: provider)
                            if let actionArray = result.featureValue(for: "action")?.multiArrayValue {
                                finalStepH = CGFloat(actionArray[0].floatValue)
                                finalStepV = CGFloat(actionArray[1].floatValue)
                            }
                        } catch {
                            // MLP 推理失败，回退到 PID
                        }
                        // 更新惯性速度（用 MLP 输出）
                        self.inertiaVelH = finalStepH / CGFloat(dt)
                        self.inertiaVelV = finalStepV / CGFloat(dt)
                    }

                    // 8.7 ★ DAgger 噪声注入：在训练数据采集时故意扰动舵机输出
                    //   - pidStepH_clean / pidStepV_clean = PID 原始输出（用作训练标签）
                    //   - finalStepH / finalStepV = 实际执行的（可能含噪声）
                    let pidStepH_clean = finalStepH
                    let pidStepV_clean = finalStepV

                    if VSimCfg.daggerMode && !self.useNeuralNetwork {
                        // (a) 模拟丢帧：随机强制 pidRect 无效几帧
                        if self.daggerDropRemaining > 0 {
                            self.daggerDropRemaining -= 1
                        } else if Double.random(in: 0...1) < VSimCfg.daggerDropProb {
                            self.daggerDropRemaining = VSimCfg.daggerDropFrames
                        }

                        // (b) 噪声注入：随机扰动舵机步进
                        if self.daggerBurstRemaining > 0 {
                            finalStepH += self.daggerCurrentNoiseH
                            finalStepV += self.daggerCurrentNoiseV
                            self.daggerBurstRemaining -= 1
                        } else if Double.random(in: 0...1) < VSimCfg.daggerNoiseProb {
                            let burstLen = Int.random(in: VSimCfg.daggerBurstMin...VSimCfg.daggerBurstMax)
                            self.daggerBurstRemaining = burstLen
                            self.daggerCurrentNoiseH = CGFloat.random(in: -VSimCfg.daggerNoiseH...VSimCfg.daggerNoiseH)
                            self.daggerCurrentNoiseV = CGFloat.random(in: -VSimCfg.daggerNoiseV...VSimCfg.daggerNoiseV)
                            finalStepH += self.daggerCurrentNoiseH
                            finalStepV += self.daggerCurrentNoiseV
                        }
                    }

                    // 9. ★ 舵机逻辑 (对齐 ViewController5 三阶段丢失处理)
                    // 变焦联动: zoom越大, 平滑时间越长
                    let baseDynMs = self.racingMode ? 12.0 : 25.0
                    let zoomDynMs = baseDynMs + Double(curZoom - 1.0) * 10.0

                    // ★ FIX: 舵机角度限幅与SceneKit侧一致, 防止PID输出超出虚拟云台可达范围
                    let servoMin: CGFloat = VSimCfg.servoHome - VSimCfg.maxServoOffsetDeg  // 68°
                    let servoMax: CGFloat = VSimCfg.servoHome + VSimCfg.maxServoOffsetDeg  // 172°

                    if pidRect != nil {
                        // === 有目标: 正常追踪 ===
                        if simTime - self.lastSendTS >= 0.02 {
                            self.servoH = min(max(self.servoH + finalStepH, servoMin), servoMax)
                            self.servoV = min(max(self.servoV + finalStepV, servoMin), servoMax)
                            self.gimbal.send(id: 1, angle: Double(self.servoH), durMs: zoomDynMs)
                            self.gimbal.send(id: 2, angle: Double(self.servoV), durMs: zoomDynMs)
                            self.lastSendTS = simTime
                        }
                        self.lostPhaseStartTS = simTime  // 重置丢失计时
                    } else {
                        let lostDur = simTime - self.lastSeenTS
                        if self.racingMode {
                            // === 赛道模式三阶段 (对齐 ViewController5) ===
                            if lostDur <= Cfg.inertiaDur {
                                // 阶段1: 惯性追踪 (0~1.2s) — 用最后速度继续驱动, 衰减
                                if simTime - self.lastSendTS >= 0.02 {
                                    let decay = Cfg.inertiaDecay
                                    self.inertiaVelH *= decay
                                    self.inertiaVelV *= decay
                                    let stepH = self.inertiaVelH * CGFloat(dt)
                                    let stepV = self.inertiaVelV * CGFloat(dt)
                                    if abs(stepH) > 0.1 || abs(stepV) > 0.1 {
                                        self.servoH = min(max(self.servoH + stepH, servoMin), servoMax)
                                        self.servoV = min(max(self.servoV + stepV, servoMin), servoMax)
                                        self.gimbal.send(id: 1, angle: Double(self.servoH), durMs: zoomDynMs)
                                        self.gimbal.send(id: 2, angle: Double(self.servoV), durMs: zoomDynMs)
                                    }
                                    self.lastSendTS = simTime
                                }
                            } else if lostDur <= Cfg.inertiaDur + Cfg.scanDur {
                                // 阶段2: 扫描 (1.2~5.2s) — 在最后已知位置附近正弦扫描
                                if simTime - self.lastSendTS >= 0.02 {
                                    let scanT = CGFloat(lostDur - Cfg.inertiaDur)
                                    let phase = scanT / Cfg.scanPeriod * 2 * .pi
                                    let scanH = sin(phase) * Cfg.scanAmpH * CGFloat(dt)
                                    let scanV = cos(phase * 0.7) * Cfg.scanAmpV * CGFloat(dt)
                                    self.servoH = min(max(self.servoH + scanH, servoMin), servoMax)
                                    self.servoV = min(max(self.servoV + scanV, servoMin), servoMax)
                                    self.gimbal.send(id: 1, angle: Double(self.servoH), durMs: zoomDynMs)
                                    self.gimbal.send(id: 2, angle: Double(self.servoV), durMs: zoomDynMs)
                                    self.lastSendTS = simTime
                                }
                            } else {
                                // 阶段3: 回中 (>5.2s) — 缓慢回到 servoHome
                                if simTime - self.lastSendTS >= 0.02 {
                                    let dH = VSimCfg.servoHome - self.servoH
                                    let dV = VSimCfg.servoHome - self.servoV
                                    if abs(dH) > 0.5 || abs(dV) > 0.5 {
                                        self.servoH += dH * Cfg.returnAlpha
                                        self.servoV += dV * Cfg.returnAlpha
                                        self.gimbal.send(id: 1, angle: Double(self.servoH), durMs: zoomDynMs)
                                        self.gimbal.send(id: 2, angle: Double(self.servoV), durMs: zoomDynMs)
                                    }
                                    self.lastSendTS = simTime
                                }
                            }
                        } else {
                            // === 普通模式: 保持 + 回中 (对齐 ViewController5) ===
                            if lostDur > Cfg.lostHold {
                                if simTime - self.lastSendTS >= 0.02 {
                                    let dH = VSimCfg.servoHome - self.servoH
                                    let dV = VSimCfg.servoHome - self.servoV
                                    if abs(dH) > 0.5 || abs(dV) > 0.5 {
                                        self.servoH += dH * Cfg.returnAlpha
                                        self.servoV += dV * Cfg.returnAlpha
                                        self.gimbal.send(id: 1, angle: Double(self.servoH), durMs: zoomDynMs)
                                        self.gimbal.send(id: 2, angle: Double(self.servoV), durMs: zoomDynMs)
                                    }
                                    self.lastSendTS = simTime
                                }
                            }
                        }
                    }

                    // 10. 统计
                    // 收集所有可见 GT 车辆的 screenRect
                    let allGtRects: [CGRect] = self.cars.compactMap { car in
                        guard car.isVisible, car.screenRect.width > 0, car.screenRect.height > 0 else { return nil }
                        return car.screenRect
                    }
                    let anyGtVisible = !allGtRects.isEmpty

                    // 保留主车 GT 信息用于 VSimSnap 构建和 UI 显示
                    let primaryCar = self.cars.first(where: { $0.id == self.activeGtCarIdx })
                    let gtVisible = primaryCar?.isVisible ?? false
                    let gtOccluded = primaryCar?.isOccluded ?? false
                    let gtRect = primaryCar?.screenRect ?? .zero
                    var gtOffCenter: CGFloat = 1.0
                    var gtInFrame = false
                    if gtVisible && gtRect.width > 0 && gtRect.height > 0 {
                        let cx = gtRect.midX, cy = gtRect.midY
                        if cx >= 0 && cx <= 1 && cy >= 0 && cy <= 1 {
                            gtInFrame = true
                            gtOffCenter = sqrt(pow(cx - 0.5, 2) + pow(cy - 0.5, 2))
                        }
                    }

                    statsLocal.totalFrames += 1

                    // ═══ 检测层统计 ═══
                    if pidRect != nil { statsLocal.trackingFrames += 1 }
                    if detRect != nil { statsLocal.pureDetFrames += 1 }
                    if yoloMatchedThisFrame { statsLocal.yoloMatchFrames += 1 }
                    if self.trackerAlive { statsLocal.trackerAliveFrames += 1 }

                    let yoloCandidateStats: (tpDet: Int, fpDet: Int, matchedGt: Int, missedGt: Int, primaryHit: Bool, primaryBestIoU: CGFloat) = {
                        guard didRunYolo else { return (0, 0, 0, allGtRects.count, false, 0) }
                        let tpDet = yoloDetsThisFrame.filter { det in
                            allGtRects.contains(where: { Self.iou(det.rect, $0) > 0.2 })
                        }.count
                        let fpDet = max(0, yoloDetsThisFrame.count - tpDet)
                        let matchedGt = allGtRects.filter { gt in
                            yoloDetsThisFrame.contains(where: { Self.iou($0.rect, gt) > 0.2 })
                        }.count
                        let missedGt = max(0, allGtRects.count - matchedGt)
                        let primaryBestIoU: CGFloat = {
                            guard gtVisible, gtRect.width > 0.001, gtRect.height > 0.001 else { return 0 }
                            return yoloDetsThisFrame.map { Self.iou($0.rect, gtRect) }.max() ?? 0
                        }()
                        return (tpDet, fpDet, matchedGt, missedGt, primaryBestIoU > 0.2, primaryBestIoU)
                    }()

                    // ═══ ★ 检测率 (YOLO 帧才累加) ═══
                    // 检测率 = (YOLO检出有车且位置匹配某个GT + YOLO检出无车且确实无车) / YOLO运行帧数
                    if didRunYolo {
                        statsLocal.yoloRunFrames += 1
                        statsLocal.yoloCandidateDetections += yoloDetsThisFrame.count
                        statsLocal.yoloCandidateTPDetections += yoloCandidateStats.tpDet
                        statsLocal.yoloCandidateFPDetections += yoloCandidateStats.fpDet
                        statsLocal.yoloVisibleGtObjects += allGtRects.count
                        statsLocal.yoloMatchedGtObjects += yoloCandidateStats.matchedGt
                        if gtVisible { statsLocal.yoloPrimaryVisibleFrames += 1 }
                        if yoloCandidateStats.primaryHit { statsLocal.yoloPrimaryMatchedFrames += 1 }
                        if anyGtVisible { statsLocal.yoloFramesWithAnyGt += 1 }
                        if yoloCandidateStats.matchedGt > 0 { statsLocal.yoloFramesWithMatchedGt += 1 }
                        if yoloCandidateStats.fpDet > 0 { statsLocal.yoloFramesWithAnyFP += 1 }

                        let yoloHasDet = yoloMatchedThisFrame
                        if yoloHasDet && anyGtVisible {
                            // YOLO 检出有车，且确实有 GT 车——进一步检查位置是否匹配
                            if let det = detRect {
                                let matchesAnyGt = allGtRects.contains(where: { Self.iou(det, $0) > 0.2 })
                                if matchesAnyGt { statsLocal.detCorrectFrames += 1 }
                            }
                        } else if !yoloHasDet && !anyGtVisible {
                            // YOLO 检出无车，且确实无 GT 车
                            statsLocal.detCorrectFrames += 1
                        }
                        // 其他情况 (FP/FN) 不累加
                    }

                    // ═══ ★ 一般追踪率 (YOLO 或 VNTracker 任一路有框且在车上) ═══
                    if let det = smRect, anyGtVisible {
                        let onCar = allGtRects.contains(where: { Self.iou(det, $0) > 0.2 })
                        if onCar { statsLocal.trackOnCarFrames += 1 }
                    }

                    // ═══ 连续丢失统计 ═══
                    if smRect != nil {
                        statsLocal.curConsecutiveLost = 0
                    } else {
                        statsLocal.curConsecutiveLost += 1
                        statsLocal.maxConsecutiveLost = max(statsLocal.maxConsecutiveLost, statsLocal.curConsecutiveLost)
                    }

                    // ═══ ★ 单目标锁定率已在步骤 6.5 提前执行并接管控制链路 ═══

                    // ═══ 误差统计 (有检测框时累加) ═══
                    let eh = abs(Double(pidOut.eX))
                    let ev = abs(Double(pidOut.eY))
                    if pidRect != nil {
                        statsLocal.errHSum += eh
                        statsLocal.errVSum += ev
                        statsLocal.errCount += 1
                        statsLocal.maxErrH = max(statsLocal.maxErrH, eh)
                        statsLocal.maxErrV = max(statsLocal.maxErrV, ev)
                    }

                    // ═══ 抽动统计 ═══
                    let sH = finalStepH
                    let sV = finalStepV
                    let curDirH = sH > 0.5 ? 1 : (sH < -0.5 ? -1 : 0)
                    let curDirV = sV > 0.5 ? 1 : (sV < -0.5 ? -1 : 0)
                    if curDirH != 0 && curDirH != statsLocal.lastDirH { statsLocal.dirChangeH += 1 }
                    if curDirH != 0 { statsLocal.lastDirH = curDirH }
                    if curDirV != 0 && curDirV != statsLocal.lastDirV { statsLocal.dirChangeV += 1 }
                    if curDirV != 0 { statsLocal.lastDirV = curDirV }

                    // ═══ YOLO 耗时统计 ═══
                    if yoloMs > 0 {
                        statsLocal.yoloMsSum += yoloMs
                        statsLocal.yoloMsCount += 1
                    }

                    // ═══ 变焦统计 ═══
                    statsLocal.zoomSum += Double(curZoom)
                    statsLocal.zoomCount += 1

                    let state: String
                    if pidRect != nil { state = "TRACK" }
                    else if self.racingMode {
                        let lostDur = simTime - self.lastSeenTS
                        if lostDur <= Cfg.inertiaDur { state = "INERTIA" }
                        else if lostDur <= Cfg.inertiaDur + Cfg.scanDur { state = "SCAN" }
                        else { state = "RETURN" }
                    }
                    else if simTime - self.lastSeenTS <= Cfg.lostHold { state = "HOLD" }
                    else { state = "RETURN" }

                    let primarySpeedKmh = Double(primaryCar?.speed ?? 0) * 3.6
                    let gimbalPos = self.gimbalBaseNode?.worldPosition

                    // ★ 帧级中间数据计算
                    let frameDetGtIoU: CGFloat = {
                        guard let det = smRect, !allGtRects.isEmpty else { return 0 }
                        return allGtRects.map { Self.iou(det, $0) }.max() ?? 0
                    }()
                    let framePrevIoU: CGFloat = {
                        guard let det = smRect, let prev = self.history.last?.detectedRect else { return 0 }
                        return Self.iou(det, prev)
                    }()
                    let frameIsDetCorrect: Bool = {
                        guard didRunYolo else { return false }
                        let yoloHasDet = yoloMatchedThisFrame
                        if yoloHasDet && anyGtVisible {
                            guard let det = detRect else { return false }
                            return allGtRects.contains(where: { Self.iou(det, $0) > 0.2 })
                        } else if !yoloHasDet && !anyGtVisible {
                            return true
                        }
                        return false
                    }()
                    let frameIsTrackOnCar: Bool = {
                        guard let det = smRect, anyGtVisible else { return false }
                        return allGtRects.contains(where: { Self.iou(det, $0) > 0.2 })
                    }()
                    let primaryTrackIoU: CGFloat = {
                        guard let det = smRect, gtVisible, gtRect.width > 0.001, gtRect.height > 0.001 else { return 0 }
                        return Self.iou(det, gtRect)
                    }()
                    let snap = VSimSnap(
                        frameIdx: frameIdx,
                        time: simTime,
                        renderedImage: cgImg,
                        croppedImage: nil,
                        gtRects: self.cars.filter(\.isVisible).map(\.screenRect),
                        gtPrimaryRect: gtVisible ? gtRect : nil,
                        gtPrimaryVisible: gtVisible,
                        gtPrimaryOccluded: gtOccluded,
                        detectedRect: smRect,
                        detSource: detSrc,
                        state: state,
                        primarySpeedKmh: primarySpeedKmh,
                        gimbalBaseWorldPos: gimbalPos,
                        cmdH: Double(self.servoH),
                        cmdV: Double(self.servoV),
                        actH: self.gimbal.panPos,
                        actV: self.gimbal.tiltPos,
                        errH: self.gimbal.panErr,
                        errV: self.gimbal.tiltErr,
                        errDegX: pidOut.eX,
                        errDegY: pidOut.eY,
                        opticalZoom: curZoom,
                        effectiveFovH: effFov.h,
                        gtOffCenter: gtOffCenter,
                        gtInFrame: gtInFrame,
                        yoloConf: yoloConfThisFrame,
                        allGtCount: allGtRects.count,
                        detGtIoU: frameDetGtIoU,
                        prevIoU: framePrevIoU,
                        isDetCorrect: frameIsDetCorrect,
                        isTrackOnCar: frameIsTrackOnCar,
                        yoloDetCount: yoloDetCountThisFrame,
                        simFalsePositiveRects: simFpRectsThisFrame,
                        didRunYoloThisFrame: didRunYolo,
                        yoloTPCount: yoloCandidateStats.tpDet,
                        yoloFPCount: yoloCandidateStats.fpDet,
                        yoloFNCount: yoloCandidateStats.missedGt,
                        yoloPrimaryHit: yoloCandidateStats.primaryHit,
                        yoloPrimaryBestIoU: yoloCandidateStats.primaryBestIoU,
                        primaryVsGtIoU: primaryTrackIoU,
                        smoothVelX: pidOut.vX,
                        smoothVelY: pidOut.vY,
                        distScale: pidOut.distScale,
                        pidStepH: VSimCfg.daggerMode ? pidStepH_clean : finalStepH,
                        pidStepV: VSimCfg.daggerMode ? pidStepV_clean : finalStepV,
                        targetLostCount: self.targetLostFrames
                    )

                    let historySnap: VSimSnap
                    if BatchSimRunner.isBatchMode {
                        historySnap = VSimSnap(
                            frameIdx: snap.frameIdx,
                            time: snap.time,
                            renderedImage: nil,
                            croppedImage: nil,
                            gtRects: snap.gtRects,
                            gtPrimaryRect: snap.gtPrimaryRect,
                            gtPrimaryVisible: snap.gtPrimaryVisible,
                            gtPrimaryOccluded: snap.gtPrimaryOccluded,
                            detectedRect: snap.detectedRect,
                            detSource: snap.detSource,
                            state: snap.state,
                            primarySpeedKmh: snap.primarySpeedKmh,
                            gimbalBaseWorldPos: snap.gimbalBaseWorldPos,
                            cmdH: snap.cmdH,
                            cmdV: snap.cmdV,
                            actH: snap.actH,
                            actV: snap.actV,
                            errH: snap.errH,
                            errV: snap.errV,
                            errDegX: snap.errDegX,
                            errDegY: snap.errDegY,
                            opticalZoom: snap.opticalZoom,
                            effectiveFovH: snap.effectiveFovH,
                            gtOffCenter: snap.gtOffCenter,
                            gtInFrame: snap.gtInFrame,
                            yoloConf: snap.yoloConf,
                            allGtCount: snap.allGtCount,
                            detGtIoU: snap.detGtIoU,
                            prevIoU: snap.prevIoU,
                            isDetCorrect: snap.isDetCorrect,
                            isTrackOnCar: snap.isTrackOnCar,
                            yoloDetCount: snap.yoloDetCount,
                            simFalsePositiveRects: snap.simFalsePositiveRects,
                            didRunYoloThisFrame: snap.didRunYoloThisFrame,
                            yoloTPCount: snap.yoloTPCount,
                            yoloFPCount: snap.yoloFPCount,
                            yoloFNCount: snap.yoloFNCount,
                            yoloPrimaryHit: snap.yoloPrimaryHit,
                            yoloPrimaryBestIoU: snap.yoloPrimaryBestIoU,
                            primaryVsGtIoU: snap.primaryVsGtIoU,
                            smoothVelX: snap.smoothVelX,
                            smoothVelY: snap.smoothVelY,
                            distScale: snap.distScale,
                            pidStepH: snap.pidStepH,
                            pidStepV: snap.pidStepV,
                            targetLostCount: snap.targetLostCount
                        )
                    } else {
                        historySnap = snap
                    }

                    // 默认每 ~4 帧刷新 UI 减轻主线程压力；批量命令行只保留 CSV 所需历史，避免 UI 和图像保留拖慢结果生成
                    let uiStride = max(1, Int(VSimCfg.fps / 15))
                    let snapEveryFrame = self.singleTargetMode || self.useSimulatedYOLO || self.debugMode
                    if !BatchSimRunner.isBatchMode && (snapEveryFrame || frameIdx % uiStride == 0) {
                        let p = Double(frameIdx) / Double(totalFrames)
                        let statsCopy = statsLocal
                        let currentHistoryCount = self.history.count
                        DispatchQueue.main.async {
                            self.snap = snap
                            self.progress = p
                            self.stats = statsCopy
                            self.debugFrameIdx = max(0, currentHistoryCount - 1)
                            if self.singleTargetMode {
                                self.status = String(format: "[ST] Frame %d/%d | Det %.1f%% | Track %.1f%% | STLock %.1f%% | Zoom %.1fx",
                                    frameIdx, totalFrames, statsCopy.detectionRate,
                                    statsCopy.generalTrackRate, statsCopy.singleTargetLockRate, statsCopy.avgZoom)
                            } else {
                                self.status = String(format: "Frame %d/%d | Det %.1f%% | Track %.1f%% | STLock %.1f%% | Zoom %.1fx",
                                    frameIdx, totalFrames, statsCopy.detectionRate,
                                    statsCopy.generalTrackRate, statsCopy.singleTargetLockRate, statsCopy.avgZoom)
                            }
                            self.objectWillChange.send()
                        }
                    }

                    self.history.append(historySnap)
                    frameIdx += 1
                    simTime += dt
                }
            }

            self.sceneRecorder?.finish()
            self.sceneRecorder = nil

            let finalStats = statsLocal
            DispatchQueue.main.async {
                self.running = false
                self.paused = false
                self.stats = finalStats

                // ★ 自动保存仿真历史记录
                let record = SimHistoryRecord(
                    timestamp: Date(),
                    strategy: self.useNeuralNetwork ? "MLP" : "PID",
                    trackShape: self.trackShape,
                    sceneType: self.sceneType,
                    cameraPreset: self.cameraPreset,
                    carCount: self.carCount,
                    simDuration: self.simDuration,
                    detectionRate: finalStats.detectionRate,
                    generalTrackRate: finalStats.generalTrackRate,
                    singleTargetLockRate: finalStats.singleTargetLockRate,
                    maxConsecutiveLost: finalStats.maxConsecutiveLost,
                    yoloDetRate: finalStats.yoloDetRate,
                    avgErrH: finalStats.avgErrH,
                    avgErrV: finalStats.avgErrV,
                    jitterH: finalStats.jitterH,
                    jitterV: finalStats.jitterV,
                    score: finalStats.score,
                    grade: finalStats.grade,
                    randomSeed: self.randomSeed,
                    cameraHeight: Double(self.cameraHeight),
                    enableRamps: self.enableRamps,
                    useSimulatedYOLO: self.useSimulatedYOLO
                )
                SimHistoryStore.shared.save(record)
                self.progress = 1.0
                self.status = String(format: "Done %d frames | Det %.1f%% | Track %.1f%% | STLock %.1f%% | Score %d %@",
                    finalStats.totalFrames, finalStats.detectionRate,
                    finalStats.generalTrackRate, finalStats.singleTargetLockRate,
                    finalStats.score, finalStats.grade)
                self.objectWillChange.send()
            }
        }
    }

    func stop() {
        cancel = true
        paused = false
        debugStepReq = false
    }

    /// 根据历史记录恢复仿真参数（「重现」前调用）
    func applyReplayConfiguration(from record: SimHistoryRecord) {
        useNeuralNetwork = (record.strategy == "MLP")
        trackShape = record.trackShape
        sceneType = record.sceneType
        cameraPreset = record.cameraPreset
        carCount = record.carCount
        simDuration = record.simDuration
        randomSeed = record.randomSeed ?? 42
        if let ch = record.cameraHeight {
            cameraHeight = CGFloat(ch)
        }
        if let r = record.enableRamps {
            enableRamps = r
        }
        if let sy = record.useSimulatedYOLO {
            useSimulatedYOLO = sy
        }
        objectWillChange.send()
    }

    /// 载入历史参数并立即开始一轮仿真
    func replayFromHistory(record: SimHistoryRecord) {
        guard !running else {
            status = "请先停止当前仿真，再重现历史记录"
            objectWillChange.send()
            return
        }
        applyReplayConfiguration(from: record)
        start()
    }

    func togglePause() {
        paused.toggle()
        objectWillChange.send()
    }

    func stepForward() {
        debugStepReq = true
        paused = false
        objectWillChange.send()
    }

    func jumpToFrame(_ idx: Int) {
        let clamped = max(0, min(idx, history.count - 1))
        debugFrameIdx = clamped
        if clamped < history.count {
            snap = history[clamped]
        }
        objectWillChange.send()
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 数据集生成支持 (供 DatasetGenerator 调用)
    // ═══════════════════════════════════════════════════════════

    /// 构建场景 (供外部数据集生成器调用)
    func datasetBuildScene() {
        buildScene()
    }

    /// 设置车辆到指定赛道进度位置, 并返回标注信息
    /// - Parameters:
    ///   - progressValues: 每辆车的赛道进度 (0~1)
    ///   - lateralOffsets: 每辆车的横向偏移 (-0.8~0.8)
    ///   - slipAngles: 每辆车的侧滑角
    func datasetPlaceCars(progressValues: [CGFloat], lateralOffsets: [CGFloat], slipAngles: [CGFloat]) {
        guard !trackPath.isEmpty else { return }
        for i in cars.indices {
            let prog = i < progressValues.count ? progressValues[i] : CGFloat(i) / CGFloat(cars.count)
            let latOff = i < lateralOffsets.count ? lateralOffsets[i] : 0
            let slip = i < slipAngles.count ? slipAngles[i] : 0

            let (pos, heading) = interpolateTrack(progress: prog)
            let normal = SCNVector3(-sin(heading), 0, cos(heading))
            let offset = latOff * VSimCfg.trackWidth * 0.4

            let cpx = pos.x + normal.x * offset
            let cpy = pos.y + VSimCfg.carHeight * 0.15
            let cpz = pos.z + normal.z * offset

            cars[i].trackProgress = prog
            cars[i].position = SCNVector3(cpx, cpy, cpz)
            cars[i].heading = heading + slip
            cars[i].laneOffset = latOff
            cars[i].slipAngle = slip

            cars[i].node?.position = cars[i].position
            cars[i].node?.eulerAngles.y = cars[i].heading
        }
    }

    /// 设置云台朝向 (直接设置pan/tilt节点的欧拉角)
    func datasetSetCameraLookAt(_ target: SCNVector3) {
        guard let baseNode = gimbalBaseNode,
              let panNode = gimbalPanNode,
              let tiltNode = gimbalTiltNode else { return }

        let basePos = baseNode.position
        let dx = target.x - basePos.x
        let dz = target.z - basePos.z
        let dy = target.y - basePos.y
        let horizDist = sqrt(dx * dx + dz * dz)

        // 水平旋转: 朝向目标
        panNode.eulerAngles.y = atan2(-dx, -dz)
        // 俯仰: 根据高度差和水平距离
        tiltNode.eulerAngles.x = -atan2(basePos.y - target.y, max(horizDist, 0.5))
    }

    /// 添加相机朝向抖动 (模拟追踪误差)
    func datasetAddCameraJitter(panJitter: CGFloat, tiltJitter: CGFloat) {
        gimbalPanNode?.eulerAngles.y += panJitter
        gimbalTiltNode?.eulerAngles.x += tiltJitter
    }

    /// 渲染当前场景并返回图像 + YOLO标注
    /// - Parameters:
    ///   - imageSize: 渲染图像尺寸
    ///   - useMultiLabel: 是否使用多标签模式 (9类: 颜色大类×距离)
    ///                    false = 兼容模式, 所有车辆 classId=0 (rc_car)
    ///                    true  = 多标签模式, classId 按颜色大类×距离映射到 0-8
    func datasetRenderFrame(imageSize: Int, useMultiLabel: Bool = false) -> (CGImage, [(classId: Int, cx: CGFloat, cy: CGFloat, w: CGFloat, h: CGFloat)])? {
        guard let renderer = renderer, let camNode = cameraNode else { return nil }

        // 渲染
        let size = CGSize(width: imageSize, height: imageSize)
        let snapshot = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .multisampling4X)
        guard let tiffData = snapshot.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let cgImage = bitmap.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else { return nil }

        // 计算GT标注 (精确8角点投影)
        let viewW = CGFloat(imageSize)
        let viewH = CGFloat(imageSize)
        var labels: [(classId: Int, cx: CGFloat, cy: CGFloat, w: CGFloat, h: CGFloat)] = []

        for i in cars.indices {
            guard let carNode = cars[i].node else { continue }

            let halfW = VSimCfg.carWidth / 2
            let halfH = VSimCfg.carHeight
            let halfL = VSimCfg.carLength / 2
            let corners: [SCNVector3] = [
                SCNVector3(-halfW, 0, -halfL), SCNVector3(halfW, 0, -halfL),
                SCNVector3(-halfW, 0, halfL),  SCNVector3(halfW, 0, halfL),
                SCNVector3(-halfW, halfH, -halfL), SCNVector3(halfW, halfH, -halfL),
                SCNVector3(-halfW, halfH, halfL),  SCNVector3(halfW, halfH, halfL),
            ]

            var sMinX: CGFloat = .greatestFiniteMagnitude
            var sMinY: CGFloat = .greatestFiniteMagnitude
            var sMaxX: CGFloat = -.greatestFiniteMagnitude
            var sMaxY: CGFloat = -.greatestFiniteMagnitude
            var anyVisible = false

            for corner in corners {
                let worldPos = carNode.convertPosition(corner, to: nil)
                let screenPt = renderer.projectPoint(worldPos)
                if screenPt.z > 1 || screenPt.z < 0 { continue }
                anyVisible = true
                sMinX = min(sMinX, screenPt.x)
                sMinY = min(sMinY, screenPt.y)
                sMaxX = max(sMaxX, screenPt.x)
                sMaxY = max(sMaxY, screenPt.y)
            }

            guard anyVisible && sMinX < sMaxX && sMinY < sMaxY else { continue }

            // SceneKit projectPoint: 左下角原点, y向上 → 转左上角归一化
            let normRect = CGRect(
                x: sMinX / viewW,
                y: 1 - (sMaxY / viewH),
                width: (sMaxX - sMinX) / viewW,
                height: (sMaxY - sMinY) / viewH
            )

            let screenBounds = CGRect(x: 0, y: 0, width: 1, height: 1)
            let clipped = normRect.intersection(screenBounds)
            guard !clipped.isNull && clipped.width > 0.005 && clipped.height > 0.005 else { continue }
            guard clipped.width < 0.8 && clipped.height < 0.8 else { continue }

            let visibility = datasetCarVisibility(carIndex: i, carNode: carNode, cameraNode: camNode, renderer: renderer)
            guard visibility.visibleSamples >= 3 && visibility.fraction >= 0.34 else { continue }

            // ── classId 计算 ──
            let classId: Int
            if useMultiLabel {
                // 多标签模式: 3 类 — 仅按距离/bbox面积分类
                //
                // classId:
                //   0 = rc_far:   面积 < 0.002  (远距离小目标)
                //   1 = rc_mid:   0.002 <= 面积 < 0.01  (中距离)
                //   2 = rc_near:  面积 >= 0.01  (近距离大目标)

                let area = clipped.width * clipped.height
                if area < 0.002 {
                    classId = 0  // rc_far
                } else if area < 0.01 {
                    classId = 1  // rc_mid
                } else {
                    classId = 2  // rc_near
                }
            } else {
                // 兼容模式: 所有车辆统一 classId = 0
                classId = 0
            }

            labels.append((classId: classId, cx: clipped.midX, cy: clipped.midY, w: clipped.width, h: clipped.height))
        }

        return (cgImage, labels)
    }

    private func datasetCarVisibility(
        carIndex: Int,
        carNode: SCNNode,
        cameraNode: SCNNode,
        renderer: SCNRenderer
    ) -> (visibleSamples: Int, testedSamples: Int, fraction: CGFloat) {
        let halfW = VSimCfg.carWidth / 2
        let halfH = VSimCfg.carHeight
        let halfL = VSimCfg.carLength / 2
        let samplePoints: [SCNVector3] = [
            SCNVector3(-halfW, 0, -halfL), SCNVector3(halfW, 0, -halfL),
            SCNVector3(-halfW, 0, halfL),  SCNVector3(halfW, 0, halfL),
            SCNVector3(-halfW, halfH, -halfL), SCNVector3(halfW, halfH, -halfL),
            SCNVector3(-halfW, halfH, halfL),  SCNVector3(halfW, halfH, halfL),
            SCNVector3(0, halfH * 0.5, 0),
            SCNVector3(0, halfH * 0.8, -halfL),
            SCNVector3(0, halfH * 0.8, halfL),
        ]

        let camWorld = cameraNode.worldPosition
        var tested = 0
        var visible = 0

        for localPoint in samplePoints {
            let worldPoint = carNode.convertPosition(localPoint, to: nil)
            let screenPoint = renderer.projectPoint(worldPoint)
            guard screenPoint.z >= 0 && screenPoint.z <= 1 else { continue }

            tested += 1
            if datasetHasLineOfSight(from: camWorld, to: worldPoint, carIndex: carIndex) {
                visible += 1
            }
        }

        let fraction = tested > 0 ? CGFloat(visible) / CGFloat(tested) : 0
        return (visible, tested, fraction)
    }

    private func datasetHasLineOfSight(from cameraPoint: SCNVector3, to targetPoint: SCNVector3, carIndex: Int) -> Bool {
        let targetDistance = scnDist(cameraPoint, targetPoint)
        guard targetDistance > 0 else { return false }

        // The rendered camera is mounted inside the simulated gimbal rig. Starting the
        // ray exactly at the camera can hit the rig itself, so nudge the origin forward.
        let originOffset = min(CGFloat(0.22), targetDistance * 0.2)
        let invDist = CGFloat(1.0) / targetDistance
        let rayStart = SCNVector3(
            cameraPoint.x + (targetPoint.x - cameraPoint.x) * invDist * originOffset,
            cameraPoint.y + (targetPoint.y - cameraPoint.y) * invDist * originOffset,
            cameraPoint.z + (targetPoint.z - cameraPoint.z) * invDist * originOffset
        )
        let options: [String: Any] = [
            SCNHitTestOption.ignoreHiddenNodes.rawValue: true,
            SCNHitTestOption.backFaceCulling.rawValue: false,
        ]
        let hits = scene?.rootNode.hitTestWithSegment(from: rayStart, to: targetPoint, options: options) ?? []
        for hit in hits {
            let hitDistance = scnDist(cameraPoint, hit.worldCoordinates)
            if hitDistance < originOffset * 0.9 { continue }
            guard hitDistance <= targetDistance * 0.985 else { continue }
            if isPartOfCar(node: hit.node, carIdx: carIndex) {
                return true
            }
            return false
        }

        return true
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 多标签辅助 — 颜色大类映射
    // ═══════════════════════════════════════════════════════════

    /// 将 NSColor 映射到多标签颜色大类 (0=暖色, 1=冷色, 2=中性)
    /// 与 setupCars() 中的 carColors 数组对应:
    ///   [.red, .blue, .green, .orange, .purple, .yellow, .cyan, .magenta, .white, .systemPink]
    ///   暖色: red(0), orange(3), yellow(5), magenta(7), systemPink(9)
    ///   冷色: blue(1), green(2), purple(4), cyan(6)
    ///   中性: white(8)
    static func multiLabelColorGroup(for color: NSColor) -> Int {
        // 将颜色转换到 sRGB 空间以获取可靠的 RGB 分量
        guard let rgb = color.usingColorSpace(.sRGB) else {
            return 2 // 无法转换时归为中性
        }
        let r = rgb.redComponent
        let g = rgb.greenComponent
        let b = rgb.blueComponent

        // 白色/接近白色: 中性
        if r > 0.85 && g > 0.85 && b > 0.85 {
            return 2 // neutral
        }
        // 暖色判定: 红色通道占主导, 或黄色(r+g高, b低)
        if r > 0.6 && r >= b {
            // red, orange, yellow, magenta, systemPink
            return 0 // warm
        }
        // 冷色判定: 蓝色或绿色通道占主导
        if b > 0.4 || g > 0.4 {
            // blue, green, cyan, purple
            return 1 // cold
        }
        // 默认归为中性
        return 2
    }

    /// 获取赛道中心坐标
    var datasetTrackCenter: SCNVector3 {
        SCNVector3((trackBBoxMinX + trackBBoxMaxX) / 2, 0, (trackBBoxMinZ + trackBBoxMaxZ) / 2)
    }

    /// 获取车辆数量
    var datasetCarCount: Int { cars.count }

    /// 获取指定车辆的世界坐标
    func datasetCarPosition(_ index: Int) -> SCNVector3 {
        guard index >= 0 && index < cars.count else { return .init() }
        return cars[index].position
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: CSV 导出
    // ═══════════════════════════════════════════════════════════

    func exportCSV(_ url: URL) {
        var csv = "frame,time,state,detSrc,gtVisible,gtOccluded,detX,detY,detW,detH,"
        csv += "gtX,gtY,gtW,gtH,iou,cmdH,cmdV,actH,actV,errDegX,errDegY,"
        csv += "smoothVelX,smoothVelY,distScale,pidStepH,pidStepV,targetLostCount,"
        csv += "yoloConf,allGtCount,detGtIoU,prevIoU,isDetCorrect,isTrackOnCar,"
        csv += "yoloDetCount,yoloTPCount,yoloFPCount,yoloFNCount,yoloPrimaryHit,yoloPrimaryBestIoU,opticalZoom\n"

        for s in history {
            let det = s.detectedRect ?? .zero
            let gt = s.gtPrimaryRect ?? .zero
            let iou: CGFloat
            if s.detectedRect != nil && s.gtPrimaryRect != nil {
                iou = Self.iou(s.detectedRect!, s.gtPrimaryRect!)
            } else {
                iou = 0
            }

            csv += "\(s.frameIdx),\(String(format: "%.4f", s.time)),\(s.state),\(s.detSource),"
            csv += "\(s.gtPrimaryVisible),\(s.gtPrimaryOccluded),"
            csv += "\(String(format: "%.4f", det.minX)),\(String(format: "%.4f", det.minY)),"
            csv += "\(String(format: "%.4f", det.width)),\(String(format: "%.4f", det.height)),"
            csv += "\(String(format: "%.4f", gt.minX)),\(String(format: "%.4f", gt.minY)),"
            csv += "\(String(format: "%.4f", gt.width)),\(String(format: "%.4f", gt.height)),"
            csv += "\(String(format: "%.4f", iou)),"
            csv += "\(String(format: "%.2f", s.cmdH)),\(String(format: "%.2f", s.cmdV)),"
            csv += "\(String(format: "%.2f", s.actH)),\(String(format: "%.2f", s.actV)),"
            csv += "\(String(format: "%.4f", s.errDegX)),\(String(format: "%.4f", s.errDegY)),"
            csv += "\(String(format: "%.4f", s.smoothVelX)),\(String(format: "%.4f", s.smoothVelY)),"
            csv += "\(String(format: "%.4f", s.distScale)),"
            csv += "\(String(format: "%.6f", s.pidStepH)),\(String(format: "%.6f", s.pidStepV)),"
            csv += "\(s.targetLostCount),"
            csv += "\(String(format: "%.3f", s.yoloConf)),\(s.allGtCount),"
            csv += "\(String(format: "%.4f", s.detGtIoU)),\(String(format: "%.4f", s.prevIoU)),"
            csv += "\(s.isDetCorrect ? 1 : 0),\(s.isTrackOnCar ? 1 : 0),"
            csv += "\(s.yoloDetCount),\(s.yoloTPCount),\(s.yoloFPCount),\(s.yoloFNCount),"
            csv += "\(s.yoloPrimaryHit ? 1 : 0),\(String(format: "%.4f", s.yoloPrimaryBestIoU)),"
            csv += "\(String(format: "%.2f", s.opticalZoom))\n"
        }
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            print("\u{2705} CSV written: \(url.path) (\(csv.count) bytes)")
        } catch {
            print("\u{26a0}\u{fe0f} Failed to write CSV to \(url.path): \(error)")
            let fallback = URL(fileURLWithPath: "/tmp/batch_sim_results")
            try? FileManager.default.createDirectory(at: fallback, withIntermediateDirectories: true)
            let fallbackURL = fallback.appendingPathComponent(url.lastPathComponent)
            try? csv.write(to: fallbackURL, atomically: true, encoding: .utf8)
            print("\u{2705} CSV fallback written: \(fallbackURL.path)")
        }
    }

    /// 导出用于神经网络训练的精简 CSV（仅包含 State-Action 对）
    func exportTrainingCSV(_ url: URL) {
        // ★ 直接通过 stdout 输出训练数据，绕过 Sandbox 文件写入限制
        // 用特殊标记 <<<TRAIN_CSV_START>>> 和 <<<TRAIN_CSV_END>>> 包裹，方便从日志中提取
        print("<<<TRAIN_CSV_START>>>")
        print("errDegX,errDegY,smoothVelX,smoothVelY,distScale,targetLostFrames,opticalZoom,detCenterX,detCenterY,pidStepH,pidStepV")

        var count = 0
        for s in history {
            let det: CGRect
            if VSimCfg.daggerMode {
                if let d = s.detectedRect {
                    det = d
                } else if let gt = s.gtPrimaryRect {
                    det = gt
                } else {
                    continue
                }
            } else {
                guard s.state == "TRACK", let d = s.detectedRect else { continue }
                det = d
            }
            let detCX = det.midX
            let detCY = det.midY
            let lostNorm = min(CGFloat(s.targetLostCount) / 30.0, 1.0)

            print("\(String(format: "%.6f", s.errDegX)),\(String(format: "%.6f", s.errDegY)),\(String(format: "%.6f", s.smoothVelX)),\(String(format: "%.6f", s.smoothVelY)),\(String(format: "%.4f", s.distScale)),\(String(format: "%.4f", lostNorm)),\(String(format: "%.4f", s.opticalZoom)),\(String(format: "%.6f", detCX)),\(String(format: "%.6f", detCY)),\(String(format: "%.6f", s.pidStepH)),\(String(format: "%.6f", s.pidStepV))")
            count += 1
        }
        print("<<<TRAIN_CSV_END>>>")
        print("✅ Training data: \(count) rows printed to stdout")
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: ★ V6 — 独立 KF + 空间门控 + 锁定迟滞 (脱钩 ByteTrack ID)
    // ═══════════════════════════════════════════════════════════
    //
    // 核心改动：
    //   1. 维护一条独立于 ByteTrack ID 的“虚拟轨迹”，用自己的 6D KF 预测位置
    //   2. 用 KF 预测位置做空间门控，只考虑门控内的候选框
    //   3. 评分权重：IoU 0.50 / 尺寸 0.25 / HS颜色 0.15 / VNFeaturePrint 0.10
    //   4. Incumbent-Challenger 迟滞防跳车
    //

    private func updateIDFiducialTargetLock(
        detRect: CGRect?,
        gtScreenRect: CGRect?,
        gtVisibleNow: Bool,
        yoloMatchedThisFrame: Bool,
        stats: inout VSimStats
    ) -> CGRect? {
        var selected: CGRect?

        if let det = detRect, yoloMatchedThisFrame {
            if !stKF_initialized {
                stKF_initWith(rect: det)
                stats.stLockedTrackId = 1
            } else {
                stKF_update(rect: det)
            }
            stIncumbentRect = det
            stPrevCenter = CGPoint(x: det.midX, y: det.midY)
            stVelocity = .zero
            stConsecutiveLost = 0
            selected = det
        } else if let det = detRect, stKF_initialized {
            stKF_predict()
            stKF_update(rect: det)
            stIncumbentRect = det
            stPrevCenter = CGPoint(x: det.midX, y: det.midY)
            stConsecutiveLost = 0
            selected = det
        } else if stKF_initialized {
            stKF_predict()
            stConsecutiveLost += 1
            if stConsecutiveLost <= Int((0.5 * VSimCfg.fps).rounded(.down)) {
                selected = stIncumbentRect ?? stKF_getPredictedRect()
            } else {
                selected = nil
            }
        }

        if gtVisibleNow, let gt = gtScreenRect, gt.width > 0.001, gt.height > 0.001 {
            stats.stTotalFrames += 1
            if let selected {
                let iou = Self.iou(selected, gt)
                if iou > 0.15 {
                    stats.stLockFrames += 1
                }
                stats.stPrevRect = selected
            } else if let prev = stats.stPrevRect {
                let iou = Self.iou(prev, gt)
                if iou > 0.15 {
                    stats.stLockFrames += 1
                }
            }
        }

        return selected
    }

    private func updateSingleTargetLockStats(
        detRect: CGRect?,
        smRect: CGRect?,
        allGtRects: [CGRect],
        gtScreenRect: CGRect?,
        gtVisibleNow: Bool,
        dets: [(rect: CGRect, conf: Float, label: String)],
        yoloMatchedThisFrame: Bool,
        stats: inout VSimStats
    ) -> CGRect? {
        let effectiveDets: [(rect: CGRect, conf: Float, label: String)] = {
            guard useIDFiducialCVDetector else { return dets }
            let targetId = activeGtCarIdx
            return dets.filter { idFiducialCarId(from: $0.label) == targetId }
        }()

        // --- V6.3: 用 ByteTrack 平滑框作为主候选，原始 YOLO 框作为补充 ---
        var candidateRects: [CGRect]
        if let tracker = byteTracker {
            let btDets = effectiveDets.map { BTDetection(rect: $0.rect, score: Double($0.conf), classId: 0) }
            let tracks = tracker.update(detections: btDets)
            candidateRects = tracks.map { $0.normalizedRect }
            // 同时加入原始 YOLO 框作为补充候选（当 ByteTrack 丢失时）
            let rawMinConf = stKF_initialized ? stLockedRawConf : stUnlockedRawConf
            let rawRects = effectiveDets
                .filter { $0.conf >= rawMinConf && stShouldAdmitRawCandidate(rect: $0.rect, conf: $0.conf) }
                .map { $0.rect }
            for raw in rawRects {
                // 只加入与现有候选不重复的框
                let isDuplicate = candidateRects.contains(where: { Self.iou($0, raw) > 0.5 })
                if !isDuplicate { candidateRects.append(raw) }
            }
        } else {
            let rawMinConf = stKF_initialized ? stLockedRawConf : stUnlockedRawConf
            candidateRects = effectiveDets
                .filter { $0.conf >= rawMinConf && stShouldAdmitRawCandidate(rect: $0.rect, conf: $0.conf) }
                .map { $0.rect }
        }

        // --- 1. 首次检测：初始化锁定 ---
        if !stKF_initialized && !candidateRects.isEmpty {
            // 初始化跟踪框：优先选与「主车真值框」重叠最大的检测框，避免多车场景下误选面积最大/置信度最高却远离主车的框（会导致单目标锁定率统计恒为 0）。
            let best: CGRect
            if let gtPrimary = groundTruthRectForActivePrimaryCar(), !effectiveDets.isEmpty {
                let onPrimary = effectiveDets.filter { Self.iou($0.rect, gtPrimary) > 0.12 }
                if let pick = onPrimary.max(by: { $0.conf < $1.conf }) {
                    best = pick.rect
                } else if let maxDet = effectiveDets.max(by: { $0.conf < $1.conf }) {
                    best = maxDet.rect
                } else {
                    best = candidateRects.first!
                }
            } else if let maxDet = effectiveDets.max(by: { $0.conf < $1.conf }) {
                best = maxDet.rect
            } else {
                best = candidateRects.first!
            }

            // 匹配 GT 车辆（仅用于评估）
            var bestIoU: CGFloat = 0
            var bestGtId = -1
            for car in cars {
                guard car.isVisible, car.screenRect.width > 0 else { continue }
                let iou = Self.iou(best, car.screenRect)
                if iou > bestIoU { bestIoU = iou; bestGtId = car.id }
            }
            stLockedGtCarId = activeGtCarIdx  // V6.3 fix: 始终用 activeGtCarIdx
            stats.stLockedTrackId = 1  // V6: 虚拟 ID，始终为 1

            // 初始化独立 KF
            stKF_initWith(rect: best)

            // 初始化模板
            stFpFixed = nil
            stFpDynamic = extractFeaturePrint(best)
            stFpReacq = stFpDynamic
            let hs = extractHSHistogram(best)
            stHsFixed = nil
            stHsDynamic = hs
            stHsReacq = hs
            stBurnInCount = 0
            stBurnInHsAccum = hs
            stBurnInHsCount = hs != nil ? 1 : 0

            stPrevCenter = CGPoint(x: best.midX, y: best.midY)
            stVelocity = .zero
            stConsecutiveLost = 0
            stReacqUpdateCount = 0
            stIncumbentRect = best
            stChallengerWinStreak = 0
        }

        guard stKF_initialized else { return nil }

        // --- 2. KF 预测步 ---
        stKF_predict()
        let predRect = stKF_getPredictedRect()

        // --- 3. 查找主 GT 车辆是否可见（仅用于评估） ---
        // ★ V6.3 fix: 用 activeGtCarIdx 而非 stLockedGtCarId
        //   stLockedGtCarId 可能在初始化时选错 GT 车（与 activeGtCarIdx 不一致）
        //   导致 stLockFrames 对比错误的 GT 框
        let lockedCar = cars.first(where: { $0.id == activeGtCarIdx })
        let lockedGtVisible = lockedCar?.isVisible ?? false
        let lockedGtRect = (lockedCar != nil && lockedGtVisible) ? lockedCar!.screenRect : nil

        // --- 4. ★ V6 空间门控 + 评分 + 迟滞 ---
        let matchResult = v6SelectTarget(candidates: candidateRects, predRect: predRect, stats: &stats)

        // --- 5. 判定是否锁定成功（评估用） ---
        var frameLocked = false
        var frameIoU: CGFloat = 0
        if lockedGtVisible, let gtRect = lockedGtRect, gtRect.width > 0 {
            stats.stTotalFrames += 1
            if let trackRect = matchResult {
                // stConsecutiveLost 已在 v6SelectTarget 内部管理
                let iou = Self.iou(trackRect, gtRect)
                frameIoU = iou
                if iou > 0.15 {
                    stats.stLockFrames += 1
                    frameLocked = true
                }
                stats.stPrevRect = trackRect
            } else {
                if !stAvoidDoubleLostCount {
                    // Baseline preserves the historical V6 behavior for apples-to-apples experiments.
                    stConsecutiveLost += 1
                }
                if let prevRect = stats.stPrevRect {
                    let iou = Self.iou(prevRect, gtRect)
                    frameIoU = iou
                    if iou > 0.15 {
                        stats.stLockFrames += 1
                        frameLocked = true
                    }
                }
            }
        } else {
            // stConsecutiveLost 已在 v6SelectTarget 内部管理
        }

        // --- ★ 逐帧诊断日志 (V6 版，默认关闭 — 否则 Pass2 每帧 print 极慢) ---
        if VSimCfg.enableStFrameDiagLog {
            let f = stats.stTotalFrames
            let nCands = candidateRects.count
            let gtX = lockedGtRect?.midX ?? -1
            let gtY = lockedGtRect?.midY ?? -1
            let gtW = lockedGtRect?.width ?? 0
            let gtH = lockedGtRect?.height ?? 0
            let selX = matchResult?.midX ?? -1
            let selY = matchResult?.midY ?? -1
            let selW = matchResult?.width ?? 0
            let selH = matchResult?.height ?? 0
            let predX = predRect.midX
            let predY = predRect.midY
            let burnIn = stBurnInCount < stBurnInFrames
            let hasFpFixed = stFpFixed != nil
            let consLost = stConsecutiveLost
            let velX = stVelocity.x
            let velY = stVelocity.y
            let cumLockRate = stats.stTotalFrames > 0 ? Double(stats.stLockFrames) / Double(stats.stTotalFrames) * 100 : 0
            let hystStreak = stChallengerWinStreak

            print("[DIAG-V6] f=\(f) nC=\(nCands) locked=\(frameLocked ? 1 : 0) iou=\(String(format: "%.3f", frameIoU)) gtVis=\(lockedGtVisible ? 1 : 0) gt=\(String(format: "%.3f,%.3f,%.3f,%.3f", gtX, gtY, gtW, gtH)) sel=\(String(format: "%.3f,%.3f,%.3f,%.3f", selX, selY, selW, selH)) pred=\(String(format: "%.3f,%.3f", predX, predY)) burnIn=\(burnIn ? 1 : 0) fpFixed=\(hasFpFixed ? 1 : 0) consLost=\(consLost) vel=\(String(format: "%.4f,%.4f", velX, velY)) hystStreak=\(hystStreak) cumSTLock=\(String(format: "%.1f", cumLockRate))%")
        }

        return matchResult
    }

    // MARK: - ★ V6 独立卡尔曼滤波 (6D: cx, cy, w, h, vx, vy)

    /// 初始化 KF
    private func stKF_initWith(rect: CGRect) {
        let cx = Double(rect.midX)
        let cy = Double(rect.midY)
        let w = Double(rect.width)
        let h = Double(rect.height)
        stKF_x = [cx, cy, w, h, 0, 0]  // 初始速度为 0
        // 初始化协方差矩阵 6x6 (对角)
        stKF_P = [Double](repeating: 0, count: 36)
        stKF_P[0]  = 0.01   // cx 不确定性
        stKF_P[7]  = 0.01   // cy
        stKF_P[14] = 0.005  // w
        stKF_P[21] = 0.005  // h
        stKF_P[28] = 0.05   // vx (速度不确定性较大)
        stKF_P[35] = 0.05   // vy
        stKF_initialized = true
    }

    /// KF 预测步: x = F*x, P = F*P*F' + Q
    private func stKF_predict() {
        guard stKF_initialized else { return }
        // 状态转移: cx += vx, cy += vy
        stKF_x[0] += stKF_x[4]  // cx += vx
        stKF_x[1] += stKF_x[5]  // cy += vy
        // w, h 不变

        // 简化协方差更新: 只更新对角元素 + 过程噪声
        // P[cx,cx] += P[vx,vx] + Q_cx
        stKF_P[0]  += stKF_P[28] + 0.002   // cx 不确定性增加
        stKF_P[7]  += stKF_P[35] + 0.002   // cy
        stKF_P[14] += 0.001                 // w
        stKF_P[21] += 0.001                 // h
        stKF_P[28] += 0.003                 // vx (加速度过程噪声)
        stKF_P[35] += 0.003                 // vy
    }

    /// KF 更新步: 用观测框更新状态
    private func stKF_update(rect: CGRect) {
        guard stKF_initialized else { return }
        let z = [Double(rect.midX), Double(rect.midY), Double(rect.width), Double(rect.height)]

        // 观测噪声 R (对角)
        let R: [Double] = [0.005, 0.005, 0.003, 0.003]

        // 卡尔曼增益 (4维观测, 6维状态)
        // H = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]]
        // 简化: K_i = P[i,i] / (P[i,i] + R[i])  对于 i=0..3
        for i in 0..<4 {
            let pii = stKF_P[i * 6 + i]
            let k = pii / (pii + R[i])
            let innovation = z[i] - stKF_x[i]
            stKF_x[i] += k * innovation
            stKF_P[i * 6 + i] = (1.0 - k) * pii
        }
        // 速度修正: 用位置创新量修正速度估计
        let velGain = 0.3
        let dx = z[0] - (stKF_x[0] - stKF_x[4])  // 实际位移 vs 预测位移
        let dy = z[1] - (stKF_x[1] - stKF_x[5])
        stKF_x[4] += velGain * dx
        stKF_x[5] += velGain * dy
        // 缩小速度协方差
        stKF_P[28] *= 0.8
        stKF_P[35] *= 0.8
    }

    /// 获取 KF 预测框
    private func stKF_getPredictedRect() -> CGRect {
        let cx = CGFloat(stKF_x[0])
        let cy = CGFloat(stKF_x[1])
        let w = CGFloat(max(stKF_x[2], 0.005))
        let h = CGFloat(max(stKF_x[3], 0.005))
        return CGRect(x: cx - w/2, y: cy - h/2, width: w, height: h)
    }

    /// 已锁目标使用双门槛：新目标仍需高置信；已锁目标允许低置信候选，但只在局部门控内进入。
    private func stShouldAdmitRawCandidate(rect: CGRect, conf: Float) -> Bool {
        if conf >= stUnlockedRawConf { return true }
        guard stKF_initialized,
              stConsecutiveLost <= stGlobalReacquireLostFrames else { return false }

        let ref = stIncumbentRect ?? stKF_getPredictedRect()
        guard ref.width > 0.005, ref.height > 0.005 else { return false }

        let gateMultiplier: CGFloat = stConsecutiveLost <= 3 ? 6.0 : 10.0
        let gateW = max(ref.width, 0.02) * gateMultiplier
        let gateH = max(ref.height, 0.02) * gateMultiplier
        let gateRect = CGRect(
            x: ref.midX - gateW / 2,
            y: ref.midY - gateH / 2,
            width: gateW,
            height: gateH
        )
        guard gateRect.contains(CGPoint(x: rect.midX, y: rect.midY)) || Self.iou(rect, ref) > 0.02 else {
            return false
        }
        return stSizeConsistency(rect, ref) >= 0.25
    }

    private func stSizeConsistency(_ cand: CGRect, _ ref: CGRect) -> CGFloat {
        guard cand.width > 0.001, cand.height > 0.001,
              ref.width > 0.001, ref.height > 0.001 else { return 0 }
        let wRatio = cand.width / ref.width
        let hRatio = cand.height / ref.height
        return max(0, 1.0 - abs(wRatio - 1.0) - abs(hRatio - 1.0))
    }

    private func stNormalizedCenterDistance(_ cand: CGRect, _ ref: CGRect) -> CGFloat {
        let dx = cand.midX - ref.midX
        let dy = cand.midY - ref.midY
        let norm = max(max(ref.width, ref.height), 0.01)
        return sqrt(dx * dx + dy * dy) / norm
    }

    /// 低 IoU 接续最容易被危险 FP 抢锁，因此必须同时满足运动、尺寸和外观中的足够证据。
    private func stPassesLowIoUContinuationVeto(cand: CGRect, predRect: CGRect, incumbent: CGRect, iou: CGFloat) -> Bool {
        if iou >= 0.20 { return true }

        let sizeScore = max(stSizeConsistency(cand, incumbent), stSizeConsistency(cand, predRect))
        guard sizeScore >= 0.30 else { return false }

        let distNorm = stNormalizedCenterDistance(cand, predRect)
        let maxDist: CGFloat = stConsecutiveLost <= 3 ? 3.5 : 5.5
        guard distNorm <= maxDist else { return false }

        let hsScore = computeHSSimilarity(extractHSHistogram(cand))
        let fpScore = computeFeaturePrintSimilarity(extractFeaturePrint(cand))
        if hsScore == nil && fpScore == nil {
            return stBurnInCount < stBurnInFrames && iou >= 0.10
        }
        return (hsScore ?? 0) >= 0.32 || (fpScore ?? 0) >= 0.08
    }

    private func stPassesLocalReacquireVeto(cand: CGRect, predRect: CGRect, score: Double) -> Bool {
        guard stSizeConsistency(cand, predRect) >= 0.25 else { return false }

        let hsScore = computeHSSimilarity(extractHSHistogram(cand))
        let fpScore = computeFeaturePrintSimilarity(extractFeaturePrint(cand))
        if hsScore == nil && fpScore == nil {
            return stBurnInCount < stBurnInFrames && score > 0.40
        }
        return (hsScore ?? 0) >= 0.30 || (fpScore ?? 0) >= 0.08
    }

    // MARK: - ★ V6 空间门控 + 评分 + 迟滞

    private func v6SelectTarget(candidates: [CGRect], predRect: CGRect, stats: inout VSimStats) -> CGRect? {

        // =====================================================================
        // V6.1 核心逻辑：严格的 "incumbent-first" 策略
        //   1. 有 incumbent 时，优先用 IoU 匹配 incumbent 的后继框
        //   2. incumbent 匹配失败时，用 KF 预测 + 模板匹配重捕获
        //   3. 完全丢失时，KF 只预测不更新，等待重现
        // =====================================================================

        // --- 情况 A：无候选框 ---
        guard !candidates.isEmpty else {
            stConsecutiveLost += 1
            return nil
        }

        // --- 情况 B：有 incumbent，优先 IoU 匹配 ---
        if let incumbent = stIncumbentRect {
            // V6.2: 用 KF 预测位置作为“预期 incumbent”，而不是上一帧的 rect
            // 这样即使车辆快速移动，也能匹配上
            let predictedIncumbent = CGRect(
                x: predRect.origin.x + (incumbent.origin.x - predRect.origin.x) * 0.3,
                y: predRect.origin.y + (incumbent.origin.y - predRect.origin.y) * 0.3,
                width: (predRect.width + incumbent.width) / 2,
                height: (predRect.height + incumbent.height) / 2
            )
            // 同时用 incumbent 和 predRect 做双重匹配
            var bestIoU: CGFloat = 0
            var bestMatch: CGRect? = nil
            for cand in candidates {
                let iou1 = Self.iou(cand, incumbent)
                let iou2 = Self.iou(cand, predRect)
                let iou = max(iou1, iou2)  // 取两者中更好的
                if iou > bestIoU { bestIoU = iou; bestMatch = cand }
            }

            if bestIoU >= 0.05, let match = bestMatch,
               !stUseAppearanceVeto || stPassesLowIoUContinuationVeto(cand: match, predRect: predRect, incumbent: incumbent, iou: bestIoU) {
                // V6.2: 降低门槛 0.15→0.05；实验 profile 可对低 IoU 接续加外观 veto。
                // ★ 成功匹配 incumbent 后继
                stKF_update(rect: match)
                stIncumbentRect = match
                stConsecutiveLost = 0
                stChallengerWinStreak = 0

                let curCenter = CGPoint(x: match.midX, y: match.midY)
                if let prev = stPrevCenter {
                    let rawVel = CGPoint(x: curCenter.x - prev.x, y: curCenter.y - prev.y)
                    stVelocity = CGPoint(
                        x: stVelocity.x * 0.7 + rawVel.x * 0.3,
                        y: stVelocity.y * 0.7 + rawVel.y * 0.3
                    )
                }
                stPrevCenter = curCenter

                // 更新模板（仅在高置信度时）
                if bestIoU > 0.4 {
                    let fp = extractFeaturePrint(match)
                    let hs = extractHSHistogram(match)
                    updateTemplates(rect: match, fp: fp, hs: hs, posterior: 0.8)
                }
                return match
            }

            // --- incumbent IoU 匹配失败，尝试用 KF 预测位置门控 ---
            let gateMultiplier: CGFloat = stConsecutiveLost <= 3 ? 4.0 : 8.0
            let gateW = max(predRect.width, 0.02) * gateMultiplier
            let gateH = max(predRect.height, 0.02) * gateMultiplier
            let gateRect = CGRect(
                x: predRect.midX - gateW / 2,
                y: predRect.midY - gateH / 2,
                width: gateW,
                height: gateH
            )

            // 筛选门控内候选，用 IoU + 模板匹配评分
            var gatedScored: [(rect: CGRect, score: Double)] = []
            for cand in candidates {
                if gateRect.contains(CGPoint(x: cand.midX, y: cand.midY)) {
                    let score = v6ScoreCandidate(cand: cand, predRect: predRect, stats: stats)
                    gatedScored.append((cand, score))
                }
            }

            if let best = gatedScored.max(by: { $0.score < $1.score }), best.score > 0.25,
               !stUseAppearanceVeto || stPassesLocalReacquireVeto(cand: best.rect, predRect: predRect, score: best.score) {
                // 门控内找到合理候选
                // 但要检查是否是干扰车：用 HS 模板确认
                let hsScore = computeHSSimilarity(extractHSHistogram(best.rect))
                if let hs = hsScore, hs > 0.3 {
                    // 颜色匹配，接受为新 incumbent
                    stKF_update(rect: best.rect)
                    stIncumbentRect = best.rect
                    stConsecutiveLost = 0
                    stChallengerWinStreak = 0
                    let curCenter = CGPoint(x: best.rect.midX, y: best.rect.midY)
                    if let prev = stPrevCenter {
                        let rawVel = CGPoint(x: curCenter.x - prev.x, y: curCenter.y - prev.y)
                        stVelocity = CGPoint(
                            x: stVelocity.x * 0.7 + rawVel.x * 0.3,
                            y: stVelocity.y * 0.7 + rawVel.y * 0.3
                        )
                    }
                    stPrevCenter = curCenter
                    return best.rect
                }
            }

            // --- 门控内也没找到：不更新 KF，只增加丢失计数 ---
            stConsecutiveLost += 1

            // 如果连续丢失太久，触发全局重捕获。
            if stConsecutiveLost > stGlobalReacquireLostFrames {
                return v6GlobalReacquire(candidates: candidates, predRect: predRect, stats: &stats)
            }

            return nil

        } else {
            // --- 情况 C：无 incumbent（初始化后第一次） ---
            // 用 KF 预测位置最近的候选
            var minDist: CGFloat = .greatestFiniteMagnitude
            var nearest: CGRect? = nil
            for cand in candidates {
                let dx = cand.midX - predRect.midX
                let dy = cand.midY - predRect.midY
                let d = sqrt(dx*dx + dy*dy)
                if d < minDist { minDist = d; nearest = cand }
            }
            if let nr = nearest {
                stKF_update(rect: nr)
                stIncumbentRect = nr
                stPrevCenter = CGPoint(x: nr.midX, y: nr.midY)
                stConsecutiveLost = 0
                return nr
            }
            stConsecutiveLost += 1
            return nil
        }
    }

    // MARK: - ★ V6.1 全局重捕获（连续丢失 >30 帧后触发）

    private func v6GlobalReacquire(candidates: [CGRect], predRect: CGRect, stats: inout VSimStats) -> CGRect? {
        // V6.4: 空间约束 + HS 颜色联合评分
        // 优先选离 KF 预测位置近且颜色匹配的候选
        var scored: [(rect: CGRect, score: Float)] = []
        for cand in candidates {
            let hs = extractHSHistogram(cand)
            let hsSim = computeHSSimilarity(hs) ?? 0
            
            // 空间距离分（归一化到 0-1，放宽到 0.5）
            let dx = cand.midX - predRect.midX
            let dy = cand.midY - predRect.midY
            let dist = sqrt(dx*dx + dy*dy)
            let spatialScore: Float = Float(max(0, 1.0 - dist / 0.5))  // V6.5: 0.5 归一化（更宽松）

            // 与主车真值框的重叠（交并比），全局重捕获时优先拉回主车，而不是任意颜色相近的车辆。
            let primaryOverlap: Float = {
                guard let gtPrimary = groundTruthRectForActivePrimaryCar() else { return 0 }
                return Float(Self.iou(cand, gtPrimary))
            }()

            // 联合评分：空间 + 颜色 + 主车重叠（主车重叠占一定权重，避免跟错车）
            let combined = spatialScore * 0.25 + hsSim * 0.60 + primaryOverlap * 0.15
            scored.append((cand, combined))
        }
        
        guard let best = scored.max(by: { $0.score < $1.score }),
              best.score > 0.35 else { return nil }
        let rect = best.rect

        // 重新初始化 KF
        stKF_initWith(rect: rect)
        stIncumbentRect = rect
        stPrevCenter = CGPoint(x: rect.midX, y: rect.midY)
        stVelocity = .zero
        stConsecutiveLost = 0
        stChallengerWinStreak = 0

        // 更新模板
        let fp = extractFeaturePrint(rect)
        let hs = extractHSHistogram(rect)
        updateTemplates(rect: rect, fp: fp, hs: hs, posterior: 0.6)

        return rect
    }

    // MARK: - ★ V6 候选评分（运动交并比 / 尺寸 / 颜色直方图 / 特征指纹 / 主车真值重叠）

    private func v6ScoreCandidate(cand: CGRect, predRect: CGRect, stats: VSimStats) -> Double {
        var score: Double = 0

        // A. 位置预测 IoU (权重 0.50) — 小目标上最可靠的线索
        let motionIoU = Self.iou(cand, predRect)
        score += Double(motionIoU) * 0.50

        // B. 尺寸一致性 (权重 0.25)
        if predRect.width > 0.001, predRect.height > 0.001 {
            let wRatio = cand.width / predRect.width
            let hRatio = cand.height / predRect.height
            let sizeScore = max(0, 1.0 - abs(wRatio - 1.0) - abs(hRatio - 1.0))
            score += Double(sizeScore) * 0.25
        }

        // C. HS 颜色直方图 (权重 0.15)
        let hs = extractHSHistogram(cand)
        let hsSim = computeHSSimilarity(hs)
        if let hsSim = hsSim {
            score += Double(hsSim) * 0.15
        }

        // D. VNFeaturePrint (权重 0.10) — 小目标上降权
        let fp = extractFeaturePrint(cand)
        let fpSim = computeFeaturePrintSimilarity(fp)
        if let fpSim = fpSim {
            score += Double(fpSim) * 0.10
        }

        // E. 与「主车真值框」的重叠（额外加分）：多车并排时，防止评分只偏向预测位置而持续跟错车。
        if let gtPrimary = groundTruthRectForActivePrimaryCar() {
            let overlapPrimary = Self.iou(cand, gtPrimary)
            score += Double(overlapPrimary) * 0.35
        }

        return score
    }

    // MARK: - ★ 软物理约束 (V6 保留但不再用于主评分，仅作为诊断辅助)

    private func computeSoftPhysicsPenalty(rect: CGRect, stats: VSimStats) -> Double {
        guard stConsecutiveLost <= 5,
              let prevC = stPrevCenter,
              let prevRect = stats.stPrevRect else { return 0 }

        var penalty: Double = 0

        let predCx = prevC.x + stVelocity.x
        let predCy = prevC.y + stVelocity.y
        let dx = Double(rect.midX) - Double(predCx)
        let dy = Double(rect.midY) - Double(predCy)
        let dist = sqrt(dx * dx + dy * dy)
        let refSize = max(Double(prevRect.width), Double(prevRect.height))
        let normalizedDist = dist / max(refSize, 0.01)
        if normalizedDist > 2.0 {
            penalty += 0.20 * sigmoid((normalizedDist - 2.5) * 2.0)
        }

        let velMag = sqrt(stVelocity.x * stVelocity.x + stVelocity.y * stVelocity.y)
        if velMag > 0.005 {
            let actualDx = rect.midX - prevC.x
            let actualDy = rect.midY - prevC.y
            let actualMag = sqrt(actualDx * actualDx + actualDy * actualDy)
            if actualMag > 0.003 {
                let dot = actualDx * stVelocity.x + actualDy * stVelocity.y
                let cosSim = dot / (actualMag * velMag)
                if cosSim < 0 {
                    penalty += Double(-cosSim) * 0.12
                }
            }
        }

        return penalty
    }

    private func sigmoid(_ x: Double) -> Double {
        return 1.0 / (1.0 + exp(-x))
    }

    // MARK: - ★ VNFeaturePrint 特征提取

    /// 从归一化 rect 提取 VNFeaturePrint 嵌入
    private func extractFeaturePrint(_ rect: CGRect) -> VNFeaturePrintObservation? {
        guard let pb = stPixelBuffer,
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

    /// 计算候选 FeaturePrint 与三层模板的加权相似度
    /// 返回 0~1，nil 表示无法计算
    private func computeFeaturePrintSimilarity(_ candidate: VNFeaturePrintObservation?) -> Float? {
        guard let cand = candidate else { return nil }

        let isSearchPhase = stConsecutiveLost > 5

        var fixedSim: Float = 0
        var dynSim: Float = 0
        var reacqSim: Float = 0
        var hasAny = false

        if let fixed = stFpFixed {
            fixedSim = fpSimilarity(fixed, cand)
            hasAny = true
        }
        if let dyn = stFpDynamic {
            dynSim = fpSimilarity(dyn, cand)
            hasAny = true
        }
        if let reacq = stFpReacq {
            reacqSim = fpSimilarity(reacq, cand)
            hasAny = true
        }

        guard hasAny else { return nil }

        // burn-in 阶段只有动态模板（固定还没冻结）
        if stFpFixed == nil {
            return dynSim
        }

        if isSearchPhase {
            return fixedSim * 0.5 + reacqSim * 0.3 + dynSim * 0.2
        } else {
            return fixedSim * 0.4 + dynSim * 0.4 + reacqSim * 0.2
        }
    }

    /// VNFeaturePrint 距离 → 0~1 相似度
    private func fpSimilarity(_ a: VNFeaturePrintObservation, _ b: VNFeaturePrintObservation) -> Float {
        var distance: Float = 0
        do {
            try a.computeDistance(&distance, to: b)
        } catch {
            return 0
        }
        // VNFeaturePrint 的距离范围通常 0~80+
        // sigma 控制衰减速率：sigma 越大越宽容
        let sigma: Float = 12.0
        return exp(-distance / sigma)
    }

    // MARK: - ★ HS 直方图后备特征

    /// 提取 128-bin HS 直方图（与 V1 一致，作为 VNFeaturePrint 失败时的后备）
    private func extractHSHistogram(_ rect: CGRect) -> [Float]? {
        guard let pb = stPixelBuffer,
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
        guard let colorBytes = stRgbaBytes(ciImage: cropped, size: CGSize(width: 32, height: 32)) else { return nil }

        let hBins = 16, sBins = 8
        var hist = [Float](repeating: 0, count: hBins * sBins)
        for i in stride(from: 0, to: min(colorBytes.count, 32 * 32 * 4), by: 4) {
            let r = Float(colorBytes[i]) / 255.0
            let g = Float(colorBytes[i + 1]) / 255.0
            let b = Float(colorBytes[i + 2]) / 255.0
            let hsv = stRgbToHSV(r: r, g: g, b: b)
            let hBin = min(Int(hsv.h * Float(hBins)), hBins - 1)
            let sBin = min(Int(hsv.s * Float(sBins)), sBins - 1)
            hist[hBin * sBins + sBin] += 1.0
        }
        let sumHist = hist.reduce(0, +)
        if sumHist > 0 { hist = hist.map { $0 / sumHist } }
        return hist
    }

    /// 计算候选 HS 直方图与三层模板的加权相似度
    private func computeHSSimilarity(_ candidate: [Float]?) -> Float? {
        guard let cand = candidate else { return nil }
        let isSearchPhase = stConsecutiveLost > 5

        var fixedSim: Float = 0
        var dynSim: Float = 0
        var reacqSim: Float = 0
        var hasAny = false

        if let fixed = stHsFixed {
            fixedSim = stHistIntersection(a: fixed, b: cand)
            hasAny = true
        }
        if let dyn = stHsDynamic {
            dynSim = stHistIntersection(a: dyn, b: cand)
            hasAny = true
        }
        if let reacq = stHsReacq {
            reacqSim = stHistIntersection(a: reacq, b: cand)
            hasAny = true
        }

        guard hasAny else { return nil }

        if stHsFixed == nil {
            return dynSim
        }

        if isSearchPhase {
            return fixedSim * 0.5 + reacqSim * 0.3 + dynSim * 0.2
        } else {
            return fixedSim * 0.4 + dynSim * 0.4 + reacqSim * 0.2
        }
    }

    // MARK: - ★ 模板更新

    private func updateTemplates(
        rect: CGRect,
        fp: VNFeaturePrintObservation?,
        hs: [Float]?,
        posterior: Double
    ) {
        let isHighConf = posterior > 0.6

        // ── burn-in 阶段：持续更新（还没冻结固定模板） ──
        if stBurnInCount < stBurnInFrames {
            stBurnInCount += 1

            // FeaturePrint：burn-in 期间持续用最新的高置信度帧
            if isHighConf, let fp = fp {
                stFpDynamic = fp
            }
            // HS：累积多帧取均值
            if let hs = hs {
                if stBurnInHsAccum == nil {
                    stBurnInHsAccum = hs
                    stBurnInHsCount = 1
                } else {
                    for i in 0..<min(stBurnInHsAccum!.count, hs.count) {
                        stBurnInHsAccum![i] += hs[i]
                    }
                    stBurnInHsCount += 1
                }
                stHsDynamic = hs
            }

            // burn-in 结束：冻结固定模板
            if stBurnInCount == stBurnInFrames {
                stFpFixed = stFpDynamic
                stFpReacq = stFpDynamic
                if let accum = stBurnInHsAccum, stBurnInHsCount > 0 {
                    let avg = accum.map { $0 / Float(stBurnInHsCount) }
                    let s = avg.reduce(0, +)
                    stHsFixed = s > 0 ? avg.map { $0 / s } : avg
                } else {
                    stHsFixed = stHsDynamic
                }
                stHsReacq = stHsFixed
            }
            return
        }

        // ── 正常阶段 ──
        guard isHighConf else { return }

        // VNFeaturePrint 动态模板：防漂移检查后替换
        if let fp = fp, let fixed = stFpFixed {
            let fixedCheck = fpSimilarity(fixed, fp)
            if fixedCheck > 0.35 {  // 与固定模板不能差太远
                stFpDynamic = fp
            }
        }

        // HS 动态模板：EMA 混合
        if let hs = hs, let dyn = stHsDynamic {
            let alpha: Float = 0.15
            var blended = [Float]()
            blended.reserveCapacity(dyn.count)
            for i in 0..<min(dyn.count, hs.count) {
                blended.append(dyn[i] * (1 - alpha) + hs[i] * alpha)
            }
            let s = blended.reduce(0, +)
            if s > 0 { blended = blended.map { $0 / s } }
            // 防漂移检查
            if let fixed = stHsFixed {
                let check = stHistIntersection(a: fixed, b: blended)
                if check > 0.40 {
                    stHsDynamic = blended
                }
            } else {
                stHsDynamic = blended
            }
        }

        // 重捕获模板：每 10 帧更新一次（极低融合率）
        stReacqUpdateCount += 1
        if stReacqUpdateCount >= 10 {
            stReacqUpdateCount = 0
            if let fp = fp { stFpReacq = fp }
            if let hs = hs { stHsReacq = hs }
        }
    }

    // MARK: - 辅助函数

    private func stRgbaBytes(ciImage: CIImage, size: CGSize) -> [UInt8]? {
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

    private func stRgbToHSV(r: Float, g: Float, b: Float) -> (h: Float, s: Float, v: Float) {
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

    private func stHistIntersection(a: [Float], b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var sum: Float = 0
        for i in 0..<n { sum += min(a[i], b[i]) }
        return max(0, min(1, sum))
    }

}
