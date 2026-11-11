// RCTrackingSimulator.swift
// 单文件 macOS 仿真器 — 真实 YOLO 推理 + 二轴耦合舵机 + BLE延迟
//
// Xcode:
//   1. 新建 macOS App (SwiftUI, Swift)
//   2. 删除自动生成的 App.swift / ContentView.swift
//   3. 将本文件加入项目
//   4. 将 yolo11_640imgsz.mlmodelc 和/或 Detector.mlmodelc 拖入项目
//   5. Target: macOS 14.0+
//   6. Signing & Capabilities → App Sandbox:
//        ✅ File Access → User Selected File → Read Only
//      或直接编辑 .entitlements:
//        com.apple.security.files.user-selected.read-only = YES
//   7. 如仍无法打开视频，可尝试关闭 Sandbox（开发阶段）
//
// 视频导入: 点击 Video… 按钮 或 直接拖放到画面区域

import SwiftUI
import Combine
import AppKit
import AVFoundation
import Vision
import CoreML
import CoreImage
import CoreGraphics
import CoreVideo
import UniformTypeIdentifiers
import SceneKit
import simd      // ★ 用于高速 3x3 矩阵求解

// ═══════════════════════════════════════════════════════════════
// MARK: - 配置（与 iOS CinematicConfig 同步）
// ═══════════════════════════════════════════════════════════════

struct Cfg {
    // ── YOLO 置信度 ──
    static let yoloConf: Float = 0.85          // 普通模式 — 与 iOS CinematicConfig 对齐
    static var rYoloConf: Float = 0.30         // 赛道模式 — 低阈值, 快速捕获小车；eval 可用 --conf 覆盖
    static let evalYoloFPS: Double = 30.0      // 评测模式固定 YOLO 调度，避免复跑随机分叉
    static let searchCooldown: CFTimeInterval = 1.2   // 0.8→1.2  丢目标后多留时间给 YOLO/KF 重新捕获（同一点丢多为 Tracker 先掉）
    static let servoHome: CGFloat = 120
    static let servoSendInterval: CFTimeInterval = 0.02

    // PID 普通模式 (自运动补偿后可适当降低增益，减轻残余振荡)
    static let kpH: CGFloat = 1.2              // 2.0→1.2  配合自运动补偿与 One-Euro 修复
    static let kffH: CGFloat = 0.5             // 0.8→0.5  前馈略降；若回调死区后出现抖动可再略降以吸收
    static let kpV: CGFloat = 0.8              // 1.2→0.8  俯仰增益同步降低
    static let kffV: CGFloat = 0.2             // 0.3→0.2
    static let velAlpha: CGFloat = 0.3         // 0.7→0.3  速度滤波加重，减少噪声
    static let safeMarginX: CGFloat = 0.2
    static let safeMarginY: CGFloat = 0.15
    static let edgeKpH: CGFloat = 2.0    // 5→2  边缘补偿降低，减少过冲
    static let edgeKpV: CGFloat = 1.5   // 4→1.5
    static let leadTime: CGFloat = 0.04 // 0.12→0.04  前瞻不超过系统延迟一半，避免预测过头
    static let targetRatio: CGFloat = 0.40
    static let maxDPS: CGFloat = 380
    static let minStep: CGFloat = 0.05         // 0.15→0.05  回调死区，避免小偏移永陷死区

    // ── Lead Room 构图余量 (速度方向多留空间) ──
    static let leadRoomGain: CGFloat = 0.05       // 普通模式: 最大偏移 (归一化, 0.05=画面5%)
    static let rLeadRoomGain: CGFloat = 0.08      // 赛道模式: 更大余量适应高速
    static let leadRoomVelThresh: CGFloat = 0.15   // 速度门槛 (归一化/s), 低于此不偏移
    static let leadRoomSmooth: CGFloat = 0.08      // 偏移量 EMA 平滑系数 (越小越稳)
    static let leadRoomMaxShift: CGFloat = 0.12    // 最大偏移量上限 (归一化)

    // PID 赛道模式 (已降低增益减轻振荡，仍比普通模式更跟手)
    static let rKpH: CGFloat = 3.0      // 5.5→3.0  进一步降低，减轻过冲
    static let rKffH: CGFloat = 0.8     // 1.6→0.8  前馈降低
    static let rKpV: CGFloat = 2.5      // 4.0→2.5
    static let rKffV: CGFloat = 0.6     // 1.2→0.6
    static let rLead: CGFloat = 0.22     // 0.18→0.22  稍增前瞻，补偿更大的系统延迟
    static let rMaxDPS: CGFloat = 500
    static let rMinStep: CGFloat = 0.02  // 0.04→0.02  回调死区，避免小偏移永陷死区

    // 惯性 & 丢失 (赛道模式使用; 普通模式与 iOS 对齐: 只有 HOLD + RETURN)
    static let inertiaDur: CFTimeInterval = 1.2
    static let inertiaDecay: CGFloat = 0.92
    static let inertiaMax: CGFloat = 2.5
    static let lostHold: CFTimeInterval = 2.0
    static let scanDur: CFTimeInterval = 4.0
    static let scanAmpH: CGFloat = 18
    static let scanAmpV: CGFloat = 10
    static let scanPeriod: CGFloat = 1.6
    static let returnAlpha: CGFloat = 0.05

    // 卡尔曼 (仅赛道模式)
    static let kfMaxPredict: Int = 8
    static let kfMaxUncert: CGFloat = 0.2

    // 目标匹配 — 赛道模式外观匹配阈值
    static let searchRelaxedAcceptScore: Float = 0.25  // ★ 搜索中更宽容
    static let targetAcceptScore: Float = 0.40         // ★ 降低正常接受阈值
    static let targetStrongScore: Float = 0.65         // ★ 降低强匹配阈值, 更易更新模板
    static let targetHoldFramesMax: Int = 14   // 8→14  多帧内延续 lastSeen，避免 Tracker 刚掉就判 isLost
    // ★ 赛道模式: Tracker 死亡后的重新捕获阈值 (更宽松)
    static let trackerDeadAcceptScore: Float = 0.30
    // ★ 赛道模式: 运动区分 — 防止锁定静止车辆
    static let proximitySigma: CGFloat = 0.15            // 位置接近度高斯核宽度 (归一化坐标)
    static let movingVelThreshold: CGFloat = 0.3         // 判定"目标在运动"的最低速度 (归一化/s)
    static let stationaryPenalty: Float = 0.35           // 静止候选惩罚 (目标运动但候选不动)
    // 普通模式 IoU 匹配阈值 (对齐 iOS)
    static let normalIoURefresh: CGFloat = 0.6         // IoU>此值: 仅刷新时间戳, 不重建tracker
    static let normalIoUReinit: CGFloat = 0.3          // IoU>此值: 重建tracker

    // ── Ground Truth SAHI 检测 (仅统计, 不参与控制) ──
    static let gtSahiTileSize: Int = 640               // 分片像素尺寸 (正方形)
    static let gtSahiOverlap: Double = 0.25            // 分片重叠率 25%
    static let gtSahiConf: Float = 0.25                // GT 检测用低阈值, 尽量高召回
    static let gtSahiNmsIoU: CGFloat = 0.5             // 跨分片 NMS 合并阈值

    // ── 变焦仿真 (对齐 iOS AutoZoomEngine + MomentumGimbalEngine) ──
    static let opticalMin: CGFloat = 1.0               // 光学变焦下限
    static let opticalMax: CGFloat = 2.5               // 光学变焦上限
    static let digitalZoomMin: CGFloat = 1.0           // 数字变焦下限
    static let digitalZoomMax: CGFloat = 4.0           // 数字变焦上限
    static let zoomTargetRatio: CGFloat = 0.15        // 目标占比 (高度)，默认 15%
    static let zoomDeadzone: CGFloat = 0.25            // 死区 (光学)
    static let zoomSmoothing: CGFloat = 0.05           // 光学 EMA 平滑系数
    static let zoomStabilityN: Int = 8                 // 稳定性阈值帧数
    static let zoomMaxStep: CGFloat = 0.02             // 光学单帧最大步进
    static let zoomRecoverySpeed: CGFloat = 0.01       // 丢失回退速度
    static let digitalSmoothFactor: CGFloat = 0.05     // 数字变焦平滑
    static let digitalDeadzoneX: CGFloat = 0.02        // 数字云台死区 X
    static let digitalDeadzoneY: CGFloat = 0.05        // 数字云台死区 Y
    static let digitalVisualSmooth: CGFloat = 0.15     // 数字云台平滑
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 变焦引擎 (对齐 iOS AutoZoomEngine + MomentumGimbalEngine)
// ═══════════════════════════════════════════════════════════════

/// 运行时可调的变焦配置 (UI 滑块 → 此结构 → 变焦引擎)
struct ZoomConfig {
    var opticalMin: CGFloat  = Cfg.opticalMin         // 光学变焦下限
    var opticalMax: CGFloat  = Cfg.opticalMax         // 光学变焦上限
    var digitalMin: CGFloat  = Cfg.digitalZoomMin     // 数字变焦下限
    var digitalMax: CGFloat  = Cfg.digitalZoomMax     // 数字变焦上限
    var targetRatio: CGFloat = Cfg.zoomTargetRatio    // 目标占比 (高度)
    var deadzone: CGFloat    = Cfg.zoomDeadzone       // 光学变焦死区
    var maxStep: CGFloat     = Cfg.zoomMaxStep        // 光学单帧最大步进
}

/// 光学变焦仿真: 根据目标大小动态调整, 含 EMA + 死区 + 稳定计数 + 步进限制
final class SimAutoZoom {
    private var smoothedAbsH: CGFloat? = nil
    private var stableCount: Int = 0

    func reset() { smoothedAbsH = nil; stableCount = 0 }

    /// 计算下一帧光学变焦倍率
    func update(targetRect: CGRect?, currentZoom: CGFloat, isLost: Bool, zc: ZoomConfig) -> CGFloat {
        // 丢失 → 极慢回退
        if isLost {
            smoothedAbsH = nil; stableCount = 0
            let target = zc.opticalMin
            if currentZoom > target {
                return max(target, currentZoom - Cfg.zoomRecoverySpeed)
            }
            return target
        }
        guard let rect = targetRect else { return currentZoom }

        // 还原到 1x 时的绝对高度
        let rawAbsH = rect.height * currentZoom
        var cleanH = rawAbsH
        if let prev = smoothedAbsH {
            cleanH = prev * (1.0 - Cfg.zoomSmoothing) + rawAbsH * Cfg.zoomSmoothing
        }
        smoothedAbsH = cleanH

        let safeH = max(cleanH, 0.01)
        let ideal = zc.targetRatio / safeH
        let clamped = min(max(ideal, zc.opticalMin), zc.opticalMax)

        // 死区
        let diff = clamped - currentZoom
        if abs(diff) < zc.deadzone { stableCount = 0; return currentZoom }

        // 稳定计数
        stableCount += 1
        if stableCount < Cfg.zoomStabilityN { return currentZoom }

        // 步进限制
        if diff > 0 { return min(currentZoom + zc.maxStep, clamped) }
        else        { return max(currentZoom - zc.maxStep, clamped) }
    }
}

/// 数字云台仿真: 软件二次裁切, 对齐 iOS MomentumGimbalEngine
struct DigitalCropState {
    var centerTL: CGPoint    // 数字云台中心 (归一化, TL 坐标系)
    var zoom: CGFloat        // 数字变焦倍率
    var hasTarget: Bool
}

final class SimDigitalGimbal {
    private var center = CGPoint(x: 0.5, y: 0.5)
    private var zoom: CGFloat = 1.0

    func reset() { center = CGPoint(x: 0.5, y: 0.5); zoom = 1.0 }

    func update(targetRectTL: CGRect?, dt: Double, zc: ZoomConfig) -> DigitalCropState {
        var targetZoom: CGFloat = zc.digitalMin
        if let rect = targetRectTL {
            let w = max(rect.width, 0.01)
            let ideal = zc.targetRatio / w
            targetZoom = min(max(ideal, zc.digitalMin), zc.digitalMax)
        }
        zoom += (targetZoom - zoom) * Cfg.digitalSmoothFactor

        var dx: CGFloat = 0, dy: CGFloat = 0
        if let r = targetRectTL {
            let rawDx = r.midX - center.x
            let rawDy = r.midY - center.y
            if abs(rawDx) > Cfg.digitalDeadzoneX {
                dx = (rawDx > 0 ? rawDx - Cfg.digitalDeadzoneX : rawDx + Cfg.digitalDeadzoneX) * Cfg.digitalVisualSmooth
            }
            if abs(rawDy) > Cfg.digitalDeadzoneY {
                dy = (rawDy > 0 ? rawDy - Cfg.digitalDeadzoneY : rawDy + Cfg.digitalDeadzoneY) * Cfg.digitalVisualSmooth
            }
        } else {
            dx = (0.5 - center.x) * 0.05
            dy = (0.5 - center.y) * 0.05
        }
        let maxMove = 4.0 * CGFloat(dt)
        dx = min(max(dx, -maxMove), maxMove)
        dy = min(max(dy, -maxMove), maxMove)
        center.x += dx; center.y += dy

        let half = 0.5 / zoom
        center.x = min(max(center.x, half), 1.0 - half)
        center.y = min(max(center.y, half), 1.0 - half)

        return DigitalCropState(centerTL: center, zoom: zoom, hasTarget: targetRectTL != nil)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - One-Euro 滤波器 (标定链路防抖)
// ═══════════════════════════════════════════════════════════════

/// 一维 One-Euro 滤波（标准公式：速度越快 cutoff 越高 → 越灵敏）
final class OneEuroFilter1D {
    private var xPrev: CGFloat?
    private var dxPrev: CGFloat = 0
    private var tPrev: Double = 0
    let minCutoff: CGFloat
    let dCutoff: CGFloat
    let beta: CGFloat

    init(minCutoff: CGFloat = 1.0, dCutoff: CGFloat = 0.5, beta: CGFloat = 0.007) {
        self.minCutoff = minCutoff
        self.dCutoff = dCutoff
        self.beta = beta
    }

    func run(_ x: CGFloat, t: Double) -> CGFloat {
        guard let xp = xPrev else {
            xPrev = x
            tPrev = t
            return x
        }
        var dt = t - tPrev
        if !dt.isFinite || dt <= 0 { dt = 0.016 }
        dt = min(max(dt, 0.001), 0.2)
        let dRaw = (x - xp) / CGFloat(dt)
        let alphaD = 1.0 / (1.0 + 1.0 / (2.0 * .pi * dCutoff * CGFloat(dt)))
        let dSmooth = dxPrev + alphaD * (dRaw - dxPrev)
        // ★ 正确公式: 速度越快 → cutoff 越高 → tau 越小 → alpha 越大 → 更灵敏
        let cutoff = minCutoff + beta * abs(dSmooth)
        let tau = 1.0 / (2.0 * .pi * cutoff)
        let alpha = 1.0 / (1.0 + tau / CGFloat(dt))
        let out = xp + alpha * (x - xp)
        xPrev = out
        dxPrev = dSmooth
        tPrev = t
        return out
    }

    func reset() { xPrev = nil; dxPrev = 0; tPrev = 0 }
}

/// 矩形 One-Euro：对 midX, midY, width, height 分别滤波
final class OneEuroRectFilter {
    private let mx = OneEuroFilter1D(minCutoff: 1.2, dCutoff: 0.6)
    private let my = OneEuroFilter1D(minCutoff: 1.2, dCutoff: 0.6)
    private let mw = OneEuroFilter1D(minCutoff: 0.8, dCutoff: 0.4)
    private let mh = OneEuroFilter1D(minCutoff: 0.8, dCutoff: 0.4)

    func run(_ r: CGRect?, t: Double) -> CGRect? {
        guard let r = r else {
            mx.reset(); my.reset(); mw.reset(); mh.reset()
            return nil
        }
        let cx = mx.run(r.midX, t: t)
        let cy = my.run(r.midY, t: t)
        let w = max(0.02, mw.run(r.width, t: t))
        let h = max(0.02, mh.run(r.height, t: t))
        return CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
    }

    func reset() { mx.reset(); my.reset(); mw.reset(); mh.reset() }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 单帧测距与增益缩放
// ═══════════════════════════════════════════════════════════════

/// 单帧测距: 已知目标实际尺寸 + 焦距 + 像素高度 → 距离；并给出 PID 增益缩放（远处增益大）
struct TargetDistanceEstimator {
    /// RC 车实际高度 (米)，典型 1:10 遥控车约 0.12~0.18m
    var objectHeight: Double = 0.15

    /// 从目标检测框估算距离 (米)
    func estimate(rectHeight: CGFloat, focalLength: Double, imageHeight: Double) -> Double {
        let pixelH = Double(rectHeight) * imageHeight
        guard pixelH > 2 else { return 999 }
        return (objectHeight * focalLength) / pixelH
    }

    /// 基于距离的 PID 增益缩放：远处像素运动慢需更大增益，近处降低防过冲。参考距离 3m 处增益=1.0
    func gainScale(distance: Double) -> CGFloat {
        let refDistance = 3.0
        let scale = distance / refDistance
        return CGFloat(min(max(scale, 0.5), 2.0))
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 标定变焦引擎 (纯占比驱动)
// ═══════════════════════════════════════════════════════════════

/// 基于目标在画面中的占比调节变焦，无硬编码物距
final class CalibratedAutoZoom {
    private var smoothedAbsH: CGFloat?
    private var stableCount: Int = 0

    func reset() { smoothedAbsH = nil; stableCount = 0 }

    func update(targetRect: CGRect?, currentZoom: CGFloat, isLost: Bool, fovMap: CalibratedFOVMap, zc: ZoomConfig) -> CGFloat {
        if isLost {
            smoothedAbsH = nil
            stableCount = 0
            let target = zc.opticalMin
            return currentZoom > target ? max(target, currentZoom - Cfg.zoomRecoverySpeed) : target
        }
        guard let rect = targetRect else { return currentZoom }

        let rawAbsH = rect.height * currentZoom
        var cleanH = rawAbsH
        if let prev = smoothedAbsH {
            cleanH = prev * (1.0 - Cfg.zoomSmoothing) + rawAbsH * Cfg.zoomSmoothing
        }
        smoothedAbsH = cleanH

        let idealZoom = (zc.targetRatio / max(cleanH, 0.01)) * currentZoom
        let clamped = min(max(idealZoom, zc.opticalMin), zc.opticalMax)

        let diff = clamped - currentZoom
        if abs(diff) < zc.deadzone { stableCount = 0; return currentZoom }
        stableCount += 1
        if stableCount < Cfg.zoomStabilityN { return currentZoom }
        if diff > 0 { return min(currentZoom + zc.maxStep, clamped) }
        return max(currentZoom - zc.maxStep, clamped)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - BLE 延迟模型
// ═══════════════════════════════════════════════════════════════

struct BLEConfig {
    var minMs: Double = 8
    var maxMs: Double = 35
    var jitterStd: Double = 5
    var lossRate: Double = 0.02

    func sampleSec() -> Double {
        let base = (minMs + maxMs) / 2
        let u1 = Double.random(in: 0.001...1)
        let u2 = Double.random(in: 0.001...1)
        let g = sqrt(-2 * log(u1)) * cos(2 * .pi * u2) * jitterStd
        return max(minMs, min(maxMs, base + g)) / 1000
    }

    func drop() -> Bool {
        return Double.random(in: 0...1) < lossRate
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 二轴耦合舵机仿真 (修正版)
// ═══════════════════════════════════════════════════════════════

// ★ 修正 1: 为 ServoPhys 增加阻尼参考时间
struct ServoPhys {
    var maxDPS: Double = 380
    var accelDPS2: Double = 3000
    var damping: Double = 0.95
    /// damping 的参考时间步长 (秒)
    /// 含义: 每 dampingRefDt 秒，速度衰减为原来的 damping 倍
    /// 默认 0.02s (20ms) 是典型舵机控制周期
    var dampingRefDt: Double = 0.02
    var minDeg: Double = 0
    var maxDeg: Double = 240
    var deadBand: Double = 0.1
}

// ★ 修正 2: 为 CouplingCfg 增加摩擦力参考时间
struct CouplingCfg {
    var gravityScale: Double = 0.15
    var centrifugalScale: Double = 0.08
    var inertiaVarScale: Double = 0.20
    var staticFriction: Double = 0.5
    var dynamicFriction: Double = 0.02
    /// dynamicFriction 的参考时间步长 (秒)
    var frictionRefDt: Double = 0.02
}

private struct PendCmd {
    let angle: Double
    let arrival: Double
}

// ★ 修正 3: 替换整个 CoupledGimbal 类以使用新的 driveAxis 函数
final class CoupledGimbal {
    let panPhy: ServoPhys
    let tiltPhy: ServoPhys
    let ble: BLEConfig
    let coup: CouplingCfg

    private(set) var panPos: Double
    private(set) var panVel: Double = 0
    private(set) var tiltPos: Double
    private(set) var tiltVel: Double = 0
    private var panTgt: Double
    private var tiltTgt: Double
    private var panAcc: Double = 0
    private var tiltAcc: Double = 0
    private var panQ: [PendCmd] = []
    private var tiltQ: [PendCmd] = []
    private(set) var time: Double = 0
    private let step: Double = 0.0005

    init(home: Double = 120, pan: ServoPhys = ServoPhys(), tilt: ServoPhys = ServoPhys(),
         ble: BLEConfig = BLEConfig(), coup: CouplingCfg = CouplingCfg()) {
        self.panPhy = pan
        self.tiltPhy = tilt
        self.ble = ble
        self.coup = coup
        self.panPos = home
        self.panTgt = home
        self.tiltPos = home
        self.tiltTgt = home
    }

    func send(id: UInt8, angle: Double, durMs: Double = 25) {
        guard !ble.drop() else { return }
        let a: Double
        if id == 1 {
            a = min(max(angle, panPhy.minDeg), panPhy.maxDeg)
        } else {
            a = min(max(angle, tiltPhy.minDeg), tiltPhy.maxDeg)
        }
        let cmd = PendCmd(angle: a, arrival: time + ble.sampleSec())
        if id == 1 {
            panQ.append(cmd)
            panQ.sort { $0.arrival < $1.arrival }
        } else {
            tiltQ.append(cmd)
            tiltQ.sort { $0.arrival < $1.arrival }
        }
    }

    func advanceTo(_ t: Double) {
        while time < t {
            let dt = min(step, t - time)
            time += dt
            tick(dt)
        }
    }

    private func tick(_ dt: Double) {
        while let c = panQ.first, c.arrival <= time {
            panQ.removeFirst()
            panTgt = c.angle
        }
        while let c = tiltQ.first, c.arrival <= time {
            tiltQ.removeFirst()
            tiltTgt = c.angle
        }

        let tOff = abs(tiltPos - 120) * .pi / 180
        let panIM = 1.0 + coup.inertiaVarScale * sin(tOff) * sin(tOff)
        let grav = coup.gravityScale * sin(tOff) * (tiltPos > 120 ? -1.0 : 1.0)
        let cent = -coup.centrifugalScale * panAcc * sin(tOff)

        driveAxis(p: &panPos, v: &panVel, a: &panAcc,
                  tgt: panTgt, phy: panPhy, im: panIM, ext: 0, dt: dt)
        driveAxis(p: &tiltPos, v: &tiltVel, a: &tiltAcc,
                  tgt: tiltTgt, phy: tiltPhy, im: 1, ext: grav + cent, dt: dt)
    }

    // ★★★ 核心修正: driveAxis 函数完整替换 ★★★
    private func driveAxis(p: inout Double, v: inout Double, a: inout Double,
                           tgt: Double, phy: ServoPhys, im: Double, ext: Double, dt: Double) {
        let err = tgt - p

        if abs(err) < phy.deadBand {
            let brakeFactor = pow(0.3, dt / 0.02) // 原始 0.3 对应 20ms 参考
            v *= brakeFactor
            a = 0
            return
        }

        let desV = min(max(err * 50, -phy.maxDPS), phy.maxDPS)  // 150→50 真实舵机位置环增益较低，减少过冲
        let aLim = phy.accelDPS2 / im
        let servo = min(max((desV - v) / max(dt, 1e-9), -aLim), aLim)
        a = servo + ext

        if abs(v) < 0.1 && abs(err) < coup.staticFriction {
            a = ext
        }

        v += a * dt

        // ★ 修正: 动态摩擦力改为与时间步长无关 (dt-independent)
        let fricFactor = pow(1.0 - coup.dynamicFriction, dt / coup.frictionRefDt)
        v *= fricFactor

        v = min(max(v, -phy.maxDPS), phy.maxDPS)

        // ★ 核心修正: 阻尼改为与时间步长无关 (dt-independent)
        // 原始代码 v *= damping 在每个0.5ms tick执行，导致速度被过度压制
        // 修正后，damping=0.95的含义是"每dampingRefDt秒速度衰减到95%"
        let dampFactor = pow(phy.damping, dt / phy.dampingRefDt)
        v *= dampFactor

        p += v * dt

        if p <= phy.minDeg {
            p = phy.minDeg
            v = 0
        } else if p >= phy.maxDeg {
            p = phy.maxDeg
            v = 0
        }
    }

    var panErr: Double { panTgt - panPos }
    var tiltErr: Double { tiltTgt - tiltPos }

    func reset(_ home: Double = 120) {
        time = 0
        panPos = home; panVel = 0; panTgt = home; panAcc = 0
        tiltPos = home; tiltVel = 0; tiltTgt = home; tiltAcc = 0
        panQ.removeAll()
        tiltQ.removeAll()
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - FOV 映射 (修正版)
// ═══════════════════════════════════════════════════════════════

struct FOVMap {
    var srcH: Double = 120
    var srcV: Double = 70
    var panC: Double = 120
    var tiltC: Double = 120
    var camH: Double = 60
    var camV: Double = 34

    func crop(pan: Double, tilt: Double) -> CGRect {
        return crop(pan: pan, tilt: tilt, opticalZoom: 1.0)
    }

    func crop(pan: Double, tilt: Double, opticalZoom: CGFloat) -> CGRect {
        let z = max(Double(opticalZoom), 1.0)
        let halfBaseH = (camH * .pi / 180.0) / 2.0
        let halfBaseV = (camV * .pi / 180.0) / 2.0
        let effH = 2.0 * atan(tan(halfBaseH) / z) * 180.0 / .pi
        let effV = 2.0 * atan(tan(halfBaseV) / z) * 180.0 / .pi

        let cx = 0.5 - (pan  - panC) / srcH
        let cy = 0.5 + (tilt - tiltC) / srcV
        let w = effH / srcH
        let h = effV / srcV

        // 钳位逻辑本身是正确的，防止裁切框移出源视频边界
        let sx = min(max(cx, w / 2), 1 - w / 2)
        let sy = min(max(cy, h / 2), 1 - h / 2)
        return CGRect(x: sx - w / 2, y: sy - h / 2, width: w, height: h)
    }

    func effectiveFov(opticalZoom: CGFloat) -> (h: Double, v: Double) {
        let z = max(Double(opticalZoom), 1.0)
        let halfH = (camH * .pi / 180.0) / 2.0
        let halfV = (camV * .pi / 180.0) / 2.0
        let effH = 2.0 * atan(tan(halfH) / z) * 180.0 / .pi
        let effV = 2.0 * atan(tan(halfV) / z) * 180.0 / .pi
        return (effH, effV)
    }

    // ★ 新增: 计算有效行程范围 (供 UI 显示和诊断)
    func effectiveRange(opticalZoom: CGFloat = 1.0) -> (panMin: Double, panMax: Double, tiltMin: Double, tiltMax: Double) {
        let z = max(Double(opticalZoom), 1.0)
        let halfBaseH = (camH * .pi / 180.0) / 2.0
        let halfBaseV = (camV * .pi / 180.0) / 2.0
        let effH = 2.0 * atan(tan(halfBaseH) / z) * 180.0 / .pi
        let effV = 2.0 * atan(tan(halfBaseV) / z) * 180.0 / .pi
        let wNorm = effH / srcH
        let hNorm = effV / srcV

        // cx = 0.5 - (pan - panC)/srcH, 有效范围 cx ∈ [w/2, 1-w/2]
        // → pan ∈ [panC - (0.5-w/2)*srcH, panC + (0.5-w/2)*srcH]
        let panHalf = (0.5 - wNorm / 2) * srcH
        let tiltHalf = (0.5 - hNorm / 2) * srcV
        return (panC - panHalf, panC + panHalf, tiltC - tiltHalf, tiltC + tiltHalf)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 卡尔曼滤波器
// ═══════════════════════════════════════════════════════════════

// MARK: - 8D 卡尔曼滤波器 (独立预测中心点与宽高尺度)
final class KF {
    private var x_p: [CGFloat] = [0.5, 0.5, 0, 0] // 空间状态: [cx, cy, vx, vy]
    private var P_p: [CGFloat] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    private var x_s: [CGFloat] = [0.1, 0.1, 0, 0] // 尺度状态: [w, h, vw, vh]
    private var P_s: [CGFloat] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    private var ok = false
    private var ts: CFTimeInterval = 0

    // 位置的噪声参数
    private let qP: CGFloat = 0.0001
    private let qV: CGFloat = 0.08
    private let rP: CGFloat = 0.002

    // ★ 尺度的噪声参数 (尺度变化通常比位置平缓，所以过程噪声更小)
    private let qP_s: CGFloat = 0.00005
    private let qV_s: CGFloat = 0.01
    private let rP_s: CGFloat = 0.005

    var alive: Bool { ok }
    var center: CGPoint { CGPoint(x: x_p[0], y: x_p[1]) }
    var velocity: CGPoint { CGPoint(x: x_p[2], y: x_p[3]) }
    var size: CGSize { CGSize(width: max(x_s[0], 0.01), height: max(x_s[1], 0.01)) } // 防止尺度输出负数
    var unc: CGFloat { sqrt(max(P_p[0], 0) + max(P_p[5], 0)) }

    func start(center c: CGPoint, size s: CGSize, _ t: CFTimeInterval) {
        x_p = [c.x, c.y, 0, 0]
        P_p = [0.01, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5]
        x_s = [s.width, s.height, 0, 0]
        P_s = [0.01, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.2]
        ok = true
        ts = t
    }

    func predict(_ t: CFTimeInterval) {
        guard ok else { return }
        var dt = CGFloat(t - ts)
        if !dt.isFinite || dt <= 0 { dt = 0.016 }
        dt = min(dt, 0.2)
        ts = t

        predict4D(&x_p, &P_p, dt: dt, qP: qP, qV: qV)
        predict4D(&x_s, &P_s, dt: dt, qP: qP_s, qV: qV_s)
    }

    func update(center c: CGPoint, size s: CGSize, _ t: CFTimeInterval) {
        guard ok else { start(center: c, size: s, t); return }
        predict(t)
        update4D(&x_p, &P_p, zx: c.x, zy: c.y, rP: rP)
        update4D(&x_s, &P_s, zx: s.width, zy: s.height, rP: rP_s)
    }

    // 提取出的 4x4 矩阵推演复用逻辑
    private func predict4D(_ x: inout [CGFloat], _ P: inout [CGFloat], dt: CGFloat, qP: CGFloat, qV: CGFloat) {
        x[0] += x[2] * dt
        x[1] += x[3] * dt
        let d2 = dt * dt
        P[0] += 2 * dt * P[2] + d2 * P[10] + qP
        P[1] += dt * P[3] + dt * P[9] + d2 * P[11]
        P[2] += dt * P[10]
        P[3] += dt * P[11]
        P[4] = P[1]
        P[5] += 2 * dt * P[7] + d2 * P[15] + qP
        P[6] += dt * P[14]
        P[7] += dt * P[15]
        P[8] = P[2]
        P[9] = P[6]
        P[10] += qV
        P[12] = P[3]
        P[13] = P[7]
        P[15] += qV
    }

    private func update4D(_ x: inout [CGFloat], _ P: inout [CGFloat], zx: CGFloat, zy: CGFloat, rP: CGFloat) {
        let ezx = zx - x[0]; let ezy = zy - x[1]
        let s00 = P[0] + rP; let s01 = P[1]; let s10 = P[4]; let s11 = P[5] + rP
        let det = s00 * s11 - s01 * s10
        guard abs(det) > 1e-12 else { return }
        let id = 1 / det
        let i00 = s11 * id; let i01 = -s01 * id; let i10 = -s10 * id; let i11 = s00 * id
        let k00 = P[0]*i00 + P[1]*i10; let k01 = P[0]*i01 + P[1]*i11
        let k10 = P[4]*i00 + P[5]*i10; let k11 = P[4]*i01 + P[5]*i11
        let k20 = P[8]*i00 + P[9]*i10; let k21 = P[8]*i01 + P[9]*i11
        let k30 = P[12]*i00 + P[13]*i10; let k31 = P[12]*i01 + P[13]*i11
        x[0] += k00*ezx + k01*ezy; x[1] += k10*ezx + k11*ezy
        x[2] += k20*ezx + k21*ezy; x[3] += k30*ezx + k31*ezy
        let p = P
        P[0] = (1-k00)*p[0] - k01*p[4]; P[1] = (1-k00)*p[1] - k01*p[5]
        P[2] = (1-k00)*p[2] - k01*p[6]; P[3] = (1-k00)*p[3] - k01*p[7]
        P[4] = -k10*p[0] + (1-k11)*p[4]; P[5] = -k10*p[1] + (1-k11)*p[5]
        P[6] = -k10*p[2] + (1-k11)*p[6]; P[7] = -k10*p[3] + (1-k11)*p[7]
        P[8] = -k20*p[0] - k21*p[4] + p[8]; P[9] = -k20*p[1] - k21*p[5] + p[9]
        P[10] = -k20*p[2] - k21*p[6] + p[10]; P[11] = -k20*p[3] - k21*p[7] + p[11]
        P[12] = -k30*p[0] - k31*p[4] + p[12]; P[13] = -k30*p[1] - k31*p[5] + p[13]
        P[14] = -k30*p[2] - k31*p[6] + p[14]; P[15] = -k30*p[3] - k31*p[7] + p[15]
    }

    func reset() {
        ok = false
        x_p = [0.5, 0.5, 0, 0]; P_p = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        x_s = [0.1, 0.1, 0, 0]; P_s = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 时序轨迹预测 (二次多项式拟合)
// ═══════════════════════════════════════════════════════════════
final class TrajectoryPredictor {
    struct Point { let t: Double; let x: Double; let y: Double }
    private var history: [Point] = []
    private let maxHistory = 15 // 收集过去 15 帧的数据 (约 0.5 秒)

    // 拟合参数: x(t) = ax*t^2 + bx*t + cx
    private var coeffX = simd_double3(0, 0, 0)
    private var coeffY = simd_double3(0, 0, 0)
    private(set) var isValid = false

    var lastTime: Double? { history.last?.t }

    func push(t: Double, x: Double, y: Double) {
        // 如果时间跳变过大（目标丢失很久后重新捕获），清空历史
        if let last = history.last, t - last.t > 0.5 { history.removeAll() }

        history.append(Point(t: t, x: x, y: y))
        if history.count > maxHistory { history.removeFirst() }
        fit()
    }

    private func fit() {
        // 至少需要 5 帧才能稳定拟合二次曲线
        guard history.count >= 5 else { isValid = false; return }

        let t0 = history.first!.t

        // 构造最小二乘法矩阵方程 A^T * A * C = A^T * B
        var s_t0: Double = Double(history.count)
        var s_t1: Double = 0, s_t2: Double = 0, s_t3: Double = 0, s_t4: Double = 0
        var s_x: Double = 0, s_tx: Double = 0, s_t2x: Double = 0
        var s_y: Double = 0, s_ty: Double = 0, s_t2y: Double = 0

        for p in history {
            let dt = p.t - t0
            let dt2 = dt * dt
            s_t1 += dt; s_t2 += dt2; s_t3 += dt2 * dt; s_t4 += dt2 * dt2

            s_x += p.x; s_tx += dt * p.x; s_t2x += dt2 * p.x
            s_y += p.y; s_ty += dt * p.y; s_t2y += dt2 * p.y
        }

        let M = simd_double3x3(
            simd_double3(s_t0, s_t1, s_t2),
            simd_double3(s_t1, s_t2, s_t3),
            simd_double3(s_t2, s_t3, s_t4)
        )

        // 如果矩阵奇异（几乎静止），退回无效状态
        if abs(M.determinant) < 1e-10 { isValid = false; return }

        let M_inv = M.inverse
        coeffX = M_inv * simd_double3(s_x, s_tx, s_t2x)
        coeffY = M_inv * simd_double3(s_y, s_ty, s_t2y)
        isValid = true
    }

    func predict(targetTime: Double) -> (x: Double, y: Double)? {
        guard isValid, let first = history.first else { return nil }
        let dt = targetTime - first.t
        let dt2 = dt * dt
        let px = coeffX[0] + coeffX[1] * dt + coeffX[2] * dt2
        let py = coeffY[0] + coeffY[1] * dt + coeffY[2] * dt2
        return (px, py)
    }

    func reset() { history.removeAll(); isValid = false }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - PID 舵机控制器
// ═══════════════════════════════════════════════════════════════

final class PIDCtrl {
    private var ts: CFTimeInterval = 0
    private var eX: CGFloat = 0
    private var eY: CGFloat = 0
    private var vX: CGFloat = 0
    private var vY: CGFloat = 0
    private var dX: Int = 0
    private var dY: Int = 0
    var racing = false
    // ── Lead Room 平滑偏移 ──
    private var lrX: CGFloat = 0     // 当前平滑后的水平偏移 (归一化)
    private var lrY: CGFloat = 0     // 当前平滑后的垂直偏移 (归一化)

    struct Out {
        let sH: CGFloat
        let sV: CGFloat
        let eX: CGFloat
        let eY: CGFloat
        let vX: CGFloat
        let vY: CGFloat
        // ── Debug 中间变量 ──
        let rawVX: CGFloat     // 补偿前水平速度 (度/s)，用于自运动补偿符号验证
        let compensatedVX: CGFloat
        let predX: CGFloat     // 前瞻预测误差 X (度)
        let predY: CGFloat     // 前瞻预测误差 Y (度)
        let outX: CGFloat      // PID 输出速度 X (度/s)
        let outY: CGFloat      // PID 输出速度 Y (度/s)
        let gainScale: CGFloat // 距离自适应增益
        let distScale: CGFloat // 距离缩放因子
        let edgeX: CGFloat     // 边缘补偿 X
        let edgeY: CGFloat     // 边缘补偿 Y
        // ── Lead Room 构图偏移 ──
        let leadRoomX: CGFloat // 水平构图偏移 (归一化, 正=目标点右移)
        let leadRoomY: CGFloat // 垂直构图偏移 (归一化, 正=目标点下移)
    }

    func reset(_ t: CFTimeInterval) {
        ts = t; eX = 0; eY = 0; vX = 0; vY = 0; dX = 0; dY = 0
        lrX = 0; lrY = 0
    }

    func run(center: CGPoint?, rect: CGRect?, t: CFTimeInterval, hFov: CGFloat, vFov: CGFloat, gainScale: CGFloat = 1.0, servoVelH: CGFloat = 0, servoVelV: CGFloat = 0) -> Out {
        var dt = t - ts
        if !dt.isFinite || dt <= 0 { dt = 0.016 }
        dt = min(max(dt, 0.005), 0.1)
        ts = t

        guard let c = center else {
            vX = 0; vY = 0; dX = 0; dY = 0
            lrX *= 0.9; lrY *= 0.9
            return Out(sH: 0, sV: 0, eX: 0, eY: 0, vX: 0, vY: 0,
                       rawVX: 0, compensatedVX: 0,
                       predX: 0, predY: 0, outX: 0, outY: 0,
                       gainScale: 0, distScale: 0, edgeX: 0, edgeY: 0,
                       leadRoomX: lrX, leadRoomY: lrY)
        }

        var curKpH = racing ? Cfg.rKpH : Cfg.kpH
        var curKffH = racing ? Cfg.rKffH : Cfg.kffH
        var curKpV = racing ? Cfg.rKpV : Cfg.kpV
        var curKffV = racing ? Cfg.rKffV : Cfg.kffV
        let gsMul = max(0.3, min(3.0, gainScale))
        curKpH *= gsMul; curKffH *= gsMul; curKpV *= gsMul; curKffV *= gsMul
        let curMaxDPS = racing ? Cfg.rMaxDPS : Cfg.maxDPS
        let curMinStep = racing ? Cfg.rMinStep : Cfg.minStep
        let curLead = racing ? Cfg.rLead : Cfg.leadTime

        let sz = min(max(rect?.height ?? 0.01, 0.01), 0.6)
        let ds = min(max(Cfg.targetRatio / sz, 1), 3.0)
        let gs = racing ? max(0.85, 1.0 / ds) : max(0.35, 1.0 / ds)
        let dynMin = curMinStep * ds
        let hyst = (racing ? 0.03 : 0.2) * ds

        let errX = (c.x - 0.5) * hFov
        let errY = (c.y - 0.5) * vFov
        let rawVX = (errX - eX) / CGFloat(dt)
        let rawVY = (errY - eY) / CGFloat(dt)
        eX = errX
        eY = errY

        // ★ 补偿相机自运动: 减去舵机角速度在误差上的投影，消除正反馈振荡
        // 符号已修正: 与 cenX/cenY 及 FOVMap 取景方向一致后，应为减 H、加 V
        let compensatedVX = rawVX - servoVelH
        let compensatedVY = rawVY + servoVelV

        let alp = Cfg.velAlpha
        vX = vX * (1 - alp) + compensatedVX * alp
        vY = vY * (1 - alp) + compensatedVY * alp

        // ── Lead Room 构图余量: 速度方向多留空间 ──
        // 将度/s 速度转回归一化/s, 用于判断运动幅度
        let lrGain = racing ? Cfg.rLeadRoomGain : Cfg.leadRoomGain
        let normVX = vX / max(hFov, 1)
        let normVY = vY / max(vFov, 1)
        let normSpd = sqrt(normVX * normVX + normVY * normVY)
        var tgtLrX: CGFloat = 0
        var tgtLrY: CGFloat = 0
        if normSpd > Cfg.leadRoomVelThresh {
            // 速度越快偏移越大, 但不超过 maxShift
            tgtLrX = min(max(normVX * lrGain, -Cfg.leadRoomMaxShift), Cfg.leadRoomMaxShift)
            tgtLrY = min(max(normVY * lrGain, -Cfg.leadRoomMaxShift), Cfg.leadRoomMaxShift)
        }
        lrX += (tgtLrX - lrX) * Cfg.leadRoomSmooth
        lrY += (tgtLrY - lrY) * Cfg.leadRoomSmooth

        // PID 用的误差 = 目标相对于偏移后目标点的偏差
        // lrX>0 (车往右开) → 目标点右移 → 车在画面偏左 → 右侧留空间
        let ctrlErrX = (c.x - (0.5 + lrX)) * hFov
        let ctrlErrY = (c.y - (0.5 + lrY)) * vFov

        var leadH = curLead
        if (ctrlErrX > 0 && vX < 0) || (ctrlErrX < 0 && vX > 0) { leadH *= 0.5 }
        var leadV = curLead
        if (ctrlErrY > 0 && vY < 0) || (ctrlErrY < 0 && vY > 0) { leadV *= 0.5 }

        let predX = ctrlErrX + vX * leadH
        let predY = ctrlErrY + vY * leadV

        // ★ 与 iOS ServoOuterController 符号完全一致：
        //   目标右侧(errX>0) → outX 负 → servoH↓ → FOVMap cx 增大 → 取景右移追上目标
        //   目标下方(errY>0) → outY 正 → servoV↑ → FOVMap cy 增大 → 取景下移追上目标
        var outX = -(predX * curKpH * gs + vX * curKffH * gs)
        var outY = +(predY * curKpV * gs + vY * curKffV * gs)

        let edgeKpH = Cfg.edgeKpH * gsMul
        let edgeKpV = Cfg.edgeKpV * gsMul
        let limX = (0.5 - Cfg.safeMarginX) * hFov
        var edgeBX: CGFloat = 0
        if predX > limX {
            edgeBX = -(predX - limX) * edgeKpH
            outX += edgeBX   // 目标偏右超边界 → outX 更负（向右追更猛）
        } else if predX < -limX {
            edgeBX = (-predX - limX) * edgeKpH
            outX += edgeBX
        }
        let limY = (0.5 - Cfg.safeMarginY) * vFov
        var edgeBY: CGFloat = 0
        if predY > limY {
            edgeBY = (predY - limY) * edgeKpV
            outY += edgeBY   // 目标偏下超边界 → outY 更正（向下追更猛）
        } else if predY < -limY {
            edgeBY = -(-predY - limY) * edgeKpV
            outY += edgeBY
        }

        let dbgOutX = outX  // 记录限幅前的 PID 输出
        let dbgOutY = outY

        // 限幅独立于 gs：增益随距离自适应，但指令速度上限不被人为压低，充分利用舵机物理能力
        let maxS = curMaxDPS
        outX = min(max(outX, -maxS), maxS)
        outY = min(max(outY, -maxS), maxS)

        var stepH = outX * CGFloat(dt)
        var stepV = outY * CGFloat(dt)
        if abs(stepH) < dynMin { stepH = 0 }
        if abs(stepV) < dynMin { stepV = 0 }

        let dirX = stepH == 0 ? 0 : (stepH > 0 ? 1 : -1)
        if dirX != 0 && dirX != dX && abs(predX) < hyst {
            stepH = 0
        } else if dirX != 0 {
            dX = dirX
        }
        let dirY = stepV == 0 ? 0 : (stepV > 0 ? 1 : -1)
        if dirY != 0 && dirY != dY && abs(predY) < hyst {
            stepV = 0
        } else if dirY != 0 {
            dY = dirY
        }

        return Out(sH: stepH, sV: stepV, eX: errX, eY: errY, vX: vX, vY: vY,
                   rawVX: rawVX, compensatedVX: compensatedVX,
                   predX: predX, predY: predY, outX: dbgOutX, outY: dbgOutY,
                   gainScale: gs, distScale: ds, edgeX: edgeBX, edgeY: edgeBY,
                   leadRoomX: lrX, leadRoomY: lrY)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 轨迹平滑
// ═══════════════════════════════════════════════════════════════

final class Smoother {
    private var r: CGRect?
    let a: CGFloat
    init(_ alpha: CGFloat = 0.6) { a = alpha }

    func run(_ t: CGRect?) -> CGRect? {
        guard let raw = t else { r = nil; return nil }
        guard let p = r else { r = raw; return raw }
        let nr = CGRect(
            x: p.minX + (raw.minX - p.minX) * a,
            y: p.minY + (raw.minY - p.minY) * a,
            width: p.width + (raw.width - p.width) * a,
            height: p.height + (raw.height - p.height) * a
        )
        r = nr
        return nr
    }

    func reset() { r = nil }
}

final class OutputCropSmoother {
    private var last: CGRect?
    private var lastTime: Double?
    private let maxCenterSpeed: CGFloat
    private let maxSizeSpeed: CGFloat

    init(maxCenterSpeed: CGFloat = 0.90, maxSizeSpeed: CGFloat = 0.45) {
        self.maxCenterSpeed = maxCenterSpeed
        self.maxSizeSpeed = maxSizeSpeed
    }

    func run(_ raw: CGRect, t: Double) -> CGRect {
        let target = clamp(raw)
        guard let prev = last, let prevTime = lastTime else {
            last = target
            lastTime = t
            return target
        }

        var dt = t - prevTime
        if !dt.isFinite || dt <= 0 { dt = 1.0 / 60.0 }
        dt = min(max(dt, 1.0 / 240.0), 0.10)

        let maxCenterStep = maxCenterSpeed * CGFloat(dt)
        let dx = target.midX - prev.midX
        let dy = target.midY - prev.midY
        let dist = sqrt(dx * dx + dy * dy)
        let centerScale = dist > maxCenterStep && dist > 0 ? maxCenterStep / dist : 1
        var cx = prev.midX + dx * centerScale
        var cy = prev.midY + dy * centerScale

        let maxSizeStep = maxSizeSpeed * CGFloat(dt)
        let w = prev.width + clamped(target.width - prev.width, limit: maxSizeStep)
        let h = prev.height + clamped(target.height - prev.height, limit: maxSizeStep)

        let halfW = w / 2
        let halfH = h / 2
        cx = min(max(cx, halfW), 1 - halfW)
        cy = min(max(cy, halfH), 1 - halfH)

        let out = clamp(CGRect(x: cx - halfW, y: cy - halfH, width: w, height: h))
        last = out
        lastTime = t
        return out
    }

    func reset() {
        last = nil
        lastTime = nil
    }

    private func clamped(_ value: CGFloat, limit: CGFloat) -> CGFloat {
        min(max(value, -limit), limit)
    }

    private func clamp(_ rect: CGRect) -> CGRect {
        let w = min(max(rect.width, 0.02), 1.0)
        let h = min(max(rect.height, 0.02), 1.0)
        let x = min(max(rect.minX, 0), 1 - w)
        let y = min(max(rect.minY, 0), 1 - h)
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 帧快照 & 统计
// ═══════════════════════════════════════════════════════════════

struct Snap {
    let idx: Int
    let t: Double
    let trackingCrop: CGRect
    let crop: CGRect
    let det: CGRect?
    let state: String
    let kfSt: String
    let detSrc: String
    let yoloMs: Double
    let cmdH: Double
    let cmdV: Double
    let actH: Double
    let actV: Double
    let errH: Double
    let errV: Double
    let velH: Double
    let velV: Double
    let eX: CGFloat
    let eY: CGFloat
    // ★ 变焦状态
    let optZoom: CGFloat            // 光学变焦倍率
    let digZoom: CGFloat            // 数字变焦倍率
    let effFovH: Double             // 等效水平 FOV (度)
    let srcImg: CGImage?
    let cropImg: CGImage?
    let yoloCandidates: [DebugCandidate]   // 本帧 YOLO 候选框（主仿真器显示与检测率统计共用）
    let yoloRanThisFrame: Bool             // 本帧是否执行了 YOLO
    let dbg: DebugInfo?                    // ★ 逐帧 Debug 数据（debug模式开启时填充详细中间量）
}

struct Stats {
    var frames: Int = 0
    // tracking 已移除 — "追踪率(含惯性)"不再使用
    var maxEH: Double = 0
    var maxEV: Double = 0
    var avgEH: Double = 0
    var avgEV: Double = 0
    var maxTD: Double = 0
    var avgTD: Double = 0
    var dirH: Int = 0
    var dirV: Int = 0
    var peakVH: Double = 0
    var peakVV: Double = 0
    var avgYolo: Double = 0

    // ── Ground Truth SAHI 统计 ──
    var gtDetectable: Int = 0     // SAHI 地毯式检测到目标的帧数 (理论最大可追踪)
    var trkOnlyFrames: Int = 0    // VNTracker 纯追踪成功帧数 (confidence > 0.5)
    var yoloHitFrames: Int = 0    // YOLO 命中 GT 可见帧数（任一 YOLO 候选与 GT 的 IoU > 阈值）
    var gtSahiMs: Double = 0      // SAHI 平均耗时 (仅供参考)

    // ── ★ 标注 GT 统计 (对齐虚拟赛道口径) ──
    var gtAnnotationFrames: Int = 0   // 标注 GT 中有框的帧数 (GT 可见帧)
    var trackOnCarFrames: Int = 0     // 输出框与 GT 框 IoU > 阈值的帧 (一般追踪率分子)
    var stLockOnCarFrames: Int = 0    // 单目标锁定成功且与 GT IoU > 阈值的帧 (单目标锁定率分子)
    var stEvalFrames: Int = 0         // 单目标锁定评估总帧 (GT 可见帧中 ST 参与的帧)

    // trackPct 已移除 — "追踪率(含惯性)"不再作为指标
    // ★ 一般追踪率 = trackOnCarFrames / totalFrames (对齐虚拟赛道)
    var generalTrackRate: Double {
        frames > 0 ? Double(trackOnCarFrames) / Double(frames) * 100 : 0
    }
    // ★ 有效追踪率 = trackOnCarFrames / gtAnnotationFrames (以GT可见帧为分母，消除取景框外帧的影响)
    var effectiveTrackRate: Double {
        gtAnnotationFrames > 0 ? Double(trackOnCarFrames) / Double(gtAnnotationFrames) * 100 : 0
    }
    // ★ YOLO 检测率（GT 可见口径） = yoloHitFrames / gtAnnotationFrames
    var yoloDetectRateVisible: Double {
        gtAnnotationFrames > 0 ? Double(yoloHitFrames) / Double(gtAnnotationFrames) * 100 : 0
    }
    // ★ 单目标锁定率 = stLockOnCarFrames / stEvalFrames (对齐虚拟赛道)
    var singleTargetLockRate: Double {
        stEvalFrames > 0 ? Double(stLockOnCarFrames) / Double(stEvalFrames) * 100 : 0
    }
    // ★ GT 标注可见率 = gtAnnotationFrames / totalFrames
    var gtAnnotationVisiblePct: Double {
        frames > 0 ? Double(gtAnnotationFrames) / Double(frames) * 100 : 0
    }
    // ★ 纯 VNTracker 追踪率 (只算 tracker 真正返回的帧)
    var pureTrkPct: Double {
        frames > 0 ? Double(trkOnlyFrames) / Double(frames) * 100 : 0
    }
    // ★ SAHI GT 可见率 (目标在源画面中被检测到的比例)
    var gtVisiblePct: Double {
        frames > 0 ? Double(gtDetectable) / Double(frames) * 100 : 0
    }
    // ★ 追踪效率 = 纯追踪帧 / GT可见帧 (tracker 在目标可见时能跟住多少)
    var trkEfficiency: Double {
        gtDetectable > 0 ? Double(trkOnlyFrames) / Double(gtDetectable) * 100 : 0
    }
    var jitH: Double {
        frames > 0 ? Double(dirH) / Double(frames) * 100 : 0
    }
    var jitV: Double {
        frames > 0 ? Double(dirV) / Double(frames) * 100 : 0
    }
}

struct EvalFrameStatus {
    let frame: Int
    let time: Double
    let gtVisible: Bool
    let yoloHit: Bool
    let trackOnCar: Bool
    let stOnCar: Bool
    let yoloRan: Bool
    let yoloCandidateCount: Int
    let bestYoloIoU: Double
    let bestTrackIoU: Double
    let bestSTIoU: Double
    let gtInCrop: Bool
    let gtCenterInCrop: Bool
    let cropX: Double
    let cropY: Double
    let cropW: Double
    let cropH: Double
    let detSource: String
    let state: String
    let stLostFrames: Int
    let targetLostFrames: Int
    let yoloMs: Double
}

enum EvalOracleMode {
    case off
    case crop
    case source
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 逐帧 Debug 数据结构
// ═══════════════════════════════════════════════════════════════

/// 单个 YOLO 候选框的完整评分详情
struct DebugCandidate {
    let rectTL: CGRect          // 归一化 top-left 坐标
    let label: String           // YOLO 类别名
    let confidence: Float       // YOLO 置信度
    let appSim: Float           // 外观相似度 (gray+HSV 加权)，未计算时 = -1
    let motionSim: Float        // 运动 IoU (与预测位置的重叠)
    let sizeSim: Float          // 尺寸相似度
    let totalScore: Float       // 加权总分 = 0.6*app + 0.25*motion + 0.15*size
    let accepted: Bool          // 是否被接受为当前目标
    let rejectReason: String    // 被拒绝的原因 ("" 表示已接受)
}

/// 一帧完整的 debug 快照 — 所有中间变量及其语义
struct DebugInfo {
    // ── A. 检测层 (YOLO + Tracker) ──
    var yoloCandidates: [DebugCandidate] = []  // 本帧所有 YOLO 候选 (含评分)
    var trackerRect: CGRect? = nil              // VNTracker 原始输出框
    var trackerConf: Float = 0                  // VNTracker 置信度
    var trackerAlive: Bool = false              // VNTracker 是否存活
    var selectedIdx: Int = -1                   // 被选中的候选索引 (-1=无)
    var matchMethod: String = ""                // 匹配方法: "Appearance" / "IoU" / "None" / "ColdStart"
    var yoloRanThisFrame: Bool = false          // 本帧是否执行了 YOLO

    // ── B. 控制层 (PID 中间变量) ──
    var errDegX: CGFloat = 0       // 原始水平误差(度) = (cx-0.5)*hFov (相对画面中心, 不含leadRoom偏移)
    var errDegY: CGFloat = 0       // 原始俯仰误差(度) = (cy-0.5)*vFov (相对画面中心, 不含leadRoom偏移)
    var smoothVelX: CGFloat = 0    // EMA 平滑水平速度(度/s)
    var smoothVelY: CGFloat = 0    // EMA 平滑俯仰速度(度/s)
    var predErrX: CGFloat = 0      // 前瞻预测水平误差(度) = errX + velX*leadTime
    var predErrY: CGFloat = 0      // 前瞻预测俯仰误差(度) = errY + velY*leadTime
    var gainScale: CGFloat = 0     // 距离自适应增益缩放因子 (0.35~1.0)
    var distScale: CGFloat = 0     // 距离缩放 = targetRatio / objectHeight (1.0~3.0)
    var pidOutX: CGFloat = 0       // PID 水平输出(度/s), 负=向右追
    var pidOutY: CGFloat = 0       // PID 俯仰输出(度/s), 正=向下追
    var stepH: CGFloat = 0         // 最终水平步进(度), 正=servoH增大
    var stepV: CGFloat = 0         // 最终俯仰步进(度), 正=servoV增大
    var edgeBoostX: CGFloat = 0    // 水平边缘补偿量
    var edgeBoostY: CGFloat = 0    // 俯仰边缘补偿量
    var leadRoomX: CGFloat = 0     // Lead Room 水平偏移 (归一化, 正=目标点右移)
    var leadRoomY: CGFloat = 0     // Lead Room 垂直偏移 (归一化, 正=目标点下移)

    // ── C. 状态层 ──
    var state: String = ""              // 舵机阶段: TRACK/INERTIA/HOLD/SCAN/RETURN
    var kfState: String = ""            // 卡尔曼状态: TRK/KF1~8/KF-L/IDLE
    var kfCenter: CGPoint? = nil        // 卡尔曼估计位置
    var kfVelocity: CGPoint? = nil      // 卡尔曼估计速度
    var kfUncertainty: CGFloat = 0      // 卡尔曼不确定性半径
    var kfPredictCount: Int = 0         // 连续卡尔曼纯预测帧数
    var kfSize: CGSize? = nil           // ★ 卡尔曼预测尺度
    var useEnhancedAlgo: Bool = false   // ★ 当前是否开启增强算法
    var hasTemplateFixed: Bool = false   // 是否有固定外观模板
    var hasTemplateDyn: Bool = false     // 是否有动态外观模板
    var targetLabel: String? = nil       // 当前目标类别标签
    var targetLostFrames: Int = 0        // 连续丢失帧计数
    var bestAppScore: Float = 0          // 本帧最佳外观匹配得分

    // ── D. 舵机层 ──
    var cmdH: CGFloat = 120     // 指令角度 H
    var cmdV: CGFloat = 120     // 指令角度 V
    var actH: Double = 120      // 实际角度 H (经物理仿真)
    var actV: Double = 120      // 实际角度 V
    var servoErrH: Double = 0   // 舵机跟随误差 H = cmd - act
    var servoErrV: Double = 0   // 舵机跟随误差 V
    var servoVelH: Double = 0   // 舵机实际角速度 H (度/秒)
    var servoVelV: Double = 0   // 舵机实际角速度 V

    // ── E. FOV/Crop ──
    var cropRegion: CGRect = .zero  // 当前裁切区域 (归一化)
    var inertiaVelH: CGFloat = 0    // 惯性水平速度
    var inertiaVelV: CGFloat = 0    // 惯性俯仰速度
    // ── F. 变焦 ──
    var opticalZoom: CGFloat = 1.0  // 光学变焦倍率
    var digitalZoom: CGFloat = 1.0  // 数字变焦倍率
    var effectiveFovH: Double = 60  // 等效水平 FOV (度)
    var trajCurve: [CGPoint] = []   // ★ 预测的时序曲线
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 目标外观特征 (与 iOS TargetFeature 同步)
// ═══════════════════════════════════════════════════════════════

private struct TargetFeature {
    let gray: [Float]       // 16x16 灰度缩略图
    let grayNorm: Float     // L2 范数
    let hsv: [Float]        // HS 直方图 (16*8 bins)
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 模型选项 (与 iOS 同步)
// ═══════════════════════════════════════════════════════════════

struct ModelOption: Identifiable, Hashable {
    let id: String          // resourceName
    let displayName: String
    let resourceName: String
    let url: URL
}

private final class SimYOLOThresholdProvider: MLFeatureProvider {
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
// MARK: - 仿真引擎
// ═══════════════════════════════════════════════════════════════

final class SimEngine: ObservableObject {
    // 子系统
    private var gimbal: CoupledGimbal!
    var fov = FOVMap()  // 内部可见，供 3D 场景概览等使用
    private let pid = PIDCtrl()
    private let smooth = Smoother()
    private let kf = KF()
    private let ci = CIContext(options: [.cacheIntermediates: false])
    // ★ 单目标追踪器 (V7: 支持自主推理 + 协同增强算法)
    private let singleTargetTracker = SingleTargetTracker(useByteTrack: true)
    private var stYoloReq: VNCoreMLRequest?  // ★ ST 专用 YOLO 请求 (独立模型实例, 不与控制 YOLO 共享)
    private var yoloInputSide: CGFloat = 640
    var stStats = STStats()

    // YOLO
    private var yoloReq: VNCoreMLRequest?
    private var trackReq: VNTrackObjectRequest?
    // ★ GT SAHI: 完全独立的模型实例 + 专用队列 (避免与控制 YOLO 互踩)
    private var gtModel: VNCoreMLModel?
    private let gtQueue = DispatchQueue(label: "gt.sahi", qos: .utility)
    private var gtBusy = false               // 防止堆积
    private var gtModelURL: URL?             // 记录模型路径, 用于创建独立实例
    private let statsLock = NSLock()         // _stats 多线程更新时加锁 (process 线程 + gtQueue)

    // 追踪状态
    private var curRect: CGRect?
    private var smRect: CGRect?
    private var lastSeenTS: CFTimeInterval = 0
    private var servoH: CGFloat = Cfg.servoHome
    private var servoV: CGFloat = Cfg.servoHome
    private var lastSendTS: CFTimeInterval = 0
    private var racing = false

    // ★ 变焦仿真
    private let autoZoom = SimAutoZoom()
    private let digitalGimbal = SimDigitalGimbal()
    private let outputCropSmoother = OutputCropSmoother()
    private var opticalZoom: CGFloat = 1.0          // 当前光学变焦倍率
    @Published var enableZoom: Bool = true           // 变焦开关 (UI 可切换)
    private var zoomCfg = ZoomConfig()               // 运行时变焦配置 (UI 可调)
    @Published var useEnhancedAlgo: Bool = true      // ★ 算法切换 (高斯权重+8D尺度)
    /// 取景追速强度：PID 增益倍数。离限幅远但追不上时调大，让指令更猛直至接近限幅。
    @Published var pidGainScale: CGFloat = 1.0
    /// 目标获取方式：false=自动(YOLO)，true=框选后以用户框为初始目标（也可与自动并存，框选优先）
    @Published var useBoxSelection: Bool = false
    /// 标定链路：true=使用 CalibratedFOVMap 取景+PID+变焦，false=沿用原 FOVMap
    @Published var useCalibratedFOV: Bool = true   // 标定与舵机中心(panC/tiltC)已对齐，project/inverseProject 均为舵机角度
    /// 镜头径向畸变 k1（Brown-Conrady）。运动相机约 -0.28；已去畸变源设 0。边缘跟踪不准时可调 -0.4~0
    @Published var distortionK1: Double = 0

    // ── 场景标定 ──
    /// 外部标定 JSON 文件 URL（用户通过 UI 选择）
    @Published var sceneCalibrationURL: URL? = nil
    /// 是否使用外部标定（优先级高于 fromFOVMap 自动构建）
    @Published var useSceneCalibration: Bool = false
    /// 相机俯仰角（度），从标定获取或用户手动设置。正值=向下
    @Published var cameraTiltDeg: Double = 0
    /// 相机高度（米），影响测距精度
    @Published var cameraHeightM: Double = 2.5

    // ★ 标定链路：首帧根据视频分辨率与 fov 参数构建，取景与控制统一几何
    private var videoResolution: CGSize?
    private var lastBuiltDistortionK1: Double?
    private var lastBuiltTilt: Double?
    private var lastBuiltHeight: Double?
    private var lastBuiltUseSceneCal: Bool?
    private var lastBuiltSceneCalURL: URL?
    private var calibratedFOVMap: CalibratedFOVMap?
    private var oneEuroRect: OneEuroRectFilter?
    private let calibratedAutoZoom = CalibratedAutoZoom()
    private let distEstimator = TargetDistanceEstimator()

    // 用户框选注入（与自动获取并存：有框选则优先用框选）
    private let userRectLock = NSLock()
    private var _userSelectedRectTL: CGRect?

    // 丢失状态
    private var searchOn = false
    private var inertiaOn = false
    private var ancH: CGFloat = 120
    private var ancV: CGFloat = 120
    private var iVH: CGFloat = 0
    private var iVV: CGFloat = 0
    private var scanTS: CFTimeInterval = 0
    private var lastVel: CGPoint = .zero

    // 卡尔曼
    private var kfPN = 0
    private var tSz = CGSize(width: 0.15, height: 0.10)

    // YOLO 帧率模拟
    private var yoloBusy = false
    private var lastYoloTS: CFTimeInterval = 0
    private var yoloInterval: Double = 0.04

    // 目标容忍 (与 iOS 同步)
    private var targetLostFrames = 0
    private var trackerJustDied = false        // ★ Tracker 本帧刚死亡，帮助 YOLO 放松匹配

    // YOLO 时间补偿 (赛道模式, 与 iOS 同步)
    private var yoloDispatchServoH: CGFloat = 0
    private var yoloDispatchServoV: CGFloat = 0
    private var yoloDispatchTS: CFTimeInterval = 0

    // 目标外观特征 (与 iOS 同步)
    private var targetTemplateFixed: TargetFeature?   // 首次锁定时的模板（不变）
    private var targetTemplateDyn: TargetFeature?     // 动态更新的模板
    private var targetLabel: String?                  // YOLO 类别标签
    private var latestCroppedPB: CVPixelBuffer?       // 最新裁切帧（供特征提取）
    /// 上一帧裁切图（仅增强算法下保留），用于局部光流
    private var prevCroppedPBForFlow: CVPixelBuffer?
    /// 局部光流得到的速度 (归一化/s)，与位置速度混合
    private var lastFlowVelTL: CGPoint = .zero
    private var lastProcessTime: CFTimeInterval = 0   // 上一帧时间，用于光流 dt

    // 目标运动状态 (与 iOS updateTargetMotion 同步)
    private var lastTargetCenterTL: CGPoint?
    private var lastTargetVelTL: CGPoint = .zero
    private var lastTargetUpdateTS: CFTimeInterval = 0

    private let trajPredictor = TrajectoryPredictor() // ★ 时序曲线预测器

    // ═══ 仿真结论 ═══
    struct SimResult {
        let frames: Int
        let durationSec: Double
        let avgYoloMs: Double
        let avgErrH: Double
        let maxErrH: Double
        let avgErrV: Double
        let maxErrV: Double
        let jitterH: Double
        let jitterV: Double
        let peakVelH: Double
        let peakVelV: Double
        let avgTgtDeg: Double
        let maxTgtDeg: Double
        // ★ GT SAHI 追踪分析
        let gtVisiblePct: Double
        let pureTrkPct: Double
        let trkEfficiency: Double
        // ★ 标注 GT 评测 (对齐虚拟赛道口径)
        let generalTrackRate: Double      // 一般追踪率 = trackOnCarFrames / totalFrames
        let effectiveTrackRate: Double     // 有效追踪率 = trackOnCarFrames / gtAnnotationFrames (消除取景框外帧影响)
        let singleTargetLockRate: Double  // 单目标锁定率 = stLockOnCarFrames / stEvalFrames
        let gtAnnotationVisiblePct: Double // GT 标注可见率
        let hasAnnotationGT: Bool          // 是否有标注 GT 数据
        // 综合评分 0-100
        var score: Int {
            if hasAnnotationGT {
                // ★ 有标注 GT 时用有效追踪率(以GT可见帧为分母)参与评分，消除取景框外帧的影响
                let trk = min(effectiveTrackRate, 100) * 0.30            // 有效追踪率 30%
                let stl = min(singleTargetLockRate, 100) * 0.20        // 单目标锁定率 20%
                let e  = max(0, 100 - avgErrH * 4) * 0.15             // H误差 15%
                let ev = max(0, 100 - avgErrV * 4) * 0.10             // V误差 10%
                let j  = max(0, 100 - (jitterH + jitterV) * 0.8) * 0.15  // 抖动 15%
                let y  = max(0, 100 - (avgYoloMs - 15)) * 0.10        // 推理速度 10%
                return Int(trk + stl + e + ev + j + y)
            } else {
                // 无标注 GT 时用原有评分体系
                let pt = min(pureTrkPct, 100) * 0.25
                let te = min(trkEfficiency, 100) * 0.15
                let e  = max(0, 100 - avgErrH * 4) * 0.20
                let ev = max(0, 100 - avgErrV * 4) * 0.15
                let j  = max(0, 100 - (jitterH + jitterV) * 0.8) * 0.15
                let y  = max(0, 100 - (avgYoloMs - 15)) * 0.10
                return Int(pt + te + e + ev + j + y)
            }
        }
        var grade: String {
            switch score {
            case 85...: return "优秀 ★★★"
            case 70...: return "良好 ★★☆"
            case 55...: return "一般 ★☆☆"
            default:    return "需优化 ☆☆☆"
            }
        }
        var trackAssessment: String {
            // ★ 用 GT 数据给出更精准的诊断
            if trkEfficiency > 85 {
                return "Tracker 续航优秀（效率 \(String(format: "%.0f", trkEfficiency))%），目标可见时几乎不丢失"
            }
            if gtVisiblePct < 50 {
                return "目标大部分时间不可见（GT \(String(format: "%.0f", gtVisiblePct))%），追踪率低非 Tracker 问题"
            }
            if trkEfficiency < 50 && gtVisiblePct > 70 {
                return "Tracker 续航差（效率 \(String(format: "%.0f", trkEfficiency))%），目标可见但频繁丢失，需优化匹配"
            }
            if pureTrkPct < 30 {
                return "VNTracker 几乎无法维持追踪，过度依赖 YOLO 兜底"
            }
            // fallback: 用一般追踪率做兜底诊断
            switch generalTrackRate {
            case 90...: return "追踪非常稳定，全程几乎无丢失"
            case 75...: return "追踪较稳定，偶有短暂丢失"
            case 50...: return "追踪中等，存在多次丢失"
            default:    return "追踪较弱，建议调整 PID 或 YOLO 参数"
            }
        }
        var servoAssessment: String {
            let good = avgErrH < 3 && avgErrV < 3
            let jitBad = jitterH > 25 || jitterV > 25
            if good && !jitBad { return "舵机响应流畅，误差在合理范围内" }
            if jitBad          { return "检测到较高抖动（Jitter），建议降低 KpH/KpV 或增大 minStep" }
            return "舵机误差偏大，建议增大 KpH/KpV 或减少 BLE 延迟"
        }
        var bleAssessment: String {
            if avgYoloMs < 25 { return "BLE 延迟对系统影响较小" }
            return "BLE 延迟可能造成响应滞后，建议降低 Max ms"
        }
    }

    // 发布
    // ═══════════════════════════════════════════════════════════
    // MARK: UI 可观察状态
    // ═══════════════════════════════════════════════════════════
    // ★ 线程安全策略:
    //   - 只有从 UI 控件 (Toggle/Picker) 直接绑定的属性用 @Published
    //   - 其余属性用普通 var，统一由 notifyUI() 在主线程触发更新
    //   - 这彻底消除 "Publishing changes from background threads" 警告

    // ── 需要双向绑定的 (只在主线程写) ──
    @Published var debugMode = false
    @Published var debugInspecting = false
    @Published var enableGtSahi = false         // ★ 默认关闭，避免阻塞帧处理
    @Published var enableSingleTarget = true    // ★ 单目标追踪模式 (V7: 自主推理 + 协同增强) — 默认开启
    @Published var bundledModels: [ModelOption] = []
    @Published var selectedModelId: String = ""
    var evalOracleMode: EvalOracleMode = .off

    // ★ 标注 GT 导入 (真实视频评测用)
    var annotationProjectURL: URL?
    var annotationProject: AnnProject?
    /// 加载标注 JSON 文件，返回是否成功
    func loadAnnotationJSON(_ url: URL) -> Bool {
        guard let data = try? Data(contentsOf: url),
              let proj = try? JSONDecoder().decode(AnnProject.self, from: data) else {
            status = "⚠ 标注 JSON 加载失败"
            return false
        }
        annotationProject = proj
        annotationProjectURL = url
        let annotatedCount = proj.frames.values.filter({ !$0.isEmpty }).count
        status = "✅ 标注已加载: \(annotatedCount)/\(proj.totalFrames) 帧有框"
        return true
    }
    // ★ 自动查找标注文件 (按优先级搜索多个候选路径)
    @discardableResult
    func autoFindAndLoadAnnotation(for videoURL: URL) -> Bool {
        // 如果已经加载了标注，跳过
        if annotationProject != nil { return true }

        let fm = FileManager.default
        let videoName = videoURL.deletingPathExtension().lastPathComponent
        let videoDir = videoURL.deletingLastPathComponent()

        // ── 候选路径列表 (按优先级) ──
        var candidates: [URL] = []

        // 1) 侧车文件: 优先修订版，其次原始版
        candidates.append(
            videoDir.appendingPathComponent("\(videoName).corrected.annotations.json")
        )
        candidates.append(
            videoDir.appendingPathComponent("\(videoName).annotations.json")
        )

        // 2) SAM3 输出目录 (从 annotate_config.json 读取)
        let configPaths = [
            Bundle.main.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent("annotate_config.json").path,
            NSHomeDirectory() + "/Documents/MacRCSimulator/annotate_config.json"
        ]
        var sam3OutputDir: String?
        for cp in configPaths {
            if fm.fileExists(atPath: cp),
               let data = try? Data(contentsOf: URL(fileURLWithPath: cp)),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let od = json["output_dir"] as? String {
                sam3OutputDir = od
                break
            }
        }
        let sam3Base = sam3OutputDir ?? (NSHomeDirectory() + "/Desktop/SAM3_Annotations")

        // 2a) SAM3_Annotations/视频名/优先 corrected 修订版，再兼容 annotator / SAM3 原始命名
        let sam3Dir = URL(fileURLWithPath: sam3Base)
            .appendingPathComponent(videoName)
        candidates.append(
            sam3Dir.appendingPathComponent("\(videoName).corrected.annotations.json")
        )
        candidates.append(
            sam3Dir.appendingPathComponent("\(videoName).annotations.json")
        )
        candidates.append(
            sam3Dir.appendingPathComponent("\(videoName)_annotations.json")
        )

        // 3) 项目内置 SAM3_Annotations 目录 (Bundle 旁)
        let bundleSAM3Dir = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent("SAM3_Annotations")
            .appendingPathComponent(videoName)
        candidates.append(bundleSAM3Dir.appendingPathComponent("\(videoName).corrected.annotations.json"))
        candidates.append(bundleSAM3Dir.appendingPathComponent("\(videoName).annotations.json"))
        candidates.append(bundleSAM3Dir.appendingPathComponent("\(videoName)_annotations.json"))

        // 4) 视频同目录下 legacy 文件名 (无子文件夹)
        candidates.append(
            videoDir.appendingPathComponent("\(videoName)_annotations.json")
        )

        // ── 按候选顺序尝试加载 ──
        for url in candidates {
            if fm.fileExists(atPath: url.path) {
                if loadAnnotationJSON(url) {
                    print("[AutoAnnotation] ✅ 自动加载标注: \(url.path)")
                    return true
                }
            }
        }

        print("[AutoAnnotation] ℹ️ 未找到视频 \"\(videoName)\" 的标注文件")
        return false
    }

    /// 查询指定帧的 GT 框 (归一化坐标)
    func gtBoxes(forFrame idx: Int) -> [AnnBox] {
        annotationProject?.boxes(forLocalFrame: idx) ?? []
    }
    /// 查询指定帧的 GT 像素框 (CGRect)
    func gtRects(forFrame idx: Int, imgW: CGFloat, imgH: CGFloat) -> [CGRect] {
        gtBoxes(forFrame: idx).map { $0.pixelRect(imgW: imgW, imgH: imgH) }
    }

    // ── 只读展示 / 由引擎驱动的 (可能从后台间接触发) ──
    var running = false
    var paused = false
    var progress: Double = 0
    var snap: Snap?
    var stats = Stats()
    var status = "Ready"
    /// 当前输入视频的帧率 (fps)，在 start() 加载视频轨道后设置；0 表示未知或可变帧率
    var videoFrameRate: Float = 0
    var simResult: SimResult?
    var debugFrameIdx: Int = 0

    // ★ 线程安全: 后台线程读写，不触发 UI
    var debugStepReq = false
    var lastDebugInfo: DebugInfo?
    private var _stats = Stats()

    /// ★ 集中通知 UI 刷新 — 必须在主线程调用
    func notifyUI() {
        assert(Thread.isMainThread, "notifyUI must be called on main thread")
        objectWillChange.send()
    }

    // 舵机轨迹 (最近 600 帧 ≈ 10s@60fps, 供曲线图和3D尾迹)
    struct ServoSample {
        let t: Double
        let cmdH: Double; let cmdV: Double
        let actH: Double; let actV: Double
        let velH: Double; let velV: Double   // 舵机实际角速度 °/s，用于曲线图饱和线
        let state: String
    }
    // ★ 不用 @Published — 手动在 snap 更新时一起刷新，避免双重触发
    var servoTrail: [ServoSample] = []
    private var trailBuf: [ServoSample] = []   // 后台线程写
    private let trailMax = 600
    private var lastUITime: CFTimeInterval = 0  // UI 节流

    /// RC 小车估算的世界坐标轨迹 (供 3D 场景使用)
    var carWorldTrail: [(x: Double, z: Double)] = []
    private var carWorldTrailBuf: [(x: Double, z: Double)] = []

    private(set) var history: [Snap] = []
    private var evalFrameHistory: [EvalFrameStatus] = []
    private var fIdx = 0
    /// Processing reader order -> PTS/display order. Built with the same pixel-buffer output as the main reader.
    private var readOrderToPtsIndexMap: [Int: Int] = [:]
    private var ldH = 0
    private var ldV = 0
    private var cancel = false
    private let q = DispatchQueue(label: "sim", qos: .userInitiated)

    init() {
        scanBundledModels()
    }

    // MARK: 扫描 Bundle 中的模型 (与 iOS setupVision 同步)
    func scanBundledModels() {
        let candidateNames = ["yolo11_640imgsz", "Detector"]   // ★ 默认改为 YOLO11-640 优先
        var models: [ModelOption] = []
        for name in candidateNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                let display: String
                if name.contains("640imgsz") {
                    display = "YOLO11-640"
                } else {
                    display = name
                }
                models.append(ModelOption(id: name, displayName: display, resourceName: name, url: url))
            }
        }
        bundledModels = models
        if let first = models.first {
            selectedModelId = first.id
            _ = loadModel(first.url)
        }
    }

    // MARK: 选择内置模型
    func selectBundledModel(_ id: String) {
        guard let m = bundledModels.first(where: { $0.id == id }) else { return }
        selectedModelId = id
        _ = loadModel(m.url)
    }

    // MARK: 加载模型
    func loadModel(_ url: URL) -> Bool {
        let c = MLModelConfiguration()
        c.computeUnits = .all                   // ★ GPU + ANE
        let modelURL: URL
        do {
            let ext = url.pathExtension.lowercased()
            modelURL = (ext == "mlpackage" || ext == "mlmodel")
                ? try MLModel.compileModel(at: url)
                : url
        } catch {
            status = "Model compile failed"; notifyUI()
            return false
        }
        guard let mlm = try? MLModel(contentsOf: modelURL, configuration: c),
              let vnm = try? VNCoreMLModel(for: mlm) else {
            status = "Model load failed"; notifyUI()
            return false
        }
        if let imageInput = mlm.modelDescription.inputDescriptionsByName.values.first(where: { $0.type == .image }),
           let pixelsWide = imageInput.imageConstraint?.pixelsWide,
           pixelsWide > 0 {
            yoloInputSide = CGFloat(pixelsWide)
        } else {
            yoloInputSide = url.lastPathComponent.contains("960") ? 960 : 640
        }
        let inputNames = Set(mlm.modelDescription.inputDescriptionsByName.keys)
        if inputNames.contains("confidenceThreshold") || inputNames.contains("iouThreshold") {
            vnm.featureProvider = SimYOLOThresholdProvider(conf: 0.01, iou: 0.45)
        }
        let r = VNCoreMLRequest(model: vnm)
        r.imageCropAndScaleOption = .scaleFill
        yoloReq = r

        // ★ GT SAHI: 完全独立的 MLModel 实例 → 独立 VNCoreMLModel
        //   这样两条推理管线不共享任何状态, 可以安全并发
        gtModelURL = modelURL
        let gc = MLModelConfiguration()
        gc.computeUnits = .cpuAndNeuralEngine   // ★ ANE 优先, 不抢 GPU 给渲染
        if let gmlm = try? MLModel(contentsOf: modelURL, configuration: gc),
           let gvnm = try? VNCoreMLModel(for: gmlm) {
            if inputNames.contains("confidenceThreshold") || inputNames.contains("iouThreshold") {
                gvnm.featureProvider = SimYOLOThresholdProvider(conf: 0.01, iou: 0.45)
            }
            gtModel = gvnm
        }

        // ★ SingleTargetTracker 专用模型实例 — 独立推理管线
        //   使用真实视频训练的检测模型, 让 ST 在主 YOLO 未触发的帧也能自主获取候选框
        let stMLConfig = MLModelConfiguration()
        stMLConfig.computeUnits = .all
        if let stMLModel = try? MLModel(contentsOf: modelURL, configuration: stMLConfig),
           let stVNModel = try? VNCoreMLModel(for: stMLModel) {
            if inputNames.contains("confidenceThreshold") || inputNames.contains("iouThreshold") {
                stVNModel.featureProvider = SimYOLOThresholdProvider(conf: 0.01, iou: 0.45)
            }
            let stReq = VNCoreMLRequest(model: stVNModel)
            stReq.imageCropAndScaleOption = .scaleFill
            stYoloReq = stReq
            // 设置检测提供器: ST 可自主调用真实检测模型推理
            singleTargetTracker.setDetectionProvider { [weak self] pb in
                guard let self = self, let req = self.stYoloReq else { return [] }
                let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
                try? handler.perform([req])
                return self.detections(from: req.results, confThreshold: Cfg.rYoloConf)
            }
        }

        status = "Model: \(url.lastPathComponent)"; notifyUI()
        // ★ 后台预热 YOLO：首次推理时 CoreML 会编译/加载，导致首帧卡死；提前跑一次 dummy 推理
        q.async { [weak self] in self?.prewarmYOLO() }
        return true
    }

    /// 在后台队列执行一次轻量推理，预热 CoreML/ANE，避免 Run 后首帧卡死
    private func prewarmYOLO() {
        guard let req = yoloReq else { return }
        let w = 320, h = 320
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferWidthKey: w,
            kCVPixelBufferHeightKey: h,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ]
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let buf = pb else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: buf, orientation: .up, options: [:])
        try? handler.perform([req])
        // ★ 同步预热 ST 专用模型
        if let stReq = stYoloReq {
            try? handler.perform([stReq])
        }
    }

    private func detections(from observations: [VNObservation]?, confThreshold: Float) -> [(rect: CGRect, conf: Float, label: String)] {
        guard let observations = observations else { return [] }
        if let recognized = observations as? [VNRecognizedObjectObservation] {
            return recognized.compactMap { obs in
                guard let lbl = obs.labels.first, obs.confidence > confThreshold else { return nil }
                let r = obs.boundingBox
                return (
                    rect: CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height),
                    conf: obs.confidence,
                    label: lbl.identifier
                )
            }
        }

        var parsed: [(rect: CGRect, conf: Float, label: String)] = []
        for observation in observations {
            guard let feature = observation as? VNCoreMLFeatureValueObservation,
                  let array = feature.featureValue.multiArrayValue else { continue }
            let shape = array.shape.map { $0.intValue }
            let rows: Int
            let cols: Int
            let rowMajor: Bool
            if shape.count >= 3, shape[shape.count - 1] >= 6 {
                rows = shape[shape.count - 2]
                cols = shape[shape.count - 1]
                rowMajor = true
            } else if shape.count >= 3, shape[shape.count - 2] >= 6 {
                rows = shape[shape.count - 1]
                cols = shape[shape.count - 2]
                rowMajor = false
            } else if shape.count == 2, shape[1] >= 6 {
                rows = shape[0]
                cols = shape[1]
                rowMajor = true
            } else if shape.count == 2, shape[0] >= 6 {
                rows = shape[1]
                cols = shape[0]
                rowMajor = false
            } else {
                continue
            }

            func value(row: Int, col: Int) -> Double {
                let idx = rowMajor ? row * cols + col : col * rows + row
                guard idx >= 0 && idx < array.count else { return 0 }
                return array[idx].doubleValue
            }

            for row in 0..<rows {
                let rawConf = Float(value(row: row, col: 4))
                guard rawConf > confThreshold else { continue }
                var x1 = CGFloat(value(row: row, col: 0))
                var y1 = CGFloat(value(row: row, col: 1))
                var x2 = CGFloat(value(row: row, col: 2))
                var y2 = CGFloat(value(row: row, col: 3))
                if x2 < x1 || y2 < y1 {
                    let cx = x1, cy = y1, w = max(0, x2), h = max(0, y2)
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                }
                let maxCoord = max(abs(x1), abs(y1), abs(x2), abs(y2))
                if maxCoord > 1.5 {
                    let side = max(yoloInputSide, 1)
                    x1 /= side
                    x2 /= side
                    y1 /= side
                    y2 /= side
                }
                x1 = min(max(x1, 0), 1)
                y1 = min(max(y1, 0), 1)
                x2 = min(max(x2, 0), 1)
                y2 = min(max(y2, 0), 1)
                let rect = CGRect(x: x1, y: y1, width: max(0, x2 - x1), height: max(0, y2 - y1))
                guard rect.width > 0.001, rect.height > 0.001 else { continue }
                let cls = cols > 5 ? Int(value(row: row, col: 5)) : 0
                parsed.append((rect: rect, conf: rawConf, label: "\(cls)"))
            }
        }
        return parsed
    }

    // MARK: 启动
    func start(videoURL: URL, racing: Bool, speed: Double,
               ble: BLEConfig, pan: ServoPhys, tilt: ServoPhys,
               coup: CouplingCfg, fovMap: FOVMap, zoom: ZoomConfig) {

        // 主线程重置状态
        self.racing = racing
        self.fov = fovMap
        self.zoomCfg = zoom

        pid.racing = racing
        pid.reset(0)
        smooth.reset()
        kf.reset()
        curRect = nil; smRect = nil
        servoH = 120; servoV = 120
        lastSendTS = 0; lastSeenTS = 0
        searchOn = false; inertiaOn = false
        lastVel = .zero; kfPN = 0
        tSz = CGSize(width: 0.15, height: 0.10)
        trackReq = nil; stats = Stats(); _stats = Stats()
        history.removeAll()
        evalFrameHistory.removeAll()
        fIdx = 0; readOrderToPtsIndexMap = [:]; ldH = 0; ldV = 0
        cancel = false; yoloBusy = false; lastYoloTS = 0
        targetLostFrames = 0; trackerJustDied = false
        yoloDispatchTS = 0; yoloDispatchServoH = 0; yoloDispatchServoV = 0
        targetTemplateFixed = nil; targetTemplateDyn = nil; targetLabel = nil
        latestCroppedPB = nil; prevCroppedPBForFlow = nil; lastFlowVelTL = .zero; lastProcessTime = 0
        singleTargetTracker.reset(); stStats = STStats()  // ★ 单目标追踪重置
        lastTargetCenterTL = nil; lastTargetVelTL = .zero; lastTargetUpdateTS = 0
        trajPredictor.reset()
        servoTrail = []; trailBuf = []; lastUITime = 0
        carWorldTrail = []; carWorldTrailBuf = []
        // ★ 变焦重置
        autoZoom.reset(); digitalGimbal.reset(); outputCropSmoother.reset(); opticalZoom = zoomCfg.opticalMin
        calibratedAutoZoom.reset()
        videoResolution = nil
        calibratedFOVMap = nil
        lastBuiltTilt = nil
        lastBuiltHeight = nil
        lastBuiltUseSceneCal = nil
        lastBuiltSceneCalURL = nil
        oneEuroRect = nil
        simResult = nil
        debugStepReq = false          // ★ 确保 step 请求干净
        running = true; paused = false
        status = "Loading video…"
        notifyUI()                    // ★ 通知 UI 状态已重置

        // ★ 自动查找并加载标注 (如果尚未手动指定)
        autoFindAndLoadAnnotation(for: videoURL)

        let spd = speed
        let bleC = ble, panC = pan, tiltC = tilt, coupC = coup

        // ★ 后台队列处理 — 使用现代 async API 替代已废弃的同步方法
        q.async { [weak self] in
            guard let self = self else { return }

            // ── 1. Security-Scoped Resource ──
            let scoped = videoURL.startAccessingSecurityScopedResource()
            defer { if scoped { videoURL.stopAccessingSecurityScopedResource() } }

            // 先检查文件可达
            guard FileManager.default.isReadableFile(atPath: videoURL.path) else {
                DispatchQueue.main.async {
                    self.running = false
                    self.status = "⚠ File not readable – check sandbox entitlements"
                    self.notifyUI()
                }
                return
            }

            let asset = AVAsset(url: videoURL)

            // ── 2. 加载视频轨道 ──
            // ★ 修复卡死: Task{} + sem.wait() 在 Swift concurrency 下可能死锁
            //   改用 Task.detached 避免继承 actor 上下文 + 加超时兜底
            var videoTrack: AVAssetTrack?
            var dur: Double = 0
            var loadedFPS: Float = 0
            var loadError: String?

            let sem = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                do {
                    let tracks = try await asset.loadTracks(withMediaType: .video)
                    videoTrack = tracks.first
                    if let vt = tracks.first {
                        loadedFPS = vt.nominalFrameRate
                    }
                    let cmDur = try await asset.load(.duration)
                    dur = CMTimeGetSeconds(cmDur)
                } catch {
                    loadError = error.localizedDescription
                }
                sem.signal()
            }
            let waitResult = sem.wait(timeout: .now() + 10.0)
            if waitResult == .timedOut {
                loadError = "Asset load timed out — 视频加载超时"
            }

            guard let vt = videoTrack else {
                let msg = loadError ?? "No video track – try .mp4 / .mov format"
                DispatchQueue.main.async {
                    self.running = false
                    self.status = "⚠ \(msg)"
                    self.notifyUI()
                }
                return
            }

            // ── 2.5 保存视频帧率供 UI 显示 ──
            let fpsToShow = loadedFPS
            DispatchQueue.main.async {
                self.videoFrameRate = fpsToShow
                self.notifyUI()
            }

            // ── 3. 初始化舵机仿真 ──
            self.gimbal = CoupledGimbal(home: 120, pan: panC, tilt: tiltC, ble: bleC, coup: coupC)
            DispatchQueue.main.async { self.status = "Running…"; self.notifyUI() }

            let processingOutputSettings: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]

            // ── 3.5 构建处理读序→PTS 帧序映射 ──
            // AVAssetReader 的 compressed sample 顺序可能是 DTS，pixel-buffer 输出通常已经是 PTS。
            // 这里必须用和主处理 reader 相同的输出设置，否则 B 帧视频会把标注帧号再次打乱。
            do {
                let preReader = try AVAssetReader(asset: asset)
                let preOut = AVAssetReaderTrackOutput(track: vt, outputSettings: processingOutputSettings)
                preOut.alwaysCopiesSampleData = false
                if preReader.canAdd(preOut) {
                    preReader.add(preOut)
                    if preReader.startReading() {
                        var readOrderPTS: [CMTime] = []
                        while let sb = preOut.copyNextSampleBuffer() {
                            readOrderPTS.append(CMSampleBufferGetPresentationTimeStamp(sb))
                        }
                        preReader.cancelReading()
                        // 按 PTS 排序后建立映射: processing read index -> pts/display index
                        let indexed = readOrderPTS.enumerated().map { ($0.offset, $0.element) }
                        let sorted = indexed.sorted { CMTimeCompare($0.1, $1.1) < 0 }
                        var map: [Int: Int] = [:]
                        for (ptsIdx, pair) in sorted.enumerated() {
                            map[pair.0] = ptsIdx
                        }
                        self.readOrderToPtsIndexMap = map
                        let movedCount = map.filter { $0.key != $0.value }.count
                        let orderNote = movedCount > 0 ? "(处理读序重排 \(movedCount) 帧)" : "(处理读序已是 PTS)"
                        print("[SimEngine] ★ PTS 帧序映射已建立: \(map.count) 帧 \(orderNote)")
                    } else {
                        preReader.cancelReading()
                    }
                }
            } catch {
                print("[SimEngine] ⚠ PTS 映射构建失败: \(error), 将使用处理读序")
            }

            // ── 4. AVAssetReader ──
            guard let reader = try? AVAssetReader(asset: asset) else {
                DispatchQueue.main.async { self.running = false; self.status = "⚠ AVAssetReader init failed"; self.notifyUI() }
                return
            }
            let out = AVAssetReaderTrackOutput(
                track: vt,
                outputSettings: processingOutputSettings
            )
            out.alwaysCopiesSampleData = false
            guard reader.canAdd(out) else {
                DispatchQueue.main.async { self.running = false; self.status = "⚠ Cannot add reader output"; self.notifyUI() }
                return
            }
            reader.add(out)
            guard reader.startReading() else {
                let err = reader.error?.localizedDescription ?? "unknown"
                DispatchQueue.main.async { self.running = false; self.status = "⚠ Read failed: \(err)"; self.notifyUI() }
                return
            }

            // ── 5. 逐帧处理 ──
            while !self.cancel {
                var shouldBreak = false
                autoreleasepool {
                // ★ Debug 模式: 逐帧暂停，等待 step 请求
                //   也响应 debugMode 被关闭 / cancel 来防止卡死
                if self.debugMode && self.fIdx > 0 {
                    DispatchQueue.main.async { self.paused = true; self.notifyUI() }
                    while !self.debugStepReq && !self.cancel && self.debugMode {
                        Thread.sleep(forTimeInterval: 0.02)
                    }
                    self.debugStepReq = false
                    // ★ debugMode 被关闭 → 自动退出暂停继续运行
                    if !self.debugMode {
                        DispatchQueue.main.async { self.paused = false; self.notifyUI() }
                    }
                }

                while self.paused && !self.debugStepReq && !self.cancel {
                    Thread.sleep(forTimeInterval: 0.02)
                }
                if self.debugStepReq {
                    self.debugStepReq = false
                    DispatchQueue.main.async { self.paused = true; self.notifyUI() }
                }

                guard let sb = out.copyNextSampleBuffer(),
                      let pb = CMSampleBufferGetImageBuffer(sb) else { shouldBreak = true; return }
                let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb))

                let s = self.process(pb: pb, time: pts)

                // 后台线程积累舵机轨迹
                let sample = SimEngine.ServoSample(
                    t: s.t, cmdH: s.cmdH, cmdV: s.cmdV,
                    actH: s.actH, actV: s.actV, velH: s.velH, velV: s.velV, state: s.state
                )
                self.trailBuf.append(sample)
                if self.trailBuf.count > self.trailMax {
                    self.trailBuf.removeFirst(self.trailBuf.count - self.trailMax)
                }

                // ★ UI 节流: ~15fps，不是每帧都刷 (debug模式下每帧刷)
                let now = CACurrentMediaTime()
                if self.debugMode || now - self.lastUITime >= 0.066 {
                    self.lastUITime = now
                    let p = dur > 0 ? pts / dur : 0
                    let trailCopy = self.trailBuf  // 快照
                    self.statsLock.lock()
                    let statsCopy = self._stats    // ★ 快照统计（与 GT 异步更新一致）
                    self.statsLock.unlock()
                    DispatchQueue.main.async {
                        self.snap = s
                        self.progress = min(1, p)
                        self.servoTrail = trailCopy
                        self.carWorldTrail = self.carWorldTrailBuf
                        self.stats = statsCopy
                        self.debugFrameIdx = self.history.count - 1
                        self.notifyUI()
                    }
                }
                self.fIdx += 1
                if spd > 0 {
                    Thread.sleep(forTimeInterval: 1.0 / 60.0 / spd)
                }
                }  // autoreleasepool 结束
                if shouldBreak { break }
            }
            reader.cancelReading()
            self.statsLock.lock()
            let finalStats = self._stats   // ★ 从后台线程捕获最终统计
            self.statsLock.unlock()
            DispatchQueue.main.async {
                self.running = false
                self.stats = finalStats
                let st = finalStats
                let hasAnnGT = self.annotationProject != nil
                let result = SimEngine.SimResult(
                    frames: st.frames,
                    durationSec: dur,
                    avgYoloMs: st.avgYolo,
                    avgErrH: st.avgEH,
                    maxErrH: st.maxEH,
                    avgErrV: st.avgEV,
                    maxErrV: st.maxEV,
                    jitterH: st.jitH,
                    jitterV: st.jitV,
                    peakVelH: st.peakVH,
                    peakVelV: st.peakVV,
                    avgTgtDeg: st.avgTD,
                    maxTgtDeg: st.maxTD,
                    gtVisiblePct: st.gtVisiblePct,
                    pureTrkPct: st.pureTrkPct,
                    trkEfficiency: st.trkEfficiency,
                    generalTrackRate: st.generalTrackRate,
                    effectiveTrackRate: st.effectiveTrackRate,
                    singleTargetLockRate: st.singleTargetLockRate,
                    gtAnnotationVisiblePct: st.gtAnnotationVisiblePct,
                    hasAnnotationGT: hasAnnGT
                )
                self.simResult = result
                if hasAnnGT {
                    self.status = String(format: "Done %d帧 | 有效追踪率 %.1f%% | 单目标锁定率 %.1f%% | GT可见 %.1f%%",
                        st.frames, st.effectiveTrackRate, st.singleTargetLockRate, st.gtAnnotationVisiblePct)
                } else {
                    self.status = String(format: "Done %d frames | VNTrk %.1f%% | GT可见 %.1f%% | 效率 %.1f%%",
                        st.frames, st.pureTrkPct, st.gtVisiblePct, st.trkEfficiency)
                }
                self.notifyUI()
            }
        }
    }

    func stop() { cancel = true }
    func togglePause() { paused.toggle(); notifyUI() }
    /// 外部标定文件变更时调用，触发下一帧重建 CalibratedFOVMap
    func invalidateSceneCalibration() {
        calibratedFOVMap = nil
        lastBuiltSceneCalURL = nil
    }
    func stepForward() {
        debugStepReq = true
        paused = false
        notifyUI()
    }
    func jumpToFrame(_ idx: Int) {
        let clamped = max(0, min(idx, history.count - 1))
        debugFrameIdx = clamped
        if clamped < history.count {
            snap = history[clamped]
            notifyUI()
        }
    }

    // MARK: 核心闭环
    private func process(pb: CVPixelBuffer, time: Double) -> Snap {
        gimbal.advanceTo(time)

        // ★ 标定链路: 首帧根据视频分辨率与 fov 参数构建 CalibratedFOVMap；k1 变化时重建
        if videoResolution == nil {
            let w = CGFloat(CVPixelBufferGetWidth(pb))
            let h = CGFloat(CVPixelBufferGetHeight(pb))
            videoResolution = CGSize(width: w, height: h)
        }
        if useCalibratedFOV, let res = videoResolution {
            let k1Changed = lastBuiltDistortionK1 == nil || abs(distortionK1 - lastBuiltDistortionK1!) > 1e-6
            let tiltChanged = lastBuiltTilt == nil || abs(cameraTiltDeg - lastBuiltTilt!) > 1e-6
            let heightChanged = lastBuiltHeight == nil || abs(cameraHeightM - lastBuiltHeight!) > 1e-6
            let sceneCalChanged = lastBuiltUseSceneCal != useSceneCalibration || lastBuiltSceneCalURL != sceneCalibrationURL

            if calibratedFOVMap == nil || k1Changed || tiltChanged || heightChanged || sceneCalChanged {
                let cal: CameraCalibration

                if cameraTiltDeg != 0 {
                    // ★ 使用用户设置的仰俯角 + 快速标定
                    cal = CameraCalibration.quickCalibrate(
                        videoResolution: res,
                        sourceFovH: fov.camH,
                        cameraTiltDeg: cameraTiltDeg,
                        cameraHeightM: cameraHeightM,
                        k1: distortionK1
                    )
                } else {
                    // ★ 后备: 从 FOVMap 参数反推（与现有行为一致）
                    cal = CameraCalibration.fromFOVMap(
                        resolution: res,
                        srcH: fov.srcH, srcV: fov.srcV,
                        panC: fov.panC, tiltC: fov.tiltC,
                        camH: fov.camH, camV: fov.camV,
                        k1: distortionK1, k2: 0, k3: 0
                    )
                }

                calibratedFOVMap = CalibratedFOVMap(calibration: cal)
                lastBuiltDistortionK1 = distortionK1
                lastBuiltTilt = cameraTiltDeg
                lastBuiltHeight = cameraHeightM
                lastBuiltUseSceneCal = useSceneCalibration
                lastBuiltSceneCalURL = sceneCalibrationURL
                if oneEuroRect == nil { oneEuroRect = OneEuroRectFilter() }
            }
        }

        // ★ 变焦感知裁切: 标定链路用 project()，否则用 FOVMap.crop
        let curZoom = enableZoom ? opticalZoom : 1.0
        let rawCropRgn: CGRect
        if useCalibratedFOV, let cf = calibratedFOVMap {
            let raw = cf.project(panDegrees: gimbal.panPos, tiltDegrees: gimbal.tiltPos, zoom: Double(curZoom))
            // ★ 安全钳位: 防止标定投影返回越界或过小的裁切区域
            let fallbackCrop = fov.crop(pan: gimbal.panPos, tilt: gimbal.tiltPos, opticalZoom: curZoom)
            let minW: CGFloat = fallbackCrop.width * 0.3
            let minH: CGFloat = fallbackCrop.height * 0.3
            if raw.width < minW || raw.height < minH || raw.minX < -0.5 || raw.minY < -0.5
                || raw.maxX > 1.5 || raw.maxY > 1.5 {
                rawCropRgn = fallbackCrop
            } else {
                let cw = max(raw.width, minW)
                let ch = max(raw.height, minH)
                let cx = min(max(raw.minX, 0), max(1 - cw, 0))
                let cy = min(max(raw.minY, 0), max(1 - ch, 0))
                rawCropRgn = CGRect(x: cx, y: cy, width: min(cw, 1 - cx), height: min(ch, 1 - cy))
            }
        } else {
            rawCropRgn = fov.crop(pan: gimbal.panPos, tilt: gimbal.tiltPos, opticalZoom: curZoom)
        }
        let trackingCropRgn = rawCropRgn
        let outputCropRgn = outputCropSmoother.run(rawCropRgn, t: time)
        let croppedPB = cropPB(pb, trackingCropRgn)
        let usePB = croppedPB ?? pb
        let outputPB = cropPB(pb, outputCropRgn) ?? usePB
        let displayFrameIndex = readOrderToPtsIndexMap[fIdx] ?? fIdx

        // ★ 用户框选：若本帧有注入的框，则以此为初始目标（与自动获取并存，框选优先）
        if let userRect = consumeUserSelectedRect() {
            applyUserSelectedTarget(rectTL: userRect, pb: pb, time: time)
        }

        // ★ 局部光流已关闭 — 纯位置差分 + 卡尔曼速度估计足够，光流引入滞后
        lastFlowVelTL = .zero

        let isLost = time - lastSeenTS > Cfg.searchCooldown
        var detected: CGRect? = nil
        var detSrc = "NONE"
        let handler = VNImageRequestHandler(cvPixelBuffer: usePB, orientation: .up, options: [:])
        let oracleRect = oracleTargetRect(frameIndex: displayFrameIndex, crop: trackingCropRgn)
        let oracleActive = evalOracleMode != .off

        // VNTracker
        var trkConf: Float = 0
        var trkAlive = false
        trackerJustDied = false   // ★ 每帧重置
        if let oracleRect {
            detected = oracleRect
            detSrc = evalOracleMode == .source ? "GT_ORACLE_SRC" : "GT_ORACLE_CROP"
            trackReq = nil
            trackerJustDied = false
        } else if let req = trackReq, !oracleActive {
            try? handler.perform([req])
            if let res = req.results?.first as? VNDetectedObjectObservation,
               res.confidence > 0.5 {
                let r = res.boundingBox
                detected = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                detSrc = "TRK"
                trkConf = res.confidence
                trkAlive = true
            } else {
                // ★ Tracker 死亡: 立即清除 trackReq，标记 trackerJustDied
                //   让后续 YOLO 可以放松匹配条件进行重新捕获
                trackReq = nil
                trackerJustDied = true
                // ★ 赛道模式: 不立即清除 curRect (让 KF 接管)
                // ★ 普通模式: 也不立即清除 (让 YOLO 有 IoU 参考)
            }
        }

        if let r = detected {
            curRect = r
            lastSeenTS = time
            targetLostFrames = 0
            updateTargetMotion(rectTL: r, now: time)
        } else if isLost {
            curRect = nil
        }

        // 保存最新裁切帧供外观特征提取
        latestCroppedPB = usePB

        // ★ 局部光流已关闭，不再复制帧缓冲（节省内存和 CPU）
        prevCroppedPBForFlow = nil
        lastProcessTime = time

        // YOLO（交互模式模拟 iPhone 20-40fps 波动；评测模式必须确定性）
        var yoloMs: Double = 0
        var yoloRanThisFrame = false
        if CommandLine.arguments.contains("--eval") {
            yoloInterval = 1.0 / Cfg.evalYoloFPS
        } else if Int(time * 10) % 3 == 0 {
            yoloInterval = 1.0 / Double.random(in: 20...40)
        }
        // ★ Tracker 刚死亡时立即触发 YOLO，不等待间隔
        let forceYolo = trackerJustDied && !yoloBusy
        if !oracleActive && !yoloBusy && (forceYolo || (time - lastYoloTS) >= yoloInterval) {
            yoloBusy = true
            yoloRanThisFrame = true
            // 记录 YOLO 发起时的时间和舵机状态 (与 iOS 同步)
            yoloDispatchTS = time
            yoloDispatchServoH = servoH
            yoloDispatchServoV = servoV
            let t0 = CACurrentMediaTime()
            doYOLO(handler: handler, time: time)
            yoloMs = (CACurrentMediaTime() - t0) * 1000
            lastYoloTS = time
            yoloBusy = false
            if detected == nil && curRect != nil {
                detSrc = "YOLO"
            }
        }

        // ★ 单目标追踪 (V7 + 协同增强 + 自主推理)
        if oracleActive {
            stStats.stTotalFrames += 1
            if let oracleRect {
                stStats.stPrevRect = oracleRect
                stStats.stLockFrames += 1
            } else {
                stStats.stPrevRect = nil
            }
        } else if enableSingleTarget && racing {
            // 收集当前帧所有 YOLO 检测结果
            var yoloDets: [(rect: CGRect, conf: Float, label: String)] = []
            if yoloRanThisFrame, let req = yoloReq {
                yoloDets.append(contentsOf: detections(from: req.results, confThreshold: Cfg.rYoloConf))
            }
            // ★ 协同模式: 增强算法提供轨迹预测 → ST 用于更精准的空间门控
            //   未开启增强算法时传 nil, ST 退化为独立 6D KF 预测
            let enhPred: CGRect? = useEnhancedAlgo ? predictedRect(now: time) : nil
            // ★ 调用单目标追踪器 (传入裁切帧 + 增强预测)
            //   当 yoloDets 为空时, ST 自动调用专用检测模型推理
            let stResult = singleTargetTracker.process(
                detRect: curRect,
                yoloDets: yoloDets,
                pb: usePB,
                enhancedPrediction: enhPred
            )
            // 单目标追踪结果覆盖 curRect
            if let stRect = stResult {
                curRect = stRect
                lastSeenTS = time
                targetLostFrames = 0
                if detSrc == "NONE" { detSrc = "ST" }
                stStats.stLockFrames += 1
                stStats.stPrevRect = stRect   // ★ 记录 ST 输出框，供 GT 评测 IoU 判定
            } else if singleTargetTracker.lostFrames > 5 {
                // ST 丢失超过 5 帧，不干涉原始管线
            }
            stStats.stTotalFrames += 1
        }

        // 赛道卡尔曼
        var kfState = ""
        if racing {
            kfState = runKF(time)
        }

        // 平滑: 标定链路用 One-Euro，否则用原 Smoother
        if useCalibratedFOV, let oer = oneEuroRect {
            smRect = oer.run(curRect, t: time)
        } else {
            smRect = smooth.run(curRect)
        }

        // ★ 变焦更新: 标定链路用 CalibratedAutoZoom(焦距/物距)，否则用 SimAutoZoom
        if enableZoom {
            if useCalibratedFOV, let cf = calibratedFOVMap {
                opticalZoom = calibratedAutoZoom.update(targetRect: smRect, currentZoom: opticalZoom, isLost: isLost, fovMap: cf, zc: zoomCfg)
            } else {
                opticalZoom = autoZoom.update(targetRect: smRect, currentZoom: opticalZoom, isLost: isLost, zc: zoomCfg)
            }
        }

        // ★ PID: 标定链路用 inverseProject 得到角度差再合成虚拟中心，与取景同一几何
        let effFov: (h: Double, v: Double)
        let hF: CGFloat
        let vF: CGFloat
        let ctr: CGPoint?
        if useCalibratedFOV, let cf = calibratedFOVMap {
            effFov = cf.effectiveFov(opticalZoom: curZoom)
            hF = CGFloat(effFov.h)
            vF = CGFloat(effFov.v)
            if let sr = smRect {
                let tgtSrcX = trackingCropRgn.minX + Double(sr.midX) * trackingCropRgn.width
                let tgtSrcY = trackingCropRgn.minY + Double(sr.midY) * trackingCropRgn.height
                let (idealPan, idealTilt) = cf.inverseProjectNormalized(normalizedX: tgtSrcX, normalizedY: tgtSrcY)
                let cenX = 0.5 - CGFloat(idealPan - gimbal.panPos) / hF
                let cenY = 0.5 + CGFloat(idealTilt - gimbal.tiltPos) / vF  // tilt 与 FOVMap.crop 的 cy 符号一致（加号）
                ctr = CGPoint(x: cenX, y: cenY)
            } else {
                ctr = nil
            }
        } else {
            effFov = fov.effectiveFov(opticalZoom: curZoom)
            hF = CGFloat(effFov.h)
            vF = CGFloat(effFov.v)
            ctr = smRect.map { CGPoint(x: $0.midX, y: $0.midY) }
        }
        // ★ 暂时禁用距离自适应增益 — 远处目标 gainScale=2.0 导致增益翻倍引发振荡；
        // 且 estimate 用裁切内归一化高度 × 源图高度，未考虑裁切框，距离会算错。
        // 若将来重做：远处应降增益减噪声，近处可适当提高；基础跟踪稳定后再优化。
        // if useCalibratedFOV, let sr = smRect, let cf = calibratedFOVMap {
        //     let effF = cf.calibration.focalLength * Double(curZoom)
        //     let imgH = cf.calibration.resolution.height
        //     let dist = distEstimator.estimate(rectHeight: sr.height, focalLength: effF, imageHeight: imgH)
        //     pidGainScale = distEstimator.gainScale(distance: dist)
        // }
        pidGainScale = 1.0
        let po = pid.run(center: ctr, rect: smRect, t: time, hFov: hF, vFov: vF, gainScale: pidGainScale, servoVelH: CGFloat(gimbal.panVel), servoVelV: CGFloat(gimbal.tiltVel))

        // 舵机逻辑
        let st = servoLogic(pid: po, t: time, hFov: hF, vFov: vF)

        // 发给仿真舵机 — 动态平滑时间: zoom 越大, 平滑越长 (对齐 iOS)
        let baseDynMs = racing ? 12.0 : 25.0
        let zoomDynMs = baseDynMs + Double(curZoom - 1.0) * 10.0
        gimbal.send(id: 1, angle: Double(servoH), durMs: zoomDynMs)
        gimbal.send(id: 2, angle: Double(servoV), durMs: zoomDynMs)

        // ★ 数字云台 (对齐 iOS MomentumGimbalEngine, 用于 Camera View 渲染)
        let dt = fIdx > 0 ? (time - (history.last?.t ?? time)) : 0.033
        let digCrop = digitalGimbal.update(targetRectTL: smRect, dt: dt, zc: zoomCfg)

        // ★ 只在会发布到 UI 的帧创建 CGImage（昂贵操作）—— debug 模式下每帧都创建
        // ★ FIX: 始终生成 srcImg/cropImg — 之前 willPublish 用 CACurrentMediaTime()
        //   与外层发布循环的时间判断存在竞态: process() 耗时(YOLO推理)导致内外判断不一致,
        //   cropImg 被设 nil 但 snap 仍被发布 → 第二窗口黑屏。
        // ★ 取景框始终用 usePB：cropPB 失败时 croppedPB 为 nil，若用 flatMap(croppedPB) 则
        //   cropImg 为 nil 导致取景框黑屏；改用 usePB 保证取景框与当前推理画面一致。
        let srcImg = cgImg(pb)
        let cropImg = cgImg(outputPB)

        // ★ 构建 DebugInfo
        var dbg: DebugInfo? = nil
        if debugMode {
            var d = DebugInfo()
            // A. 检测层
            d.yoloCandidates = lastYoloDebugCandidates
            d.trackerRect = detected
            d.trackerConf = trkConf
            d.trackerAlive = trkAlive
            d.selectedIdx = lastYoloDebugCandidates.firstIndex(where: { $0.accepted }) ?? -1
            d.matchMethod = lastYoloMatchMethod
            d.yoloRanThisFrame = yoloRanThisFrame
            // B. 控制层
            d.errDegX = po.eX
            d.errDegY = po.eY
            d.smoothVelX = po.vX
            d.smoothVelY = po.vY
            d.predErrX = po.predX
            d.predErrY = po.predY
            d.gainScale = po.gainScale
            d.distScale = po.distScale
            d.pidOutX = po.outX
            d.pidOutY = po.outY
            d.stepH = po.sH
            d.stepV = po.sV
            d.edgeBoostX = po.edgeX
            d.edgeBoostY = po.edgeY
            d.leadRoomX = po.leadRoomX
            d.leadRoomY = po.leadRoomY
            // C. 状态层
            d.state = st
            d.kfState = kfState
            d.kfCenter = kf.alive ? kf.center : nil
            d.kfVelocity = kf.alive ? kf.velocity : nil
            d.kfUncertainty = kf.alive ? kf.unc : 0
            d.kfPredictCount = kfPN
            d.kfSize = kf.alive ? kf.size : nil
            d.useEnhancedAlgo = self.useEnhancedAlgo
            d.hasTemplateFixed = targetTemplateFixed != nil
            d.hasTemplateDyn = targetTemplateDyn != nil
            d.targetLabel = targetLabel
            d.targetLostFrames = targetLostFrames
            d.bestAppScore = lastYoloBestScore
            // D. 舵机层
            d.cmdH = servoH; d.cmdV = servoV
            d.actH = gimbal.panPos; d.actV = gimbal.tiltPos
            d.servoErrH = gimbal.panErr; d.servoErrV = gimbal.tiltErr
            d.servoVelH = gimbal.panVel; d.servoVelV = gimbal.tiltVel
            // E. FOV
            d.cropRegion = outputCropRgn
            d.inertiaVelH = iVH; d.inertiaVelV = iVV
            // F. 变焦
            d.opticalZoom = curZoom; d.digitalZoom = digCrop.zoom
            d.effectiveFovH = effFov.h

            // ★ G. 时序预测曲线可视化 (采样未来 0.4 秒)
            if useEnhancedAlgo && trajPredictor.isValid {
                var curve: [CGPoint] = []
                for f in stride(from: 0.0, to: 0.4, by: 0.04) {
                    if let pt = trajPredictor.predict(targetTime: time + f) {
                        curve.append(CGPoint(x: pt.x, y: pt.y))
                    }
                }
                d.trajCurve = curve
            }

            dbg = d
        }

        let s = Snap(
            // Use PTS/display frame index so SAM3 annotation keys and rendered review frames stay aligned.
            idx: displayFrameIndex, t: time, trackingCrop: trackingCropRgn, crop: outputCropRgn, det: smRect,
            state: st, kfSt: kfState, detSrc: detSrc, yoloMs: yoloMs,
            cmdH: Double(servoH), cmdV: Double(servoV),
            actH: gimbal.panPos, actV: gimbal.tiltPos,
            errH: gimbal.panErr, errV: gimbal.tiltErr,
            velH: gimbal.panVel, velV: gimbal.tiltVel,
            eX: po.eX, eY: po.eY,
            optZoom: curZoom, digZoom: digCrop.zoom, effFovH: effFov.h,
            srcImg: srcImg, cropImg: cropImg,
            yoloCandidates: lastYoloDebugCandidates, yoloRanThisFrame: yoloRanThisFrame,
            dbg: dbg
        )

        updStats(s, yoloMs, detSrc: detSrc, trkAlive: trkAlive, srcCG: srcImg)

        // ★ 更新小车世界坐标轨迹 (用于 3D 场景概览)
        if let det = smRect {
            let gimbalPan = (gimbal.panPos - 120.0)
            let gimbalTilt = (gimbal.tiltPos - 120.0)
            let targetPanDeg = gimbalPan + Double(det.midX - 0.5) * effFov.h
            let targetTiltDeg = cameraTiltDeg + gimbalTilt + Double(det.midY - 0.5) * effFov.v
            let tiltRad = targetTiltDeg * .pi / 180.0
            let panRad = targetPanDeg * .pi / 180.0
            if targetTiltDeg > 1 {
                let groundDist = cameraHeightM / tan(tiltRad)
                let wx = groundDist * sin(panRad)
                let wz = groundDist * cos(panRad)
                if abs(wx) < 20 && wz > 0 && wz < 30 {
                    carWorldTrailBuf.append((x: wx, z: wz))
                    if carWorldTrailBuf.count > 500 {
                        carWorldTrailBuf.removeFirst(carWorldTrailBuf.count - 500)
                    }
                }
            }
        }

        // 轻量历史（无图，含debug）
        if history.count < 100000 {
            history.append(Snap(
                idx: s.idx, t: s.t, trackingCrop: s.trackingCrop, crop: s.crop, det: s.det,
                state: s.state, kfSt: s.kfSt, detSrc: s.detSrc, yoloMs: s.yoloMs,
                cmdH: s.cmdH, cmdV: s.cmdV, actH: s.actH, actV: s.actV,
                errH: s.errH, errV: s.errV, velH: s.velH, velV: s.velV,
                eX: s.eX, eY: s.eY,
                optZoom: s.optZoom, digZoom: s.digZoom, effFovH: s.effFovH,
                srcImg: nil, cropImg: nil,
                yoloCandidates: s.yoloCandidates, yoloRanThisFrame: s.yoloRanThisFrame,
                dbg: s.dbg
            ))
        }
        return s
    }

    private func oracleTargetRect(frameIndex: Int, crop: CGRect) -> CGRect? {
        guard evalOracleMode != .off,
              let annProj = annotationProject,
              let res = videoResolution else { return nil }
        let boxes = annProj.boxes(forLocalFrame: frameIndex)
        guard !boxes.isEmpty else { return nil }

        let imgW = res.width
        let imgH = res.height
        let sourceRects = boxes.map { box -> CGRect in
            let p = box.pixelRect(imgW: imgW, imgH: imgH)
            return CGRect(
                x: p.minX / imgW,
                y: p.minY / imgH,
                width: p.width / imgW,
                height: p.height / imgH
            )
        }

        let chosen: CGRect?
        switch evalOracleMode {
        case .off:
            chosen = nil
        case .source:
            chosen = sourceRects.max { ($0.width * $0.height) < ($1.width * $1.height) }
        case .crop:
            chosen = sourceRects
                .filter { !$0.intersection(crop).isNull && !$0.intersection(crop).isEmpty }
                .max { Self.iou($0, crop) < Self.iou($1, crop) }
        }
        guard let src = chosen, crop.width > 0.001, crop.height > 0.001 else { return nil }

        return CGRect(
            x: (src.minX - crop.minX) / crop.width,
            y: (src.minY - crop.minY) / crop.height,
            width: src.width / crop.width,
            height: src.height / crop.height
        )
    }

    // MARK: YOLO
    // ★ 修复：普通模式也使用外观匹配（与 iOS handleYoloResults 对齐）
    // 逻辑: 先检查是否有外观模板 → 有则走外观匹配（不分模式）
    //       → 无模板则走纯 IoU 匹配 + 首次锁定时建立模板
    //       → 赛道模式额外做 YOLO 时间补偿
    private var lastYoloDebugCandidates: [DebugCandidate] = []
    private var lastYoloMatchMethod: String = ""
    private var lastYoloBestScore: Float = 0

    private func doYOLO(handler: VNImageRequestHandler, time: CFTimeInterval) {
        lastYoloDebugCandidates = []
        lastYoloMatchMethod = "None"
        lastYoloBestScore = 0

        guard let req = yoloReq else { return }
        try? handler.perform([req])
        guard let results = req.results as? [VNRecognizedObjectObservation] else { return }

        // ★ 根据模式使用不同的置信度阈值
        let confThreshold = racing ? Cfg.rYoloConf : Cfg.yoloConf
        let valid = results.filter { obs in
            guard obs.labels.first != nil else { return false }
            return obs.confidence > confThreshold
        }
        guard !valid.isEmpty else {
            targetLostFrames += 1
            if targetLostFrames <= Cfg.targetHoldFramesMax { lastSeenTS = time }
            return
        }

        if racing {
            // ══════════════════════════════════════════════════════════
            // ★ 赛道模式: 外观匹配 + Tracker死亡重新捕获
            // ══════════════════════════════════════════════════════════
            doYOLO_Racing(valid: valid, time: time)
        } else {
            // ══════════════════════════════════════════════════════════
            // ★ 普通模式: 对齐 iOS handleYoloResults 逻辑
            //   - 有当前目标 → IoU 匹配
            //     - IoU > 0.6 → 仅刷新时间戳, 保持现有 tracker
            //     - IoU > 0.3 → 重建 tracker
            //   - 无当前目标 → 冷启动选最高置信度
            // ══════════════════════════════════════════════════════════
            doYOLO_Normal(valid: valid, time: time)
        }
    }

    // ─────────────────────────────────────────────────────────────
    // MARK: YOLO 普通模式 (对齐 iOS handleYoloResults)
    // ─────────────────────────────────────────────────────────────
    private func doYOLO_Normal(valid: [VNRecognizedObjectObservation], time: CFTimeInterval) {
        lastYoloMatchMethod = "IoU-iOS"

        if let cur = curRect {
            // ── 已有追踪目标: IoU 匹配 ──
            var bestObs: VNRecognizedObjectObservation?
            var bestIoU: CGFloat = 0

            for obs in valid {
                let r = obs.boundingBox
                let rectTL = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                let iou = Self.iou(cur, rectTL)
                lastYoloDebugCandidates.append(DebugCandidate(
                    rectTL: rectTL, label: obs.labels.first?.identifier ?? "?",
                    confidence: obs.confidence,
                    appSim: -1, motionSim: Float(iou), sizeSim: -1,
                    totalScore: Float(iou), accepted: false,
                    rejectReason: ""
                ))
                if iou > bestIoU { bestIoU = iou; bestObs = obs }
            }

            if let obs = bestObs, bestIoU > Cfg.normalIoURefresh {
                // ★ iOS 逻辑: IoU > 0.6 → tracker 仍然有效, 仅刷新时间戳
                lastSeenTS = time
                targetLostFrames = 0
                lastYoloBestScore = Float(bestIoU)
                // 标记接受
                if let idx = lastYoloDebugCandidates.indices.last(where: { lastYoloDebugCandidates[$0].motionSim == Float(bestIoU) }) {
                    let c = lastYoloDebugCandidates[idx]
                    lastYoloDebugCandidates[idx] = DebugCandidate(
                        rectTL: c.rectTL, label: c.label, confidence: c.confidence,
                        appSim: c.appSim, motionSim: c.motionSim, sizeSim: c.sizeSim,
                        totalScore: c.totalScore, accepted: true, rejectReason: "IoU>\(Cfg.normalIoURefresh) 保持tracker"
                    )
                }
                return
            }

            if let obs = bestObs, bestIoU > Cfg.normalIoUReinit {
                // ★ IoU > 0.3 → 重建 tracker
                let r = obs.boundingBox
                let rectTL = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                let nr = VNTrackObjectRequest(detectedObjectObservation: obs)
                nr.trackingLevel = .accurate
                trackReq = nr
                curRect = rectTL
                lastSeenTS = time
                targetLostFrames = 0
                lastYoloBestScore = Float(bestIoU)
                // 标记接受
                if let idx = lastYoloDebugCandidates.indices.last(where: { lastYoloDebugCandidates[$0].motionSim == Float(bestIoU) }) {
                    let c = lastYoloDebugCandidates[idx]
                    lastYoloDebugCandidates[idx] = DebugCandidate(
                        rectTL: c.rectTL, label: c.label, confidence: c.confidence,
                        appSim: c.appSim, motionSim: c.motionSim, sizeSim: c.sizeSim,
                        totalScore: c.totalScore, accepted: true, rejectReason: ""
                    )
                }
                return
            }

            // IoU 太低 — 不匹配
            for i in lastYoloDebugCandidates.indices {
                if lastYoloDebugCandidates[i].rejectReason.isEmpty {
                    let c = lastYoloDebugCandidates[i]
                    lastYoloDebugCandidates[i] = DebugCandidate(
                        rectTL: c.rectTL, label: c.label, confidence: c.confidence,
                        appSim: c.appSim, motionSim: c.motionSim, sizeSim: c.sizeSim,
                        totalScore: c.totalScore, accepted: false,
                        rejectReason: "IoU=\(String(format:"%.2f",c.motionSim)) < \(Cfg.normalIoUReinit)"
                    )
                }
            }

        } else {
            // ── 无当前目标: 冷启动 ──
            lastYoloMatchMethod = "ColdStart-iOS"
            if time - lastSeenTS < Cfg.searchCooldown { return }

            let best = valid.max { $0.confidence < $1.confidence }
            if let best = best {
                let r = best.boundingBox
                let rectTL = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                let nr = VNTrackObjectRequest(detectedObjectObservation: best)
                nr.trackingLevel = .accurate
                trackReq = nr
                curRect = rectTL
                lastSeenTS = time
                targetLostFrames = 0
                lastYoloBestScore = best.confidence
                lastYoloDebugCandidates.append(DebugCandidate(
                    rectTL: rectTL, label: best.labels.first?.identifier ?? "?",
                    confidence: best.confidence,
                    appSim: -1, motionSim: -1, sizeSim: -1,
                    totalScore: best.confidence,
                    accepted: true, rejectReason: ""
                ))
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // MARK: YOLO 赛道模式 (外观匹配 + Tracker死亡修复)
    // ─────────────────────────────────────────────────────────────
    private func doYOLO_Racing(valid: [VNRecognizedObjectObservation], time: CFTimeInterval) {

        // ── 有外观模板 → 走外观匹配 ──
        if let fixed = targetTemplateFixed, let pb = latestCroppedPB {
            lastYoloMatchMethod = "Appearance"
            let candidates = valid.sorted {
                if $0.confidence != $1.confidence { return $0.confidence > $1.confidence }
                if $0.boundingBox.width * $0.boundingBox.height != $1.boundingBox.width * $1.boundingBox.height {
                    return $0.boundingBox.width * $0.boundingBox.height > $1.boundingBox.width * $1.boundingBox.height
                }
                if $0.boundingBox.minX != $1.boundingBox.minX { return $0.boundingBox.minX < $1.boundingBox.minX }
                return $0.boundingBox.minY < $1.boundingBox.minY
            }.prefix(10)
            let predRect = predictedRect(now: time)

            var bestObs: VNRecognizedObjectObservation?
            var bestRectTL: CGRect?
            var bestFeat: TargetFeature?
            var bestScore: Float = -1
            var bestMotion: Float = 0
            var bestIdx: Int = -1

            for (idx, obs) in candidates.enumerated() {
                // 类别过滤
                if let label = targetLabel, let id = obs.labels.first?.identifier, id != label {
                    lastYoloDebugCandidates.append(DebugCandidate(
                        rectTL: CGRect(x: obs.boundingBox.minX, y: 1 - obs.boundingBox.maxY,
                                       width: obs.boundingBox.width, height: obs.boundingBox.height),
                        label: obs.labels.first?.identifier ?? "?",
                        confidence: obs.confidence,
                        appSim: -1, motionSim: -1, sizeSim: -1, totalScore: -1,
                        accepted: false, rejectReason: "类别不匹配: \(id) ≠ \(label)"
                    ))
                    continue
                }
                let r = obs.boundingBox
                let rectTL = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                guard let feat = extractFeature(pb: pb, rectTL: rectTL) else {
                    lastYoloDebugCandidates.append(DebugCandidate(
                        rectTL: rectTL, label: obs.labels.first?.identifier ?? "?",
                        confidence: obs.confidence,
                        appSim: -1, motionSim: -1, sizeSim: -1, totalScore: -1,
                        accepted: false, rejectReason: "特征提取失败"
                    ))
                    continue
                }

                let appSim = appearanceSimilarity(feat: feat, fixed: fixed, dyn: targetTemplateDyn)
                let iouSim: Float = predRect != nil ? Float(Self.iou(rectTL, predRect!)) : 0.0
                // ★ 新增: 高斯位置接近度 (比 IoU 更鲁棒 — 无重叠时仍有梯度)
                let proxSim: Float
                if let pr = predRect {
                    proxSim = proximityScore(
                        candidate: CGPoint(x: rectTL.midX, y: rectTL.midY),
                        predicted: CGPoint(x: pr.midX, y: pr.midY))
                } else if kf.alive {
                    // KF 仍存活 — 用其预测中心
                    proxSim = proximityScore(
                        candidate: CGPoint(x: rectTL.midX, y: rectTL.midY),
                        predicted: kf.center)
                } else {
                    proxSim = 0.5  // 无预测信息, 中性值
                }
                // ★ 运动得分: IoU 和位置接近度取较大值
                let motionSim = max(iouSim, proxSim)

                var sizeSim: Float = 0.5
                if let base = curRect, base.height > 0 {
                    let diff = abs(rectTL.height - base.height) / base.height
                    sizeSim = Float(max(0.0, 1.0 - min(diff, 1.0)))
                }

                // ★ 静止惩罚: 目标此前在运动, 但候选框几乎没动 → 大概率是停在赛道上的静止车
                let sPenalty = stationaryPenaltyFor(
                    candidateCenter: CGPoint(x: rectTL.midX, y: rectTL.midY), now: time)

                // ★ 赛道模式权重 — 重新设计: tracker死亡时仍保留位置接近度权重
                var score: Float
                let prolongedLoss = targetLostFrames > Cfg.targetHoldFramesMax
                if trackerJustDied || prolongedLoss {
                    // Tracker 死亡 / 持续丢失: 降低外观权重, 保持位置接近度
                    // (外观对所有同类RC车都相似, 位置才能区分运动vs静止)
                    score = 0.50 * appSim + 0.35 * motionSim + 0.15 * sizeSim
                } else if predRect == nil && !kf.alive {
                    // 真正无预测信息 (极少见): 回退到外观为主
                    score = 0.75 * appSim + 0.25 * sizeSim
                } else {
                    score = 0.55 * appSim + 0.30 * motionSim + 0.15 * sizeSim
                }
                // 减去静止惩罚
                score = max(0, score - sPenalty)

                lastYoloDebugCandidates.append(DebugCandidate(
                    rectTL: rectTL,
                    label: obs.labels.first?.identifier ?? "?",
                    confidence: obs.confidence,
                    appSim: appSim, motionSim: motionSim, sizeSim: sizeSim,
                    totalScore: score, accepted: false, rejectReason: sPenalty > 0 ? "静止惩罚-\(String(format:"%.2f",sPenalty))" : ""
                ))

                if score > bestScore {
                    bestScore = score
                    bestObs = obs
                    bestRectTL = rectTL
                    bestFeat = feat
                    bestMotion = motionSim
                    bestIdx = lastYoloDebugCandidates.count - 1
                }
            }

            lastYoloBestScore = bestScore

            if let finalObs = bestObs, var rectTL = bestRectTL {
                // ★ Tracker 死亡 / 持续丢失 → 更宽松的接受阈值
                let currentAcceptScore: Float
                let prolongedLoss2 = targetLostFrames > Cfg.targetHoldFramesMax
                if trackerJustDied || prolongedLoss2 {
                    currentAcceptScore = Cfg.trackerDeadAcceptScore  // 0.30 — 极宽松
                } else if searchOn {
                    currentAcceptScore = Cfg.searchRelaxedAcceptScore
                } else {
                    currentAcceptScore = Cfg.targetAcceptScore
                }

                // ★ 核心修复：打破“Tracker 劫持”死锁！
                // 只要外观长得足够像 (appSim > 0.45)，就无视物理距离，强制信任 YOLO (Appearance Overrule)
                let bestAppSim: Float = (bestIdx >= 0 && bestIdx < lastYoloDebugCandidates.count) ? lastYoloDebugCandidates[bestIdx].appSim : 0
                let appearanceOverrule = bestAppSim > 0.45

                let motionOk: Bool
                if predRect == nil && !kf.alive {
                    motionOk = true   // 真正无预测: 放行
                } else if trackerJustDied || prolongedLoss2 || searchOn {
                    motionOk = bestMotion >= 0.02 || bestScore >= Cfg.targetStrongScore || appearanceOverrule
                } else {
                    motionOk = bestMotion >= 0.05 || bestScore >= Cfg.targetStrongScore || appearanceOverrule
                }

                let accept = bestScore >= currentAcceptScore && motionOk

                if accept {
                    // 标记被接受
                    if bestIdx >= 0 && bestIdx < lastYoloDebugCandidates.count {
                        lastYoloDebugCandidates[bestIdx] = DebugCandidate(
                            rectTL: lastYoloDebugCandidates[bestIdx].rectTL,
                            label: lastYoloDebugCandidates[bestIdx].label,
                            confidence: lastYoloDebugCandidates[bestIdx].confidence,
                            appSim: lastYoloDebugCandidates[bestIdx].appSim,
                            motionSim: lastYoloDebugCandidates[bestIdx].motionSim,
                            sizeSim: lastYoloDebugCandidates[bestIdx].sizeSim,
                            totalScore: lastYoloDebugCandidates[bestIdx].totalScore,
                            accepted: true, rejectReason: ""
                        )
                    }

                    if targetLabel == nil {
                        targetLabel = finalObs.labels.first?.identifier
                    }
                    let nr = VNTrackObjectRequest(detectedObjectObservation: finalObs)
                    nr.trackingLevel = .fast
                    trackReq = nr

                    // ★ YOLO 时间补偿 (赛道模式)
                    let yoloDt = CGFloat(time - yoloDispatchTS)
                    if yoloDt > 0.005 && yoloDt < 0.3 {
                        let vel: CGPoint = kf.alive ? kf.velocity : lastTargetVelTL
                        let compX = rectTL.origin.x + vel.x * yoloDt
                        let compY = rectTL.origin.y + vel.y * yoloDt
                        let cx = min(max(compX + rectTL.width / 2, rectTL.width / 2), 1.0 - rectTL.width / 2) - rectTL.width / 2
                        let cy = min(max(compY + rectTL.height / 2, rectTL.height / 2), 1.0 - rectTL.height / 2) - rectTL.height / 2
                        rectTL = CGRect(x: cx, y: cy, width: rectTL.width, height: rectTL.height)
                    }
                    let center = CGPoint(x: rectTL.midX, y: rectTL.midY)
                    let size = CGSize(width: rectTL.width, height: rectTL.height)
                    if kf.alive {
                        kf.update(center: center, size: size, time)
                    } else {
                        kf.start(center: center, size: size, time)
                    }

                    curRect = rectTL
                    lastSeenTS = time
                    updateTargetMotion(rectTL: rectTL, now: time)
                    targetLostFrames = 0
                    tSz = CGSize(width: rectTL.width, height: rectTL.height)

                    // 动态模板更新 — 增加静止检查
                    let templatePenalty = stationaryPenaltyFor(
                        candidateCenter: CGPoint(x: rectTL.midX, y: rectTL.midY), now: time)
                    if let feat = bestFeat, bestScore >= Cfg.targetStrongScore && templatePenalty == 0 {
                        if let dyn = targetTemplateDyn {
                            targetTemplateDyn = blendFeature(old: dyn, new: feat, alpha: 0.2)
                        } else {
                            targetTemplateDyn = feat
                        }
                    }
                } else {
                    // 标记拒绝原因
                    if bestIdx >= 0 && bestIdx < lastYoloDebugCandidates.count {
                        var reason = "score=\(String(format:"%.2f",bestScore))"
                        if bestScore < currentAcceptScore { reason += " < 阈值\(String(format:"%.2f",currentAcceptScore))" }
                        if !motionOk { reason += " + 运动不匹配" }
                        lastYoloDebugCandidates[bestIdx] = DebugCandidate(
                            rectTL: lastYoloDebugCandidates[bestIdx].rectTL,
                            label: lastYoloDebugCandidates[bestIdx].label,
                            confidence: lastYoloDebugCandidates[bestIdx].confidence,
                            appSim: lastYoloDebugCandidates[bestIdx].appSim,
                            motionSim: lastYoloDebugCandidates[bestIdx].motionSim,
                            sizeSim: lastYoloDebugCandidates[bestIdx].sizeSim,
                            totalScore: lastYoloDebugCandidates[bestIdx].totalScore,
                            accepted: false, rejectReason: reason
                        )
                    }
                    targetLostFrames += 1
                    if targetLostFrames <= Cfg.targetHoldFramesMax {
                        lastSeenTS = time
                    }
                }
            } else {
                targetLostFrames += 1
                if targetLostFrames <= Cfg.targetHoldFramesMax {
                    lastSeenTS = time
                }
            }
            return
        }

        // ── 无外观模板 → IoU 匹配 + 首次锁定时建立模板 ──
        if let cur = curRect {
            lastYoloMatchMethod = "IoU"
            var bestObs: VNRecognizedObjectObservation?
            var bestIoU: CGFloat = 0
            for o in valid {
                let r = o.boundingBox
                let rt = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                let iou = Self.iou(cur, rt)
                lastYoloDebugCandidates.append(DebugCandidate(
                    rectTL: rt, label: o.labels.first?.identifier ?? "?",
                    confidence: o.confidence,
                    appSim: -1, motionSim: Float(iou), sizeSim: -1, totalScore: Float(iou),
                    accepted: false, rejectReason: iou > 0.3 ? "" : "IoU=\(String(format:"%.2f",iou)) < 0.3"
                ))
                if iou > bestIoU { bestIoU = iou; bestObs = o }
            }
            if let o = bestObs, bestIoU > 0.3 {
                let r = o.boundingBox
                let rectTL = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                let nr = VNTrackObjectRequest(detectedObjectObservation: o)
                nr.trackingLevel = .fast
                trackReq = nr
                lastSeenTS = time
                targetLabel = o.labels.first?.identifier
                // 建立外观模板
                if let pb = latestCroppedPB, let feat = extractFeature(pb: pb, rectTL: rectTL) {
                    targetTemplateFixed = feat
                    targetTemplateDyn = feat
                }
                updateTargetMotion(rectTL: rectTL, now: time)
                curRect = rectTL
                targetLostFrames = 0
                tSz = CGSize(width: rectTL.width, height: rectTL.height)
                kf.start(center: CGPoint(x: rectTL.midX, y: rectTL.midY), size: CGSize(width: rectTL.width, height: rectTL.height), time)
                if let idx = lastYoloDebugCandidates.indices.last(where: { lastYoloDebugCandidates[$0].motionSim == Float(bestIoU) }) {
                    let c = lastYoloDebugCandidates[idx]
                    lastYoloDebugCandidates[idx] = DebugCandidate(
                        rectTL: c.rectTL, label: c.label, confidence: c.confidence,
                        appSim: c.appSim, motionSim: c.motionSim, sizeSim: c.sizeSim,
                        totalScore: c.totalScore, accepted: true, rejectReason: ""
                    )
                }
                lastYoloBestScore = Float(bestIoU)
            }
        } else {
            lastYoloMatchMethod = "ColdStart"
            if time - lastSeenTS < Cfg.searchCooldown { return }

            // ★ 冷启动改进: 如果有历史速度信息 (之前跟踪过运动目标),
            //   优先选接近预测位置的候选, 并惩罚静止候选
            let hasVelInfo = sqrt(lastTargetVelTL.x * lastTargetVelTL.x +
                                  lastTargetVelTL.y * lastTargetVelTL.y) > Cfg.movingVelThreshold
                             && lastTargetCenterTL != nil

            var best: VNRecognizedObjectObservation?
            var bestColdScore: Float = -1

            for obs in valid {
                let r = obs.boundingBox
                let rt = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                let conf = obs.confidence

                if hasVelInfo {
                    // 有历史运动信息 → 综合: 置信度 + 位置接近度 - 静止惩罚
                    let dt = CGFloat(time - lastTargetUpdateTS)
                    let predCenter: CGPoint
                    if kf.alive {
                        predCenter = kf.center
                    } else if let lc = lastTargetCenterTL, dt > 0 && dt < 2.0 {
                        predCenter = CGPoint(x: lc.x + lastTargetVelTL.x * dt,
                                             y: lc.y + lastTargetVelTL.y * dt)
                    } else {
                        predCenter = CGPoint(x: 0.5, y: 0.5)
                    }
                    let prox = proximityScore(
                        candidate: CGPoint(x: rt.midX, y: rt.midY),
                        predicted: predCenter)
                    let sPen = stationaryPenaltyFor(
                        candidateCenter: CGPoint(x: rt.midX, y: rt.midY), now: time)
                    let coldScore = 0.4 * conf + 0.45 * prox + 0.15 * Float(0.5) - sPen
                    lastYoloDebugCandidates.append(DebugCandidate(
                        rectTL: rt, label: obs.labels.first?.identifier ?? "?",
                        confidence: conf,
                        appSim: -1, motionSim: prox, sizeSim: -1,
                        totalScore: coldScore,
                        accepted: false, rejectReason: sPen > 0 ? "静止惩罚" : ""
                    ))
                    if coldScore > bestColdScore { bestColdScore = coldScore; best = obs }
                } else {
                    // 无历史运动信息 (真正首次) → 纯置信度
                    let coldScore = conf
                    lastYoloDebugCandidates.append(DebugCandidate(
                        rectTL: rt, label: obs.labels.first?.identifier ?? "?",
                        confidence: conf,
                        appSim: -1, motionSim: -1, sizeSim: -1,
                        totalScore: coldScore,
                        accepted: false, rejectReason: ""
                    ))
                    if coldScore > bestColdScore { bestColdScore = coldScore; best = obs }
                }
            }

            if let best = best {
                let r = best.boundingBox
                let rt = CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
                let nr = VNTrackObjectRequest(detectedObjectObservation: best)
                nr.trackingLevel = .fast
                trackReq = nr
                curRect = rt
                lastSeenTS = time
                targetLabel = best.labels.first?.identifier
                if let pb = latestCroppedPB, let feat = extractFeature(pb: pb, rectTL: rt) {
                    targetTemplateFixed = feat
                    targetTemplateDyn = feat
                }
                updateTargetMotion(rectTL: rt, now: time)
                tSz = CGSize(width: rt.width, height: rt.height)
                kf.start(center: CGPoint(x: rt.midX, y: rt.midY), size: CGSize(width: rt.width, height: rt.height), time)
                // 标记被接受
                if let idx = lastYoloDebugCandidates.indices.last(where: { !lastYoloDebugCandidates[$0].accepted }) {
                    let c = lastYoloDebugCandidates[idx]
                    lastYoloDebugCandidates[idx] = DebugCandidate(
                        rectTL: c.rectTL, label: c.label, confidence: c.confidence,
                        appSim: c.appSim, motionSim: c.motionSim, sizeSim: c.sizeSim,
                        totalScore: c.totalScore, accepted: true, rejectReason: ""
                    )
                }
                lastYoloBestScore = bestColdScore
            }
        }
    }

    // MARK: 目标运动跟踪 (改进版: 去除光流混合, 加快 EMA 响应)
    private func updateTargetMotion(rectTL: CGRect, now: CFTimeInterval) {
        let center = CGPoint(x: rectTL.midX, y: rectTL.midY)
        if let lastCenter = lastTargetCenterTL {
            let dt = CGFloat(now - lastTargetUpdateTS)
            if dt > 0 {
                let rawVX = (center.x - lastCenter.x) / dt
                let rawVY = (center.y - lastCenter.y) / dt
                // ★ 加快 EMA 响应: 0.35 → 0.50, 减少速度估计滞后
                let alpha: CGFloat = 0.50
                let vel = CGPoint(
                    x: lastTargetVelTL.x * (1 - alpha) + rawVX * alpha,
                    y: lastTargetVelTL.y * (1 - alpha) + rawVY * alpha
                )
                // ★ 光流混合已移除 — 纯位置差分速度, 响应更快
                lastTargetVelTL = vel
            }
        }
        lastTargetCenterTL = center
        lastTargetUpdateTS = now
        lastVel = lastTargetVelTL

        // ★ 增强算法：时序二次曲线拟合仍保留（与光流无关，独立有效）
        if useEnhancedAlgo {
            trajPredictor.push(t: now, x: center.x, y: center.y)
        }
    }

    // MARK: 用户框选注入（与自动获取并存）
    func setInitialTargetFromUser(rectTL: CGRect) {
        userRectLock.lock()
        _userSelectedRectTL = rectTL
        userRectLock.unlock()
    }

    private func consumeUserSelectedRect() -> CGRect? {
        userRectLock.lock()
        let r = _userSelectedRectTL
        _userSelectedRectTL = nil
        userRectLock.unlock()
        return r
    }

    /// 将用户框选的区域设为当前追踪目标，并建立 VNTracker + 卡尔曼
    private func applyUserSelectedTarget(rectTL: CGRect, pb: CVPixelBuffer, time: CFTimeInterval) {
        let box = CGRect(x: rectTL.minX, y: 1.0 - rectTL.maxY, width: rectTL.width, height: rectTL.height)
        let obs = VNDetectedObjectObservation(boundingBox: box)
        let req = VNTrackObjectRequest(detectedObjectObservation: obs)
        req.trackingLevel = .fast
        trackReq = req
        curRect = rectTL
        lastSeenTS = time
        targetLostFrames = 0
        targetLabel = "用户框选"
        updateTargetMotion(rectTL: rectTL, now: time)
        tSz = CGSize(width: rectTL.width, height: rectTL.height)
        kf.start(center: CGPoint(x: rectTL.midX, y: rectTL.midY), size: tSz, time)
        if let feat = extractFeature(pb: pb, rectTL: rectTL) {
            targetTemplateFixed = feat
            targetTemplateDyn = feat
        }
    }

    // MARK: 预测目标位置 (与 iOS predictedRect 同步)
    private func predictedRect(now: CFTimeInterval) -> CGRect? {
        guard let base = curRect else { return nil }
        let dt = CGFloat(now - lastTargetUpdateTS)
        guard dt > 0 && dt < 1.0 else { return base }

        // ★ 升级版预测：如果开启增强算法，且拟合成功，使用二次曲线推演未来位置
        if useEnhancedAlgo, let pt = trajPredictor.predict(targetTime: now) {
            return CGRect(
                x: CGFloat(pt.x) - base.width / 2,
                y: CGFloat(pt.y) - base.height / 2,
                width: base.width,
                height: base.height
            )
        }

        // 降级版预测：原有的恒定速度直线外推
        let predX = base.midX + lastTargetVelTL.x * dt
        let predY = base.midY + lastTargetVelTL.y * dt
        return CGRect(
            x: predX - base.width / 2,
            y: predY - base.height / 2,
            width: base.width,
            height: base.height
        )
    }

    // MARK: ★ 位置接近度评分 — 高斯距离 (比 IoU 更适合重新捕获: 即使无重叠也有梯度)
    private func proximityScore(candidate: CGPoint, predicted: CGPoint) -> Float {
        let dx = candidate.x - predicted.x
        let dy = candidate.y - predicted.y
        let dist2 = dx * dx + dy * dy
        let sigma = Cfg.proximitySigma
        return Float(exp(-Double(dist2) / Double(2 * sigma * sigma)))
    }

    // MARK: ★ 静止候选检测 — 判断候选框是否可能是静止物体
    //   逻辑: 如果之前的目标有明显速度, 而这个候选框离 "上一次位置" 很近
    //         (而不是离 "预测位置" 近), 说明它没有跟着运动 → 大概率是静止车
    private func stationaryPenaltyFor(candidateCenter: CGPoint, now: CFTimeInterval) -> Float {
        let velMag = sqrt(lastTargetVelTL.x * lastTargetVelTL.x + lastTargetVelTL.y * lastTargetVelTL.y)
        // 只有在目标此前确实在运动时才施加惩罚
        guard velMag > Cfg.movingVelThreshold else { return 0 }
        guard let lastCenter = lastTargetCenterTL else { return 0 }
        let dt = CGFloat(now - lastTargetUpdateTS)
        guard dt > 0.02 && dt < 2.0 else { return 0 }

        // 候选框到 "上次位置" 的距离
        let dxLast = candidateCenter.x - lastCenter.x
        let dyLast = candidateCenter.y - lastCenter.y
        let distToLast = sqrt(dxLast * dxLast + dyLast * dyLast)

        // 目标在这段时间应该移动的距离
        let expectedDisp = velMag * dt

        // 如果候选框几乎没动（距离上次位置 < 预期位移的 30%），很可能是静止的
        if distToLast < expectedDisp * 0.3 {
            return Cfg.stationaryPenalty
        }
        return 0
    }

    // ─────────────────────────────────────────────────────────────
    // MARK: Ground Truth SAHI 地毯式检测 (仅统计, 不参与控制)
    // ─────────────────────────────────────────────────────────────
    // 在 **源广角 CGImage** 上做分片检测, 作为 "目标是否可见" 的 ground truth
    // ★ 使用 CGImage 而非 CVPixelBuffer — 避免占用 AVAssetReader buffer pool
    // ★ 使用独立 gtModel — 每片创建新 VNCoreMLRequest, 不与控制 YOLO 共享
    // ★ 并发推理 — DispatchQueue.concurrentPerform 充分利用多核 + ANE pipeline
    private func runGtSahiYolo(srcCG: CGImage) -> [CGRect] {
        guard let model = gtModel else { return [] }
        let imgW = CGFloat(srcCG.width)
        let imgH = CGFloat(srcCG.height)
        guard imgW > 0, imgH > 0 else { return [] }

        let tile = CGFloat(Cfg.gtSahiTileSize)
        let overlap = CGFloat(Cfg.gtSahiOverlap)
        let stride = tile * (1.0 - overlap)

        // ── 生成分片坐标 (像素坐标, CGImage: 左上原点) ──
        var tiles: [CGRect] = []
        var y: CGFloat = 0
        while y < imgH {
            var ty = y
            if ty + tile > imgH { ty = max(imgH - tile, 0) }
            var x: CGFloat = 0
            while x < imgW {
                var tx = x
                if tx + tile > imgW { tx = max(imgW - tile, 0) }
                let tw = min(tile, imgW - tx)
                let th = min(tile, imgH - ty)
                if tw > 32 && th > 32 {
                    tiles.append(CGRect(x: tx, y: ty, width: tw, height: th))
                }
                x += stride
                if tx + tile >= imgW { break }
            }
            y += stride
            if ty + tile >= imgH { break }
        }

        // ── 并发逐片推理 ──
        struct RawDet {
            let rect: CGRect   // 全图归一化, TL 坐标系
            let conf: Float
        }
        let lock = NSLock()
        var allDets: [RawDet] = []

        DispatchQueue.concurrentPerform(iterations: tiles.count) { idx in
            let tileRect = tiles[idx]
            guard let tileCG = srcCG.cropping(to: tileRect) else { return }

            // ★ 每片独立 request — 避免 results 互踩
            let req = VNCoreMLRequest(model: model)
            req.imageCropAndScaleOption = .scaleFill
            let handler = VNImageRequestHandler(cgImage: tileCG, orientation: .up, options: [:])
            try? handler.perform([req])

            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            var localDets: [RawDet] = []
            for obs in results {
                guard obs.labels.first != nil, obs.confidence > Cfg.gtSahiConf else { continue }
                let bb = obs.boundingBox  // Vision BL 坐标系, 归一化于 tile
                // → 转为全图像素坐标 (CGImage TL 坐标系)
                let px = tileRect.minX + bb.minX * tileRect.width
                let pw = bb.width * tileRect.width
                let ph = bb.height * tileRect.height
                // Vision BL → CGImage TL: py_tl = tileRect.minY + (1 - bb.maxY) * tileRect.height
                let py = tileRect.minY + (1.0 - bb.maxY) * tileRect.height
                // → 全图归一化 TL 坐标系
                let nx = px / imgW
                let ny = py / imgH
                let nw = pw / imgW
                let nh = ph / imgH
                localDets.append(RawDet(rect: CGRect(x: nx, y: ny, width: nw, height: nh), conf: obs.confidence))
            }
            if !localDets.isEmpty {
                lock.lock()
                allDets.append(contentsOf: localDets)
                lock.unlock()
            }
        }

        // ── 跨分片 NMS (按置信度降序, greedy) ──
        let sorted = allDets.sorted { $0.conf > $1.conf }
        var kept: [CGRect] = []
        var suppressed = [Bool](repeating: false, count: sorted.count)
        for i in sorted.indices {
            if suppressed[i] { continue }
            kept.append(sorted[i].rect)
            for j in (i + 1)..<sorted.count {
                if suppressed[j] { continue }
                if Self.iou(sorted[i].rect, sorted[j].rect) > Cfg.gtSahiNmsIoU {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    // MARK: 外观特征提取 (带 2D 高斯空间权重)
    private func extractFeature(pb: CVPixelBuffer, rectTL: CGRect) -> TargetFeature? {
        let base = CIImage(cvPixelBuffer: pb)
        let w = base.extent.width
        let h = base.extent.height
        // TL→CI 坐标: y 翻转
        let rectCI = CGRect(x: rectTL.minX * w, y: (1.0 - rectTL.maxY) * h,
                            width: rectTL.width * w, height: rectTL.height * h)
        if rectCI.width < 4 || rectCI.height < 4 { return nil }
        let cropped = base.cropped(to: rectCI)

        // ── 1. 灰度 16x16 (应用高斯权重) ──
        guard let grayBytes = rgbaBytes(ciImage: cropped, size: CGSize(width: 16, height: 16)) else { return nil }
        var gray: [Float] = []
        gray.reserveCapacity(16 * 16)
        var sumSq: Float = 0
        let sigmaG: Float = 6.0 // 16x16 的高斯核标准差

        for i in stride(from: 0, to: grayBytes.count, by: 4) {
            let idx = i / 4
            let px = Float(idx % 16)
            let py = Float(idx / 16)
            // 计算到中心点 (7.5, 7.5) 的空间距离，生成高斯权重
            let dx = px - 7.5
            let dy = py - 7.5
            // ★ 撤销高斯权重，避免“半车身”时中心是赛道导致外观分被压低、无法触发 Appearance Overrule
            let weight: Float = 1.0

            let r = Float(grayBytes[i]) / 255.0
            let g = Float(grayBytes[i + 1]) / 255.0
            let b = Float(grayBytes[i + 2]) / 255.0
            // ★ 特征值乘以空间权重，压制边缘背景
            let v = (0.299 * r + 0.587 * g + 0.114 * b) * weight
            gray.append(v)
            sumSq += v * v
        }
        let grayNorm = sqrt(max(sumSq, 1e-6))

        // ── 2. HS 直方图 32x32 (应用高斯权重) ──
        guard let colorBytes = rgbaBytes(ciImage: cropped, size: CGSize(width: 32, height: 32)) else { return nil }
        let hBins = 16, sBins = 8
        var hist = [Float](repeating: 0, count: hBins * sBins)
        let sigmaC: Float = 12.0 // 32x32 的高斯核标准差

        for i in stride(from: 0, to: colorBytes.count, by: 4) {
            let idx = i / 4
            let px = Float(idx % 32)
            let py = Float(idx / 32)
            // 计算到中心点 (15.5, 15.5) 的距离
            let dx = px - 15.5
            let dy = py - 15.5
            // ★ 同上：直方图也取消空间衰减
            let weight: Float = 1.0

            let r = Float(colorBytes[i]) / 255.0
            let g = Float(colorBytes[i + 1]) / 255.0
            let b = Float(colorBytes[i + 2]) / 255.0
            let hsv = rgbToHSV(r: r, g: g, b: b)
            let hBin = min(Int(hsv.h * Float(hBins)), hBins - 1)
            let sBin = min(Int(hsv.s * Float(sBins)), sBins - 1)
            // ★ 投票时累加空间权重，边缘背景对直方图的贡献微乎其微
            hist[hBin * sBins + sBin] += weight
        }
        let sumHist = hist.reduce(0, +)
        if sumHist > 0 { hist = hist.map { $0 / sumHist } }

        return TargetFeature(gray: gray, grayNorm: grayNorm, hsv: hist)
    }

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

    private func appearanceSimilarity(feat: TargetFeature, fixed: TargetFeature, dyn: TargetFeature?) -> Float {
        let simFixed = 0.6 * cosineSim(a: feat.gray, normA: feat.grayNorm, b: fixed.gray, normB: fixed.grayNorm)
                     + 0.4 * histIntersection(a: feat.hsv, b: fixed.hsv)
        if let dyn = dyn {
            let simDyn = 0.6 * cosineSim(a: feat.gray, normA: feat.grayNorm, b: dyn.gray, normB: dyn.grayNorm)
                       + 0.4 * histIntersection(a: feat.hsv, b: dyn.hsv)
            return max(simFixed, simDyn)
        }
        return simFixed
    }

    private func cosineSim(a: [Float], normA: Float, b: [Float], normB: Float) -> Float {
        if normA < 1e-6 || normB < 1e-6 { return 0 }
        let n = min(a.count, b.count)
        var dot: Float = 0
        for i in 0..<n { dot += a[i] * b[i] }
        return max(0, min(1, dot / (normA * normB)))
    }

    private func histIntersection(a: [Float], b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var sum: Float = 0
        for i in 0..<n { sum += min(a[i], b[i]) }
        return max(0, min(1, sum))
    }

    private func blendFeature(old: TargetFeature, new: TargetFeature, alpha: Float) -> TargetFeature {
        let a = max(0, min(alpha, 1))
        let n = min(old.gray.count, new.gray.count)
        var gray: [Float] = []
        gray.reserveCapacity(n)
        var sumSq: Float = 0
        for i in 0..<n {
            let v = old.gray[i] * (1 - a) + new.gray[i] * a
            gray.append(v)
            sumSq += v * v
        }
        let grayNorm = sqrt(max(sumSq, 1e-6))
        let m = min(old.hsv.count, new.hsv.count)
        var hsv: [Float] = []
        hsv.reserveCapacity(m)
        for i in 0..<m { hsv.append(old.hsv[i] * (1 - a) + new.hsv[i] * a) }
        let sum = hsv.reduce(0, +)
        if sum > 0 { hsv = hsv.map { $0 / sum } }
        return TargetFeature(gray: gray, grayNorm: grayNorm, hsv: hsv)
    }

    // MARK: 卡尔曼增强
    private func runKF(_ t: CFTimeInterval) -> String {
        if let r = curRect {
            let c = CGPoint(x: r.midX, y: r.midY)
            let s = CGSize(width: r.width, height: r.height)
            if kf.alive { kf.update(center: c, size: s, t) } else { kf.start(center: c, size: s, t) }
            kfPN = 0
            return "TRK"
        } else if kf.alive && kfPN < Cfg.kfMaxPredict {
            kf.predict(t)
            if kf.unc < Cfg.kfMaxUncert {
                let pc = kf.center
                // ★ 算法对比: 开启增强时用 8D 卡尔曼的动态尺度，关闭时用上次 YOLO 锁定的固定尺度 (tSz)
                let w = useEnhancedAlgo ? kf.size.width : tSz.width
                let h = useEnhancedAlgo ? kf.size.height : tSz.height
                let cx = min(max(pc.x, w / 2), 1 - w / 2)
                let cy = min(max(pc.y, h / 2), 1 - h / 2)
                curRect = CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
                lastSeenTS = t
                kfPN += 1
                return "KF\(kfPN)"
            }
            return "KF-L"
        }
        return "IDLE"
    }

    // MARK: 舵机三阶段
    private func servoLogic(pid p: PIDCtrl.Out, t: CFTimeInterval, hFov: CGFloat, vFov: CGFloat) -> String {
        if smRect != nil {
            if t - lastSendTS >= Cfg.servoSendInterval {
                servoH = min(max(servoH + p.sH, 0), 240)
                servoV = min(max(servoV + p.sV, 0), 240)
                lastSendTS = t
            }
            searchOn = false
            inertiaOn = false
            return "TRACK"
        }

        let dur = t - lastSeenTS

        // ★ 普通模式: 对齐 iOS — 只有 HOLD (保持) + RETURN (回中), 无惯性/扫描
        if !racing {
            if dur <= Cfg.lostHold {
                return "HOLD"
            } else {
                // 缓慢回中 (与 iOS servoReturnAlpha 逻辑一致)
                if t - lastSendTS >= Cfg.servoSendInterval {
                    let dH = Cfg.servoHome - servoH
                    let dV = Cfg.servoHome - servoV
                    if abs(dH) > 0.5 || abs(dV) > 0.5 {
                        servoH += dH * Cfg.returnAlpha
                        servoV += dV * Cfg.returnAlpha
                    }
                    lastSendTS = t
                }
                return "RETURN"
            }
        }

        // ★ 赛道模式: 完整的 惯性 → HOLD → SCAN → RETURN 状态机
        if dur <= Cfg.lostHold {
            if !inertiaOn {
                inertiaOn = true
                ancH = servoH
                ancV = servoV
                iVH = -lastVel.x * hFov
                iVV =  lastVel.y * vFov
            }
            if dur <= Cfg.inertiaDur {
                if t - lastSendTS >= Cfg.servoSendInterval {
                    if kf.alive {
                        // 赛道模式：用卡尔曼预测位置驱动舵机
                        let predCenter = kf.center
                        let errX = (predCenter.x - 0.5) * hFov
                        let errY = (predCenter.y - 0.5) * vFov
                        let kpH = Cfg.rKpH * 0.5
                        let kpV = Cfg.rKpV * 0.5
                        let dtSend = CGFloat(Cfg.servoSendInterval)
                        var sH = -errX * kpH * dtSend
                        var sV =  errY * kpV * dtSend
                        let mx = Cfg.inertiaMax * 1.5
                        sH = min(max(sH, -mx), mx)
                        sV = min(max(sV, -mx), mx)
                        servoH = min(max(servoH + sH, 0), 240)
                        servoV = min(max(servoV + sV, 0), 240)
                    } else {
                        iVH *= Cfg.inertiaDecay
                        iVV *= Cfg.inertiaDecay
                        let dtSend = CGFloat(Cfg.servoSendInterval)
                        let mx = Cfg.inertiaMax
                        let sH = min(max(iVH * dtSend, -mx), mx)
                        let sV = min(max(iVV * dtSend, -mx), mx)
                        servoH = min(max(servoH + sH, 0), 240)
                        servoV = min(max(servoV + sV, 0), 240)
                    }
                    ancH = servoH
                    ancV = servoV
                    lastSendTS = t
                }
                return kf.alive ? "KF-INERTIA" : "INERTIA"
            }
            return "HOLD"
        } else if dur <= Cfg.lostHold + Cfg.scanDur {
            if !searchOn {
                searchOn = true
                scanTS = t
            }
            if t - lastSendTS >= Cfg.servoSendInterval {
                let elapsed = CGFloat(t - scanTS)
                let phase = elapsed / Cfg.scanPeriod * 2.0 * .pi
                servoH = min(max(ancH + sin(phase) * Cfg.scanAmpH, 0), 240)
                servoV = min(max(ancV + sin(phase * 0.5) * Cfg.scanAmpV, 0), 240)
                lastSendTS = t
            }
            return "SCAN"
        } else {
            searchOn = false
            inertiaOn = false
            if t - lastSendTS >= Cfg.servoSendInterval {
                let dH = Cfg.servoHome - servoH
                let dV = Cfg.servoHome - servoV
                if abs(dH) > 0.5 || abs(dV) > 0.5 {
                    servoH += dH * Cfg.returnAlpha
                    servoV += dV * Cfg.returnAlpha
                }
                lastSendTS = t
            }
            return "RETURN"
        }
    }

    // MARK: 局部光流 (增强算法) — 在目标邻域内算光流，得到速度用于与位置速度混合
    private func runLocalFlow(prev: CVPixelBuffer, curr: CVPixelBuffer, curRect: CGRect, time: CFTimeInterval) -> CGPoint {
        let pad: CGFloat = 0.25
        let localRoi = CGRect(
            x: max(0, curRect.minX - pad),
            y: max(0, curRect.minY - pad),
            width: min(1 - max(0, curRect.minX - pad), curRect.width + 2 * pad),
            height: min(1 - max(0, curRect.minY - pad), curRect.height + 2 * pad)
        )
        guard localRoi.width >= 0.12, localRoi.height >= 0.12 else { return .zero }
        guard let prevPatch = cropPB(prev, localRoi), let currPatch = cropPB(curr, localRoi) else { return .zero }
        let pw = CVPixelBufferGetWidth(currPatch)
        let ph = CVPixelBufferGetHeight(currPatch)
        guard pw >= 32, ph >= 32 else { return .zero }

        let req = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: prevPatch)
        req.computationAccuracy = .low
        let handler = VNImageRequestHandler(cvPixelBuffer: currPatch, orientation: .up)
        guard (try? handler.perform([req])) != nil,
              let obs = req.results?.first as? VNPixelBufferObservation else { return .zero }
        let flowPB = obs.pixelBuffer
        let fw = CVPixelBufferGetWidth(flowPB)
        let fh = CVPixelBufferGetHeight(flowPB)
        guard fw > 0, fh > 0 else { return .zero }

        CVPixelBufferLockBaseAddress(flowPB, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowPB, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(flowPB) else { return .zero }
        let bpr = CVPixelBufferGetBytesPerRow(flowPB)
        let cx = fw / 2
        let cy = fh / 2
        let ptr = base.advanced(by: cy * bpr + cx * 8).assumingMemoryBound(to: Float32.self)
        let dx = CGFloat(ptr[0])
        let dy = CGFloat(ptr[1])

        let dt = time - lastProcessTime
        guard dt > 0.001, dt < 0.2 else { return .zero }
        let velX = (dx / CGFloat(pw)) * localRoi.width / dt
        let velY = (dy / CGFloat(ph)) * localRoi.height / dt
        return CGPoint(x: velX, y: velY)
    }

    private func copyPixelBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src)
        let h = CVPixelBufferGetHeight(src)
        var out: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferWidthKey: w,
            kCVPixelBufferHeightKey: h,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out) == kCVReturnSuccess,
              let dest = out else { return nil }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dest, [])
        defer {
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
            CVPixelBufferUnlockBaseAddress(dest, [])
        }
        let srcPtr = CVPixelBufferGetBaseAddress(src)!
        let destPtr = CVPixelBufferGetBaseAddress(dest)!
        let bpr = CVPixelBufferGetBytesPerRow(src)
        let destBpr = CVPixelBufferGetBytesPerRow(dest)
        if bpr == destBpr {
            memcpy(destPtr, srcPtr, bpr * h)
        } else {
            for y in 0..<h {
                memcpy(destPtr.advanced(by: y * destBpr), srcPtr.advanced(by: y * bpr), min(bpr, destBpr))
            }
        }
        return dest
    }

    // MARK: 工具
    private func cropPB(_ src: CVPixelBuffer, _ rgn: CGRect) -> CVPixelBuffer? {
        let w = CGFloat(CVPixelBufferGetWidth(src))
        let h = CGFloat(CVPixelBufferGetHeight(src))
        let cx = rgn.minX * w
        let cy = rgn.minY * h
        let cw = rgn.width * w
        let ch = rgn.height * h
        let fy = h - cy - ch
        let rect = CGRect(x: cx, y: fy, width: cw, height: ch)
        let cropped = CIImage(cvPixelBuffer: src)
            .cropped(to: rect)
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
        let ow = max(Int(cw), 64)
        let oh = max(Int(ch), 64)
        guard ow > 10 && oh > 10 else { return nil }
        var out: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferWidthKey: ow,
            kCVPixelBufferHeightKey: oh,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ]
        CVPixelBufferCreate(nil, ow, oh, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out)
        guard let o = out else { return nil }
        ci.render(cropped, to: o)
        return o
    }

    private func cgImg(_ pb: CVPixelBuffer) -> CGImage? {
        let ciImg = CIImage(cvPixelBuffer: pb)
        return ci.createCGImage(ciImg, from: ciImg.extent)
    }

    static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let i = a.intersection(b)
        if i.isNull || i.isEmpty { return 0 }
        let u = a.width * a.height + b.width * b.height - i.width * i.height
        return u > 0 ? i.width * i.height / u : 0
    }

    private func updStats(_ s: Snap, _ ym: Double, detSrc: String, trkAlive: Bool, srcCG: CGImage?) {
        statsLock.lock()
        _stats.frames += 1
        let n = Double(_stats.frames)
        // tracking 计数已移除 — "追踪率(含惯性)"不再使用
        if trkAlive { _stats.trkOnlyFrames += 1 }

        let eh = abs(s.errH)
        let ev = abs(s.errV)
        _stats.maxEH = max(_stats.maxEH, eh)
        _stats.maxEV = max(_stats.maxEV, ev)
        _stats.avgEH = _stats.avgEH * ((n-1)/n) + eh / n
        _stats.avgEV = _stats.avgEV * ((n-1)/n) + ev / n
        let ed = sqrt(Double(s.eX * s.eX + s.eY * s.eY))
        _stats.maxTD = max(_stats.maxTD, ed)
        _stats.avgTD = _stats.avgTD * ((n-1)/n) + ed / n
        _stats.peakVH = max(_stats.peakVH, abs(s.velH))
        _stats.peakVV = max(_stats.peakVV, abs(s.velV))
        let dh = s.velH > 1 ? 1 : (s.velH < -1 ? -1 : 0)
        if dh != 0 && Int(dh) != ldH { _stats.dirH += 1 }
        if dh != 0 { ldH = Int(dh) }
        let dv = s.velV > 1 ? 1 : (s.velV < -1 ? -1 : 0)
        if dv != 0 && Int(dv) != ldV { _stats.dirV += 1 }
        if dv != 0 { ldV = Int(dv) }
        if ym > 0 { _stats.avgYolo = _stats.avgYolo * ((n-1)/n) + ym / n }

        // ★ 标注 GT 评测 (对齐虚拟赛道口径)
        var frameGtVisible = false
        var frameYoloHit = false
        var frameTrackOnCar = false
        var frameSTOnCar = false
        var frameBestYoloIoU: CGFloat = 0
        var frameBestTrackIoU: CGFloat = 0
        var frameBestSTIoU: CGFloat = 0
        var frameGtInCrop = false
        var frameGtCenterInCrop = false
        if let annProj = annotationProject, let res = videoResolution {
            let gtBoxesThisFrame = annProj.boxes(forLocalFrame: s.idx)
            let gtVisible = !gtBoxesThisFrame.isEmpty
            frameGtVisible = gtVisible
            if gtVisible {
                _stats.gtAnnotationFrames += 1
                let imgW = res.width
                let imgH = res.height
                let allGtRects = gtBoxesThisFrame.map { $0.pixelRect(imgW: imgW, imgH: imgH) }
                let allGtNormRects = allGtRects.map {
                    CGRect(
                        x: $0.minX / imgW,
                        y: $0.minY / imgH,
                        width: $0.width / imgW,
                        height: $0.height / imgH
                    )
                }
                frameGtInCrop = allGtNormRects.contains { Self.iou($0, s.crop) > 0 }
                frameGtCenterInCrop = allGtNormRects.contains {
                    s.crop.contains(CGPoint(x: $0.midX, y: $0.midY))
                }

                // ── YOLO 检测率（GT 可见口径）: 本帧任一 YOLO 候选框与任一 GT 框 IoU > 0.2 ──
                do {
                    var bestIoU_yolo: CGFloat = 0
                    for cand in s.yoloCandidates {
                        let sourceRect = CGRect(
                            x: s.trackingCrop.minX + cand.rectTL.minX * s.trackingCrop.width,
                            y: s.trackingCrop.minY + cand.rectTL.minY * s.trackingCrop.height,
                            width: cand.rectTL.width * s.trackingCrop.width,
                            height: cand.rectTL.height * s.trackingCrop.height
                        )
                        let yoloPixel = CGRect(
                            x: sourceRect.minX * imgW,
                            y: sourceRect.minY * imgH,
                            width: sourceRect.width * imgW,
                            height: sourceRect.height * imgH
                        )
                        bestIoU_yolo = max(bestIoU_yolo, allGtRects.map { SimEngine.iou(yoloPixel, $0) }.max() ?? 0)
                    }
                    frameBestYoloIoU = bestIoU_yolo
                    frameYoloHit = bestIoU_yolo > 0.2
                    if frameYoloHit { _stats.yoloHitFrames += 1 }
                }

                // ── 一般追踪率: 输出框 (smRect) 与任一 GT 框 IoU > 0.2 ──
                // ★ 方案B辅助修复: 同时考虑 ST 输出框，取两者中较大 IoU
                do {
                    var bestIoU_gen: CGFloat = 0
                    var detPixel: CGRect = .zero

                    // 1) smRect 与 GT 的 IoU
                    if let det = s.det {
	                        detPixel = CGRect(
	                            x: (s.trackingCrop.minX + det.minX * s.trackingCrop.width) * imgW,
	                            y: (s.trackingCrop.minY + det.minY * s.trackingCrop.height) * imgH,
	                            width: det.width * s.trackingCrop.width * imgW,
	                            height: det.height * s.trackingCrop.height * imgH
	                        )
                        bestIoU_gen = allGtRects.map { SimEngine.iou(detPixel, $0) }.max() ?? 0
                    }

                    // 2) ★ ST 输出框与 GT 的 IoU（当 ST 有有效输出时）
                    var bestIoU_stForGen: CGFloat = 0
                    if enableSingleTarget, let stRect = stStats.stPrevRect {
	                        let stPixelForGen = CGRect(
	                            x: (s.trackingCrop.minX + stRect.minX * s.trackingCrop.width) * imgW,
	                            y: (s.trackingCrop.minY + stRect.minY * s.trackingCrop.height) * imgH,
	                            width: stRect.width * s.trackingCrop.width * imgW,
	                            height: stRect.height * s.trackingCrop.height * imgH
	                        )
                        bestIoU_stForGen = allGtRects.map { SimEngine.iou(stPixelForGen, $0) }.max() ?? 0
                    }

                    // 3) 取两者中较大的 IoU 判定是否追踪成功
                    let effectiveIoU = max(bestIoU_gen, bestIoU_stForGen)
                    frameBestTrackIoU = effectiveIoU
                    let onCar = effectiveIoU > 0.2
                    frameTrackOnCar = onCar
                    if onCar { _stats.trackOnCarFrames += 1 }

                    // ★ 调试日志: 每10帧输出一次坐标和 IoU
                    if s.idx % 10 == 0, let det = s.det {
                        let gt0 = allGtRects.first ?? .zero
	                        let dbgLine = "[GT-DBG] f=\(s.idx) det=\(String(format: "%.1f,%.1f,%.1f,%.1f", det.minX, det.minY, det.width, det.height)) crop=\(String(format: "%.3f,%.3f,%.3f,%.3f", s.trackingCrop.minX, s.trackingCrop.minY, s.trackingCrop.width, s.trackingCrop.height)) detPx=\(String(format: "%.0f,%.0f,%.0f,%.0f", detPixel.minX, detPixel.minY, detPixel.width, detPixel.height)) gt0Px=\(String(format: "%.0f,%.0f,%.0f,%.0f", gt0.minX, gt0.minY, gt0.width, gt0.height)) iou=\(String(format: "%.3f", bestIoU_gen)) stIoU=\(String(format: "%.3f", bestIoU_stForGen)) effIoU=\(String(format: "%.3f", effectiveIoU)) img=\(String(format: "%.0fx%.0f", imgW, imgH))\n"
                        let logPath = NSTemporaryDirectory() + "gt_debug.log"
                        if let fh = FileHandle(forWritingAtPath: logPath) {
                            fh.seekToEndOfFile()
                            fh.write(dbgLine.data(using: .utf8)!)
                            fh.closeFile()
                        } else {
                            FileManager.default.createFile(atPath: logPath, contents: dbgLine.data(using: .utf8))
                        }
                    }
                }

                // ── 单目标锁定率: ST 锁定框与 GT IoU > 0.15 ──
                if enableSingleTarget {
                    _stats.stEvalFrames += 1
                    if let stRect = stStats.stPrevRect {
                        // stPrevRect 是裁切内归一化，转为源图像素
	                        let stPixel = CGRect(
	                            x: (s.trackingCrop.minX + stRect.minX * s.trackingCrop.width) * imgW,
	                            y: (s.trackingCrop.minY + stRect.minY * s.trackingCrop.height) * imgH,
	                            width: stRect.width * s.trackingCrop.width * imgW,
	                            height: stRect.height * s.trackingCrop.height * imgH
	                        )
                        let bestIoU_st = allGtRects.map { SimEngine.iou(stPixel, $0) }.max() ?? 0
                        frameBestSTIoU = bestIoU_st
                        let stOnCar = bestIoU_st > 0.15
                        frameSTOnCar = stOnCar
                        if stOnCar { _stats.stLockOnCarFrames += 1 }

                        // ★ 调试日志: ST 框对比
                        if s.idx % 10 == 0 {
                            let gt0 = allGtRects.first ?? .zero
                            let dbgLine = "[ST-DBG] f=\(s.idx) stRect=\(String(format: "%.1f,%.1f,%.1f,%.1f", stRect.minX, stRect.minY, stRect.width, stRect.height)) stPx=\(String(format: "%.0f,%.0f,%.0f,%.0f", stPixel.minX, stPixel.minY, stPixel.width, stPixel.height)) gt0Px=\(String(format: "%.0f,%.0f,%.0f,%.0f", gt0.minX, gt0.minY, gt0.width, gt0.height)) iou=\(String(format: "%.3f", bestIoU_st))\n"
                            let logPath = NSTemporaryDirectory() + "gt_debug.log"
                            if let fh = FileHandle(forWritingAtPath: logPath) {
                                fh.seekToEndOfFile()
                                fh.write(dbgLine.data(using: .utf8)!)
                                fh.closeFile()
                            } else {
                                FileManager.default.createFile(atPath: logPath, contents: dbgLine.data(using: .utf8))
                            }
                        }
                    }
                }
            }
        }
        if annotationProject != nil {
            evalFrameHistory.append(EvalFrameStatus(
                frame: s.idx,
                time: s.t,
                gtVisible: frameGtVisible,
                yoloHit: frameYoloHit,
                trackOnCar: frameTrackOnCar,
                stOnCar: frameSTOnCar,
                yoloRan: s.yoloRanThisFrame,
                yoloCandidateCount: s.yoloCandidates.count,
                bestYoloIoU: Double(frameBestYoloIoU),
                bestTrackIoU: Double(frameBestTrackIoU),
                bestSTIoU: Double(frameBestSTIoU),
                gtInCrop: frameGtInCrop,
                gtCenterInCrop: frameGtCenterInCrop,
                cropX: Double(s.crop.minX),
                cropY: Double(s.crop.minY),
                cropW: Double(s.crop.width),
                cropH: Double(s.crop.height),
                detSource: s.detSrc,
                state: s.state,
                stLostFrames: singleTargetTracker.lostFrames,
                targetLostFrames: targetLostFrames,
                yoloMs: s.yoloMs
            ))
        }

        statsLock.unlock()

        // ★ Ground Truth SAHI (独立模型 + 并发分片, 异步执行不阻塞帧处理)
        if enableGtSahi, let cg = srcCG, !gtBusy {
            gtBusy = true
            let frameN = n
            gtQueue.async { [weak self] in
                guard let self = self else { return }
                let t0 = CACurrentMediaTime()
                let gtDets = self.runGtSahiYolo(srcCG: cg)
                let gtMs = (CACurrentMediaTime() - t0) * 1000
                if !gtDets.isEmpty { self._stats.gtDetectable += 1 }
                self._stats.gtSahiMs = self._stats.gtSahiMs * ((frameN-1)/frameN) + gtMs / frameN
                self.gtBusy = false
            }
        }
    }

    // MARK: 导出 CSV
    func exportCSV(_ url: URL) {
        var s = "frame,time,state,kf,detSrc,yoloMs,cropX,cropY,cropW,cropH,"
        s += "cmdH,cmdV,actH,actV,errH,errV,velH,velV,eDegX,eDegY,optZoom,digZoom,effFovH\n"
        for f in history {
            s += "\(f.idx),\(String(format: "%.4f", f.t)),\(f.state),\(f.kfSt),"
            s += "\(f.detSrc),\(String(format: "%.1f", f.yoloMs)),"
            s += "\(String(format: "%.4f", f.crop.minX)),\(String(format: "%.4f", f.crop.minY)),"
            s += "\(String(format: "%.4f", f.crop.width)),\(String(format: "%.4f", f.crop.height)),"
            s += "\(String(format: "%.2f", f.cmdH)),\(String(format: "%.2f", f.cmdV)),"
            s += "\(String(format: "%.2f", f.actH)),\(String(format: "%.2f", f.actV)),"
            s += "\(String(format: "%.3f", f.errH)),\(String(format: "%.3f", f.errV)),"
            s += "\(String(format: "%.1f", f.velH)),\(String(format: "%.1f", f.velV)),"
            s += "\(String(format: "%.2f", f.eX)),\(String(format: "%.2f", f.eY)),"
            s += "\(String(format: "%.2f", f.optZoom)),\(String(format: "%.2f", f.digZoom)),\(String(format: "%.1f", f.effFovH))\n"
        }
        try? s.write(to: url, atomically: true, encoding: .utf8)

        let yoloURL = url.deletingLastPathComponent().appendingPathComponent("yolo_candidates.csv")
        func csvEsc(_ value: String) -> String {
            let escaped = value.replacingOccurrences(of: "\"", with: "\"\"")
            if escaped.contains(",") || escaped.contains("\n") || escaped.contains("\"") {
                return "\"\(escaped)\""
            }
            return escaped
        }
        var ys = "frame,time,yoloRan,candidateIndex,accepted,label,confidence,"
        ys += "sourceX,sourceY,sourceW,sourceH,cropX,cropY,cropW,cropH,"
        ys += "totalScore,appSim,motionSim,sizeSim,rejectReason\n"
        for f in history where f.yoloRanThisFrame {
            for (idx, cand) in f.yoloCandidates.enumerated() {
                let src = CGRect(
                    x: f.trackingCrop.minX + cand.rectTL.minX * f.trackingCrop.width,
                    y: f.trackingCrop.minY + cand.rectTL.minY * f.trackingCrop.height,
                    width: cand.rectTL.width * f.trackingCrop.width,
                    height: cand.rectTL.height * f.trackingCrop.height
                )
                ys += "\(f.idx),\(String(format: "%.4f", f.t)),\(f.yoloRanThisFrame ? 1 : 0),\(idx),"
                ys += "\(cand.accepted ? 1 : 0),\(csvEsc(cand.label)),\(String(format: "%.4f", cand.confidence)),"
                ys += "\(String(format: "%.6f", src.minX)),\(String(format: "%.6f", src.minY)),"
                ys += "\(String(format: "%.6f", src.width)),\(String(format: "%.6f", src.height)),"
                ys += "\(String(format: "%.6f", cand.rectTL.minX)),\(String(format: "%.6f", cand.rectTL.minY)),"
                ys += "\(String(format: "%.6f", cand.rectTL.width)),\(String(format: "%.6f", cand.rectTL.height)),"
                ys += "\(String(format: "%.4f", cand.totalScore)),\(String(format: "%.4f", cand.appSim)),"
                ys += "\(String(format: "%.4f", cand.motionSim)),\(String(format: "%.4f", cand.sizeSim)),"
                ys += "\(csvEsc(cand.rejectReason))\n"
            }
        }
        try? ys.write(to: yoloURL, atomically: true, encoding: .utf8)

        // ★ 附加 GT SAHI 汇总到同名 _summary.txt
        let sumURL = url.deletingPathExtension().appendingPathExtension("summary.txt")
        statsLock.lock()
        let st = _stats
        statsLock.unlock()
        var sum = "=== GT SAHI 追踪分析汇总 ===\n"
        sum += "总帧数: \(st.frames)\n"
        sum += "纯VNTracker帧: \(st.trkOnlyFrames) (\(String(format: "%.1f%%", st.pureTrkPct)))\n"
        sum += "GT SAHI可见帧: \(st.gtDetectable) (\(String(format: "%.1f%%", st.gtVisiblePct)))\n"
        sum += "追踪效率(Trk/GT): \(String(format: "%.1f%%", st.trkEfficiency))\n"
        sum += "SAHI平均耗时: \(String(format: "%.1fms", st.gtSahiMs))\n"
        sum += "SAHI配置: tile=\(Cfg.gtSahiTileSize) overlap=\(Cfg.gtSahiOverlap) conf=\(Cfg.gtSahiConf)\n"
        sum += "变焦仿真: \(enableZoom ? "启用" : "关闭") optRange=\(Cfg.opticalMin)-\(Cfg.opticalMax)x digRange=\(Cfg.digitalZoomMin)-\(Cfg.digitalZoomMax)x\n"
        try? sum.write(to: sumURL, atomically: true, encoding: .utf8)
    }

    func exportEvalFrameCSV(_ url: URL) {
        statsLock.lock()
        let rows = evalFrameHistory
        statsLock.unlock()
        var s = "frame,time,gtVisible,yoloHit,trackOnCar,stOnCar,"
        s += "yoloRan,yoloCandidateCount,bestYoloIoU,bestTrackIoU,bestSTIoU,"
        s += "gtInCrop,gtCenterInCrop,cropX,cropY,cropW,cropH,detSource,state,stLostFrames,targetLostFrames,yoloMs\n"
        for r in rows {
            s += "\(r.frame),\(String(format: "%.4f", r.time)),"
            s += "\(r.gtVisible ? 1 : 0),\(r.yoloHit ? 1 : 0),"
            s += "\(r.trackOnCar ? 1 : 0),\(r.stOnCar ? 1 : 0),"
            s += "\(r.yoloRan ? 1 : 0),\(r.yoloCandidateCount),"
            s += "\(String(format: "%.4f", r.bestYoloIoU)),"
            s += "\(String(format: "%.4f", r.bestTrackIoU)),"
            s += "\(String(format: "%.4f", r.bestSTIoU)),"
            s += "\(r.gtInCrop ? 1 : 0),\(r.gtCenterInCrop ? 1 : 0),"
            s += "\(String(format: "%.4f", r.cropX)),\(String(format: "%.4f", r.cropY)),"
            s += "\(String(format: "%.4f", r.cropW)),\(String(format: "%.4f", r.cropH)),"
            s += "\(r.detSource),\(r.state),\(r.stLostFrames),\(r.targetLostFrames),"
            s += "\(String(format: "%.2f", r.yoloMs))\n"
        }
        try? s.write(to: url, atomically: true, encoding: .utf8)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - SwiftUI App
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
// MARK: - EnvLoader (自动加载 .env 环境变量)
// ═══════════════════════════════════════════════════════════════

/// 从项目根目录 .env 文件加载环境变量，优先级：
/// 1. 系统环境变量（已设置的不会被覆盖）
/// 2. .env 文件
enum EnvLoader {
    /// 已加载的 .env 键值对缓存
    private(set) static var loaded: [String: String] = [:]

    /// 在 App 启动时调用一次
    static func load() {
        // Public package: no automatic discovery of local credential files.
        // Optional cloud annotation only reads environment values explicitly supplied by the user.
    }

    /// 获取环境变量值（优先系统环境变量 → .env 缓存）
    static func get(_ key: String) -> String? {
        return ProcessInfo.processInfo.environment[key] ?? loaded[key]
    }

    /// 获取 Replicate API Token
    static var replicateToken: String {
        return get("REPLICATE_API_TOKEN") ?? ""
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - AppDelegate (数据集模式启动器)
// ═══════════════════════════════════════════════════════════════

class RCSimAppDelegate: NSObject, NSApplicationDelegate {
    // ★ shared 指向 SwiftUI 创建的实例 (通过 appDelegate 属性设置)
    static var shared: RCSimAppDelegate?
    var datasetGen: DatasetGenerator?
    var annotateEngine: VideoAnnotationEngine?

    func applicationDidFinishLaunching(_ notification: Notification) {
        RCSimAppDelegate.shared = self  // ★ 注册 shared
        EnvLoader.load()               // ★ 自动加载 .env 环境变量
        // ── 调试日志写入文件 ──
        if !CommandLine.arguments.contains("--eval") {
            let debugLog = NSHomeDirectory() + "/Desktop/annotate_debug.log"
            let argsStr = CommandLine.arguments.joined(separator: " ")
            let logMsg = "[\(Date())] applicationDidFinishLaunching called. Args: \(argsStr)\n"
            if let data = logMsg.data(using: .utf8) {
                if FileManager.default.fileExists(atPath: debugLog) {
                    if let fh = FileHandle(forWritingAtPath: debugLog) {
                        fh.seekToEndOfFile(); fh.write(data); fh.closeFile()
                    }
                } else {
                    FileManager.default.createFile(atPath: debugLog, contents: data)
                }
            }
        }
        if CommandLine.arguments.contains("--batch-sim")
            || CommandLine.arguments.contains("--batch-newcam50")
            || CommandLine.arguments.contains("--batch-st-unseen100")
            || CommandLine.arguments.contains("--batch-st10")
            || CommandLine.arguments.contains("--batch-recall-test")
            || CommandLine.arguments.contains("--batch-recall-fp-test")
            || CommandLine.arguments.contains("--batch-recall-fp-env10-st60")
            || CommandLine.arguments.contains("--batch-recall-fp-env40-st60")
            || CommandLine.arguments.contains("--batch-tracker7w-st60")
            || CommandLine.arguments.contains("--batch-tracker7w-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-run009-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-track1-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-run009-yolo-calib-pair-st60")
            || CommandLine.arguments.contains("--batch-run009-simyolo-calib-st60")
            || CommandLine.arguments.contains("--batch-track1-30s-lock-threshold")
            || CommandLine.arguments.contains("--batch-simyolo-calib-pair")
            || CommandLine.arguments.contains("--batch-multiscene10-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-multiscene10-simyolo-aligned-st60")
            || CommandLine.arguments.contains("--batch-multiscene10-yolo-simyolo-pair-st60")
            || CommandLine.arguments.contains("--batch-marker-cv-multiscene-30s")
            || CommandLine.arguments.contains("--batch-fiducial-cv-multiscene20-30s")
            || CommandLine.arguments.contains("--batch-id-fiducial-cv-multiscene20-30s")
            || CommandLine.arguments.contains("--batch-marker-yolo-threshold-30s")
            || CommandLine.arguments.contains("--batch-marker-cv-video-samples")
            || CommandLine.arguments.contains("--batch-fiducial-cv-video-samples")
            || CommandLine.arguments.contains("--batch-id-fiducial-cv-video-samples")
            || CommandLine.arguments.contains("--batch-id-fiducial-cv-smoke5")
            || CommandLine.arguments.contains("--batch-verify-st-095-010")
            || CommandLine.arguments.contains("--export-training")
            || CommandLine.arguments.contains("--dagger-training") {
            let isTraining = CommandLine.arguments.contains("--export-training") || CommandLine.arguments.contains("--dagger-training")
            let isDagger = CommandLine.arguments.contains("--dagger-training")
            print("═══ \(isDagger ? "DAgger Training" : isTraining ? "Training Data Export" : "Batch") Mode: applicationDidFinishLaunching ═══")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                if !BatchSimRunner.shared.isRunning {
                    let config: BatchSimConfig
                    if CommandLine.arguments.contains("--public-smoke") {
                        config = .publicSmoke
                    } else if isDagger {
                        // DAgger 模式：100 轮 × 60 秒，噪声注入
                        config = .dagger
                    } else if isTraining {
                        config = BatchSimConfig(runs: Array(BatchSimConfig.deterministic100Runs.prefix(20)), simDuration: 30.0,
                                         outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/training_output")
                    } else if CommandLine.arguments.contains("--batch-recall-test") {
                        config = .recallTest
                    } else if CommandLine.arguments.contains("--batch-recall-fp-test") {
                        config = .recallFpGrid
                    } else if CommandLine.arguments.contains("--batch-track1-real-yolo-st60") {
                        config = .track1RealYoloST60
                    } else if CommandLine.arguments.contains("--batch-run009-real-yolo-st60") {
                        config = .run009RealYoloST60
                    } else if CommandLine.arguments.contains("--batch-run009-yolo-calib-pair-st60") {
                        config = .run009YoloCalibPairST60
                    } else if CommandLine.arguments.contains("--batch-run009-simyolo-calib-st60") {
                        config = .run009SimYoloCalibST60
                    } else if CommandLine.arguments.contains("--batch-recall-fp-env10-st60") {
                        config = .recallFpEnv10ST60
                    } else if CommandLine.arguments.contains("--batch-recall-fp-env40-st60") {
                        config = .recallFpEnv40ST60
                    } else if CommandLine.arguments.contains("--batch-tracker7w-st60") {
                        config = .tracker7wST60
                    } else if CommandLine.arguments.contains("--batch-tracker7w-real-yolo-st60") {
                        config = .tracker7wRealYoloST60
                    } else if CommandLine.arguments.contains("--batch-track1-30s-lock-threshold") {
                        config = .track1ThirtySecondLockThreshold
                    } else if CommandLine.arguments.contains("--batch-simyolo-calib-pair") {
                        config = .simYoloCalibPair
                    } else if CommandLine.arguments.contains("--batch-multiscene10-real-yolo-st60") {
                        config = .multiscene10RealYoloST60
                    } else if CommandLine.arguments.contains("--batch-multiscene10-simyolo-aligned-st60") {
                        config = .multiscene10SimYoloAlignedST60
                    } else if CommandLine.arguments.contains("--batch-multiscene10-yolo-simyolo-pair-st60") {
                        config = .multiscene10YoloSimYoloPairST60
                    } else if CommandLine.arguments.contains("--batch-marker-cv-video-samples") {
                        config = .markerCVVideoSamples
                    } else if CommandLine.arguments.contains("--batch-marker-cv-multiscene-30s") {
                        config = .markerCVMultiscene30
                    } else if CommandLine.arguments.contains("--batch-fiducial-cv-video-samples") {
                        config = .fiducialCVVideoSamples
                    } else if CommandLine.arguments.contains("--batch-fiducial-cv-multiscene20-30s") {
                        config = .fiducialCVMultiscene20
                    } else if CommandLine.arguments.contains("--batch-id-fiducial-cv-video-samples") {
                        config = .idFiducialCVVideoSamples
                    } else if CommandLine.arguments.contains("--batch-id-fiducial-cv-smoke5") {
                        config = .idFiducialCVSmoke5
                    } else if CommandLine.arguments.contains("--batch-id-fiducial-cv-multiscene20-30s") {
                        config = .idFiducialCVMultiscene20
                    } else if CommandLine.arguments.contains("--batch-marker-yolo-threshold-30s") {
                        config = .markerYoloThreshold30
                    } else if CommandLine.arguments.contains("--batch-verify-st-095-010") {
                        config = .stVerify095010
                    } else if CommandLine.arguments.contains("--batch-st10") {
                        config = .stDegrade10
                    } else if CommandLine.arguments.contains("--batch-st-unseen100") {
                        config = .stUnseen100
                    } else if CommandLine.arguments.contains("--batch-newcam50") {
                        config = .newCam50
                    } else {
                        config = .default
                    }
                    BatchSimRunner.shared.startBatch(config: config)
                }
                // 批量模式只保留一个主窗口，关闭额外恢复/新建窗口。
                if NSApp.windows.count > 1 {
                    for w in NSApp.windows.dropFirst() {
                        w.close()
                    }
                }
            }
            return
        }
        // ── Annotate 模式（支持命令行参数或标志文件）──
        let annotateFlag = NSHomeDirectory() + "/Desktop/.annotate_mode"
        let isAnnotateMode = CommandLine.arguments.contains("--annotate") || FileManager.default.fileExists(atPath: annotateFlag)
        if isAnnotateMode {
            // 清除标志文件
            try? FileManager.default.removeItem(atPath: annotateFlag)
            print("═══ Annotate Mode: applicationDidFinishLaunching ═══")
            UserDefaults.standard.set(false, forKey: "NSQuitAlwaysKeepsWindows")
            // 直接在 AppDelegate 中启动标注流程，不依赖 SwiftUI View 的 onAppear
            let engine = VideoAnnotationEngine()
            self.annotateEngine = engine
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.startAnnotateFromDelegate(engine: engine)
            }
            return
        }
        // ── ★ Eval 模式: 命令行真实视频评测 ──
        // 用法: macRCtrackerSimulator --eval --video <path> --annotation <path>
        if CommandLine.arguments.contains("--eval") {
            print("═══ Eval Mode: applicationDidFinishLaunching ═══")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.startEvalFromDelegate()
            }
            return
        }
        // ── GT Test 模式: 命令行纯 YOLO GT 评测 ──
        if CommandLine.arguments.contains("--gt-test") {
            print("═══ GT Test Mode: applicationDidFinishLaunching ═══")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                GTTestCLIRunner.run()
            }
            return
        }
        // 新增：--split-ablation-lite <base-dir> 精简模式（840 帧，~15 分钟）
        // 用于快速验证 session 切分有效性，供 Manus 本地 1 小时训练
        if CommandLine.arguments.contains("--split-ablation-lite") {
            print("═══ Split Ablation Mode (LITE): applicationDidFinishLaunching ═══")
            let args = CommandLine.arguments
            let baseDir: String = {
                if let idx = args.firstIndex(of: "--split-ablation-lite"),
                   idx + 1 < args.count,
                   !args[idx + 1].hasPrefix("--") {
                    return args[idx + 1]
                }
                return NSHomeDirectory() + "/Desktop/vsim_split_ablation_lite"
            }()

            let gen = DatasetGenerator()
            self.datasetGen = gen

            let useMultiLabel = CommandLine.arguments.contains("--multi-label")
            print("═══ Starting Split Ablation (LITE), base dir = \(baseDir), multiLabel=\(useMultiLabel) ═══")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                gen.runSplitAblation(baseDir: baseDir, lite: true, useMultiLabel: useMultiLabel) {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                        print("═══ Split Ablation LITE 全部完成，退出 App ═══")
                        NSApplication.shared.terminate(nil)
                    }
                }
            }
            return
        }

        // 新增：--split-ablation <base-dir> 一键跑 pool + holdout
        // 对应任务：vsim_切分方法论验证_任务书.md
        if CommandLine.arguments.contains("--split-ablation") {
            print("═══ Split Ablation Mode: applicationDidFinishLaunching ═══")
            let args = CommandLine.arguments
            let baseDir: String = {
                if let idx = args.firstIndex(of: "--split-ablation"),
                   idx + 1 < args.count,
                   !args[idx + 1].hasPrefix("--") {
                    return args[idx + 1]
                }
                return NSHomeDirectory() + "/Desktop/vsim_split_ablation"
            }()

            let gen = DatasetGenerator()
            self.datasetGen = gen

            let useMultiLabel = CommandLine.arguments.contains("--multi-label")
            print("═══ Starting Split Ablation, base dir = \(baseDir), multiLabel=\(useMultiLabel) ═══")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                gen.runSplitAblation(baseDir: baseDir, useMultiLabel: useMultiLabel) {
                    // pool + holdout 两阶段都完成后退出 app
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                        print("═══ Split Ablation 全部完成，退出 App ═══")
                        NSApplication.shared.terminate(nil)
                    }
                }
            }
            return
        }

        // 新增：--scale-effect-dataset <base-dir>
        // 生成 YOLO 数据量缩放实验需要的 train_pool / val / OOD eval。
        if CommandLine.arguments.contains("--scale-effect-dataset") {
            print("═══ Scale Effect Dataset Mode: applicationDidFinishLaunching ═══")
            let args = CommandLine.arguments
            let baseDir: String = {
                if let idx = args.firstIndex(of: "--scale-effect-dataset"),
                   idx + 1 < args.count,
                   !args[idx + 1].hasPrefix("--") {
                    return args[idx + 1]
                }
                return NSHomeDirectory() + "/Desktop/macRCSimulator/vsim_scale_effect_20260505"
            }()

            let gen = DatasetGenerator()
            self.datasetGen = gen

            print("═══ Starting Scale Effect dataset, base dir = \(baseDir) ═══")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                gen.runScaleEffectDataset(baseDir: baseDir) {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                        print("═══ Scale Effect Dataset 全部完成，退出 App ═══")
                        NSApplication.shared.terminate(nil)
                    }
                }
            }
            return
        }

        guard CommandLine.arguments.contains("--generate-dataset") else { return }
        print("═══ Dataset Mode: applicationDidFinishLaunching ═══")

        let gen = DatasetGenerator()
        self.datasetGen = gen

        let args = CommandLine.arguments
        var cfg = DatasetConfig()
        if let idx = args.firstIndex(of: "--count"), idx + 1 < args.count,
           let n = Int(args[idx + 1]) { cfg.totalImages = n }
        if let idx = args.firstIndex(of: "--output"), idx + 1 < args.count {
            cfg.outputDir = args[idx + 1]
        }
        if let idx = args.firstIndex(of: "--size"), idx + 1 < args.count,
           let n = Int(args[idx + 1]) { cfg.imageSize = n }

        // 多标签模式开关
        if args.contains("--multi-label") {
            cfg.useMultiLabel = true
        }

        print("═══ Starting generation: \(cfg.totalImages) images to \(cfg.outputDir), multiLabel=\(cfg.useMultiLabel) ═══")
        // ★ 延迟启动: 确保 SwiftUI 窗口和 RunLoop 已就绪
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            print("═══ Timer fired, calling startGeneration ═══")
            gen.startGeneration(config: cfg)
        }
    }

    // ── SAM3 Annotate: 在 AppDelegate 中直接驱动标注 ──
    private func startAnnotateFromDelegate(engine: VideoAnnotationEngine) {
        let logFile = NSHomeDirectory() + "/Desktop/annotate_debug.log"
        func alog(_ msg: String) {
            let ts = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
            let line = "[\(ts)] \(msg)\n"
            print(line, terminator: "")
            if let data = line.data(using: .utf8), let fh = FileHandle(forWritingAtPath: logFile) {
                fh.seekToEndOfFile(); fh.write(data); fh.closeFile()
            }
        }
        // 读取配置
        let srcConfigPath = NSHomeDirectory() + "/Documents/MacRCSimulator/annotate_config.json"
        guard FileManager.default.fileExists(atPath: srcConfigPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: srcConfigPath)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            alog("ERROR: 无法读取配置文件: \(srcConfigPath)")
            return
        }
        let videoDir = json["video_dir"] as? String ?? ""
        let outputDir = json["output_dir"] as? String ?? NSHomeDirectory() + "/Desktop/SAM3_Annotations"
        let token: String = {
            // 优先级: .env / 系统环境变量 → config JSON
            if let envToken = EnvLoader.get("REPLICATE_API_TOKEN"), !envToken.isEmpty {
                return envToken
            }
            return json["token"] as? String ?? ""
        }()
        let prompt = json["prompt"] as? String ?? "RC car"
        let singleVideo = json["single_video"] as? String
        let videoURLs = json["video_urls"] as? [String: String] ?? [:]
        alog("═══ SAM3 批量视频标注 (AppDelegate) ═══")
        alog("配置文件: \(srcConfigPath)")
        alog("视频目录: \(videoDir)")
        alog("输出目录: \(outputDir)")
        alog("Token: \(token.prefix(10))...")
        alog("Prompt: \(prompt)")
        alog("预上传 URL 映射: \(videoURLs.count) 个")
        // 收集视频
        var videos: [URL] = []
        if let single = singleVideo {
            let p = (videoDir as NSString).appendingPathComponent(single)
            videos = [URL(fileURLWithPath: p)]
        } else {
            let dirURL = URL(fileURLWithPath: videoDir)
            if let contents = try? FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) {
                let exts = Set(["mp4", "mov", "avi", "m4v", "MP4", "MOV"])
                videos = contents.filter { exts.contains($0.pathExtension) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            }
        }
        guard !videos.isEmpty else {
            alog("ERROR: 未找到视频文件")
            return
        }
        alog("找到 \(videos.count) 个视频文件")
        try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        // 配置 SAM3
        engine.sam3Cfg = SAM3Cfg(apiToken: token, prompt: prompt)
        // 开始处理
        processVideo(at: 0, videos: videos, engine: engine, outputDir: outputDir, videoURLs: videoURLs, alog: alog)
    }
    private func processVideo(at index: Int, videos: [URL], engine: VideoAnnotationEngine, outputDir: String, videoURLs: [String: String] = [:], alog: @escaping (String) -> Void) {
        guard index < videos.count else {
            alog("═══ ✓ 全部 \(videos.count) 个视频标注完成! ═══")
            alog("输出目录: \(outputDir)")
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                NSApplication.shared.terminate(nil)
            }
            return
        }
        let videoURL = videos[index]
        let videoName = videoURL.deletingPathExtension().lastPathComponent
        let existingJSON = URL(fileURLWithPath: outputDir).appendingPathComponent(videoName).appendingPathComponent("\(videoName)_annotations.json")
        if FileManager.default.fileExists(atPath: existingJSON.path) {
            alog("── [\(index + 1)/\(videos.count)] 跳过(已有标注): \(videoURL.lastPathComponent) ──")
            processVideo(at: index + 1, videos: videos, engine: engine, outputDir: outputDir, videoURLs: videoURLs, alog: alog)
            return
        }
        alog("── [\(index + 1)/\(videos.count)] 处理: \(videoURL.lastPathComponent) ──")
        engine.loadVideo(url: videoURL)
        waitAndAnnotate(engine: engine, videoURL: videoURL, index: index, videos: videos, outputDir: outputDir, videoURLs: videoURLs, alog: alog, retries: 0)
    }
    private func waitAndAnnotate(engine: VideoAnnotationEngine, videoURL: URL, index: Int, videos: [URL], outputDir: String, videoURLs: [String: String] = [:], alog: @escaping (String) -> Void, retries: Int) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            if engine.isLoaded {
                alog("  视频已加载: \(engine.totalFrames) 帧, \(String(format: "%.1f", engine.fps)) fps")
                engine.sam3.updateCfg(engine.sam3Cfg)
                // 优先使用预上传的远程 URL
                let videoName = videoURL.lastPathComponent
                if let remoteURL = videoURLs[videoName] {
                    alog("  使用预上传 URL: \(remoteURL)")
                    engine.runSAM3WithURL(remoteURL)
                } else {
                    alog("  无预上传 URL，使用本地上传模式")
                    engine.runSAM3()
                }
                alog("  SAM3 标注已启动...")
                self?.monitorAnnotation(engine: engine, videoURL: videoURL, index: index, videos: videos, outputDir: outputDir, videoURLs: videoURLs, alog: alog)
            } else if retries < 15 {
                self?.waitAndAnnotate(engine: engine, videoURL: videoURL, index: index, videos: videos, outputDir: outputDir, videoURLs: videoURLs, alog: alog, retries: retries + 1)
            } else {
                alog("  ERROR: 视频加载超时，跳过")
                self?.processVideo(at: index + 1, videos: videos, engine: engine, outputDir: outputDir, videoURLs: videoURLs, alog: alog)
            }
        }
    }
    private func monitorAnnotation(engine: VideoAnnotationEngine, videoURL: URL, index: Int, videos: [URL], outputDir: String, videoURLs: [String: String] = [:], alog: @escaping (String) -> Void) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            alog("  [monitor] SAM3 status: \(engine.sam3.status), progress: \(engine.sam3.progress)")
            switch engine.sam3.status {
            case .done(let n):
                alog("  ✓ SAM3 完成: \(n) 个标注框")
                // 保存结果
                let videoName = videoURL.deletingPathExtension().lastPathComponent
                let videoOutDir = URL(fileURLWithPath: outputDir).appendingPathComponent(videoName)
                let yoloDir = videoOutDir.appendingPathComponent("yolo")
                engine.exportYOLO(to: yoloDir)
                alog("  ✓ YOLO 导出: \(yoloDir.path)")
                let jsonPath = videoOutDir.appendingPathComponent("\(videoName)_annotations.json")
                engine.exportJSON(to: jsonPath)
                alog("  ✓ JSON 导出: \(jsonPath.path)")
                let s = engine.stats
                alog("  统计: \(s.annotated) 帧有标注, \(s.boxes) 个框, 覆盖率 \(String(format: "%.1f%%", s.coverage * 100))")
                // 下一个
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    engine.sam3.status = .idle
                    engine.sam3.progress = 0
                    self?.processVideo(at: index + 1, videos: videos, engine: engine, outputDir: outputDir, videoURLs: videoURLs, alog: alog)
                }
            case .failed(let msg):
                alog("  ERROR: SAM3 失败 — \(msg)")
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    engine.sam3.status = .idle
                    self?.processVideo(at: index + 1, videos: videos, engine: engine, outputDir: outputDir, videoURLs: videoURLs, alog: alog)
                }
            default:
                self?.monitorAnnotation(engine: engine, videoURL: videoURL, index: index, videos: videos, outputDir: outputDir, videoURLs: videoURLs, alog: alog)
            }
        }
    }

    // ═════════════════════════════════════════════════════════════
    // MARK: - ★ Eval 模式: 命令行真实视频评测
    // ═════════════════════════════════════════════════════════════

    var evalEngine: SimEngine?

    private func startEvalFromDelegate() {
        let args = CommandLine.arguments
        func argVal(_ key: String) -> String? {
            guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }
            return args[i + 1]
        }

        let evalLogPath = argVal("--eval-log") ?? (NSHomeDirectory() + "/Desktop/eval_result.log")
        // 清空旧日志
        FileManager.default.createFile(atPath: evalLogPath, contents: nil)
        func elog(_ msg: String) {
            let line = msg + "\n"
            print(line, terminator: "")
            if let data = line.data(using: .utf8), let fh = FileHandle(forWritingAtPath: evalLogPath) {
                fh.seekToEndOfFile(); fh.write(data); fh.closeFile()
            }
        }

        // 也支持从 eval_config.json 读取
        let srcConfigPath = NSHomeDirectory() + "/Documents/MacRCSimulator/eval_config.json"
        var cfgVideoPath: String?
        var cfgAnnotationPath: String?
        if FileManager.default.fileExists(atPath: srcConfigPath),
           let data = try? Data(contentsOf: URL(fileURLWithPath: srcConfigPath)),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            cfgVideoPath = json["video"] as? String
            cfgAnnotationPath = json["annotation"] as? String
        }

        let videoPath = argVal("--video") ?? cfgVideoPath ?? ""
        let annotationPath = argVal("--annotation") ?? cfgAnnotationPath ?? ""
        let outputDir = argVal("--output") ?? (NSHomeDirectory() + "/Desktop/eval_outputs")
        let evalSpeed = argVal("--speed").flatMap { Double($0) } ?? 0
        let oracleModeArg = (argVal("--oracle-gt") ?? "").lowercased()
        if let confArg = argVal("--conf"), let conf = Float(confArg) {
            Cfg.rYoloConf = conf
        }

        guard !videoPath.isEmpty else {
            elog("[EVAL] ERROR: 未指定视频文件 (--video <path> 或 eval_config.json)")
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { NSApplication.shared.terminate(nil) }
            return
        }

        let videoURL = URL(fileURLWithPath: videoPath)
        guard FileManager.default.fileExists(atPath: videoPath) else {
            elog("[EVAL] ERROR: 视频文件不存在: \(videoPath)")
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { NSApplication.shared.terminate(nil) }
            return
        }

        elog("[EVAL] 视频: \(videoPath)")
        elog("[EVAL] 标注: \(annotationPath.isEmpty ? "未指定" : annotationPath)")
        elog("[EVAL] YOLO conf: \(String(format: "%.3f", Cfg.rYoloConf))")

        let eng = SimEngine()
        self.evalEngine = eng
        switch oracleModeArg {
        case "source", "src", "full":
            eng.evalOracleMode = .source
            elog("[EVAL] GT Oracle: source/full-frame 上帝视角控制")
        case "crop", "visible", "in-crop":
            eng.evalOracleMode = .crop
            elog("[EVAL] GT Oracle: crop-only 当前裁切内完美检测")
        case "", "off", "false", "0":
            eng.evalOracleMode = .off
        default:
            elog("[EVAL] WARNING: 未识别 --oracle-gt \(oracleModeArg)，使用 off")
            eng.evalOracleMode = .off
        }

        // 加载模型 — 支持 --model <name> 和 --model-path <path>
        let modelPath = argVal("--model-path")
        let modelName = argVal("--model")
        var effectiveModelName = modelPath
            .map { URL(fileURLWithPath: expandHomePath($0)).deletingPathExtension().lastPathComponent }
            ?? modelName ?? "default"

        func expandHomePath(_ path: String) -> String {
            guard path == "~" || path.hasPrefix("~/") else { return path }
            return NSHomeDirectory() + String(path.dropFirst())
        }
        func existingModelURL(forPath rawPath: String) -> URL? {
            let path = expandHomePath(rawPath)
            let url = URL(fileURLWithPath: path)
            var candidates = [url]
            if url.pathExtension.isEmpty {
                candidates.append(URL(fileURLWithPath: path + ".mlmodelc"))
                candidates.append(URL(fileURLWithPath: path + ".mlpackage"))
            }
            return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
        }
        func bundledModelURL(named name: String) -> URL? {
            Bundle.main.url(forResource: name, withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: name, withExtension: "mlpackage")
        }

        var loadedModel = false
        if let modelPath = modelPath {
            if let url = existingModelURL(forPath: modelPath) {
                loadedModel = eng.loadModel(url)
                effectiveModelName = url.deletingPathExtension().lastPathComponent
                elog("[EVAL] 模型: \(url.path) (--model-path 指定)")
            } else {
                elog("[EVAL] WARNING: 未找到 --model-path \(modelPath)，尝试回落到 --model/默认模型")
            }
        }
        if !loadedModel, let modelName = modelName {
            // 用户指定了模型名，在 Bundle 中查找
            let allCandidates = ["yolo11_640imgsz", "Detector", "vsim_yolo11n", "RC_YOLO11n_320"]
            for name in allCandidates {
                if name == modelName, let url = bundledModelURL(named: name) {
                    loadedModel = eng.loadModel(url)
                    elog("[EVAL] 模型: \(name) (--model 指定)")
                    effectiveModelName = name
                    break
                }
            }
            if !loadedModel {
                // 尝试直接用名字查找
                if let url = bundledModelURL(named: modelName) {
                    loadedModel = eng.loadModel(url)
                    elog("[EVAL] 模型: \(modelName) (--model 指定)")
                    effectiveModelName = modelName
                } else {
                    elog("[EVAL] WARNING: 未找到模型 \(modelName)，使用默认模型")
                }
            }
        }
        if !loadedModel, let firstModel = eng.bundledModels.first {
            _ = eng.loadModel(firstModel.url)
            elog("[EVAL] 模型: \(firstModel.displayName) (默认)")
            effectiveModelName = firstModel.id
        } else if !loadedModel {
            elog("[EVAL] WARNING: 无内置模型")
        }

        // 加载标注
        if !annotationPath.isEmpty && FileManager.default.fileExists(atPath: annotationPath) {
            let annURL = URL(fileURLWithPath: annotationPath)
            let ok = eng.loadAnnotationJSON(annURL)
            elog("[EVAL] 标注加载: \(ok ? "✅ 成功" : "❌ 失败")")
        } else {
            // ★ 未显式指定标注 → 自动查找
            let found = eng.autoFindAndLoadAnnotation(for: videoURL)
            elog("[EVAL] 自动查找标注: \(found ? "✅ 已找到并加载" : "⚠️ 未找到")")
        }

        // 开启单目标追踪
        eng.enableSingleTarget = true
        elog("[EVAL] 单目标追踪: 已开启")

        // 启动评测
        eng.start(videoURL: videoURL, racing: true, speed: evalSpeed,
                  ble: BLEConfig(), pan: ServoPhys(), tilt: ServoPhys(),
                  coup: CouplingCfg(), fovMap: FOVMap(), zoom: ZoomConfig())
        elog("[EVAL] 评测已启动...")

        // 监控完成
        monitorEval(
            eng: eng,
            modelName: effectiveModelName,
            sceneName: videoURL.deletingPathExtension().lastPathComponent,
            outputDir: outputDir,
            elog: elog
        )
    }

    private func monitorEval(eng: SimEngine, modelName: String, sceneName: String, outputDir: String, elog: @escaping (String) -> Void) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
            if eng.running {
                let s = eng.stats
                elog("[EVAL] 进度: \(s.frames) 帧 | 一般追踪率 \(String(format: "%.1f%%", s.generalTrackRate)) | GT评测帧 \(s.gtAnnotationFrames)")
                self?.monitorEval(eng: eng, modelName: modelName, sceneName: sceneName, outputDir: outputDir, elog: elog)
            } else {
                // 完成
                let s = eng.stats
                elog("")
                elog("════════════════════════════════════════════════════════════")
                elog("★ EVAL 评测结果")
                elog("════════════════════════════════════════════════════════════")
                elog("总帧数: \(s.frames)")
                elog("一般追踪率: \(String(format: "%.1f%%", s.generalTrackRate))")
                elog("")
                if s.gtAnnotationFrames > 0 {
                    elog("── ★ 标注 GT 评测 (虚拟赛道口径) ──")
                    elog("一般追踪率: \(String(format: "%.1f%%", s.generalTrackRate)) (\(s.trackOnCarFrames)/\(s.frames))")
                    elog("有效追踪率: \(String(format: "%.1f%%", s.effectiveTrackRate)) (\(s.trackOnCarFrames)/\(s.gtAnnotationFrames))")
                    elog("YOLO检测率(GT可见): \(String(format: "%.1f%%", s.yoloDetectRateVisible)) (\(s.yoloHitFrames)/\(s.gtAnnotationFrames))")
                    elog("单目标锁定率: \(String(format: "%.1f%%", s.singleTargetLockRate)) (\(s.stLockOnCarFrames)/\(s.stEvalFrames))")
                    elog("GT标注可见率: \(String(format: "%.1f%%", s.gtAnnotationVisiblePct)) (\(s.gtAnnotationFrames)/\(s.frames))")
                    elog("")
                    elog("★ 说明: 有效追踪率以GT可见帧为分母，消除了目标不在取景框内的帧对评分的影响")
                    elog("★ 方案B: 一般追踪率已含 ST 辅助 (smRect/stPrevRect 取较大IoU); ST短暂丢失时用上次成功框维持")
                } else {
                    elog("── 无标注 GT 数据 ──")
                    elog("GT可见率 SAHI: \(String(format: "%.1f%%", s.gtVisiblePct))")
                    elog("纯Tracker率: \(String(format: "%.1f%%", s.pureTrkPct))")
                    elog("追踪效率: \(String(format: "%.1f%%", s.trkEfficiency))")
                }
                elog("")
                elog("── 舵机精度 ──")
                elog("H误差: avg \(String(format: "%.2f°", s.avgEH)) max \(String(format: "%.2f°", s.maxEH))")
                elog("V误差: avg \(String(format: "%.2f°", s.avgEV)) max \(String(format: "%.2f°", s.maxEV))")
                elog("Jitter H: \(String(format: "%.1f%%", s.jitH))  V: \(String(format: "%.1f%%", s.jitV))")
                elog("峰值速度: H \(String(format: "%.0f°/s", s.peakVH))  V \(String(format: "%.0f°/s", s.peakVV))")
                elog("")
                if let result = eng.simResult {
                    elog("── 综合评分 ──")
                    elog("评分: \(result.score)/100  \(result.grade)")
                }
                let outURL = URL(fileURLWithPath: outputDir)
                    .appendingPathComponent(modelName)
                    .appendingPathComponent(sceneName)
                try? FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)
                let frameCSV = outURL.appendingPathComponent("eval_frames.csv")
                let historyCSV = outURL.appendingPathComponent("history.csv")
                eng.exportEvalFrameCSV(frameCSV)
                eng.exportCSV(historyCSV)
                elog("")
                elog("── 导出 ──")
                elog("逐帧GT评测CSV: \(frameCSV.path)")
                elog("历史CSV: \(historyCSV.path)")
                elog("════════════════════════════════════════════════════════════")

                // 延迟退出
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    NSApplication.shared.terminate(nil)
                }
            }
        }
    }
}

@main
struct RCSimApp: App {
    @NSApplicationDelegateAdaptor(RCSimAppDelegate.self) var appDelegate
    @State private var selectedTab: Int = 2  // 启动默认切到“虚拟赛道”
    @State private var didAutoFullscreen: Bool = false

    static var isDatasetMode: Bool {
        CommandLine.arguments.contains("--generate-dataset")
            || CommandLine.arguments.contains("--split-ablation")
            || CommandLine.arguments.contains("--split-ablation-lite")
            || CommandLine.arguments.contains("--scale-effect-dataset")
    }

    // --split-ablation 由 AppDelegate.applicationDidFinishLaunching 直接驱动
    // 不复用 DatasetCLIView（避免它启动自己的 generator 与编排器竞争）
    // 进度通过 stdout 的 [SplitAblation] / [DatasetGen] 日志观察

    static var isAnnotateMode: Bool {
        CommandLine.arguments.contains("--annotate")
    }

    static var isEvalMode: Bool {
        CommandLine.arguments.contains("--eval")
    }

    static var isGTTestMode: Bool {
        CommandLine.arguments.contains("--gt-test")
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if BatchSimRunner.isBatchMode || BatchSimRunner.isTrainingMode {
                    BatchSimView()
                        .frame(minWidth: 800, minHeight: 600)
                } else if Self.isDatasetMode {
                    VStack(spacing: 16) {
                        Image(systemName: "photo.stack.fill")
                            .font(.system(size: 42))
                            .foregroundColor(.blue)
                        Text("数据集生成模式运行中")
                            .font(.title2.bold())
                        Text("实际生成由 AppDelegate 直接驱动，请查看 stdout 日志获取进度。")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(minWidth: 600, minHeight: 400)
                } else if Self.isAnnotateMode {
                    AnnotationCLIView()
                        .frame(minWidth: 700, minHeight: 500)
                } else if Self.isEvalMode {
                    EvalCLIView()
                        .frame(minWidth: 600, minHeight: 400)
                } else if Self.isGTTestMode {
                    GTTestCLIView()
                        .frame(minWidth: 600, minHeight: 400)
                } else {
                    TabView(selection: $selectedTab) {
                        MainView()
                            .tabItem { Label("仿真器 Simulator", systemImage: "scope") }
                            .tag(0)
                        VNLabView()
                            .tabItem { Label("Vision Lab 学习实验室", systemImage: "flask") }
                            .tag(1)
                        VirtualRaceView()
                            .tabItem { Label("虚拟赛道 Virtual Race", systemImage: "car.2.fill") }
                            .tag(2)
                        DatasetGeneratorView()
                            .tabItem { Label("数据集生成 Dataset", systemImage: "photo.stack.fill") }
                            .tag(3)
                        VideoAnnotationView()
                            .tabItem { Label("视频标注 Annotator", systemImage: "pencil.and.ruler") }
                            .tag(4)
                        YOLOGTTestView()
                            .tabItem { Label("GT检测测试 GT-vis", systemImage: "checkmark.shield") }
                            .tag(5)
                    }
                    .frame(minWidth: 1300, minHeight: 1000)
                }
            }
            .onAppear {
                // ★ Batch 模式下不强制全屏，避免干扰
                if BatchSimRunner.isBatchMode || BatchSimRunner.isTrainingMode { return }
                
                guard !didAutoFullscreen else { return }
                didAutoFullscreen = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                    guard let w = NSApp.windows.first else { return }
                    guard let screen = w.screen ?? NSScreen.main else { return }
                    let targetFrame = screen.visibleFrame
                    w.setFrame(targetFrame, display: true, animate: true)
                }
            }
        }
        .windowStyle(.hiddenTitleBar) // ★ 尝试统一窗口风格
    }
}

// ═════════════════════════════════════════════════════════════
// MARK: - ★ Eval CLI View
// ═════════════════════════════════════════════════════════════

struct EvalCLIView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "scope")
                .font(.system(size: 60))
                .foregroundColor(.cyan)
            Text("★ Eval 模式: 真实视频评测中...")
                .font(.title2.bold())
            Text("请查看终端输出了解评测进度和结果")
                .foregroundColor(.secondary)
            ProgressView()
                .scaleEffect(1.5)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct MainView: View {
    @StateObject private var eng = SimEngine()
    @State private var vidURL: URL?
    @State private var modURL: URL?
    @State private var showV = false
    @State private var showM = false
    @State private var showC = false
    @State private var showResult = false        // ★ 结论弹窗
    @State private var useBuiltinModel = true    // 默认用内置模型
    @State private var annURL: URL?              // ★ 标注 JSON 文件
    @State private var showAnn = false           // ★ 标注文件选择器

    @State private var racing = true             // ★ 默认赛道模式
    @State private var speed: Double = 1
    @State private var bleMin: Double = 8
    @State private var bleMax: Double = 35
    @State private var bleLoss: Double = 0.02
    @State private var srvSpd: Double = 380
    @State private var srvAcc: Double = 3000
    @State private var srcFov: Double = 120
    @State private var camFov: Double = 60
    @State private var grav: Double = 0.15
    @State private var cent: Double = 0.08
    // ── 变焦范围 ──
    @State private var optMin: Double = 1.0
    @State private var optMax: Double = 2.5
    @State private var digMin: Double = 1.0
    @State private var digMax: Double = 4.0
    @State private var zoomTgtRatio: Double = 0.15
    @State private var zoomDead: Double = 0.25
    @State private var zoomStep: Double = 0.02

    var body: some View {
        HSplitView {
            // ═══ 左：五行两列可视化 ═══
            VStack(spacing: 1) {
                // ── Row 1-2: 广角源画面 (占上部 40%) ──
                ZStack {
                    Color.black
                    if eng.snap != nil {
                        FrameV(f: eng.snap!, mode: .source,
                               useBoxSelection: eng.useBoxSelection,
                               onBoxSelected: { eng.setInitialTargetFromUser(rectTL: $0) },
                               gtBoxes: eng.gtBoxes(forFrame: eng.snap!.idx))
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "film")
                                .font(.system(size: 40))
                                .foregroundColor(.gray)
                            Text("加载视频和模型后点击 Run")
                                .foregroundColor(.gray)
                            Text("或将视频文件拖放到此处")
                                .font(.caption)
                                .foregroundColor(.gray.opacity(0.6))
                        }
                    }
                }
                .animation(.none, value: eng.snap?.idx)
                .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                    handleVideoDrop(providers)
                }

                // ── Row 3-4: 舵机取景框 (占中部 40%) ──
                ZStack {
                    Color(white: 0.03)
                    if eng.snap != nil {
                        FrameV(f: eng.snap!, mode: .crop)
                    } else {
                        Text("舵机取景框 Camera View")
                            .font(.caption).foregroundColor(.gray)
                    }
                }
                .animation(.none, value: eng.snap?.idx)
                // ★ 图例标注
                .overlay(alignment: .bottomTrailing) {
                    VStack(alignment: .trailing, spacing: 2) {
                        legendItem(color: .cyan,   dash: false, text: "青色框 — 取景裁切区域(源画面)")
                        legendItem(color: .green,  dash: false, text: "绿色实框 — 当前追踪目标")
                        legendItem(color: .red,    dash: true,  text: "红色虚框 — 被拒绝的YOLO候选")
                        legendItem(color: .yellow, dash: false, text: "黄色箭头 — PID误差方向")
                        legendItem(color: .cyan,   dash: true,  text: "青色虚箭头 — 前瞻预测误差")
                        legendItem(color: .blue,   dash: true, text: "蓝色虚框/十字 — 卡尔曼预测 (位置+尺度)")
                        legendItem(color: .white,  dash: false, text: "白色十字 — 画面中心")
                        legendItem(color: .orange, dash: false, text: "橙色十字 — 构图目标点 (Lead Room)")
                    }
                    .padding(5)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(4)
                    .padding(4)
                }

                // ── Row 5: 3D 云台 (左) + 3D 场景 (中) + 折线图 (右) ──
                HStack(spacing: 1) {
                    // 3D Gimbal（现有的，不变）
                    ZStack {
                        Color(white: 0.04)
                        if let f = eng.snap {
                            GimbalSceneView(
                                panDeg: f.actH, tiltDeg: f.actV,
                                cmdPanDeg: f.cmdH, cmdTiltDeg: f.cmdV,
                                state: f.state, trail: eng.servoTrail
                            )
                        } else {
                            VStack(spacing: 2) {
                                Image(systemName: "rotate.3d")
                                    .foregroundColor(.gray)
                                Text("3D 云台").font(.caption).foregroundColor(.gray)
                            }
                        }
                    }
                    .frame(maxWidth: 160)
                    .overlay(alignment: .topLeading) {
                        Text("3D 云台 Gimbal")
                            .font(.caption2.bold())
                            .foregroundColor(.white.opacity(0.6))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(3)
                            .padding(4)
                    }

                    // ★ 新增: 3D 场景概览
                    ZStack {
                        Color(white: 0.04)
                        if let f = eng.snap {
                            SceneOverviewView(
                                panDeg: f.actH,
                                tiltDeg: f.actV,
                                targetInCrop: f.det,
                                cropInSource: f.crop,
                                srcFovH: eng.fov.srcH,
                                srcFovV: eng.fov.srcV,
                                camFovH: f.effFovH,
                                camFovV: f.effFovH * (9.0 / 16.0), // 近似垂直FOV
                                cameraTiltDeg: eng.cameraTiltDeg,
                                cameraHeight: eng.cameraHeightM,
                                state: f.state,
                                carTrail: eng.carWorldTrail
                            )
                        } else {
                            VStack(spacing: 2) {
                                Image(systemName: "map")
                                    .foregroundColor(.gray)
                                Text("3D 场景").font(.caption).foregroundColor(.gray)
                            }
                        }
                    }
                    .frame(maxWidth: 200)
                    .overlay(alignment: .topLeading) {
                        Text("3D 场景 Scene")
                            .font(.caption2.bold())
                            .foregroundColor(.white.opacity(0.6))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(3)
                            .padding(4)
                    }

                    // Servo Plot（现有的，不变）
                    VStack(spacing: 0) {
                        ServoPlotView(
                            trail: eng.servoTrail,
                            currentTime: eng.snap?.t ?? 0,
                            maxDPS: srvSpd,
                            accelDPS2: srvAcc
                        )
                    }
                    .background(Color(white: 0.04))
                }
                .frame(minHeight: 200, maxHeight: 280)

                // 进度条
                VStack(spacing: 2) {
                    ProgressView(value: eng.progress)
                        .progressViewStyle(.linear)
                    HStack {
                        Text(String(format: "%.1f%%", eng.progress * 100))
                            .font(.caption.monospaced())
                        Spacer()
                        if eng.running {
                            Circle()
                                .fill(eng.paused ? .yellow : .red)
                                .frame(width: 8, height: 8)
                            Text(eng.paused ? "PAUSED" : "RUN")
                                .font(.caption.bold())
                        }
                    }
                    .foregroundColor(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(Color(white: 0.08))
            }
            .frame(minWidth: 700)

            // 右：控制面板
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    // ── 视频文件 ──
                    GroupBox("Video + 标注 GT") {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Button("Video…") {
                                    let panel = NSOpenPanel()
                                    panel.allowedContentTypes = [.video, .movie, .mpeg4Movie, .quickTimeMovie,
                                                                  UTType(filenameExtension: "mp4") ?? .movie,
                                                                  UTType(filenameExtension: "avi") ?? .movie,
                                                                  UTType(filenameExtension: "mkv") ?? .movie]
                                    panel.allowsMultipleSelection = false
                                    panel.canChooseDirectories = false
                                    // ★ 默认打开测试视频文件夹
                                    let defaultVideoDir = URL(fileURLWithPath: NSHomeDirectory() + "/Documents/MacRCSimulator/云台仿真器测试视频0308_副本")
                                    if FileManager.default.fileExists(atPath: defaultVideoDir.path) {
                                        panel.directoryURL = defaultVideoDir
                                    }
                                    panel.begin { response in
                                        guard response == .OK, let url = panel.url else { return }
                                        vidURL = url
                                        eng.videoFrameRate = 0
                                        eng.status = "Video: \(url.lastPathComponent)"
                                        // ★ 自动查找并加载 SAM3 标注
                                        eng.annotationProject = nil   // 清除旧标注
                                        eng.annotationProjectURL = nil
                                        eng.autoFindAndLoadAnnotation(for: url)
                                        if let annProjURL = eng.annotationProjectURL {
                                            annURL = annProjURL
                                        }
                                        eng.notifyUI()
                                    }
                                }
                                if let u = vidURL {
                                    Text(u.lastPathComponent).font(.caption).foregroundColor(.secondary)
                                }
                            }
                            if eng.videoFrameRate > 0 {
                                Text("帧率 \(String(format: "%.1f", eng.videoFrameRate)) fps")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            Divider()
                            // ★ 标注 JSON 导入
                            HStack {
                                Button("标注 JSON…") { showAnn = true }
                                if let u = annURL {
                                    Text(u.lastPathComponent).font(.caption).foregroundColor(.secondary)
                                } else {
                                    Text("未加载").font(.caption).foregroundColor(.secondary.opacity(0.5))
                                }
                            }
                            if eng.annotationProject != nil {
                                let proj = eng.annotationProject!
                                let annotated = proj.frames.values.filter({ !$0.isEmpty }).count
                                Text("★ 标注 GT: \(annotated)/\(proj.totalFrames) 帧有框")
                                    .font(.caption2)
                                    .foregroundColor(.green)
                            }
                            Text("导入 .annotations.json 后仿真将按虚拟赛道口径计算追踪率和锁定率")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                        .padding(4)
                    }

                    // ── 模型选择 (与 iOS 同步: 内置二选一 + 手动导入) ──
                    GroupBox("Model") {
                        VStack(alignment: .leading, spacing: 6) {
                            // 内置模型
                            if !eng.bundledModels.isEmpty {
                                Toggle("Use bundled model", isOn: $useBuiltinModel)
                                    .toggleStyle(.switch)

                                if useBuiltinModel {
                                    Picker("", selection: $eng.selectedModelId) {
                                        ForEach(eng.bundledModels) { m in
                                            Text(m.displayName).tag(m.id)
                                        }
                                    }
                                    .pickerStyle(.segmented)
                                    .onChange(of: eng.selectedModelId) { newId in
                                        eng.selectBundledModel(newId)
                                    }
                                }
                            }

                            // 手动导入 (兜底 / 自定义模型)
                            if !useBuiltinModel || eng.bundledModels.isEmpty {
                                HStack {
                                    Button("Import .mlmodelc…") { showM = true }
                                    if let u = modURL {
                                        Text(u.lastPathComponent).font(.caption).foregroundColor(.secondary)
                                    }
                                }
                            }

                            // 当前模型状态
                            Text(eng.status)
                                .font(.caption2).foregroundColor(.secondary)
                                .lineLimit(1)
                        }
                        .padding(4)
                    }

                    GroupBox("Mode 追踪模式") {
                        VStack(alignment: .leading, spacing: 6) {
                            Toggle("Racing 赛道模式", isOn: $racing)
                                .toggleStyle(.switch)
                            
                            Toggle("标定取景/控制", isOn: $eng.useCalibratedFOV)
                                .toggleStyle(.switch)
                            Text("基于标定几何的取景+PID+变焦，与 FOVMap 参数一致；关闭则用原线性模型。")
                                .font(.caption2).foregroundColor(.secondary)
                            if eng.useCalibratedFOV {
                                ParamSlider(label: "畸变 k1", value: $eng.distortionK1, range: -0.4...0, fmt: "%.2f")
                                Text("镜头桶形畸变校正。运动相机约 -0.28；源已去畸变设 0。边缘跟踪不准时可调。")
                                    .font(.caption2).foregroundColor(.secondary)

                                Divider()

                                // ★ 场景标定参数
                                ParamSlider(label: "俯仰角 Tilt", value: $eng.cameraTiltDeg, range: -45...45, fmt: "%.1f°")
                                Text("相机向下倾斜角度。正值=向下看。从标定脚本获取或手动调整。")
                                    .font(.caption2).foregroundColor(.secondary)

                                ParamSlider(label: "相机高度", value: $eng.cameraHeightM, range: 0.5...5.0, fmt: "%.1fm")
                                Text("相机离地面高度（米）。影响测距精度，不影响取景框映射。")
                                    .font(.caption2).foregroundColor(.secondary)

                                // ★ 可选: 加载标定 JSON
                                Toggle("使用外部标定文件", isOn: $eng.useSceneCalibration)
                                if eng.useSceneCalibration {
                                    Button("选择标定文件...") {
                                        let panel = NSOpenPanel()
                                        panel.allowedContentTypes = [.json]
                                        panel.begin { response in
                                            if response == .OK, let url = panel.url {
                                                eng.sceneCalibrationURL = url
                                                eng.invalidateSceneCalibration()
                                            }
                                        }
                                    }
                                    if let url = eng.sceneCalibrationURL {
                                        Text(url.lastPathComponent)
                                            .font(.caption2).foregroundColor(.secondary)
                                    }
                                }
                            }
                            Toggle("启用增强算法", isOn: $eng.useEnhancedAlgo)
                                .toggleStyle(.switch)
                            Text("开启以使用高斯空间权重、8D 尺度卡尔曼、时序二次曲线轨迹预测与局部光流速度融合").font(.caption2).foregroundColor(.secondary)

                            Divider()

                            Toggle("单目标追踪 (V7)", isOn: $eng.enableSingleTarget)
                                .toggleStyle(.switch)
                            Text("V7 单目标锁定: 三阶段分级确认 + 三层模板 + 自主检测器推理。与增强算法同时开启时自动协同 (轨迹预测→空间门控融合)。可单独或协同运行。仅赛道模式生效。")
                                .font(.caption2).foregroundColor(.secondary)
                            if eng.enableSingleTarget && eng.stStats.stTotalFrames > 0 {
                                HStack {
                                    Text("ST锁定率")
                                        .font(.caption2)
                                    Spacer()
                                    Text(String(format: "%.1f%% (%d/%d)",
                                                eng.stStats.stLockRate,
                                                eng.stStats.stLockFrames,
                                                eng.stStats.stTotalFrames))
                                        .font(.caption2.monospaced())
                                        .foregroundColor(eng.stStats.stLockRate > 80 ? .green : eng.stStats.stLockRate > 50 ? .yellow : .red)
                                }
                            }
                        }
                        .padding(4)
                    }

                    GroupBox("目标获取") {
                        VStack(alignment: .leading, spacing: 6) {
                            Picker("方式", selection: $eng.useBoxSelection) {
                                Text("自动 (YOLO)").tag(false)
                                Text("框选").tag(true)
                            }
                            .pickerStyle(.segmented)
                            if eng.useBoxSelection {
                                Text("在源画面上拖拽框选目标；播放中也可随时框选以重新指定目标。未框选时仍会使用 YOLO 自动检测。")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(4)
                    }

                    GroupBox("BLE 蓝牙延迟") {
                        VStack(spacing: 4) {
                            ParamSlider(label: "Min ms 最短延迟", value: $bleMin, range: 2...50, fmt: "%.0f")
                            ParamSlider(label: "Max ms 最长延迟", value: $bleMax, range: 10...100, fmt: "%.0f")
                            ParamSlider(label: "Loss 丢包率", value: $bleLoss, range: 0...0.2, fmt: "%.2f")
                        }
                        .padding(4)
                    }

                    GroupBox("PID 取景追速") {
                        VStack(alignment: .leading, spacing: 4) {
                            ParamSlider(label: "取景追速强度 (增益倍数)", value: Binding(
                                get: { Double(eng.pidGainScale) },
                                set: { eng.pidGainScale = CGFloat($0) }
                            ), range: 0.5...2.0, fmt: "%.2f")
                            Text("离限幅还远但取景框已追不上车时调大（如 1.2～1.5）。指令会更强，速度曲线会更接近 ±max；过大易抖动。")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                        .padding(4)
                    }

                    GroupBox("Servo 舵机物理参数") {
                        VStack(alignment: .leading, spacing: 4) {
                            ParamSlider(label: "°/s 最大角速度", value: $srvSpd, range: 100...800, fmt: "%.0f")
                            ParamSlider(label: "°/s² 角加速度", value: $srvAcc, range: 500...8000, fmt: "%.0f")
                            Text("追不上时：先调上面「取景追速强度」；仍不够再增大本组角速度/角加速度。曲线图速度贴齐 ±max 即已饱和。")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                        .padding(4)
                    }

                    GroupBox("Coupling 轴间耦合") {
                        VStack(spacing: 4) {
                            ParamSlider(label: "Gravity 重力耦合", value: $grav, range: 0...0.5, fmt: "%.2f")
                            ParamSlider(label: "Centrifugal 离心耦合", value: $cent, range: 0...0.3, fmt: "%.2f")
                        }
                        .padding(4)
                    }

                    GroupBox("FOV 视场角 / 取景框大小") {
                        VStack(spacing: 4) {
                            ParamSlider(label: "Source H° 广角源视场", value: $srcFov, range: 60...180, fmt: "%.0f")
                            ParamSlider(label: "Camera H° 相机视场", value: $camFov, range: 20...90, fmt: "%.0f")
                            Text("Camera FOV 决定取景框大小: 值越小 → 取景框越小(长焦)。比值 Camera/Source 即为裁切比例。当前取景比 ≈ \(String(format: "%.0f%%", camFov / srcFov * 100))")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                        .padding(4)
                    }

                    GroupBox("Speed 仿真倍速") {
                        ParamSlider(label: "x", value: $speed, range: 0.1...10, fmt: "%.1f")
                            .padding(4)
                    }

                    // ═══ GT SAHI 追踪分析 ═══
                    GroupBox("GT SAHI 地毯式检测") {
                        VStack(alignment: .leading, spacing: 6) {
                            Toggle("启用 GT SAHI", isOn: $eng.enableGtSahi)
                                .toggleStyle(.switch)
                            Text("每帧在源广角图上分片跑 YOLO，统计目标可见率。仿真速度会降低。")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                    }

                    // ═══ 变焦仿真 ═══
                    GroupBox("变焦仿真 AutoZoom") {
                        VStack(alignment: .leading, spacing: 6) {
                            Toggle("启用自动变焦", isOn: $eng.enableZoom)
                                .toggleStyle(.switch)
                            Text("模拟 iOS 光学变焦 + 数字变焦。变焦会缩小取景范围，目标在画面中更大，提升追踪成功率。")
                                .font(.caption2).foregroundColor(.secondary)

                            if eng.enableZoom {
                                Divider()
                                Text("光学变焦 Optical").font(.caption.bold()).foregroundColor(.secondary)
                                ParamSlider(label: "最小倍率", value: $optMin, range: 1.0...3.0, fmt: "%.1fx")
                                ParamSlider(label: "最大倍率", value: $optMax, range: 1.0...5.0, fmt: "%.1fx")
                                ParamSlider(label: "死区", value: $zoomDead, range: 0.05...0.5, fmt: "%.2f")
                                ParamSlider(label: "单帧步进", value: $zoomStep, range: 0.005...0.1, fmt: "%.3f")

                                Divider()
                                Text("数字变焦 Digital").font(.caption.bold()).foregroundColor(.secondary)
                                ParamSlider(label: "最小倍率", value: $digMin, range: 1.0...2.0, fmt: "%.1fx")
                                ParamSlider(label: "最大倍率", value: $digMax, range: 1.0...8.0, fmt: "%.1fx")

                                Divider()
                                Text("目标占比").font(.caption.bold()).foregroundColor(.secondary)
                                ParamSlider(label: "目标高度占比", value: $zoomTgtRatio, range: 0.10...0.70, fmt: "%.2f")
                                Text("目标在画面中期望占比，越大 → 变焦越近。光学+数字共用此值。")
                                    .font(.caption2).foregroundColor(.secondary)
                            }
                        }
                    }

                    // ═══ Debug 模式 ═══
                    GroupBox("Debug 逐帧检查") {
                        VStack(alignment: .leading, spacing: 6) {
                            Toggle("启用 Debug 模式", isOn: $eng.debugMode)
                                .toggleStyle(.switch)
                            Text("开启后自动逐帧暂停，可检查所有中间变量")
                                .font(.caption2).foregroundColor(.secondary)

                            if eng.debugMode && eng.running {
                                HStack(spacing: 8) {
                                    Button(action: { eng.stepForward() }) {
                                        Label("Step ▶", systemImage: "forward.frame.fill")
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(.blue)

                                    Button(action: {
                                        eng.debugMode = false
                                        eng.paused = false
                                        eng.notifyUI()
                                    }) {
                                        Label("Continue ▶▶", systemImage: "play.fill")
                                    }
                                    .tint(.green)
                                }
                            }

                            // 仿真结束后的历史帧浏览
                            if !eng.running && eng.history.count > 0 {
                                VStack(spacing: 2) {
                                    Text("帧浏览 Frame \(eng.debugFrameIdx) / \(eng.history.count - 1)")
                                        .font(.caption.monospaced())
                                    Slider(value: Binding(
                                        get: { Double(eng.debugFrameIdx) },
                                        set: { eng.jumpToFrame(Int($0)) }
                                    ), in: 0...Double(max(eng.history.count - 1, 1)), step: 1)
                                    HStack(spacing: 4) {
                                        Button("◀") { eng.jumpToFrame(eng.debugFrameIdx - 1) }
                                        Button("◀◀10") { eng.jumpToFrame(eng.debugFrameIdx - 10) }
                                        Spacer()
                                        Button("10▶▶") { eng.jumpToFrame(eng.debugFrameIdx + 10) }
                                        Button("▶") { eng.jumpToFrame(eng.debugFrameIdx + 1) }
                                    }
                                    .font(.caption)
                                }
                            }
                        }
                        .padding(4)
                    }

                    HStack(spacing: 10) {
                        Button(action: go) {
                            Label("Run", systemImage: "play.fill")
                        }
                        .disabled(!canRun || eng.running)
                        .buttonStyle(.borderedProminent)

                        Button(action: { eng.togglePause() }) {
                            Label(eng.paused ? "Resume" : "Pause",
                                  systemImage: eng.paused ? "play" : "pause")
                        }
                        .disabled(!eng.running)

                        Button(action: { eng.stop() }) {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .disabled(!eng.running)
                        .tint(.red)
                    }

                    if eng.stats.frames > 0 {
                        StatsPanel(s: eng.stats)
                    }

                    // ═══ Debug Inspector ═══
                    if eng.debugMode, let dbg = eng.snap?.dbg {
                        DebugInspectorPanel(dbg: dbg)
                    }

                    if !eng.running && eng.stats.frames > 0 {
                        Button("Export CSV…") { showC = true }
                    }

                    Spacer()
                }
                .padding()
            }
            .frame(minWidth: 320, maxWidth: 400)
        }
        // ★ 视频选择已改为 NSOpenPanel (在 Button 中实现，支持默认目录)
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            handleVideoDrop(providers)
        }
        .fileImporter(isPresented: $showM, allowedContentTypes: [UTType(filenameExtension: "mlmodelc") ?? .data]) { result in
            if case .success(let url) = result {
                _ = url.startAccessingSecurityScopedResource()
                modURL = url
                _ = eng.loadModel(url)
                useBuiltinModel = false
            }
        }
        // ★ 标注 JSON 导入
        .fileImporter(isPresented: $showAnn, allowedContentTypes: [.json]) { result in
            if case .success(let url) = result {
                _ = url.startAccessingSecurityScopedResource()
                annURL = url
                _ = eng.loadAnnotationJSON(url)
                eng.notifyUI()
            }
        }
        .fileExporter(isPresented: $showC, document: CSVDoc(eng: eng),
                      contentType: .commaSeparatedText, defaultFilename: "sim.csv") { _ in }
        // ★ 仿真完成 → 自动弹出结论面板
        .sheet(isPresented: $showResult) {
            if let r = eng.simResult {
                SimResultSheet(result: r, isPresented: $showResult)
            }
        }
        .onChange(of: eng.running) { isRunning in
            // running 从 true → false 且有结果时弹窗
            if !isRunning && eng.simResult != nil {
                showResult = true
            }
        }
    }

    /// 是否满足运行条件: 有视频 + 有模型(内置或手动)
    private var canRun: Bool {
        guard vidURL != nil else { return false }
        if useBuiltinModel && !eng.bundledModels.isEmpty { return true }
        return modURL != nil
    }

    // ★ 图例标注辅助
    @ViewBuilder
    private func legendItem(color: Color, dash: Bool, text: String) -> some View {
        HStack(spacing: 4) {
            if dash {
                // 虚线段
                ZStack {
                    RoundedRectangle(cornerRadius: 1)
                        .stroke(color, style: StrokeStyle(lineWidth: 2, dash: [3, 2]))
                        .frame(width: 14, height: 8)
                }
            } else {
                RoundedRectangle(cornerRadius: 1)
                    .stroke(color, lineWidth: 2)
                    .frame(width: 14, height: 8)
            }
            Text(text)
                .font(.system(size: 8).monospaced())
                .foregroundColor(.white.opacity(0.85))
        }
    }

    // MARK: 拖放视频
    private func handleVideoDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        let utID = "public.file-url"
        guard provider.hasItemConformingToTypeIdentifier(utID) else { return false }
        provider.loadItem(forTypeIdentifier: utID, options: nil) { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
            let ext = url.pathExtension.lowercased()
            let videoExts = ["mp4", "mov", "m4v", "avi", "mkv", "ts", "mts", "mpg", "mpeg", "webm"]
            guard videoExts.contains(ext) else { return }
            DispatchQueue.main.async {
                vidURL = url
                eng.videoFrameRate = 0
                eng.status = "Video: \(url.lastPathComponent)"
                eng.notifyUI()
            }
        }
        return true
    }

    private func go() {
        guard let vu = vidURL else { return }
        // 确保模型已加载
        if useBuiltinModel, let m = eng.bundledModels.first(where: { $0.id == eng.selectedModelId }) {
            _ = eng.loadModel(m.url)
        } else if let mu = modURL {
            _ = eng.loadModel(mu)
        } else {
            return
        }
        var b = BLEConfig()
        b.minMs = bleMin
        b.maxMs = bleMax
        b.lossRate = bleLoss
        var p = ServoPhys()
        p.maxDPS = srvSpd
        p.accelDPS2 = srvAcc
        var t = ServoPhys()
        t.maxDPS = srvSpd
        t.accelDPS2 = srvAcc
        var c = CouplingCfg()
        c.gravityScale = grav
        c.centrifugalScale = cent
        var f = FOVMap()
        f.srcH = srcFov
        f.srcV = srcFov * 9 / 16
        f.camH = camFov
        f.camV = camFov * 9 / 16
        var zc = ZoomConfig()
        zc.opticalMin = CGFloat(optMin)
        zc.opticalMax = CGFloat(max(optMax, optMin))   // 确保 max ≥ min
        zc.digitalMin = CGFloat(digMin)
        zc.digitalMax = CGFloat(max(digMax, digMin))
        zc.targetRatio = CGFloat(zoomTgtRatio)
        zc.deadzone = CGFloat(zoomDead)
        zc.maxStep = CGFloat(zoomStep)
        eng.start(videoURL: vu, racing: racing, speed: speed,
                  ble: b, pan: p, tilt: t, coup: c, fovMap: f, zoom: zc)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 子视图
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
// MARK: - 3D 云台可视化 (SceneKit)
// ═══════════════════════════════════════════════════════════════

struct GimbalSceneView: NSViewRepresentable {
    let panDeg: Double      // actual position 0-240
    let tiltDeg: Double
    let cmdPanDeg: Double   // commanded
    let cmdTiltDeg: Double
    let state: String
    let trail: [SimEngine.ServoSample]

    func makeNSView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = context.coordinator.scene
        v.allowsCameraControl = true
        v.backgroundColor = NSColor(white: 0.06, alpha: 1)
        v.antialiasingMode = .multisampling4X
        return v
    }
    func updateNSView(_ v: SCNView, context: Context) {
        context.coordinator.update(pan: panDeg, tilt: tiltDeg,
                                   cmdPan: cmdPanDeg, cmdTilt: cmdTiltDeg,
                                   state: state, trail: trail)
    }
    func makeCoordinator() -> Coord { Coord() }

    class Coord {
        let scene = SCNScene()
        let panPivot = SCNNode()
        let tiltPivot = SCNNode()
        let camBox = SCNNode()
        let cmdGhost = SCNNode()
        let trailNode = SCNNode()
        // ★ 固定数量的 trail dot，重用而不是每帧重建
        private let maxDots = 50
        private var dots: [SCNNode] = []

        init() { build() }

        func build() {
            // lights
            let amb = SCNNode(); amb.light = SCNLight()
            amb.light!.type = .ambient; amb.light!.intensity = 500
            amb.light!.color = NSColor(white: 0.9, alpha: 1)
            scene.rootNode.addChildNode(amb)

            let dir = SCNNode(); dir.light = SCNLight()
            dir.light!.type = .directional; dir.light!.intensity = 700
            dir.eulerAngles = SCNVector3(-CGFloat.pi / 3, CGFloat.pi / 5, 0)
            scene.rootNode.addChildNode(dir)

            // camera
            let cam = SCNNode(); cam.camera = SCNCamera()
            cam.camera!.fieldOfView = 45
            cam.position = SCNVector3(3.5, 3, 5)
            cam.look(at: SCNVector3(0, 0.8, 0))
            scene.rootNode.addChildNode(cam)

            // floor grid
            let floor = SCNFloor()
            floor.firstMaterial?.diffuse.contents = NSColor(white: 0.12, alpha: 1)
            floor.reflectivity = 0.02
            scene.rootNode.addChildNode(SCNNode(geometry: floor))

            // ── 底座 ──
            let base = SCNCylinder(radius: 0.6, height: 0.15)
            base.firstMaterial?.diffuse.contents = NSColor(white: 0.25, alpha: 1)
            let baseN = SCNNode(geometry: base)
            baseN.position = SCNVector3(0, 0.075, 0)
            scene.rootNode.addChildNode(baseN)

            // ── 水平旋转轴标记 ──
            let axisRingH = SCNTorus(ringRadius: 0.65, pipeRadius: 0.015)
            axisRingH.firstMaterial?.diffuse.contents = NSColor.systemBlue.withAlphaComponent(0.4)
            let axisRingHN = SCNNode(geometry: axisRingH)
            axisRingHN.position = SCNVector3(0, 0.2, 0)
            scene.rootNode.addChildNode(axisRingHN)

            // ── Pan 柱 ──
            panPivot.position = SCNVector3(0, 0.15, 0)
            scene.rootNode.addChildNode(panPivot)

            let col = SCNCylinder(radius: 0.1, height: 0.8)
            col.firstMaterial?.diffuse.contents = NSColor(white: 0.35, alpha: 1)
            let colN = SCNNode(geometry: col)
            colN.position = SCNVector3(0, 0.4, 0)
            panPivot.addChildNode(colN)

            // pan 方向指示箭头
            let arrow = SCNCone(topRadius: 0, bottomRadius: 0.08, height: 0.15)
            arrow.firstMaterial?.diffuse.contents = NSColor.systemBlue
            let arrowN = SCNNode(geometry: arrow)
            arrowN.eulerAngles = SCNVector3(CGFloat.pi / 2, 0, 0)
            arrowN.position = SCNVector3(0, 0.85, 0.15)
            panPivot.addChildNode(arrowN)

            // ── Tilt 枢轴 ──
            tiltPivot.position = SCNVector3(0, 0.85, 0)
            panPivot.addChildNode(tiltPivot)

            // 俯仰轴标记圆环
            let axisRingV = SCNTorus(ringRadius: 0.35, pipeRadius: 0.01)
            axisRingV.firstMaterial?.diffuse.contents = NSColor.systemOrange.withAlphaComponent(0.4)
            let axisRingVN = SCNNode(geometry: axisRingV)
            axisRingVN.eulerAngles = SCNVector3(0, 0, CGFloat.pi / 2)
            tiltPivot.addChildNode(axisRingVN)

            // 悬臂
            let arm = SCNBox(width: 0.08, height: 0.08, length: 0.9, chamferRadius: 0.02)
            arm.firstMaterial?.diffuse.contents = NSColor.systemBlue
            let armN = SCNNode(geometry: arm)
            armN.position = SCNVector3(0, 0, 0.45)
            tiltPivot.addChildNode(armN)

            // ── 相机体 ──
            let box = SCNBox(width: 0.35, height: 0.25, length: 0.45, chamferRadius: 0.04)
            box.firstMaterial?.diffuse.contents = NSColor.systemGreen
            camBox.geometry = box
            camBox.position = SCNVector3(0, 0, 0.95)
            tiltPivot.addChildNode(camBox)

            // 镜头
            let lens = SCNCylinder(radius: 0.08, height: 0.18)
            lens.firstMaterial?.diffuse.contents = NSColor(white: 0.1, alpha: 1)
            let lensN = SCNNode(geometry: lens)
            lensN.eulerAngles = SCNVector3(CGFloat.pi / 2, 0, 0)
            lensN.position = SCNVector3(0, 0, 0.3)
            camBox.addChildNode(lensN)

            // ── 指令位置（半透明镜像）──
            let ghostBox = SCNBox(width: 0.35, height: 0.25, length: 0.45, chamferRadius: 0.04)
            ghostBox.firstMaterial?.diffuse.contents = NSColor.white.withAlphaComponent(0.15)
            ghostBox.firstMaterial?.transparency = 0.8
            cmdGhost.geometry = ghostBox
            cmdGhost.position = SCNVector3(0, 0, 0.95)
            cmdGhost.opacity = 0.3
            tiltPivot.addChildNode(cmdGhost)

            // ── 预创建 trail dots (固定池) ──
            scene.rootNode.addChildNode(trailNode)
            let dotGeo = SCNSphere(radius: 0.025)
            dotGeo.segmentCount = 6  // 低多边形
            for _ in 0..<maxDots {
                let n = SCNNode(geometry: dotGeo.copy() as? SCNGeometry)
                n.isHidden = true
                trailNode.addChildNode(n)
                dots.append(n)
            }
        }

        func update(pan: Double, tilt: Double, cmdPan: Double, cmdTilt: Double, state: String, trail: [SimEngine.ServoSample]) {
            let panRad = CGFloat((pan - 120.0) * .pi / 180.0)
            let tiltRad = CGFloat((tilt - 120.0) * .pi / 180.0)
            panPivot.eulerAngles.y = -panRad
            tiltPivot.eulerAngles.x = tiltRad

            // state color
            let color: NSColor
            switch state {
            case "TRACK": color = .systemGreen
            case "INERTIA", "KF-INERTIA": color = .systemYellow
            case "HOLD": color = .systemYellow
            case "SCAN": color = .systemOrange
            case "RETURN": color = .systemRed
            default: color = .gray
            }
            camBox.geometry?.firstMaterial?.diffuse.contents = color

            // 指令位置偏移
            let cmdPanRad = CGFloat((cmdPan - 120.0) * .pi / 180.0)
            let cmdTiltRad = CGFloat((cmdTilt - 120.0) * .pi / 180.0)
            let dPan = cmdPanRad - (-panRad)
            let dTilt = cmdTiltRad - tiltRad
            cmdGhost.position = SCNVector3(sin(dPan) * 0.3, sin(dTilt) * 0.3, 0.95)
            let errMag = sqrt(dPan * dPan + dTilt * dTilt)
            cmdGhost.opacity = min(errMag * 3, 0.6)

            // ★ 更新固定 trail dots（不创建/删除节点）
            let step = max(1, trail.count / maxDots)
            for i in 0..<maxDots {
                let idx = i * step
                guard idx < trail.count else {
                    dots[i].isHidden = true
                    continue
                }
                let s = trail[idx]
                let pRad = CGFloat((s.actH - 120.0) * .pi / 180.0)
                let tRad = CGFloat((s.actV - 120.0) * .pi / 180.0)
                let r: CGFloat = 1.2
                dots[i].position = SCNVector3(
                    r * sin(-pRad) * cos(tRad),
                    0.85 + r * sin(tRad),
                    r * cos(-pRad) * cos(tRad)
                )
                let age = CGFloat(i) / CGFloat(maxDots)
                let dotColor: NSColor
                switch s.state {
                case "TRACK": dotColor = .systemGreen.withAlphaComponent(age * 0.7 + 0.2)
                case "INERTIA", "KF-INERTIA": dotColor = .systemYellow.withAlphaComponent(age * 0.7 + 0.2)
                case "SCAN": dotColor = .systemOrange.withAlphaComponent(age * 0.7 + 0.2)
                default: dotColor = .systemRed.withAlphaComponent(age * 0.5 + 0.2)
                }
                dots[i].geometry?.firstMaterial?.diffuse.contents = dotColor
                dots[i].isHidden = false
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 3D 场景概览 (SceneKit)
// ═══════════════════════════════════════════════════════════════

struct SceneOverviewView: NSViewRepresentable {
    let panDeg: Double       // 舵机实际 pan (0-240, 中心120)
    let tiltDeg: Double      // 舵机实际 tilt
    let targetInCrop: CGRect? // 目标在裁切画面中的归一化 rect
    let cropInSource: CGRect  // 裁切框在源视频中的归一化 rect
    let srcFovH: Double      // 源视频水平 FOV (度)
    let srcFovV: Double      // 源视频垂直 FOV (度)
    let camFovH: Double      // 取景水平 FOV (度)
    let camFovV: Double      // 取景垂直 FOV (度)
    let cameraTiltDeg: Double // 相机安装俯仰角 (度，正=向下)
    let cameraHeight: Double  // 相机高度 (米)
    let state: String
    let carTrail: [(x: Double, z: Double)] // 小车历史世界坐标

    func makeNSView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = context.coordinator.scene
        v.pointOfView = context.coordinator.viewCam  // ★ 与真实拍摄相机同位置同角度
        v.allowsCameraControl = true
        v.backgroundColor = NSColor(white: 0.06, alpha: 1)
        v.antialiasingMode = .multisampling4X
        return v
    }

    func updateNSView(_ v: SCNView, context: Context) {
        context.coordinator.update(
            panDeg: panDeg, tiltDeg: tiltDeg,
            targetInCrop: targetInCrop,
            cropInSource: cropInSource,
            srcFovH: srcFovH, srcFovV: srcFovV,
            camFovH: camFovH, camFovV: camFovV,
            cameraTiltDeg: cameraTiltDeg,
            cameraHeight: cameraHeight,
            state: state,
            carTrail: carTrail
        )
    }

    func makeCoordinator() -> Coord { Coord() }

    class Coord {
        let scene = SCNScene()

        // 可变节点
        let viewCam = SCNNode()        // 观察相机（与真实拍摄相机同位置同角度，update 中每帧更新）
        let cropConeNode = SCNNode()    // 取景锥
        let carNode = SCNNode()         // RC 小车
        let trailParent = SCNNode()     // 轨迹父节点
        let fovConeNode = SCNNode()     // 源 FOV 锥（半透明）

        private let maxTrailDots = 120
        private var trailDots: [SCNNode] = []

        init() { build() }

        func build() {
            // ── 灯光 ──
            let amb = SCNNode()
            amb.light = SCNLight()
            amb.light!.type = .ambient
            amb.light!.intensity = 600
            amb.light!.color = NSColor(white: 0.9, alpha: 1)
            scene.rootNode.addChildNode(amb)

            let dir = SCNNode()
            dir.light = SCNLight()
            dir.light!.type = .directional
            dir.light!.intensity = 600
            dir.eulerAngles = SCNVector3(-CGFloat.pi / 3, CGFloat.pi / 4, 0)
            scene.rootNode.addChildNode(dir)

            // ── 观察相机（与真实拍摄相机一致，位置/俯仰/FOV 在 update 中每帧更新）──
            viewCam.camera = SCNCamera()
            viewCam.camera!.fieldOfView = 60  // 默认值，update 中按 srcFovH 覆盖
            scene.rootNode.addChildNode(viewCam)

            // ── 地面 ──
            let floor = SCNFloor()
            floor.firstMaterial?.diffuse.contents = NSColor(red: 0.12, green: 0.15, blue: 0.10, alpha: 1)
            floor.reflectivity = 0.01
            scene.rootNode.addChildNode(SCNNode(geometry: floor))

            // ── 地面网格线 (跑道标记) ──
            buildGroundGrid()

            // ── 三脚架 + 相机 ──
            buildTripodAndCamera()

            // ── 源 FOV 锥 (半透明蓝) ──
            fovConeNode.opacity = 0.08
            scene.rootNode.addChildNode(fovConeNode)

            // ── 取景锥 (半透明黄) ──
            cropConeNode.opacity = 0.2
            scene.rootNode.addChildNode(cropConeNode)

            // ── RC 小车 ──
            let carBody = SCNBox(width: 0.3, height: 0.12, length: 0.5, chamferRadius: 0.03)
            carBody.firstMaterial?.diffuse.contents = NSColor.systemRed
            carNode.geometry = carBody
            carNode.position = SCNVector3(0, 0.06, 5)
            scene.rootNode.addChildNode(carNode)

            // 车轮指示（简单的四个小球）
            let wheelGeo = SCNSphere(radius: 0.06)
            wheelGeo.firstMaterial?.diffuse.contents = NSColor(white: 0.2, alpha: 1)
            for (dx, dz) in [(-0.15, -0.18), (0.15, -0.18), (-0.15, 0.18), (0.15, 0.18)] as [(CGFloat, CGFloat)] {
                let wn = SCNNode(geometry: wheelGeo.copy() as? SCNGeometry)
                wn.position = SCNVector3(dx, -0.03, dz)
                carNode.addChildNode(wn)
            }

            // ── 轨迹 ──
            scene.rootNode.addChildNode(trailParent)
            let dotGeo = SCNSphere(radius: 0.04)
            dotGeo.segmentCount = 6
            dotGeo.firstMaterial?.diffuse.contents = NSColor.systemGreen.withAlphaComponent(0.6)
            for _ in 0..<maxTrailDots {
                let n = SCNNode(geometry: dotGeo.copy() as? SCNGeometry)
                n.isHidden = true
                trailParent.addChildNode(n)
                trailDots.append(n)
            }
        }

        private func buildGroundGrid() {
            // 用细长 Box 画网格线 (比 SCNGeometrySource 更简单)
            let lineMat = SCNMaterial()
            lineMat.diffuse.contents = NSColor(white: 0.25, alpha: 1)

            // X 方向线 (每2米一条, -10到10)
            for z in stride(from: -2.0, through: 20.0, by: 2.0) {
                let line = SCNBox(width: 20, height: 0.005, length: 0.02, chamferRadius: 0)
                line.firstMaterial = z == 0 ? {
                    let m = SCNMaterial()
                    m.diffuse.contents = NSColor.systemBlue.withAlphaComponent(0.4)
                    return m
                }() : lineMat
                let n = SCNNode(geometry: line)
                n.position = SCNVector3(0, 0.003, z)
                scene.rootNode.addChildNode(n)
            }

            // Z 方向线
            for x in stride(from: -10.0, through: 10.0, by: 2.0) {
                let line = SCNBox(width: 0.02, height: 0.005, length: 22, chamferRadius: 0)
                line.firstMaterial = x == 0 ? {
                    let m = SCNMaterial()
                    m.diffuse.contents = NSColor.systemRed.withAlphaComponent(0.4)
                    return m
                }() : lineMat
                let n = SCNNode(geometry: line)
                n.position = SCNVector3(x, 0.003, 9)
                scene.rootNode.addChildNode(n)
            }
        }

        private func buildTripodAndCamera() {
            // 三脚架三条腿
            let legMat = SCNMaterial()
            legMat.diffuse.contents = NSColor(white: 0.3, alpha: 1)

            for angle in [0.0, 2.094, 4.189] { // 0°, 120°, 240°
                let leg = SCNCylinder(radius: 0.02, height: 2.5)
                leg.firstMaterial = legMat
                let n = SCNNode(geometry: leg)
                n.position = SCNVector3(
                    sin(angle) * 0.3,
                    1.25,
                    cos(angle) * 0.3
                )
                n.eulerAngles = SCNVector3(0, 0, sin(angle) * 0.08)
                scene.rootNode.addChildNode(n)
            }

            // 相机体 (在三脚架顶端)
            let camBody = SCNBox(width: 0.3, height: 0.2, length: 0.35, chamferRadius: 0.03)
            camBody.firstMaterial?.diffuse.contents = NSColor(white: 0.15, alpha: 1)
            let camBodyNode = SCNNode(geometry: camBody)
            camBodyNode.position = SCNVector3(0, 2.6, 0)
            scene.rootNode.addChildNode(camBodyNode)

            // 镜头 (指向前方)
            let lens = SCNCylinder(radius: 0.06, height: 0.15)
            lens.firstMaterial?.diffuse.contents = NSColor(white: 0.05, alpha: 1)
            let lensNode = SCNNode(geometry: lens)
            lensNode.eulerAngles = SCNVector3(CGFloat.pi / 2, 0, 0)
            lensNode.position = SCNVector3(0, 0, 0.25)
            camBodyNode.addChildNode(lensNode)

            // 相机标签
            let text = SCNText(string: "CAM", extrusionDepth: 0.01)
            text.font = NSFont.systemFont(ofSize: 0.2)
            text.firstMaterial?.diffuse.contents = NSColor.white
            let textNode = SCNNode(geometry: text)
            textNode.position = SCNVector3(-0.15, 2.85, 0)
            scene.rootNode.addChildNode(textNode)
        }

        /// 构建一个四棱锥线框代表视锥
        private func buildFrustum(hFovDeg: Double, vFovDeg: Double,
                                   near: Double, far: Double,
                                   tiltDeg: Double, panDeg: Double,
                                   cameraY: Double) -> SCNNode {
            let hHalf = tan(hFovDeg * .pi / 360.0)
            let vHalf = tan(vFovDeg * .pi / 360.0)

            let farW = far * hHalf
            let farH = far * vHalf

            // 四个远端点 (相机坐标系: X=右, Y=上, Z=前)
            let tl = SCNVector3(-farW, farH, far)
            let tr = SCNVector3(farW, farH, far)
            let bl = SCNVector3(-farW, -farH, far)
            let br = SCNVector3(farW, -farH, far)
            let origin = SCNVector3(0, 0, 0)

            // 线段: 4条从原点到远端 + 4条远端围框
            let verts: [SCNVector3] = [
                origin, tl, origin, tr, origin, bl, origin, br,
                tl, tr, tr, br, br, bl, bl, tl
            ]
            let indices: [Int32] = Array(0..<Int32(verts.count))

            let source = SCNGeometrySource(vertices: verts)
            let element = SCNGeometryElement(indices: indices, primitiveType: .line)
            let geo = SCNGeometry(sources: [source], elements: [element])
            geo.firstMaterial?.diffuse.contents = NSColor.white
            geo.firstMaterial?.isDoubleSided = true

            let node = SCNNode(geometry: geo)

            // 旋转: 先 tilt 再 pan, 然后平移到相机位置
            let tiltRad = -tiltDeg * .pi / 180.0  // 负号: 向下看
            let panRad = -(panDeg) * .pi / 180.0
            node.eulerAngles = SCNVector3(tiltRad, panRad, 0)
            node.position = SCNVector3(0, cameraY, 0)

            return node
        }

        func update(panDeg: Double, tiltDeg: Double,
                    targetInCrop: CGRect?,
                    cropInSource: CGRect,
                    srcFovH: Double, srcFovV: Double,
                    camFovH: Double, camFovV: Double,
                    cameraTiltDeg: Double, cameraHeight: Double,
                    state: String,
                    carTrail: [(x: Double, z: Double)]) {

            let camY = cameraHeight

            // ── 观察相机与真实拍摄相机一致（高度、俯仰角、FOV）──
            viewCam.camera?.fieldOfView = CGFloat(srcFovH)
            viewCam.position = SCNVector3(0, camY, 0)
            viewCam.eulerAngles = SCNVector3(-CGFloat(cameraTiltDeg * .pi / 180), 0, 0)

            // ── 更新源 FOV 锥 ──
            fovConeNode.childNodes.forEach { $0.removeFromParentNode() }
            let srcCone = buildFrustum(
                hFovDeg: srcFovH, vFovDeg: srcFovV,
                near: 0.5, far: 15,
                tiltDeg: cameraTiltDeg, panDeg: 0,
                cameraY: camY
            )
            srcCone.geometry?.firstMaterial?.diffuse.contents = NSColor.systemBlue
            fovConeNode.addChildNode(srcCone)

            // ── 更新取景锥 (随舵机转动) ──
            cropConeNode.childNodes.forEach { $0.removeFromParentNode() }
            let gimbalPan = panDeg - 120.0   // 相对于中心的偏转
            let gimbalTilt = tiltDeg - 120.0
            let cropCone = buildFrustum(
                hFovDeg: camFovH, vFovDeg: camFovV,
                near: 0.5, far: 12,
                tiltDeg: cameraTiltDeg + gimbalTilt,
                panDeg: gimbalPan,
                cameraY: camY
            )
            cropCone.geometry?.firstMaterial?.diffuse.contents = NSColor.systemYellow
            cropConeNode.addChildNode(cropCone)

            // ── 估算 RC 小车世界坐标 ──
            // 从舵机角度 + 目标在画面中的偏移 → 射线与地面交点
            if let det = targetInCrop {
                let targetCenterX = det.midX - 0.5  // -0.5~0.5
                let targetCenterY = det.midY - 0.5

                // 目标的总角度 = 舵机偏转 + 画面内偏移
                let totalPanDeg = gimbalPan + Double(targetCenterX) * camFovH
                let totalTiltDeg = cameraTiltDeg + gimbalTilt + Double(targetCenterY) * camFovV

                let totalTiltRad = totalTiltDeg * .pi / 180.0
                let totalPanRad = totalPanDeg * .pi / 180.0

                // 射线与 y=0 平面交点
                if totalTiltDeg > 1 { // 必须向下看才能与地面相交
                    let groundDist = camY / tan(totalTiltRad)
                    let worldX = groundDist * sin(totalPanRad)
                    let worldZ = groundDist * cos(totalPanRad)

                    // 限制范围避免异常值
                    let clampedX = min(max(worldX, -15), 15)
                    let clampedZ = min(max(worldZ, 0), 25)

                    carNode.position = SCNVector3(clampedX, 0.06, clampedZ)
                    carNode.eulerAngles.y = CGFloat(-totalPanRad)
                    carNode.isHidden = false
                } else {
                    carNode.isHidden = true
                }
            } else {
                carNode.isHidden = true
            }

            // 小车颜色随状态变化
            let carColor: NSColor
            switch state {
            case "TRACK": carColor = .systemRed
            case "HOLD", "INERTIA", "KF-INERTIA": carColor = .systemOrange
            default: carColor = .systemGray
            }
            carNode.geometry?.firstMaterial?.diffuse.contents = carColor

            // ── 更新轨迹 ──
            let step = max(1, carTrail.count / maxTrailDots)
            for i in 0..<maxTrailDots {
                let idx = i * step
                guard idx < carTrail.count else {
                    trailDots[i].isHidden = true
                    continue
                }
                let pt = carTrail[idx]
                trailDots[i].position = SCNVector3(pt.x, 0.04, pt.z)
                let age = CGFloat(i) / CGFloat(maxTrailDots)
                trailDots[i].geometry?.firstMaterial?.diffuse.contents =
                    NSColor.systemGreen.withAlphaComponent(age * 0.6 + 0.15)
                trailDots[i].isHidden = false
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 舵机角度曲线图
// ═══════════════════════════════════════════════════════════════

struct ServoPlotView: View {
    let trail: [SimEngine.ServoSample]
    let currentTime: Double
    var maxDPS: Double = 380
    var accelDPS2: Double = 3000

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text("舵机角度曲线 Servo Plot")
                    .font(.caption2.bold())
                    .foregroundColor(.white.opacity(0.6))
                    .padding(.trailing, 4)
                Label("Pan cmd/act", systemImage: "line.diagonal").foregroundColor(.cyan.opacity(0.7))
                Label("Tilt cmd/act", systemImage: "line.diagonal").foregroundColor(.orange.opacity(0.7))
                Spacer()
                Text("角度 0°—240° · 速度/加速度贴齐参考线=饱和").font(.caption2).foregroundColor(.gray)
            }
            .font(.caption2)
            .padding(.horizontal, 4)

            Canvas { ctx, sz in
                let w = sz.width, h = sz.height
                guard trail.count > 1 else { return }

                let window: Double = 8.0
                let tEnd = currentTime
                let tStart = tEnd - window

                // 三列一行：角度 / 速度(±maxDPS) / 加速度(±accelDPS2)
                let colW = w / 3
                let maxVel = max(1, maxDPS)
                let maxAcc = max(1, accelDPS2)

                // 列间分隔线
                [colW, 2 * colW].forEach { x in
                    ctx.stroke(Path { p in
                        p.move(to: CGPoint(x: x, y: 0))
                        p.addLine(to: CGPoint(x: x, y: h))
                    }, with: .color(.white.opacity(0.15)), lineWidth: 1)
                }

                // ─── 第 1 列：角度 0—240° ───
                let angleScaleY = h / 240.0
                func plotAngleLine(keyPath: KeyPath<SimEngine.ServoSample, Double>, color: Color, dashed: Bool) {
                    var path = Path()
                    var started = false
                    for s in trail {
                        let frac = (s.t - tStart) / window
                        guard frac >= 0 && frac <= 1 else { continue }
                        let x = CGFloat(frac) * colW
                        let y = h - CGFloat(s[keyPath: keyPath]) * angleScaleY
                        if !started { path.move(to: CGPoint(x: x, y: y)); started = true }
                        else { path.addLine(to: CGPoint(x: x, y: y)) }
                    }
                    if dashed {
                        ctx.stroke(path, with: .color(color.opacity(0.4)), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
                    } else {
                        ctx.stroke(path, with: .color(color), lineWidth: 2)
                    }
                }
                for deg in stride(from: 0.0, through: 240, by: 60) {
                    let y = h - CGFloat(deg) * angleScaleY
                    ctx.stroke(Path { p in
                        p.move(to: CGPoint(x: 0, y: y))
                        p.addLine(to: CGPoint(x: colW, y: y))
                    }, with: .color(.white.opacity(0.08)), lineWidth: 0.5)
                }
                let homeY = h - 120.0 * angleScaleY
                ctx.stroke(Path { p in
                    p.move(to: CGPoint(x: 0, y: homeY))
                    p.addLine(to: CGPoint(x: colW, y: homeY))
                }, with: .color(.white.opacity(0.2)), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
                for deg in stride(from: 0.0, through: 240, by: 60) {
                    let y = h - CGFloat(deg) * angleScaleY
                    ctx.draw(Text("\(Int(deg))°").font(.system(size: 8)).foregroundColor(.gray),
                             at: CGPoint(x: 4, y: y), anchor: .leading)
                }
                plotAngleLine(keyPath: \.cmdH, color: .cyan, dashed: true)
                plotAngleLine(keyPath: \.actH, color: .cyan, dashed: false)
                plotAngleLine(keyPath: \.cmdV, color: .orange, dashed: true)
                plotAngleLine(keyPath: \.actV, color: .orange, dashed: false)
                ctx.draw(Text("角度").font(.system(size: 8)).foregroundColor(.gray),
                         at: CGPoint(x: 4, y: 2), anchor: .leading)

                // ─── 第 2 列：速度 °/s，轴限 ±maxDPS + y 刻度 ───
                let velMidY = h / 2
                let velScale = (h / 2) / CGFloat(maxVel)
                let velTicks: [Double] = [0, maxVel / 2, maxVel, -maxVel / 2, -maxVel]
                for v in velTicks {
                    let y = velMidY - CGFloat(v) * velScale
                    ctx.stroke(Path { p in
                        p.move(to: CGPoint(x: colW, y: y))
                        p.addLine(to: CGPoint(x: 2 * colW, y: y))
                    }, with: .color(v == 0 ? .white.opacity(0.2) : .white.opacity(0.08)), lineWidth: v == 0 ? 1 : 0.5)
                    let label = v == 0 ? "0" : (v > 0 ? "\(Int(v))" : "\(Int(v))")
                    ctx.draw(Text("\(label)").font(.system(size: 7)).foregroundColor(.gray),
                             at: CGPoint(x: colW + 4, y: y), anchor: .leading)
                }
                [-1, 1].forEach { sign in
                    let y = velMidY - CGFloat(sign) * CGFloat(maxDPS) * velScale
                    ctx.stroke(Path { p in
                        p.move(to: CGPoint(x: colW, y: y))
                        p.addLine(to: CGPoint(x: 2 * colW, y: y))
                    }, with: .color(.red.opacity(0.6)), style: StrokeStyle(lineWidth: 1, dash: [4, 2]))
                }
                ctx.draw(Text("±\(Int(maxDPS))°/s").font(.system(size: 8)).foregroundColor(.red.opacity(0.9)),
                         at: CGPoint(x: 2 * colW - 42, y: velMidY - CGFloat(maxDPS) * velScale - 10), anchor: .leading)
                func velToY(_ v: Double) -> CGFloat { velMidY - CGFloat(max(-maxVel, min(maxVel, v))) * velScale }
                func plotVel(keyPath: KeyPath<SimEngine.ServoSample, Double>, color: Color) {
                    var path = Path()
                    var started = false
                    for s in trail {
                        let frac = (s.t - tStart) / window
                        guard frac >= 0 && frac <= 1 else { continue }
                        let x = colW + CGFloat(frac) * colW
                        let y = velToY(s[keyPath: keyPath])
                        if !started { path.move(to: CGPoint(x: x, y: y)); started = true }
                        else { path.addLine(to: CGPoint(x: x, y: y)) }
                    }
                    ctx.stroke(path, with: .color(color), lineWidth: 1.5)
                }
                plotVel(keyPath: \.velH, color: .cyan)
                plotVel(keyPath: \.velV, color: .orange)
                ctx.draw(Text("速度 °/s").font(.system(size: 8)).foregroundColor(.gray),
                         at: CGPoint(x: colW + 4, y: 2), anchor: .leading)

                // ─── 第 3 列：加速度 °/s²，轴限 ±accelDPS2 + y 刻度 ───
                let accMidY = h / 2
                let accScale = (h / 2) / CGFloat(maxAcc)
                let accTicks: [Double] = [0, accelDPS2 / 2, accelDPS2, -accelDPS2 / 2, -accelDPS2]
                for a in accTicks {
                    let y = accMidY - CGFloat(a) * accScale
                    ctx.stroke(Path { p in
                        p.move(to: CGPoint(x: 2 * colW, y: y))
                        p.addLine(to: CGPoint(x: w, y: y))
                    }, with: .color(a == 0 ? .white.opacity(0.2) : .white.opacity(0.08)), lineWidth: a == 0 ? 1 : 0.5)
                    let label = a == 0 ? "0" : (a > 0 ? "\(Int(a))" : "\(Int(a))")
                    ctx.draw(Text("\(label)").font(.system(size: 7)).foregroundColor(.gray),
                             at: CGPoint(x: 2 * colW + 4, y: y), anchor: .leading)
                }
                [-1, 1].forEach { sign in
                    let y = accMidY - CGFloat(sign) * CGFloat(accelDPS2) * accScale
                    ctx.stroke(Path { p in
                        p.move(to: CGPoint(x: 2 * colW, y: y))
                        p.addLine(to: CGPoint(x: w, y: y))
                    }, with: .color(.purple.opacity(0.6)), style: StrokeStyle(lineWidth: 1, dash: [4, 2]))
                }
                ctx.draw(Text("±\(Int(accelDPS2))°/s²").font(.system(size: 8)).foregroundColor(.purple.opacity(0.9)),
                         at: CGPoint(x: w - 52, y: accMidY - CGFloat(accelDPS2) * accScale - 10), anchor: .leading)
                var accPathH = Path()
                var accPathV = Path()
                var startedH = false, startedV = false
                let accSmooth: Double = 0.25
                var smH: Double? = nil, smV: Double? = nil
                for i in 1..<trail.count {
                    let dt = trail[i].t - trail[i-1].t
                    guard dt > 0 && dt < 0.5 else { continue }
                    var accH = (trail[i].velH - trail[i-1].velH) / dt
                    var accV = (trail[i].velV - trail[i-1].velV) / dt
                    accH = max(-accelDPS2, min(accelDPS2, accH))
                    accV = max(-accelDPS2, min(accelDPS2, accV))
                    if let prevH = smH { accH = prevH + (accH - prevH) * accSmooth }
                    if let prevV = smV { accV = prevV + (accV - prevV) * accSmooth }
                    smH = accH; smV = accV
                    let frac = (trail[i].t - tStart) / window
                    guard frac >= 0 && frac <= 1 else { continue }
                    let x = 2 * colW + CGFloat(frac) * colW
                    let yH = accMidY - CGFloat(accH) * accScale
                    let yV = accMidY - CGFloat(accV) * accScale
                    if !startedH { accPathH.move(to: CGPoint(x: x, y: yH)); startedH = true }
                    else { accPathH.addLine(to: CGPoint(x: x, y: yH)) }
                    if !startedV { accPathV.move(to: CGPoint(x: x, y: yV)); startedV = true }
                    else { accPathV.addLine(to: CGPoint(x: x, y: yV)) }
                }
                ctx.stroke(accPathH, with: .color(.cyan), lineWidth: 1.5)
                ctx.stroke(accPathV, with: .color(.orange), lineWidth: 1.5)
                ctx.draw(Text("加速度 °/s²").font(.system(size: 8)).foregroundColor(.gray),
                         at: CGPoint(x: 2 * colW + 4, y: 2), anchor: .leading)

                // 状态色带（整高，半透明）
                var lastX: CGFloat = 0
                var lastState = ""
                for s in trail {
                    let frac = (s.t - tStart) / window
                    guard frac >= 0 && frac <= 1 else { continue }
                    let x = CGFloat(frac) * w
                    if s.state != lastState && lastState != "" {
                        let stateColor: Color
                        switch lastState {
                        case "TRACK": stateColor = .green
                        case "INERTIA", "KF-INERTIA": stateColor = .yellow
                        case "SCAN": stateColor = .orange
                        case "RETURN": stateColor = .red
                        default: stateColor = .gray
                        }
                        ctx.fill(Path(CGRect(x: lastX, y: 0, width: x - lastX, height: h)),
                                 with: .color(stateColor.opacity(0.06)))
                    }
                    if s.state != lastState { lastX = x; lastState = s.state }
                }
                if !lastState.isEmpty {
                    let stateColor: Color
                    switch lastState {
                    case "TRACK": stateColor = .green
                    case "INERTIA", "KF-INERTIA": stateColor = .yellow
                    case "SCAN": stateColor = .orange
                    case "RETURN": stateColor = .red
                    default: stateColor = .gray
                    }
                    ctx.fill(Path(CGRect(x: lastX, y: 0, width: w - lastX, height: h)),
                             with: .color(stateColor.opacity(0.06)))
                }
            }
            .background(Color(white: 0.04))
            .cornerRadius(4)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 无闪烁图像层 (NSViewRepresentable)
// ═══════════════════════════════════════════════════════════════

/// 直接操作 CALayer，绕开 SwiftUI 动画过渡，彻底消除换帧闪烁。
struct StaticImageView: NSViewRepresentable {
    let cgImage: CGImage?

    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        v.wantsLayer = true
        v.layer?.contentsGravity = .resizeAspect
        v.layer?.backgroundColor = CGColor(gray: 0, alpha: 1)
        return v
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        // 直接赋值，无任何 CoreAnimation 隐式事务
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        nsView.layer?.contents = cgImage
        CATransaction.commit()
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 可视化视图
// ═══════════════════════════════════════════════════════════════

enum FrameMode { case source, crop }

struct FrameV: View {
    let f: Snap
    let mode: FrameMode
    var useBoxSelection: Bool = false
    var onBoxSelected: ((CGRect) -> Void)? = nil
    var gtBoxes: [AnnBox] = []   // ★ GT 标注框 (归一化坐标)

    @State private var dragStart: CGPoint?
    @State private var dragCurrent: CGPoint?

    private static func aspectFitContent(sz: CGSize, asp: CGFloat) -> (CGFloat, CGFloat, CGFloat, CGFloat) {
        let va = sz.width / sz.height
        if asp > va {
            return (sz.width, sz.width / asp, 0, (sz.height - sz.width / asp) / 2)
        } else {
            return (sz.height * asp, sz.height, (sz.width - sz.height * asp) / 2, 0)
        }
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // ★ 图像层：用 NSViewRepresentable 原地更新，无 SwiftUI 过渡动画
                let displayImg = mode == .source ? f.srcImg : f.cropImg
                StaticImageView(cgImage: displayImg)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                // 叠加绘制
                Canvas { ctx, sz in
                    let asp: CGFloat
                    if mode == .source {
                        asp = f.srcImg.map { CGFloat($0.width) / CGFloat($0.height) } ?? (16.0 / 9.0)
                    } else {
                        asp = f.cropImg.map { CGFloat($0.width) / CGFloat($0.height) } ?? (16.0 / 9.0)
                    }
                    let fit = fitRect(sz, asp)
                    let rw = fit.0, rh = fit.1, ox = fit.2, oy = fit.3

                    if mode == .source {
                        let cr = CGRect(
                            x: ox + f.crop.minX * rw,
                            y: oy + f.crop.minY * rh,
                            width: f.crop.width * rw,
                            height: f.crop.height * rh
                        )
                        ctx.stroke(Path(cr), with: .color(.cyan), lineWidth: 2)
                        // ★ 中文标注
                        ctx.draw(
                            Text("取景区域")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.cyan),
                            at: CGPoint(x: cr.minX + 4, y: cr.minY - 2),
                            anchor: .bottomLeading
                        )
                    }

                    if let d = f.det {
                        let dr: CGRect
                        if mode == .source {
                            dr = CGRect(
                                x: ox + (f.crop.minX + d.minX * f.crop.width) * rw,
                                y: oy + (f.crop.minY + d.minY * f.crop.height) * rh,
                                width: d.width * f.crop.width * rw,
                                height: d.height * f.crop.height * rh
                            )
                        } else {
                            dr = CGRect(
                                x: ox + d.minX * rw,
                                y: oy + d.minY * rh,
                                width: d.width * rw,
                                height: d.height * rh
                            )
                        }
                        // 外框 (绿色 3px)
                        ctx.stroke(Path(dr), with: .color(.green), lineWidth: 3)
                        // 内框 (黑色描边增加对比)
                        ctx.stroke(Path(dr.insetBy(dx: -1.5, dy: -1.5)), with: .color(.black.opacity(0.4)), lineWidth: 1)
                        // ★ 中文标注
                        ctx.draw(
                            Text("追踪目标")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.green),
                            at: CGPoint(x: dr.minX + 2, y: dr.minY - 2),
                            anchor: .bottomLeading
                        )
                        // 中心十字
                        let cx = dr.midX, cy = dr.midY
                        var cross = Path()
                        cross.move(to: CGPoint(x: cx - 8, y: cy))
                        cross.addLine(to: CGPoint(x: cx + 8, y: cy))
                        cross.move(to: CGPoint(x: cx, y: cy - 8))
                        cross.addLine(to: CGPoint(x: cx, y: cy + 8))
                        ctx.stroke(cross, with: .color(.green), lineWidth: 2)
                        // 中心点
                        ctx.fill(Path(ellipseIn: CGRect(x: cx - 3, y: cy - 3, width: 6, height: 6)),
                                 with: .color(.green))
                    } else if mode == .source {
                        // 未追踪时在画面中央显示 ✕
                        let cx = ox + rw / 2, cy = oy + rh / 2
                        var xMark = Path()
                        xMark.move(to: CGPoint(x: cx - 12, y: cy - 12))
                        xMark.addLine(to: CGPoint(x: cx + 12, y: cy + 12))
                        xMark.move(to: CGPoint(x: cx + 12, y: cy - 12))
                        xMark.addLine(to: CGPoint(x: cx - 12, y: cy + 12))
                        ctx.stroke(xMark, with: .color(.red.opacity(0.5)), lineWidth: 2)
                    }

                    // ★ 绘制 GT 标注框 (橙色虚线)
                    if mode == .source && !gtBoxes.isEmpty {
                        for box in gtBoxes {
                            // box.cx/cy/w/h 是归一化 [0,1]，直接映射到画面坐标
                            let gr = CGRect(
                                x: ox + (box.cx - box.w / 2) * rw,
                                y: oy + (box.cy - box.h / 2) * rh,
                                width: box.w * rw,
                                height: box.h * rh
                            )
                            // 橙色虚线框
                            ctx.stroke(Path(gr), with: .color(.orange),
                                       style: StrokeStyle(lineWidth: 2, dash: [6, 3]))
                            // 标签
                            ctx.draw(
                                Text("GT")
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundColor(.orange),
                                at: CGPoint(x: gr.minX + 2, y: gr.minY - 2),
                                anchor: .bottomLeading
                            )
                        }
                    }

                    if mode == .crop {
                        let cxP = ox + rw / 2
                        let cyP = oy + rh / 2
                        var crossPath = Path()
                        crossPath.move(to: CGPoint(x: cxP - 15, y: cyP))
                        crossPath.addLine(to: CGPoint(x: cxP + 15, y: cyP))
                        crossPath.move(to: CGPoint(x: cxP, y: cyP - 15))
                        crossPath.addLine(to: CGPoint(x: cxP, y: cyP + 15))
                        ctx.stroke(crossPath, with: .color(.white.opacity(0.4)), lineWidth: 1)
                        // 画面中心标注
                        ctx.draw(
                            Text("画面中心")
                                .font(.system(size: 8))
                                .foregroundColor(.white.opacity(0.4)),
                            at: CGPoint(x: cxP + 18, y: cyP - 2),
                            anchor: .leading
                        )

                        // ★ Lead Room 目标点 (橙色小十字, 速度方向偏移后的构图中心)
                        if let dbg = f.dbg, (abs(dbg.leadRoomX) > 0.001 || abs(dbg.leadRoomY) > 0.001) {
                            let lrPx = ox + (0.5 + dbg.leadRoomX) * rw
                            let lrPy = oy + (0.5 + dbg.leadRoomY) * rh
                            var lrCross = Path()
                            lrCross.move(to: CGPoint(x: lrPx - 10, y: lrPy))
                            lrCross.addLine(to: CGPoint(x: lrPx + 10, y: lrPy))
                            lrCross.move(to: CGPoint(x: lrPx, y: lrPy - 10))
                            lrCross.addLine(to: CGPoint(x: lrPx, y: lrPy + 10))
                            ctx.stroke(lrCross, with: .color(.orange.opacity(0.8)), lineWidth: 2)
                            ctx.fill(Path(ellipseIn: CGRect(x: lrPx - 2.5, y: lrPy - 2.5, width: 5, height: 5)),
                                     with: .color(.orange))
                            ctx.draw(
                                Text("构图目标")
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.orange.opacity(0.8)),
                                at: CGPoint(x: lrPx + 13, y: lrPy - 2),
                                anchor: .leading
                            )
                        }

                        // ★ 主仿真器显示 YOLO 检测框（不依赖 debugMode）
                        if mode == .source {
                            for (idx, cand) in f.yoloCandidates.enumerated() {
                                let cr = CGRect(
                                    x: ox + cand.rectTL.minX * rw,
                                    y: oy + cand.rectTL.minY * rh,
                                    width: cand.rectTL.width * rw,
                                    height: cand.rectTL.height * rh
                                )
                                let candColor: Color = cand.accepted ? .green : .red.opacity(0.6)
                                ctx.stroke(Path(cr), with: .color(candColor),
                                           style: StrokeStyle(lineWidth: cand.accepted ? 3 : 1.5,
                                                              dash: cand.accepted ? [] : [4, 3]))
                                let scoreStr: String
                                if cand.accepted {
                                    scoreStr = cand.appSim >= 0
                                        ? String(format: "✅ YOLO#%d %@ 得分=%.2f", idx, cand.label, cand.totalScore)
                                        : String(format: "✅ YOLO#%d %@ 置信=%.2f", idx, cand.label, cand.confidence)
                                } else {
                                    scoreStr = cand.appSim >= 0
                                        ? String(format: "❌ YOLO#%d %@ 得分=%.2f", idx, cand.label, cand.totalScore)
                                        : String(format: "❌ YOLO#%d %@ 置信=%.2f", idx, cand.label, cand.confidence)
                                }
                                ctx.draw(
                                    Text(scoreStr)
                                        .font(.system(size: 9, weight: .bold).monospaced())
                                        .foregroundColor(cand.accepted ? .green : .red),
                                    at: CGPoint(x: cr.minX + 2, y: cr.minY - 3),
                                    anchor: .bottomLeading
                                )
                            }
                        }

                        // ★ Debug Overlay: PID 箭头 + 预测信息
                        if let dbg = f.dbg {
                            // ★ 时序二次曲线轨迹 (紫色平滑曲线)
                            if !dbg.trajCurve.isEmpty {
                                var trajPath = Path()
                                let first = dbg.trajCurve[0]
                                trajPath.move(to: CGPoint(x: ox + first.x * rw, y: oy + first.y * rh))
                                for pt in dbg.trajCurve.dropFirst() {
                                    trajPath.addLine(to: CGPoint(x: ox + pt.x * rw, y: oy + pt.y * rh))
                                }
                                ctx.stroke(trajPath, with: .color(.purple), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))

                                let last = dbg.trajCurve.last!
                                ctx.draw(
                                    Text("时序曲率预测")
                                        .font(.system(size: 8, weight: .bold))
                                        .foregroundColor(.purple),
                                    at: CGPoint(x: ox + last.x * rw + 5, y: oy + last.y * rh),
                                    anchor: .leading
                                )
                            }

                            // PID 误差箭头 (从画面中心指向目标)
                            let errEndX = cxP + dbg.errDegX * rw / 30  // 缩放可视化
                            let errEndY = cyP + dbg.errDegY * rh / 20
                            var errArrow = Path()
                            errArrow.move(to: CGPoint(x: cxP, y: cyP))
                            errArrow.addLine(to: CGPoint(x: errEndX, y: errEndY))
                            ctx.stroke(errArrow, with: .color(.yellow), lineWidth: 2)
                            // 黄色箭头标注
                            ctx.draw(
                                Text(String(format: "误差 %.1f°", sqrt(dbg.errDegX*dbg.errDegX + dbg.errDegY*dbg.errDegY)))
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.yellow),
                                at: CGPoint(x: errEndX + 3, y: errEndY - 3),
                                anchor: .topLeading
                            )

                            // 预测误差箭头 (虚线)
                            let predEndX = cxP + dbg.predErrX * rw / 30
                            let predEndY = cyP + dbg.predErrY * rh / 20
                            var predArrow = Path()
                            predArrow.move(to: CGPoint(x: cxP, y: cyP))
                            predArrow.addLine(to: CGPoint(x: predEndX, y: predEndY))
                            ctx.stroke(predArrow, with: .color(.cyan),
                                       style: StrokeStyle(lineWidth: 1.5, dash: [3, 2]))

                            // ★ 升级版可视化: 卡尔曼预测位置与尺度 (蓝色虚线框 + 十字)
                            if let kfc = dbg.kfCenter {
                                let kx = ox + kfc.x * rw
                                let ky = oy + kfc.y * rh
                                
                                // 1. 始终绘制中心十字
                                var kfMark = Path()
                                kfMark.move(to: CGPoint(x: kx - 8, y: ky))
                                kfMark.addLine(to: CGPoint(x: kx + 8, y: ky))
                                kfMark.move(to: CGPoint(x: kx, y: ky - 8))
                                kfMark.addLine(to: CGPoint(x: kx, y: ky + 8))
                                ctx.stroke(kfMark, with: .color(.blue), lineWidth: 2)
                                
                                // 2. 如果开启了增强算法且有尺度预测，绘制 8D 预测虚线框
                                if let kfs = dbg.kfSize, dbg.useEnhancedAlgo {
                                    let kw = kfs.width * rw
                                    let kh = kfs.height * rh
                                    let kfRect = CGRect(x: kx - kw / 2, y: ky - kh / 2, width: kw, height: kh)
                                    
                                    ctx.stroke(Path(kfRect), with: .color(.blue.opacity(0.8)), style: StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                                    
                                    ctx.draw(
                                        Text("KF 8D预测")
                                            .font(.system(size: 8, weight: .bold))
                                            .foregroundColor(.blue),
                                        at: CGPoint(x: kfRect.minX + 2, y: kfRect.minY - 3),
                                        anchor: .bottomLeading
                                    )
                                } else {
                                    // 退化为 4D 算法时，只标注预测中心
                                    ctx.draw(
                                        Text("KF预测中心")
                                            .font(.system(size: 8, weight: .bold))
                                            .foregroundColor(.blue),
                                        at: CGPoint(x: kx + 10, y: ky - 2),
                                        anchor: .leading
                                    )
                                }
                            }
                        }
                    }
                }

                // 标签
                VStack {
                    HStack(spacing: 4) {
                        // ★ 视图标识 (替代外层 overlay，避免重叠)
                        Text(mode == .source ? "广角源视频 Source" : "舵机取景框 Camera View")
                            .font(.caption2.bold())
                            .foregroundColor(.white.opacity(0.7))
                            .padding(.horizontal, 5).padding(.vertical, 2)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(3)
                        // 状态 badge：中文对照
                        Badge(stateLabel(f.state), stateColor(f.state))
                        if !f.kfSt.isEmpty {
                            Badge(kfLabel(f.kfSt), f.kfSt.hasPrefix("KF") ? .indigo : .blue)
                        }
                        if mode == .source {
                            Badge(detSrcLabel(f.detSrc), f.detSrc == "TRK" ? .green : (f.detSrc == "YOLO" ? .purple : .gray))
                        }
                        Spacer()
                        Text(String(format: "%.2fs #%d", f.t, f.idx))
                            .font(.caption.monospaced())
                            .foregroundColor(.white)
                            .padding(3)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(3)
                    }
                    Spacer()
                    if mode == .crop {
                        HStack {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(String(format: "偏差(Err) H%.1f° V%.1f°  YOLO推理 %.0fms", f.eX, f.eY, f.yoloMs))
                                Text(String(format: "舵机(Srv) H%.1f° V%.1f°  速度(Vel) H%.0f V%.0f°/s",
                                            f.actH, f.actV, f.velH, f.velV))
                                Text(String(format: "变焦 Opt %.2fx  Dig %.2fx  FOV %.0f°", f.optZoom, f.digZoom, f.effFovH))
                            }
                            .font(.caption2.monospaced())
                            .foregroundColor(.green)
                            .padding(3)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(3)
                            Spacer()
                        }
                    }
                }
                .padding(6)
            }
            .overlay {
                if mode == .source, useBoxSelection, let callback = onBoxSelected {
                    let asp = f.srcImg.map { CGFloat($0.width) / CGFloat($0.height) } ?? (16.0 / 9.0)
                    GeometryReader { g in
                        let (rw, rh, ox, oy) = Self.aspectFitContent(sz: g.size, asp: asp)
                        ZStack {
                            Color.clear.contentShape(Rectangle())
                                .gesture(
                                    DragGesture()
                                        .onChanged { value in
                                            if dragStart == nil { dragStart = value.startLocation }
                                            dragCurrent = value.location
                                        }
                                        .onEnded { value in
                                            let s = value.startLocation
                                            let e = value.location
                                            let nx1 = min(max((s.x - ox) / rw, 0), 1)
                                            let ny1 = min(max((s.y - oy) / rh, 0), 1)
                                            let nx2 = min(max((e.x - ox) / rw, 0), 1)
                                            let ny2 = min(max((e.y - oy) / rh, 0), 1)
                                            let rectTL = CGRect(
                                                x: min(nx1, nx2),
                                                y: min(ny1, ny2),
                                                width: max(0.01, abs(nx2 - nx1)),
                                                height: max(0.01, abs(ny2 - ny1))
                                            )
                                            callback(rectTL)
                                            dragStart = nil
                                            dragCurrent = nil
                                        }
                                )
                            if let s = dragStart, let e = dragCurrent {
                                let nx1 = (s.x - ox) / rw
                                let ny1 = (s.y - oy) / rh
                                let nx2 = (e.x - ox) / rw
                                let ny2 = (e.y - oy) / rh
                                let r = CGRect(
                                    x: ox + min(nx1, nx2) * rw,
                                    y: oy + min(ny1, ny2) * rh,
                                    width: abs(nx2 - nx1) * rw,
                                    height: abs(ny2 - ny1) * rh
                                )
                                Rectangle()
                                    .stroke(Color.orange, lineWidth: 3)
                                    .frame(width: r.width, height: r.height)
                                    .position(x: r.midX, y: r.midY)
                            }
                        }
                    }
                }
            }
        }
    }

    private func stateLabel(_ s: String) -> String {
        switch s {
        case "TRACK":      return "TRACK 追踪"
        case "INERTIA":    return "INERTIA 惯性"
        case "KF-INERTIA": return "KF-INERTIA 卡尔曼惯性"
        case "HOLD":       return "HOLD 保持"
        case "SCAN":       return "SCAN 扫描"
        case "RETURN":     return "RETURN 回中"
        default:           return s
        }
    }

    private func kfLabel(_ s: String) -> String {
        if s == "TRK"  { return "TRK 追踪" }
        if s == "KF-L" { return "KF-L 失效" }
        if s == "IDLE" { return "IDLE 空闲" }
        if s.hasPrefix("KF") { return "\(s) 卡尔曼" }
        return s
    }

    private func detSrcLabel(_ s: String) -> String {
        switch s {
        case "TRK":  return "TRK 跟踪器"
        case "YOLO": return "YOLO 检测"
        case "NONE": return "NONE 无检测"
        default:     return s
        }
    }

    private func stateColor(_ s: String) -> Color {
        switch s {
        case "TRACK": return .green
        case "INERTIA", "KF-INERTIA": return .yellow
        case "HOLD": return .yellow
        case "SCAN": return .orange
        case "RETURN": return .red
        default: return .gray
        }
    }

    private func fitRect(_ sz: CGSize, _ asp: CGFloat) -> (CGFloat, CGFloat, CGFloat, CGFloat) {
        let va = sz.width / sz.height
        if asp > va {
            return (sz.width, sz.width / asp, 0, (sz.height - sz.width / asp) / 2)
        } else {
            return (sz.height * asp, sz.height, (sz.width - sz.height * asp) / 2, 0)
        }
    }
}

struct Badge: View {
    let text: String
    let color: Color
    init(_ text: String, _ color: Color) {
        self.text = text
        self.color = color
    }
    var body: some View {
        if !text.isEmpty {
            Text(text)
                .font(.caption.bold().monospaced())
                .foregroundColor(.white)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(color.opacity(0.8))
                .cornerRadius(3)
        }
    }
}

struct StatsPanel: View {
    let s: Stats
    var body: some View {
        GroupBox("Metrics 仿真指标") {
            VStack(alignment: .leading, spacing: 3) {
                MetricRow("Frames 总帧数", "\(s.frames)")
                MetricRow("YOLO 推理延迟", String(format: "%.1fms", s.avgYolo))
                Divider()
                // ★ Ground Truth 追踪分析
                Text("── GT SAHI 追踪分析 ──").font(.caption2.bold()).foregroundColor(.orange)
                MetricRow("GT可见率 SAHI", String(format: "%.1f%% (%d/%d)", s.gtVisiblePct, s.gtDetectable, s.frames),
                          s.gtVisiblePct > 80 ? .green : (s.gtVisiblePct > 50 ? .yellow : .red))
                MetricRow("纯Tracker率 VNTrk", String(format: "%.1f%% (%d)", s.pureTrkPct, s.trkOnlyFrames),
                          s.pureTrkPct > 60 ? .green : (s.pureTrkPct > 30 ? .yellow : .red))
                MetricRow("追踪效率 Trk/GT", String(format: "%.1f%%", s.trkEfficiency),
                          s.trkEfficiency > 80 ? .green : (s.trkEfficiency > 50 ? .yellow : .red))
                MetricRow("SAHI耗时", String(format: "%.1fms", s.gtSahiMs), .secondary)
                // ★ 标注 GT 评测 (虚拟赛道口径)
                if s.gtAnnotationFrames > 0 {
                    Divider()
                    Text("── ★ 标注 GT 评测 (虚拟赛道口径) ──").font(.caption2.bold()).foregroundColor(.cyan)
                    MetricRow("一般追踪率", String(format: "%.1f%% (%d/%d)", s.generalTrackRate, s.trackOnCarFrames, s.frames),
                              s.generalTrackRate > 70 ? .green : (s.generalTrackRate > 40 ? .yellow : .red))
                    MetricRow("YOLO检测率 GT-vis", String(format: "%.1f%% (%d/%d)", s.yoloDetectRateVisible, s.yoloHitFrames, s.gtAnnotationFrames),
                              s.yoloDetectRateVisible > 70 ? .green : (s.yoloDetectRateVisible > 40 ? .yellow : .red))
                    MetricRow("单目标锁定率", String(format: "%.1f%% (%d/%d)", s.singleTargetLockRate, s.stLockOnCarFrames, s.stEvalFrames),
                              s.singleTargetLockRate > 70 ? .green : (s.singleTargetLockRate > 40 ? .yellow : .red))
                    MetricRow("GT标注可见率", String(format: "%.1f%% (%d/%d)", s.gtAnnotationVisiblePct, s.gtAnnotationFrames, s.frames),
                              s.gtAnnotationVisiblePct > 80 ? .green : .yellow)
                }
                Divider()
                MetricRow("SrvErr H 水平误差", String(format: "avg %.2f° max %.2f°", s.avgEH, s.maxEH))
                MetricRow("SrvErr V 俯仰误差", String(format: "avg %.2f° max %.2f°", s.avgEV, s.maxEV))
                Divider()
                MetricRow("TgtErr 目标跟随误差", String(format: "avg %.1f° max %.1f°", s.avgTD, s.maxTD))
                MetricRow("Jitter H 水平抖动", String(format: "%.1f%% (%d)", s.jitH, s.dirH),
                          s.jitH > 30 ? .red : (s.jitH > 15 ? .yellow : .green))
                MetricRow("Jitter V 俯仰抖动", String(format: "%.1f%% (%d)", s.jitV, s.dirV),
                          s.jitV > 30 ? .red : (s.jitV > 15 ? .yellow : .green))
                MetricRow("PeakVel 峰值速度", String(format: "H %.0f V %.0f °/s", s.peakVH, s.peakVV))
            }
            .font(.caption.monospaced())
            .padding(4)
        }
    }
}

struct MetricRow: View {
    let label: String
    let value: String
    var color: Color = .primary

    init(_ label: String, _ value: String, _ color: Color = .primary) {
        self.label = label
        self.value = value
        self.color = color
    }

    var body: some View {
        HStack {
            Text(label).foregroundColor(.secondary)
            Spacer()
            Text(value).foregroundColor(color)
        }
    }
}

struct ParamSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let fmt: String

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack {
                Text(label).font(.caption).foregroundColor(.secondary)
                Spacer()
                Text(String(format: fmt, value)).font(.caption.monospaced())
            }
            Slider(value: $value, in: range)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Debug Inspector 面板 (显示所有中间变量)
// ═══════════════════════════════════════════════════════════════

struct DebugInspectorPanel: View {
    let dbg: DebugInfo
    @State private var expandA = true
    @State private var expandB = true
    @State private var expandC = true
    @State private var expandD = true

    var body: some View {
        GroupBox("🔍 Debug Inspector") {
            ScrollView(.vertical, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 4) {

                    // ── A. 检测层 ──
                    DisclosureGroup("A. 检测层 Detection", isExpanded: $expandA) {
                        VStack(alignment: .leading, spacing: 2) {
                            DbgRow("匹配方法 matchMethod", dbg.matchMethod, tip: "Appearance=外观特征匹配 IoU=交并比匹配 ColdStart=首次检测")
                            DbgRow("YOLO本帧执行", dbg.yoloRanThisFrame ? "✅ Yes" : "❌ No", tip: "YOLO 不是每帧都跑，约20-40fps")
                            DbgRow("Tracker存活", dbg.trackerAlive ? "✅ conf=\(String(format:"%.2f",dbg.trackerConf))" : "❌ Dead",
                                   tip: "VNTrackObjectRequest 是否成功追踪")
                            DbgRow("目标丢失帧数 lostFrames", "\(dbg.targetLostFrames)/\(Cfg.targetHoldFramesMax)",
                                   tip: "连续找不到目标的帧数，超过阈值则判定丢失")
                            DbgRow("最佳外观得分 bestAppScore", String(format: "%.3f", dbg.bestAppScore),
                                   tip: "本帧最佳候选的加权匹配得分 (接受阈值=\(Cfg.targetAcceptScore))")

                            if !dbg.yoloCandidates.isEmpty {
                                Text("候选框 Candidates (\(dbg.yoloCandidates.count)):")
                                    .font(.caption2.bold()).padding(.top, 2)
                                ForEach(Array(dbg.yoloCandidates.enumerated()), id: \.offset) { idx, c in
                                    VStack(alignment: .leading, spacing: 0) {
                                        HStack(spacing: 4) {
                                            Circle().fill(c.accepted ? Color.green : Color.red).frame(width: 6, height: 6)
                                            Text("#\(idx) \(c.label)")
                                                .font(.caption2.bold().monospaced())
                                            Text(String(format: "conf=%.2f", c.confidence))
                                                .font(.system(size: 9).monospaced())
                                                .foregroundColor(.secondary)
                                        }
                                        if c.appSim >= 0 {
                                            Text(String(format: "  app=%.2f mot=%.2f sz=%.2f → total=%.2f %@",
                                                        c.appSim, c.motionSim, c.sizeSim, c.totalScore,
                                                        c.accepted ? "✅" : "❌ \(c.rejectReason)"))
                                                .font(.system(size: 9).monospaced())
                                                .foregroundColor(c.accepted ? .green : .secondary)
                                        } else if !c.rejectReason.isEmpty {
                                            Text("  \(c.rejectReason)")
                                                .font(.system(size: 9).monospaced())
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // ── B. 控制层 ──
                    DisclosureGroup("B. 控制层 PID", isExpanded: $expandB) {
                        VStack(alignment: .leading, spacing: 2) {
                            DbgRow("误差 errDeg", String(format: "X=%.2f° Y=%.2f°", dbg.errDegX, dbg.errDegY),
                                   tip: "(目标中心-0.5)*FOV  正=右/下")
                            DbgRow("平滑速度 smoothVel", String(format: "X=%.1f°/s Y=%.1f°/s", dbg.smoothVelX, dbg.smoothVelY),
                                   tip: "EMA平滑后的目标角速度")
                            DbgRow("预测误差 predErr", String(format: "X=%.2f° Y=%.2f°", dbg.predErrX, dbg.predErrY),
                                   tip: "err + vel*leadTime  前瞻补偿后的误差")
                            DbgRow("增益缩放 gainScale", String(format: "%.3f", dbg.gainScale),
                                   tip: "距离自适应: 目标远则增益高(max 1.0), 近则降低(min 0.35)")
                            DbgRow("距离因子 distScale", String(format: "%.2f", dbg.distScale),
                                   tip: "targetRatio/objectHeight, 越大=目标越远")
                            DbgRow("PID输出 pidOut", String(format: "X=%.1f°/s Y=%.1f°/s", dbg.pidOutX, dbg.pidOutY),
                                   tip: "X负=向右追  Y正=向下追  限幅前的角速度指令")
                            DbgRow("边缘补偿 edgeBoost", String(format: "X=%.1f Y=%.1f", dbg.edgeBoostX, dbg.edgeBoostY),
                                   tip: "目标靠近画面边缘时的额外加速(非零=触发边缘补偿)")
                            DbgRow("构图余量 leadRoom", String(format: "X=%.3f Y=%.3f", dbg.leadRoomX, dbg.leadRoomY),
                                   tip: "速度方向留白(归一化)  正=目标点右/下移  车在画面偏左/上")
                            DbgRow("最终步进 step", String(format: "H=%.3f° V=%.3f°", dbg.stepH, dbg.stepV),
                                   tip: "pidOut*dt  应用于servoH/V的角度增量")
                        }
                    }

                    // ── C. 状态层 ──
                    DisclosureGroup("C. 状态层 State", isExpanded: $expandC) {
                        VStack(alignment: .leading, spacing: 2) {
                            DbgRow("舵机阶段 state", dbg.state,
                                   tip: "TRACK=追踪 INERTIA=惯性 HOLD=保持 SCAN=搜索 RETURN=回中")
                            DbgRow("卡尔曼 kfState", dbg.kfState,
                                   tip: "TRK=有观测更新 KFn=纯预测第n帧 KF-L=不确定性过大 IDLE=未启用")
                            if let kfc = dbg.kfCenter {
                                DbgRow("KF位置", String(format: "(%.3f, %.3f)", kfc.x, kfc.y), tip: "卡尔曼估计中心")
                            }
                            if let kfv = dbg.kfVelocity {
                                DbgRow("KF速度", String(format: "(%.2f, %.2f)/s", kfv.x, kfv.y), tip: "卡尔曼估计速度(归一化/秒)")
                            }
                            if let kfs = dbg.kfSize {
                                DbgRow("KF尺度", String(format: "W=%.1f%% H=%.1f%%", kfs.width*100, kfs.height*100), tip: "卡尔曼估计的动态宽高")
                            }
                            DbgRow("算法版本", dbg.useEnhancedAlgo ? "增强算法 (高斯+8D+时序曲线+局部光流)" : "基础算法 (均值+4D)", tip: "增强算法含: 高斯空间权重、8D尺度卡尔曼、二次多项式轨迹预测、局部光流速度融合")
                            DbgRow("KF不确定性", String(format: "%.4f", dbg.kfUncertainty),
                                   tip: "sqrt(Pxx+Pyy) 越大=预测越不可靠 阈值=\(Cfg.kfMaxUncert)")
                            DbgRow("外观模板", "\(dbg.hasTemplateFixed ? "Fixed✅" : "Fixed❌") \(dbg.hasTemplateDyn ? "Dyn✅" : "Dyn❌")",
                                   tip: "Fixed=首次锁定时不变 Dyn=动态EMA更新")
                            DbgRow("目标标签", dbg.targetLabel ?? "(无)", tip: "YOLO类别名称")
                        }
                    }

                    // ── D. 舵机层 ──
                    DisclosureGroup("D. 舵机层 Servo", isExpanded: $expandD) {
                        VStack(alignment: .leading, spacing: 2) {
                            DbgRow("指令角度 cmd", String(format: "H=%.2f° V=%.2f°", dbg.cmdH, dbg.cmdV),
                                   tip: "PID输出的目标角度(0-240)")
                            DbgRow("实际角度 act", String(format: "H=%.2f° V=%.2f°", dbg.actH, dbg.actV),
                                   tip: "经物理仿真(BLE延迟+惯性)后的真实角度")
                            DbgRow("跟随误差 servoErr", String(format: "H=%.3f° V=%.3f°", dbg.servoErrH, dbg.servoErrV),
                                   tip: "cmd-act  BLE延迟和舵机惯性导致的滞后")
                            DbgRow("角速度 servoVel", String(format: "H=%.1f°/s V=%.1f°/s", dbg.servoVelH, dbg.servoVelV),
                                   tip: "舵机实际转速")
                            if dbg.state == "INERTIA" || dbg.state == "KF-INERTIA" {
                                DbgRow("惯性速度 inertia", String(format: "H=%.1f V=%.1f", dbg.inertiaVelH, dbg.inertiaVelV),
                                       tip: "丢失后沿最后速度方向惯性运动的速度(度/秒)")
                            }
                            // ── F. 变焦 ──
                            DbgRow("光学变焦 optZoom", String(format: "%.2fx", dbg.opticalZoom),
                                   tip: "仿真光学变焦, 缩小取景范围 (范围见右侧面板)")
                            DbgRow("数字变焦 digZoom", String(format: "%.2fx", dbg.digitalZoom),
                                   tip: "数字云台变焦, 软件二次裁切 (范围见右侧面板)")
                            DbgRow("等效FOV effFovH", String(format: "%.1f°", dbg.effectiveFovH),
                                   tip: "光学变焦后的等效水平视场角")
                        }
                    }
                }
                .font(.caption2.monospaced())
            }
            .frame(maxHeight: 500)
            .padding(4)
        }
    }
}

/// Debug 变量行: 名称 + 值 + Tooltip 解释
struct DbgRow: View {
    let name: String
    let value: String
    let tip: String

    init(_ name: String, _ value: String, tip: String = "") {
        self.name = name
        self.value = value
        self.tip = tip
    }

    var body: some View {
        HStack(alignment: .top, spacing: 4) {
            Text(name)
                .foregroundColor(.secondary)
                .frame(minWidth: 100, alignment: .leading)
            Text(value)
                .foregroundColor(.primary)
            Spacer()
        }
        .help(tip)  // macOS tooltip
    }
}

struct CSVDoc: FileDocument {
    static var readableContentTypes: [UTType] = [.commaSeparatedText]
    let eng: SimEngine

    init(eng: SimEngine) { self.eng = eng }
    init(configuration: ReadConfiguration) throws { fatalError() }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sim.csv")
        eng.exportCSV(url)
        return try FileWrapper(url: url)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 仿真结论面板
// ═══════════════════════════════════════════════════════════════

struct SimResultSheet: View {
    let result: SimEngine.SimResult
    @Binding var isPresented: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // 顶栏
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("仿真结论报告")
                        .font(.title2.bold())
                    Text(String(format: "共 %d 帧 · %.1fs · %@",
                                result.frames, result.durationSec,
                                result.grade))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
                // 综合评分圆圈
                ZStack {
                    Circle()
                        .stroke(scoreColor(result.score).opacity(0.3), lineWidth: 8)
                    Circle()
                        .trim(from: 0, to: CGFloat(result.score) / 100)
                        .stroke(scoreColor(result.score), style: StrokeStyle(lineWidth: 8, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    VStack(spacing: 0) {
                        Text("\(result.score)")
                            .font(.title.bold().monospaced())
                            .foregroundColor(scoreColor(result.score))
                        Text("/ 100")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
                .frame(width: 80, height: 80)
                .animation(.easeOut(duration: 0.6), value: result.score)
            }
            .padding()
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {

                    // ── 追踪性能 ──
                    if result.hasAnnotationGT {
                        ResultSection(title: "★ 标注 GT 评测 (虚拟赛道口径)", icon: "scope") {
                            ResultGaugeRow(label: "一般追踪率 generalTrackRate",
                                           value: result.generalTrackRate / 100,
                                           text: String(format: "%.1f%%", result.generalTrackRate),
                                           good: result.generalTrackRate > 70)
                            ResultGaugeRow(label: "单目标锁定率 singleTargetLockRate",
                                           value: result.singleTargetLockRate / 100,
                                           text: String(format: "%.1f%%", result.singleTargetLockRate),
                                           good: result.singleTargetLockRate > 70)
                            ResultGaugeRow(label: "GT 标注可见率",
                                           value: result.gtAnnotationVisiblePct / 100,
                                           text: String(format: "%.1f%%", result.gtAnnotationVisiblePct),
                                           good: result.gtAnnotationVisiblePct > 80)
                            ResultTextRow(label: "说明", value: "一般追踪率=输出框与GT重叠帧/总帧; 单目标锁定率=ST锁定框与GT重叠帧/GT可见帧")
                        }
                    }
                    // (旧口径已移除)

                    // ── 舵机精度 ──
                    ResultSection(title: "舵机精度", icon: "target") {
                        ResultGaugeRow(label: "H 平均误差",
                                       value: min(result.avgErrH / 10, 1),
                                       text: String(format: "avg %.2f°  max %.2f°", result.avgErrH, result.maxErrH),
                                       good: result.avgErrH < 3, invert: true)
                        ResultGaugeRow(label: "V 平均误差",
                                       value: min(result.avgErrV / 10, 1),
                                       text: String(format: "avg %.2f°  max %.2f°", result.avgErrV, result.maxErrV),
                                       good: result.avgErrV < 3, invert: true)
                        ResultTextRow(label: "评估", value: result.servoAssessment)
                    }

                    // ── 抖动分析 ──
                    ResultSection(title: "抖动分析 (Jitter)", icon: "waveform.path") {
                        ResultGaugeRow(label: "Jitter H",
                                       value: min(result.jitterH / 60, 1),
                                       text: String(format: "%.1f%%", result.jitterH),
                                       good: result.jitterH < 15, invert: true)
                        ResultGaugeRow(label: "Jitter V",
                                       value: min(result.jitterV / 60, 1),
                                       text: String(format: "%.1f%%", result.jitterV),
                                       good: result.jitterV < 15, invert: true)
                        ResultTextRow(label: "峰值速度",
                                      value: String(format: "H %.0f°/s  V %.0f°/s",
                                                    result.peakVelH, result.peakVelV))
                    }

                    // ── 目标跟随精度 ──
                    ResultSection(title: "目标跟随精度", icon: "arrow.triangle.2.circlepath") {
                        ResultGaugeRow(label: "目标误差",
                                       value: min(result.avgTgtDeg / 15, 1),
                                       text: String(format: "avg %.1f°  max %.1f°", result.avgTgtDeg, result.maxTgtDeg),
                                       good: result.avgTgtDeg < 5, invert: true)
                    }

                    // ── 推理速度 ──
                    ResultSection(title: "YOLO 推理速度", icon: "cpu") {
                        ResultGaugeRow(label: "平均时延",
                                       value: min(result.avgYoloMs / 60, 1),
                                       text: String(format: "%.1f ms", result.avgYoloMs),
                                       good: result.avgYoloMs < 30, invert: true)
                        ResultTextRow(label: "BLE 评估", value: result.bleAssessment)
                    }

                    // ── 综合建议 ──
                    ResultSection(title: "综合建议", icon: "lightbulb") {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(suggestions(result), id: \.self) { s in
                                HStack(alignment: .top, spacing: 6) {
                                    Image(systemName: "arrow.right.circle.fill")
                                        .foregroundColor(.accentColor)
                                        .font(.caption)
                                        .padding(.top, 1)
                                    Text(s).font(.caption)
                                }
                            }
                        }
                    }
                }
                .padding()
            }

            Divider()
            HStack {
                Spacer()
                Button("关闭") { isPresented = false }
                    .keyboardShortcut(.defaultAction)
                    .padding()
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .frame(minWidth: 520, minHeight: 600)
    }

    private func scoreColor(_ s: Int) -> Color {
        s >= 85 ? .green : (s >= 70 ? .yellow : (s >= 55 ? .orange : .red))
    }

    private func suggestions(_ r: SimEngine.SimResult) -> [String] {
        var list: [String] = []
        // ★ 标注 GT 诊断
        if r.hasAnnotationGT {
            if r.generalTrackRate < 50 {
                list.append("一般追踪率仅 \(String(format: "%.0f", r.generalTrackRate))%，大量帧输出框未命中 GT。可能原因: YOLO 检测失败、取景框未包含目标、或标注框偏差。")
            } else if r.generalTrackRate < 70 {
                list.append("一般追踪率 \(String(format: "%.0f", r.generalTrackRate))%，还有提升空间。检查目标快速移动时的丢失情况。")
            }
            if r.singleTargetLockRate < 50 && r.singleTargetLockRate > 0 {
                list.append("单目标锁定率仅 \(String(format: "%.0f", r.singleTargetLockRate))%，ST 连续性差。可能是目标外观变化大或多目标干扰。")
            }
            if r.gtAnnotationVisiblePct < 70 {
                list.append("GT 标注可见率仅 \(String(format: "%.0f", r.gtAnnotationVisiblePct))%，部分帧无标注。建议在视频标注模块中补全缺失帧的标注。")
            }
        }
        // "追踪率(含惯性)"诊断已移除，改用一般追踪率判断
        // ★ GT SAHI 诊断
        if r.gtVisiblePct < 60 {
            list.append("GT可见率仅 \(String(format: "%.0f", r.gtVisiblePct))%，目标频繁出画面/太小/被遮挡，追踪率低并非 Tracker 的问题。")
        }
        if r.trkEfficiency < 60 && r.gtVisiblePct > 60 {
            list.append("追踪效率仅 \(String(format: "%.0f", r.trkEfficiency))%（目标可见但 VNTracker 跟丢），建议降低 searchCooldown 或优化外观匹配阈值。")
        }
        if r.pureTrkPct < 40 && r.gtVisiblePct > 70 {
            list.append("VNTracker 纯追踪率极低（\(String(format: "%.0f", r.pureTrkPct))%），Tracker 续航能力差，过度依赖 YOLO 重建。")
        }
        if r.avgErrH > 5 || r.avgErrV > 5 {
            list.append("舵机误差较大，可适当增大 KpH / KpV，或减小 BLE maxMs 来降低延迟。")
        }
        if r.jitterH > 25 || r.jitterV > 25 {
            list.append("Jitter 偏高，建议增大 minStep 或降低 KpH/KpV，以减少反向振荡。")
        }
        if r.avgYoloMs > 35 {
            list.append("YOLO 推理较慢（\(String(format: "%.0f", r.avgYoloMs))ms），可尝试更小的模型输入尺寸（320px）。")
        }
        if r.peakVelH > 400 || r.peakVelV > 400 {
            list.append("峰值速度过高，若实体舵机限速，建议降低 maxDPS 参数以匹配硬件上限。")
        }
        if list.isEmpty {
            list.append("各项指标均在优秀范围，当前参数配置适合部署到实机。")
        }
        return list
    }
}

struct ResultSection<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        GroupBox {
            content()
                .padding(.top, 2)
        } label: {
            Label(title, systemImage: icon)
                .font(.subheadline.bold())
        }
    }
}

struct ResultGaugeRow: View {
    let label: String
    let value: Double        // 0-1
    let text: String
    let good: Bool
    var invert: Bool = false // true → 低值好

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
                        .frame(width: g.size.width * CGFloat(value))
                }
            }
            .frame(height: 6)
        }
    }
}

struct ResultTextRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .top) {
            Text(label).font(.caption).foregroundColor(.secondary).frame(width: 50, alignment: .leading)
            Text(value).font(.caption).foregroundColor(.primary)
            Spacer()
        }
    }
}
