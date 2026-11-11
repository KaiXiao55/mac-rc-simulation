//
//  BatchSimRunner.swift
//  macRCtrackerSimulator
//
//  自动化批量仿真运行器 — 检测启动参数 --batch-sim 后自动运行多轮仿真
//
//  Created by Manus AI on 2026/3/19.
//

import SwiftUI
import Combine
import Foundation

// ═══════════════════════════════════════════════════════════════
// MARK: - 批量仿真配置（默认 20 次：快速诊断）
// ═══════════════════════════════════════════════════════════════

struct BatchSimRunSpec: Equatable {
    let runIndex: Int
    let seed: UInt64
    /// 0...9 → Track 1...10（TrackShape.rawValue）
    let trackShape: Int
    /// 虚拟场景类型（与 VirtualRaceEngine.sceneType 一致；0–5 为带赛道场景）
    let sceneType: Int
    /// VSimCfg.cameraPresets 索引 0...8
    let cameraPreset: Int
    let carCount: Int
    let cameraHeight: CGFloat
    let enableRamps: Bool
    let recallRate: Float  // ★ 模拟YOLO召回率 (默认0.70)
    let fpRate: Float      // ★ 模拟YOLO误检率 (默认0.0)
    let forceSimulatedYOLO: Bool? // nil=按模式自动, true/false=强制

    init(runIndex: Int, seed: UInt64, trackShape: Int, sceneType: Int,
         cameraPreset: Int, carCount: Int, cameraHeight: CGFloat,
         enableRamps: Bool, recallRate: Float, fpRate: Float = 0.0,
         forceSimulatedYOLO: Bool? = nil) {
        self.runIndex = runIndex
        self.seed = seed
        self.trackShape = trackShape
        self.sceneType = sceneType
        self.cameraPreset = cameraPreset
        self.carCount = carCount
        self.cameraHeight = cameraHeight
        self.enableRamps = enableRamps
        self.recallRate = recallRate
        self.fpRate = fpRate
        self.forceSimulatedYOLO = forceSimulatedYOLO
    }
}

struct BatchSimConfig {
    let runs: [BatchSimRunSpec]
    let simDuration: Double
    let outputDir: String

    static var `default`: BatchSimConfig {
        BatchSimConfig(
            runs: BatchSimConfig.deterministic20Runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_20run_v2"
        )
    }

    /// 10 轮快速验证：仅单目标锁定 + 帧降质
    static var stDegrade10: BatchSimConfig {
        BatchSimConfig(
            runs: Array(BatchSimConfig.deterministic20Runs.prefix(10)),
            simDuration: 10.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_st10_degrade"
        )
    }

    /// 50 轮新机位组合测试：优先选择历史 batch_summary.csv 中未出现过的 (track,scene,preset) 组合
    static var newCam50: BatchSimConfig {
        let seen = historicalTestedComboKeys()
        var selected = deterministic100Runs.filter { !seen.contains(comboKey($0)) }
        if selected.count > 50 { selected = Array(selected.prefix(50)) }

        if selected.count < 50 {
            let heights: [CGFloat] = [1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
            var used = Set(selected.map(comboKey))
            var seed: UInt64 = 91001
            for t in 0..<10 {
                for s in 0..<8 {
                    for p in 0..<9 {
                        let key = "\(t)-\(s)-\(p)"
                        if seen.contains(key) || used.contains(key) { continue }
                        let h = heights[(t + s + p) % heights.count]
                        let ramps = (s == 3 || s == 4) ? ((t + p) % 2 == 0) : false
                        selected.append(BatchSimRunSpec(
                            runIndex: selected.count + 1,
                            seed: seed,
                            trackShape: t,
                            sceneType: s,
                            cameraPreset: p,
                            carCount: 5 + ((t + s + p) % 4),
                            cameraHeight: h,
                            enableRamps: ramps,
                            recallRate: 0.70
                        ))
                        used.insert(key)
                        seed += 1
                        if selected.count >= 50 { break }
                    }
                    if selected.count >= 50 { break }
                }
                if selected.count >= 50 { break }
            }
        }

        let runs = Array(selected.prefix(50)).enumerated().map { i, r in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: r.seed,
                trackShape: r.trackShape,
                sceneType: r.sceneType,
                cameraPreset: r.cameraPreset,
                carCount: r.carCount,
                cameraHeight: r.cameraHeight,
                enableRamps: r.enableRamps,
                recallRate: r.recallRate
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_newcam50"
        )
    }

    /// 从未出现在历史 `batch_summary.csv` 中的 (赛道,场景,机位) 组合里选取最多 100 组，跑 Pass1+Pass2（Pass2=单目标锁定追踪），结果写入 `batch_summary.csv` 与界面动态表
    static var stUnseen100: BatchSimConfig {
        let seen = historicalTestedComboKeys()
        let heights: [CGFloat] = [1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
        var selected: [BatchSimRunSpec] = []
        var used = Set<String>()
        var seed: UInt64 = 73_001
        outer: for t in 0..<10 {
            for s in 0..<8 {
                for p in 0..<9 {
                    let key = "\(t)-\(s)-\(p)"
                    if seen.contains(key) || used.contains(key) { continue }
                    let i = selected.count
                    let carCount = 5 + (i % 4)
                    let cameraHeight = heights[i % heights.count]
                    let enableRamps = (s == 3 || s == 4) ? (i % 2 == 0) : false
                    selected.append(BatchSimRunSpec(
                        runIndex: i + 1,
                        seed: seed,
                        trackShape: t,
                        sceneType: s,
                        cameraPreset: p,
                        carCount: carCount,
                        cameraHeight: cameraHeight,
                        enableRamps: enableRamps,
                recallRate: 0.70
                    ))
                    used.insert(key)
                    seed += 1
                    if selected.count >= 100 { break outer }
                }
            }
        }
        let runs = selected.enumerated().map { i, r in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: r.seed,
                trackShape: r.trackShape,
                sceneType: r.sceneType,
                cameraPreset: r.cameraPreset,
                carCount: r.carCount,
                cameraHeight: r.cameraHeight,
                enableRamps: r.enableRamps,
                recallRate: r.recallRate
            )
        }
        if runs.count < 100 {
            print("★ stUnseen100: 历史 batch_summary 已覆盖较多组合，本次仅能选取 \(runs.count) 个未出现过的 (赛道,场景,机位)（目标 100）")
        } else {
            print("★ stUnseen100: 已选取 100 个未在历史 batch_summary 中出现过的 (赛道,场景,机位) 组合")
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_st_unseen100"
        )
    }

    /// 5 轮快速测试：随机赛道/场景/机位
    static var quick5: BatchSimConfig {
        let runs: [BatchSimRunSpec] = (0..<5).map { i in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: UInt64(9001 + i),
                trackShape: Int.random(in: 0...9),
                sceneType: Int.random(in: 0...5),
                cameraPreset: Int.random(in: 0...8),
                carCount: Int.random(in: 5...8),
                cameraHeight: [1.5, 2.0, 2.5, 3.0, 3.5].randomElement()!,
                enableRamps: Bool.random(),
                recallRate: 0.70
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSTemporaryDirectory() + "batch_sim_quick5"
        )
    }

    /// 8 轮快速测试：随机赛道/场景/机位
    static var quick8: BatchSimConfig {
        let runs: [BatchSimRunSpec] = (0..<8).map { i in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: UInt64(8001 + i),
                trackShape: Int.random(in: 0...9),
                sceneType: Int.random(in: 0...5),
                cameraPreset: Int.random(in: 0...8),
                carCount: Int.random(in: 5...8),
                cameraHeight: [1.5, 2.0, 2.5, 3.0, 3.5].randomElement()!,
                enableRamps: Bool.random(),
                recallRate: 0.70
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSTemporaryDirectory() + "batch_sim_quick8"
        )
    }

    /// V6.5 验证：跑 deterministic20Runs 的前 5 个场景
    static var single1: BatchSimConfig {
        BatchSimConfig(
            runs: Array(BatchSimConfig.deterministic20Runs.prefix(5)),
            simDuration: 30.0,
            outputDir: NSTemporaryDirectory() + "batch_sim_single1"
        )
    }
    /// 单次：Track1 / 室内 / 机位1 / seed42，用 Track1 真 vsim_yolo11n 的 object-recall / FP-frame-rate 对齐 SimYOLO。
    static var stVerify095010: BatchSimConfig {
        let spec = BatchSimRunSpec(
            runIndex: 6,
            seed: 42,
            trackShape: 0,
            sceneType: 0,
            cameraPreset: 1,
            carCount: 5,
            cameraHeight: 2.5,
            enableRamps: false,
            recallRate: 0.8236,
            fpRate: 0.7803,
            forceSimulatedYOLO: true
        )
        return BatchSimConfig(
            runs: [spec],
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_track1_simyolo_vsim11n_aligned"
        )
    }

    /// 同心圆贴标 + 传统 CV 检测：5 个代表场景 × 30s，验证标记检测能否支撑连续单目标锁定。
    static var markerCVMultiscene30: BatchSimConfig {
        let rows: [BatchSimRunSpec] = [
            BatchSimRunSpec(runIndex: 1, seed: 42, trackShape: 0, sceneType: 0, cameraPreset: 1, carCount: 5, cameraHeight: 2.5, enableRamps: false, recallRate: 0.0, forceSimulatedYOLO: false),
            BatchSimRunSpec(runIndex: 2, seed: 6002, trackShape: 1, sceneType: 1, cameraPreset: 2, carCount: 5, cameraHeight: 2.5, enableRamps: false, recallRate: 0.0, forceSimulatedYOLO: false),
            BatchSimRunSpec(runIndex: 3, seed: 6003, trackShape: 2, sceneType: 2, cameraPreset: 4, carCount: 6, cameraHeight: 3.5, enableRamps: true, recallRate: 0.0, forceSimulatedYOLO: false),
            BatchSimRunSpec(runIndex: 4, seed: 6004, trackShape: 3, sceneType: 3, cameraPreset: 3, carCount: 5, cameraHeight: 2.5, enableRamps: false, recallRate: 0.0, forceSimulatedYOLO: false),
            BatchSimRunSpec(runIndex: 5, seed: 6006, trackShape: 5, sceneType: 5, cameraPreset: 5, carCount: 5, cameraHeight: 3.5, enableRamps: false, recallRate: 0.0, forceSimulatedYOLO: false),
        ]
        return BatchSimConfig(
            runs: rows,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_marker_cv_multiscene_30s"
        )
    }

    /// AprilTag/ArUco 类强标志物上界：deterministic20Runs 全覆盖，30s 单目标锁定。
    static var fiducialCVMultiscene20: BatchSimConfig {
        let runs = BatchSimConfig.deterministic20Runs.enumerated().map { i, base in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: base.seed,
                trackShape: base.trackShape,
                sceneType: base.sceneType,
                cameraPreset: base.cameraPreset,
                carCount: base.carCount,
                cameraHeight: base.cameraHeight,
                enableRamps: base.enableRamps,
                recallRate: 0.0,
                fpRate: 0.0,
                forceSimulatedYOLO: false
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_fiducial_cv_multiscene20_30s"
        )
    }

    /// 带唯一 ID 的强标志物：每辆车可解码 carId，目标选择按 ID 固定，验证无 YOLO 主链路上界。
    static var idFiducialCVMultiscene20: BatchSimConfig {
        let runs = BatchSimConfig.deterministic20Runs.enumerated().map { i, base in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: base.seed,
                trackShape: base.trackShape,
                sceneType: base.sceneType,
                cameraPreset: base.cameraPreset,
                carCount: base.carCount,
                cameraHeight: base.cameraHeight,
                enableRamps: base.enableRamps,
                recallRate: 0.0,
                fpRate: 0.0,
                forceSimulatedYOLO: false
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_id_fiducial_cv_multiscene20_30s"
        )
    }

    /// 视频样例：同心圆 MarkerCV 3 场景，配合 --record-scenekit 生成 mp4。
    static var markerCVVideoSamples: BatchSimConfig {
        let runs = Array(BatchSimConfig.markerCVMultiscene30.runs.prefix(3))
        return BatchSimConfig(
            runs: runs,
            simDuration: 20.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/video_marker_cv_samples_20s"
        )
    }

    /// 视频样例：强 fiducial 标志物 3 场景，配合 --record-scenekit 生成 mp4。
    static var fiducialCVVideoSamples: BatchSimConfig {
        let runs = Array(BatchSimConfig.fiducialCVMultiscene20.runs.prefix(3))
        return BatchSimConfig(
            runs: runs,
            simDuration: 20.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/video_fiducial_cv_samples_20s"
        )
    }

    /// 视频样例：带 ID 的 fiducial 标志物 3 场景，配合 --record-scenekit 生成 mp4。
    static var idFiducialCVVideoSamples: BatchSimConfig {
        let runs = Array(BatchSimConfig.idFiducialCVMultiscene20.runs.prefix(3))
        return BatchSimConfig(
            runs: runs,
            simDuration: 20.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/video_id_fiducial_cv_samples_20s"
        )
    }

    /// 快速冒烟：单场景 5s，验证像素级 ID-CV 解码和动态 ID 锁定。
    static var idFiducialCVSmoke5: BatchSimConfig {
        let run = BatchSimConfig.idFiducialCVMultiscene20.runs[0]
        return BatchSimConfig(
            runs: [run],
            simDuration: 5.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/smoke_id_fiducial_cv_5s"
        )
    }

    /// Track1 / 室内 / 机位1 / seed42：30s SimYOLO 阈值扫描，用于和 MarkerCV 基线对齐。
    static var markerYoloThreshold30: BatchSimConfig {
        let combos: [(Float, Float)] = [
            (0.80, 0.00),
            (0.85, 0.00),
            (0.90, 0.00),
            (0.93, 0.00),
            (0.95, 0.00),
            (0.90, 0.30),
            (0.93, 0.30),
            (0.95, 0.30),
            (0.95, 0.78),
        ]
        let runs = combos.enumerated().map { i, c in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: 42,
                trackShape: 0,
                sceneType: 0,
                cameraPreset: 1,
                carCount: 5,
                cameraHeight: 2.5,
                enableRamps: false,
                recallRate: c.0,
                fpRate: c.1,
                forceSimulatedYOLO: true
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_marker_yolo_threshold_30s"
        )
    }

    /// 10-run 多场景真实 YOLO 基线：deterministic20Runs 前 10 个，覆盖 10 条赛道、6 类场景、多个机位和车数。
    static var multiscene10RealYoloST60: BatchSimConfig {
        let runs = Array(BatchSimConfig.deterministic20Runs.prefix(10)).map { base in
            BatchSimRunSpec(
                runIndex: base.runIndex,
                seed: base.seed,
                trackShape: base.trackShape,
                sceneType: base.sceneType,
                cameraPreset: base.cameraPreset,
                carCount: base.carCount,
                cameraHeight: base.cameraHeight,
                enableRamps: base.enableRamps,
                recallRate: 0.8236,
                fpRate: 0.0,
                forceSimulatedYOLO: false
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_multiscene10_real_yolo_st60"
        )
    }

    /// 10-run 多场景默认 SimYOLO：与真实 YOLO 基线同一批场景，使用当前 SimYOLO 默认 recall/fp 参数。
    static var multiscene10SimYoloAlignedST60: BatchSimConfig {
        let runs = Array(BatchSimConfig.deterministic20Runs.prefix(10)).map { base in
            BatchSimRunSpec(
                runIndex: base.runIndex,
                seed: base.seed,
                trackShape: base.trackShape,
                sceneType: base.sceneType,
                cameraPreset: base.cameraPreset,
                carCount: base.carCount,
                cameraHeight: base.cameraHeight,
                enableRamps: base.enableRamps,
                recallRate: VSimCfg.simulatedRecallRate,
                fpRate: VSimCfg.simulatedFPRate,
                forceSimulatedYOLO: true
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_multiscene10_simyolo_aligned_st60"
        )
    }

    /// 20-run 合并对照：先 10 个真实 YOLO，再同场景 10 个对齐 SimYOLO，便于一次性跑完整 pair。
    static var multiscene10YoloSimYoloPairST60: BatchSimConfig {
        let real = Array(BatchSimConfig.deterministic20Runs.prefix(10)).map { base in
            BatchSimRunSpec(
                runIndex: base.runIndex,
                seed: base.seed,
                trackShape: base.trackShape,
                sceneType: base.sceneType,
                cameraPreset: base.cameraPreset,
                carCount: base.carCount,
                cameraHeight: base.cameraHeight,
                enableRamps: base.enableRamps,
                recallRate: 0.8236,
                fpRate: 0.0,
                forceSimulatedYOLO: false
            )
        }
        let sim = Array(BatchSimConfig.deterministic20Runs.prefix(10)).map { base in
            BatchSimRunSpec(
                runIndex: base.runIndex + 100,
                seed: base.seed,
                trackShape: base.trackShape,
                sceneType: base.sceneType,
                cameraPreset: base.cameraPreset,
                carCount: base.carCount,
                cameraHeight: base.cameraHeight,
                enableRamps: base.enableRamps,
                recallRate: VSimCfg.simulatedRecallRate,
                fpRate: VSimCfg.simulatedFPRate,
                forceSimulatedYOLO: true
            )
        }
        return BatchSimConfig(
            runs: real + sim,
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_multiscene10_yolo_simyolo_pair_st60"
        )
    }

    /// Track1 / 室内 / 机位1 / seed42：高召回低误检门槛扫描，用于评估 30s 连续单车锁定所需 SimYOLO 水平。
    static var track1ThirtySecondLockThreshold: BatchSimConfig {
        let combos: [(Float, Float)] = [
            // Boundary probes after the coarse 20s sweep.
            (0.775, 0.00),
            (0.775, 0.7803),
        ]
        let runs = combos.enumerated().map { i, c in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: 42,
                trackShape: 0,
                sceneType: 0,
                cameraPreset: 1,
                carCount: 5,
                cameraHeight: 2.5,
                enableRamps: false,
                recallRate: c.0,
                fpRate: c.1,
                forceSimulatedYOLO: true
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_track1_simyolo_20s_boundary"
        )
    }

    /// ★ 现实参数扫描 (8 runs × 60s): Track 1 / 室内 / 赛道边中位 (cameraPreset=1, H=2.5m, -25°)
    /// 用户截图的赛道+机位组合。recall 范围更贴近真 YOLO (0.95~0.80)
    /// 注意：`batch_summary.csv` 中的 singleTargetLockRate 仅为**当时二进制**的快照；ST/SimYOLO 逻辑迭代后须重新 `--batch-recall-fp-test`，勿与桌面旧 CSV 对照当「当前真值」。
    static var recallFpGrid: BatchSimConfig {
        // 用户截图配置: trackShape=0, sceneType=0, cameraPreset=1 (赛道边中位 H=2.5m, -25°), 5 cars, seed=42
        // 8 个 (recall, fp) 组合
        let combos: [(Float, Float)] = [
            (0.95, 0.00),  // 1: 接近真 YOLO 上限,无 FP
            (0.90, 0.00),  // 2: 90% recall 纯净
            (0.85, 0.00),  // 3: 85% recall 纯净
            (0.80, 0.00),  // 4: 80% recall 纯净
            (0.95, 0.05),  // 5: 高 recall + 5% FP
            (0.95, 0.10),  // 6: 高 recall + 10% FP（截图档；具体 ST% 以重跑批次为准）
            (0.85, 0.10),  // 7: 中 recall + FP
            (0.80, 0.10),  // 8: 双因素退化最严重
        ]
        let runs: [BatchSimRunSpec] = combos.enumerated().map { i, c in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: 42,
                trackShape: 0,                 // Track 1 (24x20m)
                sceneType: 0,                  // 室内场地
                cameraPreset: 1,               // 包络外-中位 (赛道边中位 H=2.5m, -25°)
                carCount: 5,
                cameraHeight: 2.5,
                enableRamps: false,
                recallRate: c.0,
                fpRate: c.1
            )
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 60.0,
            // 与历史目录 `batch_sim_track1_realistic_60s_round2` 分离，避免沿用旧版算法写下的 batch_summary 造成误解
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_track1_recall_fp_grid"
        )
    }

    /// 40 轮 ST-only 60s 参数扫描：5 个赛道/场景/机位组合 × 8 个 (recall, fp) 档。
    ///
    /// 只跑 Pass2-ST：单目标锁定率由 Pass2 独立初始化并评估，跳过 Pass1 不改变 ST 结果，
    /// 但能把批量耗时减半。用于回答「什么召回率/误检率能维持 60s 单目标锁定」。
    static var recallFpEnv40ST60: BatchSimConfig {
        let combos: [(Float, Float)] = [
            (0.95, 0.00),
            (0.90, 0.00),
            (0.85, 0.00),
            (0.80, 0.00),
            (0.95, 0.05),
            (0.95, 0.10),
            (0.85, 0.10),
            (0.80, 0.10),
        ]
        let envs: [(track: Int, scene: Int, preset: Int, height: CGFloat, ramps: Bool, seedBase: UInt64)] = [
            (0, 0, 1, 2.5, false, 42),      // Track1 / 室内 / 包络外-中位
            (1, 2, 2, 5.0, false, 10_142),  // Track2 / 室外柏油 / 高位看台
            (2, 5, 3, 3.5, false, 20_142),  // Track3 / 专业赛道 / 中位看台
            (3, 1, 5, 1.8, false, 30_142),  // Track4 / 室外草地 / 直道侧
            (4, 3, 6, 1.3, true, 40_142),   // Track5 / 娱乐飞坡 / 手持低
        ]
        var runs: [BatchSimRunSpec] = []
        for (envIndex, env) in envs.enumerated() {
            for combo in combos {
                runs.append(BatchSimRunSpec(
                    runIndex: runs.count + 1,
                    seed: env.seedBase,
                    trackShape: env.track,
                    sceneType: env.scene,
                    cameraPreset: env.preset,
                    carCount: 5,
                    cameraHeight: env.height,
                    enableRamps: env.ramps,
                    recallRate: combo.0,
                    fpRate: combo.1,
                    forceSimulatedYOLO: true
                ))
            }
            print("★ env40 group \(envIndex + 1): track=\(env.track) scene=\(env.scene) preset=\(env.preset)")
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_recall_fp_env40_st60s_constseed"
        )
    }

    /// 10 轮 ST-only 60s 参数扫描：用于快速完整验证，取 env40 设计的前 10 个 run。
    static var recallFpEnv10ST60: BatchSimConfig {
        let base = BatchSimConfig.recallFpEnv40ST60
        return BatchSimConfig(
            runs: Array(base.runs.prefix(10)),
            simDuration: base.simDuration,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_recall_fp_env10_st60s_constseed"
        )
    }

    private static func tracker7wRunSpecs(forceSimulatedYolo: Bool) -> [BatchSimRunSpec] {
        let envs: [(track: Int, scene: Int, preset: Int, height: CGFloat, ramps: Bool, seed: UInt64)] = [
            (0, 0, 1, 2.5, false, 70_001),
            (1, 2, 2, 5.0, false, 70_002),
            (2, 5, 3, 3.5, false, 70_003),
            (3, 1, 5, 1.8, false, 70_004),
            (4, 3, 6, 1.3, true, 70_005),
        ]
        return envs.enumerated().map { i, env in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: env.seed,
                trackShape: env.track,
                sceneType: env.scene,
                cameraPreset: env.preset,
                carCount: 5,
                cameraHeight: env.height,
                enableRamps: env.ramps,
                recallRate: 0.5464,
                fpRate: 0.2764,
                forceSimulatedYOLO: forceSimulatedYolo
            )
        }
    }

    /// 7W YOLO 水平追踪实验：固定截图近似指标，跨 5 个环境跑 60s Pass2-ST。
    ///
    /// 使用环境变量 `ST_EXPERIMENT_PROFILE` 切换追踪策略：
    /// baseline / lostfix / lowconf / veto / conservative / strict
    static var tracker7wST60: BatchSimConfig {
        let env = ProcessInfo.processInfo.environment
        let requestedRuns = Int(env["TRACKER7W_RUNS"] ?? "") ?? 5
        let requestedDuration = Double(env["TRACKER7W_DURATION"] ?? "") ?? 60.0
        let runs = tracker7wRunSpecs(forceSimulatedYolo: true)
        let profile = ProcessInfo.processInfo.environment["ST_EXPERIMENT_PROFILE"] ?? "baseline"
        return BatchSimConfig(
            runs: Array(runs.prefix(max(1, min(requestedRuns, runs.count)))),
            simDuration: requestedDuration,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_tracker7w_st60_\(profile)"
        )
    }

    /// 真实 7W YOLO 追踪实验：与 tracker7wST60 使用相同场景/seed，但走真实 CoreML 推理。
    static var tracker7wRealYoloST60: BatchSimConfig {
        let env = ProcessInfo.processInfo.environment
        let requestedRuns = Int(env["TRACKER7W_RUNS"] ?? "") ?? 5
        let requestedDuration = Double(env["TRACKER7W_DURATION"] ?? "") ?? 60.0
        let runs = tracker7wRunSpecs(forceSimulatedYolo: false)
        let profile = ProcessInfo.processInfo.environment["ST_EXPERIMENT_PROFILE"] ?? "baseline"
        return BatchSimConfig(
            runs: Array(runs.prefix(max(1, min(requestedRuns, runs.count)))),
            simDuration: requestedDuration,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_tracker7w_real_st60_\(profile)"
        )
    }

    /// 单次复核：run_009 的环境，使用正常 vsim_yolo11n 检测而非模拟 YOLO。
    static var run009RealYoloST60: BatchSimConfig {
        let spec = BatchSimRunSpec(
            runIndex: 9,
            seed: 10_142,
            trackShape: 1,
            sceneType: 2,
            cameraPreset: 2,
            carCount: 5,
            cameraHeight: 5.0,
            enableRamps: false,
            recallRate: 0.0,
            fpRate: 0.0,
            forceSimulatedYOLO: false
        )
        return BatchSimConfig(
            runs: [spec],
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_run009_real_yolo_st60"
        )
    }

    /// 单次复核：Track1 / 室内 / 机位1 / seed42，使用正常 vsim_yolo11n 检测而非模拟 YOLO。
    static var track1RealYoloST60: BatchSimConfig {
        let spec = BatchSimRunSpec(
            runIndex: 1,
            seed: 42,
            trackShape: 0,
            sceneType: 0,
            cameraPreset: 1,
            carCount: 5,
            cameraHeight: 2.5,
            enableRamps: false,
            recallRate: 0.0,
            fpRate: 0.0,
            forceSimulatedYOLO: false
        )
        return BatchSimConfig(
            runs: [spec],
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_track1_real_yolo_st60"
        )
    }

    /// run_009 校准对照：同一环境先跑真实 vsim_yolo11n，再跑校准版 SimYOLO。
    static var run009YoloCalibPairST60: BatchSimConfig {
        let real = BatchSimRunSpec(
            runIndex: 9,
            seed: 10_142,
            trackShape: 1,
            sceneType: 2,
            cameraPreset: 2,
            carCount: 5,
            cameraHeight: 5.0,
            enableRamps: false,
            recallRate: 0.0,
            fpRate: 0.0,
            forceSimulatedYOLO: false
        )
        let sim = BatchSimRunSpec(
            runIndex: 109,
            seed: 10_142,
            trackShape: 1,
            sceneType: 2,
            cameraPreset: 2,
            carCount: 5,
            cameraHeight: 5.0,
            enableRamps: false,
            recallRate: 0.5126,
            fpRate: 0.2989,
            forceSimulatedYOLO: true
        )
        return BatchSimConfig(
            runs: [real, sim],
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_run009_yolo_calib_pair_st60"
        )
    }

    /// run_009 SimYOLO 快速校准：只跑模拟检测，用于迭代拟合真实 vsim_yolo11n 指标。
    static var run009SimYoloCalibST60: BatchSimConfig {
        let spec = BatchSimRunSpec(
            runIndex: 109,
            seed: 10_142,
            trackShape: 1,
            sceneType: 2,
            cameraPreset: 2,
            carCount: 5,
            cameraHeight: 5.0,
            enableRamps: false,
            recallRate: 0.5126,
            fpRate: 0.2989,
            forceSimulatedYOLO: true
        )
        return BatchSimConfig(
            runs: [spec],
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_run009_simyolo_calib_st60"
        )
    }

    /// SimYOLO 校准双跑：同场景导出 真YOLO + SimYOLO 两份 CSV（15s）
    static var simYoloCalibPair: BatchSimConfig {
        let runs: [BatchSimRunSpec] = [
            BatchSimRunSpec(
                runIndex: 1,
                seed: 42,
                trackShape: 0,
                sceneType: 0,
                cameraPreset: 1,
                carCount: 5,
                cameraHeight: 2.5,
                enableRamps: false,
                recallRate: 0.95,
                fpRate: 0.0,
                forceSimulatedYOLO: false
            ),
            BatchSimRunSpec(
                runIndex: 2,
                seed: 42,
                trackShape: 0,
                sceneType: 0,
                cameraPreset: 1,
                carCount: 5,
                cameraHeight: 2.5,
                enableRamps: false,
                recallRate: 0.95,
                fpRate: 0.0,
                forceSimulatedYOLO: true
            )
        ]
        return BatchSimConfig(
            runs: runs,
            simDuration: 15.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/simyolo_calib_pair"
        )
    }

    /// ★ 召回率敏感性测试：5个赛道 × 5个召回率 = 25次仿真
    static var recallTest: BatchSimConfig {
        let recallRates: [Float] = [0.70, 0.50, 0.40, 0.30, 0.20]
        let baseRuns = Array(BatchSimConfig.deterministic20Runs.prefix(5))
        var runs: [BatchSimRunSpec] = []
        var idx = 1
        for recall in recallRates {
            for base in baseRuns {
                runs.append(BatchSimRunSpec(
                    runIndex: idx,
                    seed: base.seed,
                    trackShape: base.trackShape,
                    sceneType: base.sceneType,
                    cameraPreset: base.cameraPreset,
                    carCount: base.carCount,
                    cameraHeight: base.cameraHeight,
                    enableRamps: base.enableRamps,
                    recallRate: recall
                ))
                idx += 1
            }
        }
        return BatchSimConfig(
            runs: runs,
            simDuration: 30.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/batch_sim_recall_test"
        )
    }

    /// DAgger 训练配置：100 轮 × 60 秒
    static var dagger: BatchSimConfig {
        BatchSimConfig(
            runs: BatchSimConfig.deterministic20Runs,
            simDuration: 60.0,
            outputDir: NSHomeDirectory() + "/Desktop/macRCSimulator/dagger_training_results"
        )
    }

    /// 20 次固定组合：每条赛道至少跑 2 次（不同场景/机位/车数/高度），便于对比有效追踪率
    static let deterministic20Runs: [BatchSimRunSpec] = {
        let rows: [(UInt64, Int, Int, Int, Int, CGFloat, Bool)] = [
            (6001, 0, 0, 0, 5, 1.5, false),
            (6002, 1, 1, 2, 5, 2.5, false),
            (6003, 2, 2, 4, 6, 3.5, true),
            (6004, 3, 3, 3, 5, 2.5, false),
            (6005, 4, 4, 4, 7, 2.0, false),
            (6006, 5, 5, 5, 5, 3.5, false),
            (6007, 6, 0, 6, 6, 1.3, true),
            (6008, 7, 1, 7, 5, 1.6, false),
            (6009, 8, 2, 8, 7, 1.8, false),
            (6010, 9, 3, 8, 8, 4.0, true),
            (6011, 0, 4, 1, 6, 2.5, false),
            (6012, 1, 5, 2, 5, 5.0, false),
            (6013, 2, 0, 3, 5, 1.5, false),
            (6014, 3, 1, 4, 5, 3.5, true),
            (6015, 4, 2, 5, 7, 1.3, false),
            (6016, 5, 3, 6, 5, 5.0, false),
            (6017, 6, 4, 7, 6, 1.6, true),
            (6018, 7, 5, 8, 5, 4.0, false),
            (6019, 8, 0, 4, 6, 2.0, false),
            (6020, 9, 2, 0, 8, 1.5, true),
        ]
        return rows.enumerated().map { i, t in
            BatchSimRunSpec(
                runIndex: i + 1,
                seed: t.0,
                trackShape: t.1,
                sceneType: t.2,
                cameraPreset: t.3,
                carCount: t.4,
                cameraHeight: t.5,
                enableRamps: t.6,
                recallRate: 0.70
            )
        }
    }()

    /// 100 次固定组合：循环覆盖 10 条赛道、8 类场景、9 个机位
    static let deterministic100Runs: [BatchSimRunSpec] = {
        let heights: [CGFloat] = [1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
        return (0..<100).map { i in
            let trackShape = i % 10                  // 0...9
            let sceneType = i % 8                    // 0...7
            let cameraPreset = i % 9                 // 0...8
            let carCount = 5 + (i % 4)              // 5...8
            let cameraHeight = heights[i % heights.count]
            let enableRamps = (sceneType == 3 || sceneType == 4) ? (i % 2 == 0) : false
            return BatchSimRunSpec(
                runIndex: i + 1,
                seed: UInt64(7001 + i),
                trackShape: trackShape,
                sceneType: sceneType,
                cameraPreset: cameraPreset,
                carCount: carCount,
                cameraHeight: cameraHeight,
                enableRamps: enableRamps,
                recallRate: 0.70
            )
        }
    }()

    private static func comboKey(_ spec: BatchSimRunSpec) -> String {
        "\(spec.trackShape)-\(spec.sceneType)-\(spec.cameraPreset)"
    }

    private static func historicalTestedComboKeys() -> Set<String> {
        let root = URL(fileURLWithPath: NSHomeDirectory() + "/Desktop/macRCSimulator")
        var keys = Set<String>()
        let fm = FileManager.default
        /// 仅扫描根目录与各一级子目录下的 batch_summary.csv，避免对整个 macRCSimulator 树做递归枚举（目录极大时会卡死 main 线程）。
        var csvURLs: [URL] = []
        let rootCSV = root.appendingPathComponent("batch_summary.csv")
        if fm.fileExists(atPath: rootCSV.path) { csvURLs.append(rootCSV) }
        if let subs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) {
            for u in subs {
                guard (try? u.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { continue }
                let c = u.appendingPathComponent("batch_summary.csv")
                if fm.fileExists(atPath: c.path) { csvURLs.append(c) }
            }
        }
        for url in csvURLs {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            let lines = text.split(whereSeparator: \.isNewline).map(String.init)
            guard let header = lines.first else { continue }
            let cols = header.split(separator: ",").map(String.init)
            guard let ti = cols.firstIndex(of: "trackShape"),
                  let si = cols.firstIndex(of: "sceneType"),
                  let pi = cols.firstIndex(of: "cameraPreset")
            else { continue }
            for line in lines.dropFirst() {
                if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
                let f = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                if ti < f.count, si < f.count, pi < f.count {
                    keys.insert("\(f[ti])-\(f[si])-\(f[pi])")
                }
            }
        }
        return keys
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 批量仿真运行器
// ═══════════════════════════════════════════════════════════════

final class BatchSimRunner: ObservableObject {
    @Published var currentRun: Int = 0
    @Published var totalRuns: Int = 0
    @Published var isRunning: Bool = false
    @Published var statusText: String = "Ready"
    @Published var allResults: [[String: Any]] = []
    @Published var showReport: Bool = false
    @Published var currentUsesSimulatedYolo: Bool?
    @Published var currentUsesMarkerCVDetector: Bool = false
    @Published var currentRecallRate: Float = 0
    @Published var currentFpRate: Float = 0

    private var engine: VirtualRaceEngine?
    private var cancellable: AnyCancellable?
    private var runSpecs: [BatchSimRunSpec] = []
    private var config: BatchSimConfig = .default
    private let dynamicResultsKey = "batch_dynamic_results_v1"

    static let shared = BatchSimRunner()

    private init() {
        loadDynamicResults()
    }

    static var isBatchMode: Bool {
        CommandLine.arguments.contains("--batch-sim")
            || CommandLine.arguments.contains("--batch-quick5")
            || CommandLine.arguments.contains("--batch-quick8")
            || CommandLine.arguments.contains("--batch-single1")
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
            || CommandLine.arguments.contains("--batch-verify-st-095-010")
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
    }

    static var isStUnseen100Mode: Bool {
        CommandLine.arguments.contains("--batch-st-unseen100")
    }

    static var isSt10Mode: Bool {
        CommandLine.arguments.contains("--batch-st10")
    }

    static var isQuick5Mode: Bool {
        CommandLine.arguments.contains("--batch-quick5")
    }

    static var isQuick8Mode: Bool {
        CommandLine.arguments.contains("--batch-quick8")
    }

    static var isSingle1Mode: Bool {
        CommandLine.arguments.contains("--batch-single1")
    }

    static var isNewCam50Mode: Bool {
        CommandLine.arguments.contains("--batch-newcam50")
    }

    /// ★ 训练数据生成模式：用 GT 框驱动 PID，导出 State-Action 对
    static var isTrainingMode: Bool {
        CommandLine.arguments.contains("--export-training") || CommandLine.arguments.contains("--dagger-training")
    }

    /// ★ DAgger 噪声注入模式
    static var isDaggerMode: Bool {
        CommandLine.arguments.contains("--dagger-training")
    }

    /// 仅在显式要求时自动退出，默认保留窗口展示报表
    static var isRecallTestMode: Bool {
        CommandLine.arguments.contains("--batch-recall-test")
    }
    /// ★ 召回率 + 误检率全交叉测试模式
    static var isRecallFpTestMode: Bool {
        CommandLine.arguments.contains("--batch-recall-fp-test")
    }
    static var isRecallFpEnv40ST60Mode: Bool {
        CommandLine.arguments.contains("--batch-recall-fp-env40-st60")
            || CommandLine.arguments.contains("--batch-recall-fp-env10-st60")
            || CommandLine.arguments.contains("--batch-tracker7w-st60")
            || CommandLine.arguments.contains("--batch-tracker7w-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-run009-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-track1-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-run009-yolo-calib-pair-st60")
            || CommandLine.arguments.contains("--batch-run009-simyolo-calib-st60")
            || CommandLine.arguments.contains("--batch-track1-30s-lock-threshold")
            || CommandLine.arguments.contains("--batch-multiscene10-real-yolo-st60")
            || CommandLine.arguments.contains("--batch-multiscene10-simyolo-aligned-st60")
            || CommandLine.arguments.contains("--batch-multiscene10-yolo-simyolo-pair-st60")
    }
    static var isSimYoloCalibPairMode: Bool {
        CommandLine.arguments.contains("--batch-simyolo-calib-pair")
    }
    /// 单次验证：与 recall-fp 第 6 组相同 (0.95, 0.10)，Pass1+Pass2，用于核对 STLock%
    static var isStVerify095010Mode: Bool {
        CommandLine.arguments.contains("--batch-verify-st-095-010")
    }
    static var isTrack1ThirtySecondLockThresholdMode: Bool {
        CommandLine.arguments.contains("--batch-track1-30s-lock-threshold")
    }
    static var isMarkerCVMode: Bool {
        CommandLine.arguments.contains("--batch-marker-cv-multiscene-30s")
            || CommandLine.arguments.contains("--batch-marker-cv-video-samples")
    }
    static var isFiducialCVMode: Bool {
        CommandLine.arguments.contains("--batch-fiducial-cv-multiscene20-30s")
            || CommandLine.arguments.contains("--batch-fiducial-cv-video-samples")
    }
    static var isIDFiducialCVMode: Bool {
        CommandLine.arguments.contains("--batch-id-fiducial-cv-multiscene20-30s")
            || CommandLine.arguments.contains("--batch-id-fiducial-cv-video-samples")
            || CommandLine.arguments.contains("--batch-id-fiducial-cv-smoke5")
    }
    static var isMarkerYoloThresholdMode: Bool {
        CommandLine.arguments.contains("--batch-marker-yolo-threshold-30s")
    }
    static var isBatchAutoExit: Bool {
        CommandLine.arguments.contains("--batch-sim-auto-exit")
    }

    func startBatch(config: BatchSimConfig = .default) {
        // ★ 训练模式：强制用 GT 框驱动 PID
        if BatchSimRunner.isTrainingMode {
            VSimCfg.forceGtForPid = true
            print("★ TRAINING MODE: forceGtForPid = true")
        }
        // ★ DAgger 模式：额外开启噪声注入
        if BatchSimRunner.isDaggerMode {
            VSimCfg.daggerMode = true
            print("★ DAGGER MODE: noise injection enabled")
        }
        self.config = config

        // ★ Sandbox 感知目录创建: 先尝试原始路径，失败则回退到 /tmp并更新 config
        do {
            try FileManager.default.createDirectory(
                atPath: config.outputDir,
                withIntermediateDirectories: true
            )
            print("✅ Output dir created: \(config.outputDir)")
        } catch {
            print("⚠️ Failed to create output dir: \(config.outputDir) - \(error.localizedDescription)")
            let fallbackDir = NSTemporaryDirectory() + "batch_sim_results"
            try? FileManager.default.createDirectory(atPath: fallbackDir, withIntermediateDirectories: true)
            // ★ Fix: 更新 config 为 fallback 路径，确保后续 CSV 写入也用 fallback 目录
            self.config = BatchSimConfig(runs: config.runs, simDuration: config.simDuration, outputDir: fallbackDir)
            print("✅ Using fallback output dir: \(fallbackDir)")  
        }

        runSpecs = config.runs
        totalRuns = runSpecs.count
        currentRun = 0
        isRunning = true
        showReport = false
        
        // ★ stUnseen100 模式下，直接进入 Pass 2 (单目标锁定)
        if BatchSimRunner.isStUnseen100Mode || BatchSimRunner.isSt10Mode || BatchSimRunner.isRecallFpEnv40ST60Mode || BatchSimRunner.isTrack1ThirtySecondLockThresholdMode || BatchSimRunner.isMarkerCVMode || BatchSimRunner.isFiducialCVMode || BatchSimRunner.isIDFiducialCVMode || BatchSimRunner.isMarkerYoloThresholdMode {
            isPass2 = true
            let modeName: String = {
                if BatchSimRunner.isMarkerCVMode { return "marker-cv-multiscene-30s" }
                if BatchSimRunner.isFiducialCVMode { return "fiducial-cv-multiscene20-30s" }
                if BatchSimRunner.isIDFiducialCVMode { return "id-fiducial-cv-multiscene20-30s" }
                if BatchSimRunner.isMarkerYoloThresholdMode { return "marker-yolo-threshold-30s" }
                if BatchSimRunner.isTrack1ThirtySecondLockThresholdMode { return "track1-30s-lock-threshold" }
                if BatchSimRunner.isRecallFpEnv40ST60Mode { return "recallFpEnv40ST60" }
                if BatchSimRunner.isSt10Mode { return "st10" }
                return "stUnseen100"
            }()
            print("★ \(modeName) MODE: skipping Pass 1, running Pass 2 (ST Lock) only")
        } else {
            isPass2 = false
        }
        
        // ★ 保持现有结果，不重置 allResults，实现断点续传/追加
        // allResults = []
        loadDynamicResults() // 确保加载了之前的数据
        // ★ recall-fp-test / simyolo-calib-pair 模式: 清空旧持久化结果,避免污染新批次的 summary CSV
        if BatchSimRunner.isRecallFpTestMode || BatchSimRunner.isRecallFpEnv40ST60Mode || BatchSimRunner.isSimYoloCalibPairMode || BatchSimRunner.isStVerify095010Mode || BatchSimRunner.isTrack1ThirtySecondLockThresholdMode || BatchSimRunner.isMarkerCVMode || BatchSimRunner.isFiducialCVMode || BatchSimRunner.isIDFiducialCVMode || BatchSimRunner.isMarkerYoloThresholdMode {
            allResults = []
            UserDefaults.standard.removeObject(forKey: dynamicResultsKey)
            let modeName: String = {
                if BatchSimRunner.isMarkerCVMode { return "marker-cv-multiscene-30s" }
                if BatchSimRunner.isFiducialCVMode { return "fiducial-cv-multiscene20-30s" }
                if BatchSimRunner.isIDFiducialCVMode { return "id-fiducial-cv-multiscene20-30s" }
                if BatchSimRunner.isMarkerYoloThresholdMode { return "marker-yolo-threshold-30s" }
                if BatchSimRunner.isTrack1ThirtySecondLockThresholdMode { return "track1-30s-lock-threshold" }
                if BatchSimRunner.isRecallFpEnv40ST60Mode { return "recall-fp-env40-st60" }
                if BatchSimRunner.isSimYoloCalibPairMode { return "simyolo-calib-pair" }
                if BatchSimRunner.isStVerify095010Mode { return "verify-st-095-010" }
                return "recall-fp-test"
            }()
            print("★ \(modeName): cleared persisted results for clean batch")
        }
        persistDynamicResults()

        print("═══════════════════════════════════════════════════════")
        let modeDesc = isPass2 ? "Pass 2 (ST Lock) Only" : "Pass 1 + Pass 2"
        print("  Batch Simulation — \(totalRuns) runs (\(VSimCfg.modelDisplayName), zoom) | \(modeDesc)")
        print("  Output: \(config.outputDir)")
        print("═══════════════════════════════════════════════════════")

        runNext()
    }

    /// ★ 双 pass 状态: 当前是否在跑 Pass 2 (单目标锁定率)
    private var isPass2: Bool = false
    /// ★ Pass 1 统计结果缓存 (用于合并)
    private var pass1Stats: VSimStats?

    private func runNext() {
        guard currentRun < runSpecs.count else {
            finishBatch()
            return
        }

        let spec = runSpecs[currentRun]
        let passLabel = isPass2 ? "Pass2-ST" : "Pass1"
        let plannedSimYolo: Bool = {
            if let forceSim = spec.forceSimulatedYOLO { return forceSim }
            return BatchSimRunner.isRecallTestMode || BatchSimRunner.isRecallFpTestMode
        }()
        currentUsesSimulatedYolo = plannedSimYolo
        currentUsesMarkerCVDetector = BatchSimRunner.isMarkerCVMode || BatchSimRunner.isFiducialCVMode || BatchSimRunner.isIDFiducialCVMode
        currentRecallRate = spec.recallRate
        currentFpRate = plannedSimYolo ? spec.fpRate : 0
        let sourceLabel: String = {
            if BatchSimRunner.isMarkerCVMode { return "MarkerCV concentric rings" }
            if BatchSimRunner.isFiducialCVMode { return "FiducialCV tags" }
            if BatchSimRunner.isIDFiducialCVMode { return "IDFiducialCV tags" }
            if plannedSimYolo {
                return String(format: "SimYOLO r=%.1f%% fp=%.1f%%", Double(spec.recallRate * 100), Double(spec.fpRate * 100))
            }
            return "RealYOLO \(VSimCfg.modelDisplayName)"
        }()
        statusText = "Run \(spec.runIndex)/\(totalRuns) [\(passLabel)] \(sourceLabel): T\(spec.trackShape + 1) scene=\(spec.sceneType) preset=\(spec.cameraPreset) seed=\(spec.seed)"
        print("\n[\(spec.runIndex)/\(totalRuns)] [\(passLabel)] track=\(spec.trackShape) scene=\(spec.sceneType) preset=\(spec.cameraPreset) seed=\(spec.seed) cars=\(spec.carCount) camH=\(spec.cameraHeight) ramps=\(spec.enableRamps)")

        let eng = VirtualRaceEngine()
        eng.randomSeed = spec.seed
        eng.carCount = spec.carCount
        eng.cameraHeight = spec.cameraHeight
        eng.simDuration = config.simDuration
        eng.enableOcclusion = true
        eng.enableLightVariation = true
        eng.racingMode = true
        eng.enableLensDistortion = false
        eng.enableDrift = true
        eng.sceneType = spec.sceneType
        eng.trackShape = spec.trackShape
        eng.enableRamps = spec.enableRamps
        eng.cameraPreset = spec.cameraPreset
        // ★ marker-cv 模式：使用同心圆标记的传统 CV 检测，不调用真实模型。
        if BatchSimRunner.isIDFiducialCVMode {
            eng.useIDFiducialCVDetector = true
            eng.useFiducialCVDetector = false
            eng.useMarkerCVDetector = false
            eng.useSimulatedYOLO = false
            eng.simulatedRecallRate = 0
            eng.simulatedFPRate = 0
        } else if BatchSimRunner.isFiducialCVMode {
            eng.useIDFiducialCVDetector = false
            eng.useFiducialCVDetector = true
            eng.useMarkerCVDetector = false
            eng.useSimulatedYOLO = false
            eng.simulatedRecallRate = 0
            eng.simulatedFPRate = 0
        } else if BatchSimRunner.isMarkerCVMode {
            eng.useMarkerCVDetector = true
            eng.useFiducialCVDetector = false
            eng.useIDFiducialCVDetector = false
            eng.useSimulatedYOLO = false
            eng.simulatedRecallRate = 0
            eng.simulatedFPRate = 0
        } else if let forceSim = spec.forceSimulatedYOLO {
            eng.useSimulatedYOLO = forceSim
            eng.simulatedRecallRate = spec.recallRate
            eng.simulatedFPRate = forceSim ? spec.fpRate : 0.0
        } else if BatchSimRunner.isRecallTestMode || BatchSimRunner.isRecallFpTestMode {
            eng.useSimulatedYOLO = true
            eng.simulatedRecallRate = spec.recallRate
            eng.simulatedFPRate = spec.fpRate
        } else {
            eng.useSimulatedYOLO = false
            eng.simulatedFPRate = 0.0
        }
        eng.enableZoom = true
        // ★ Pass 2: 开启单目标锁定模式
        eng.singleTargetMode = isPass2
        let shouldRecordSceneKit = CommandLine.arguments.contains("--record-scenekit")
            || CommandLine.arguments.contains("--batch-run009-simyolo-calib-st60")
        if shouldRecordSceneKit && isPass2 {
            let mp4Path = "\(config.outputDir)/run_\(String(format: "%03d", spec.runIndex))_T\(spec.trackShape + 1)_sc\(spec.sceneType)_cp\(spec.cameraPreset)_pass2_st_scenekit_raw.mp4"
            eng.sceneRecordingURL = URL(fileURLWithPath: mp4Path)
        } else {
            eng.sceneRecordingURL = nil
        }
        // ★ st10 模式：强制开启帧降质，模拟真实画质
        if BatchSimRunner.isSt10Mode {
            eng.enableFrameDegrade = true
            eng.degradeIntensity = 0.7
        }
        self.engine = eng

        cancellable = eng.$running
            .dropFirst()
            .filter { !$0 }
            .first()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.onRunComplete()
            }

        eng.start()
    }

    private func onRunComplete() {
        guard let eng = engine else { return }
        let spec = runSpecs[currentRun]
        let stats = eng.stats

        if !isPass2 && !BatchSimRunner.isStUnseen100Mode {
            // ═══ Pass 1 完成: 保存统计，启动 Pass 2 ═══
            pass1Stats = stats
            print("  [Pass1] Det=\(String(format: "%.1f", stats.detectionRate))%  Track=\(String(format: "%.1f", stats.generalTrackRate))%  STLock=\(String(format: "%.1f", stats.singleTargetLockRate))%")

            // 导出 Pass 1 CSV
            let csvPath = "\(config.outputDir)/run_\(String(format: "%03d", spec.runIndex))_T\(spec.trackShape + 1)_sc\(spec.sceneType)_cp\(spec.cameraPreset)_pass1.csv"
            eng.exportCSV(URL(fileURLWithPath: csvPath))

            // ★ 训练模式：额外导出 State-Action 训练数据 CSV
            if BatchSimRunner.isTrainingMode {
                let trainingDir = "\(config.outputDir)/training_data"
                try? FileManager.default.createDirectory(atPath: trainingDir, withIntermediateDirectories: true)
                let trainingPath = "\(trainingDir)/train_\(String(format: "%03d", spec.runIndex)).csv"
                eng.exportTrainingCSV(URL(fileURLWithPath: trainingPath))
            }

            // 启动 Pass 2 (单目标锁定率)
            isPass2 = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
                self?.runNext()
            }
            return
        }

        // ═══ Pass 2 完成: 合并结果 ═══
        let stLockRate = stats.singleTargetLockRate
        let cleanLockSec = eng.longestPrimaryCleanLockSeconds()
        let trackerProfile = ProcessInfo.processInfo.environment["ST_EXPERIMENT_PROFILE"] ?? "baseline"
        print("  [Pass2-ST] STLock=\(String(format: "%.1f", stLockRate))%  (\(stats.stLockFrames)/\(stats.stTotalFrames))  CleanLock=\(String(format: "%.2f", cleanLockSec))s")
        if BatchSimRunner.isStVerify095010Mode {
            let exact = stats.stTotalFrames > 0
                ? Double(stats.stLockFrames) / Double(stats.stTotalFrames) * 100
                : 0
            print("  ★ VERIFY --batch-verify-st-095-010: 单目标锁定精确值 \(String(format: "%.4f", exact))%  (四舍五入到一位小数 ≈ \(String(format: "%.1f", stLockRate))%)")
        }

        // 导出 Pass 2 CSV
        let csvPath2 = "\(config.outputDir)/run_\(String(format: "%03d", spec.runIndex))_T\(spec.trackShape + 1)_sc\(spec.sceneType)_cp\(spec.cameraPreset)_pass2_st.csv"
        eng.exportCSV(URL(fileURLWithPath: csvPath2))

        // 合并 Pass 1 + Pass 2 结果
        let p1 = pass1Stats ?? stats
        let result: [String: Any] = [
            "run": spec.runIndex,
            "seed": Int(spec.seed),  // ★ Fix: UInt64 → Int, 否则 JSONSerialization.isValidJSONObject 返回 false
            "trackShape": spec.trackShape,
            "sceneType": spec.sceneType,
            "cameraPreset": spec.cameraPreset,
            "carCount": spec.carCount,
            "cameraHeight": Double(spec.cameraHeight),
            "enableRamps": spec.enableRamps,
            // 界面与汇总用：本轮是否走「模拟检测」（不用随应用打包的真实神经网络推理）
            "usesSimulatedYolo": eng.useSimulatedYOLO,
            "usesMarkerCVDetector": eng.useMarkerCVDetector,
            "usesFiducialCVDetector": eng.useFiducialCVDetector,
            "usesIDFiducialCVDetector": eng.useIDFiducialCVDetector,
            "trackerProfile": trackerProfile,
            "recallRate": spec.recallRate,
            "fpRate": spec.fpRate,
            "totalFrames": p1.totalFrames,
            "trackingFrames": p1.trackingFrames,
            "pureDetFrames": p1.pureDetFrames,
            "detectionRate": p1.detectionRate,
            "generalTrackRate": p1.generalTrackRate,
            "singleTargetLockRate": stLockRate,
            "cleanLockSec": cleanLockSec,
            "cleanLock10s": cleanLockSec >= 10.0,
            "maxConsecutiveLost": p1.maxConsecutiveLost,
            "trackingRate": p1.generalTrackRate,
            "yoloDetRate": p1.yoloDetRate,
            "yoloObjectRecallRate": p1.yoloObjectRecallRate,
            "yoloFrameRecallRate": p1.yoloFrameRecallRate,
            "yoloPrimaryRecallRate": p1.yoloPrimaryRecallRate,
            "yoloCandidateFPRate": p1.yoloCandidateFPRate,
            "yoloFPFrameRate": p1.yoloFPFrameRate,
            "yoloFPPerFrame": p1.yoloFPPerFrame,
            "avgErrH": p1.avgErrH,
            "avgErrV": p1.avgErrV,
            "maxErrH": p1.maxErrH,
            "maxErrV": p1.maxErrV,
            "jitterH": p1.jitterH,
            "jitterV": p1.jitterV,
            "avgYoloMs": p1.avgYoloMs,
            "avgZoom": p1.avgZoom,
            "score": p1.score,
            "grade": p1.grade
        ]
        allResults.append(result)
        persistDynamicResults()

        // 保存历史记录 (包含单目标锁定率)
        let histRecord = SimHistoryRecord(
            timestamp: Date(),
            strategy: eng.useNeuralNetwork ? "MLP" : "PID",
            trackShape: spec.trackShape,
            sceneType: spec.sceneType,
            cameraPreset: spec.cameraPreset,
            carCount: spec.carCount,
            simDuration: config.simDuration,
            detectionRate: p1.detectionRate,
            generalTrackRate: p1.generalTrackRate,
            singleTargetLockRate: stLockRate,
            maxConsecutiveLost: p1.maxConsecutiveLost,
            yoloDetRate: p1.yoloDetRate,
            avgErrH: p1.avgErrH,
            avgErrV: p1.avgErrV,
            jitterH: p1.jitterH,
            jitterV: p1.jitterV,
            score: p1.score,
            grade: p1.grade,
            randomSeed: spec.seed,
            cameraHeight: Double(spec.cameraHeight),
            enableRamps: spec.enableRamps,
            useSimulatedYOLO: eng.useSimulatedYOLO
        )
        SimHistoryStore.shared.save(histRecord)

        // 实时日志
        let logLine = "Run \(spec.runIndex)/\(totalRuns): Det=\(String(format: "%.1f", p1.detectionRate))% Track=\(String(format: "%.1f", p1.generalTrackRate))% STLock=\(String(format: "%.1f", stLockRate))% Score=\(p1.score)\n"
        let logURL = URL(fileURLWithPath: NSTemporaryDirectory() + "batch_progress.log")
        if let data = logLine.data(using: .utf8) {
            if let handle = try? FileHandle(forWritingTo: logURL) {
                handle.seekToEndOfFile()
                handle.write(data)
                handle.closeFile()
            } else {
                try? data.write(to: logURL)
            }
        }

        print("  → Det=\(String(format: "%.1f", p1.detectionRate))%  Track=\(String(format: "%.1f", p1.generalTrackRate))%  STLock=\(String(format: "%.1f", stLockRate))%  Score=\(p1.score) \(p1.grade)")

        // 重置双 pass 状态，前进到下一个场景
        // ★ Fix: ST-only 模式下保持 isPass2=true，否则后续轮次 singleTargetMode 会被关闭导致 STLock=0/0
        if BatchSimRunner.isStUnseen100Mode || BatchSimRunner.isSt10Mode || BatchSimRunner.isRecallFpEnv40ST60Mode || BatchSimRunner.isTrack1ThirtySecondLockThresholdMode || BatchSimRunner.isMarkerCVMode || BatchSimRunner.isFiducialCVMode || BatchSimRunner.isIDFiducialCVMode || BatchSimRunner.isMarkerYoloThresholdMode {
            isPass2 = true   // 保持 Pass2 模式
        } else {
            isPass2 = false
        }
        pass1Stats = nil
        currentRun += 1

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.runNext()
        }
    }

    private func finishBatch() {
        isRunning = false

        var summaryCSV = "run,seed,trackShape,sceneType,cameraPreset,carCount,cameraHeight,enableRamps,usesSimulatedYolo,usesMarkerCVDetector,usesFiducialCVDetector,usesIDFiducialCVDetector,trackerProfile,recallRate,fpRate,totalFrames,detectionRate,generalTrackRate,singleTargetLockRate,cleanLockSec,cleanLock10s,maxConsecutiveLost,trackingRate,yoloDetRate,yoloObjectRecallRate,yoloFrameRecallRate,yoloPrimaryRecallRate,yoloCandidateFPRate,yoloFPFrameRate,yoloFPPerFrame,avgErrH,avgErrV,jitterH,jitterV,avgYoloMs,avgZoom,score,grade\n"

        for r in allResults {
            summaryCSV += "\(r["run"]!),\(r["seed"]!),\(r["trackShape"]!),\(r["sceneType"]!),\(r["cameraPreset"]!),\(r["carCount"]!),\(r["cameraHeight"]!),"
            summaryCSV += "\(r["enableRamps"]!),\(r["usesSimulatedYolo"]!),\(r["usesMarkerCVDetector"] as? Bool ?? false),\(r["usesFiducialCVDetector"] as? Bool ?? false),\(r["usesIDFiducialCVDetector"] as? Bool ?? false),\(r["trackerProfile"] as? String ?? "baseline"),\(r["recallRate"] as? Float ?? 0.70),\(r["fpRate"] as? Float ?? 0.0),\(r["totalFrames"]!),"
            summaryCSV += "\(String(format: "%.2f", r["detectionRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["generalTrackRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["singleTargetLockRate"] as? Double ?? 0)),"
            summaryCSV += "\(String(format: "%.2f", r["cleanLockSec"] as? Double ?? 0)),"
            summaryCSV += "\((r["cleanLock10s"] as? Bool ?? false) ? 1 : 0),"
            summaryCSV += "\(r["maxConsecutiveLost"]!),"
            summaryCSV += "\(String(format: "%.2f", r["trackingRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["yoloDetRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["yoloObjectRecallRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["yoloFrameRecallRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["yoloPrimaryRecallRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["yoloCandidateFPRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["yoloFPFrameRate"] as! Double)),"
            summaryCSV += "\(String(format: "%.3f", r["yoloFPPerFrame"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["avgErrH"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["avgErrV"] as! Double)),"
            summaryCSV += "\(String(format: "%.1f", r["jitterH"] as! Double)),"
            summaryCSV += "\(String(format: "%.1f", r["jitterV"] as! Double)),"
            summaryCSV += "\(String(format: "%.1f", r["avgYoloMs"] as! Double)),"
            summaryCSV += "\(String(format: "%.2f", r["avgZoom"] as! Double)),"
            summaryCSV += "\(r["score"]!),\(r["grade"]!)\n"
        }

        let summaryPath = "\(config.outputDir)/batch_summary.csv"
        do {
            try summaryCSV.write(toFile: summaryPath, atomically: true, encoding: .utf8)
            print("✅ batch_summary.csv written: \(summaryPath)")
        } catch {
            print("⚠️ Failed to write batch_summary.csv: \(error.localizedDescription)")
            let fallbackPath = NSTemporaryDirectory() + "batch_sim_results/batch_summary.csv"
            try? FileManager.default.createDirectory(atPath: NSTemporaryDirectory() + "batch_sim_results", withIntermediateDirectories: true)
            try? summaryCSV.write(toFile: fallbackPath, atomically: true, encoding: .utf8)
            print("✅ batch_summary.csv fallback written: \(fallbackPath)")
        }

        let detRates = allResults.compactMap { $0["detectionRate"] as? Double }
        let trackRates = allResults.compactMap { $0["generalTrackRate"] as? Double }
        let stLockRates = allResults.compactMap { $0["singleTargetLockRate"] as? Double }
        let cleanLocks = allResults.compactMap { $0["cleanLockSec"] as? Double }
        let clean10Count = allResults.filter { ($0["cleanLock10s"] as? Bool) == true }.count
        let zooms = allResults.compactMap { $0["avgZoom"] as? Double }
        let scores = allResults.compactMap { $0["score"] as? Int }
        let yoloDets = allResults.compactMap { $0["yoloDetRate"] as? Double }
        let yoloObjRecalls = allResults.compactMap { $0["yoloObjectRecallRate"] as? Double }
        let yoloPrimaryRecalls = allResults.compactMap { $0["yoloPrimaryRecallRate"] as? Double }
        let yoloFpFrames = allResults.compactMap { $0["yoloFPFrameRate"] as? Double }
        let yoloFpPerFrames = allResults.compactMap { $0["yoloFPPerFrame"] as? Double }

        let avgDet = detRates.isEmpty ? 0 : detRates.reduce(0, +) / Double(detRates.count)
        let avgTrack = trackRates.isEmpty ? 0 : trackRates.reduce(0, +) / Double(trackRates.count)
        let avgSTLock = stLockRates.isEmpty ? 0 : stLockRates.reduce(0, +) / Double(stLockRates.count)
        let avgCleanLock = cleanLocks.isEmpty ? 0 : cleanLocks.reduce(0, +) / Double(cleanLocks.count)
        let avgZm = zooms.isEmpty ? 0 : zooms.reduce(0, +) / Double(zooms.count)
        let avgScore = scores.isEmpty ? 0 : Double(scores.reduce(0, +)) / Double(scores.count)
        let avgYoloDet = yoloDets.isEmpty ? 0 : yoloDets.reduce(0, +) / Double(yoloDets.count)
        let avgYoloObjRecall = yoloObjRecalls.isEmpty ? 0 : yoloObjRecalls.reduce(0, +) / Double(yoloObjRecalls.count)
        let avgYoloPrimaryRecall = yoloPrimaryRecalls.isEmpty ? 0 : yoloPrimaryRecalls.reduce(0, +) / Double(yoloPrimaryRecalls.count)
        let avgYoloFpFrame = yoloFpFrames.isEmpty ? 0 : yoloFpFrames.reduce(0, +) / Double(yoloFpFrames.count)
        let avgYoloFpPerFrame = yoloFpPerFrames.isEmpty ? 0 : yoloFpPerFrames.reduce(0, +) / Double(yoloFpPerFrames.count)

        print("\n═══════════════════════════════════════════════════════")
        print("  BATCH COMPLETE — \(totalRuns) runs (虚拟赛道 / \(VSimCfg.modelDisplayName) / 变焦)")
        print("═══════════════════════════════════════════════════════")
        print("  Avg Detection Rate:   \(String(format: "%.2f", avgDet))%")
        print("  Avg General Track:    \(String(format: "%.2f", avgTrack))%")
        print("  Avg ST Lock Rate:     \(String(format: "%.2f", avgSTLock))%")
        print("  Avg Clean Lock:       \(String(format: "%.2f", avgCleanLock))s  (\(clean10Count)/\(totalRuns) >=10s)")
        print("  Avg YOLO Det:         \(String(format: "%.2f", avgYoloDet))%")
        print("  Avg YOLO Obj Recall:  \(String(format: "%.2f", avgYoloObjRecall))%")
        print("  Avg YOLO Pri Recall:  \(String(format: "%.2f", avgYoloPrimaryRecall))%")
        print("  Avg YOLO FP Frames:   \(String(format: "%.2f", avgYoloFpFrame))% (\(String(format: "%.3f", avgYoloFpPerFrame)) FP/frame)")
        print("  Avg Zoom:             \(String(format: "%.2f", avgZm))x")
        print("  Avg Score:            \(String(format: "%.1f", avgScore))")
        print("  Summary CSV:          \(summaryPath)")
        print("═══════════════════════════════════════════════════════")

        statusText = String(format: "Done %d runs | Det=%.1f%% | Track=%.1f%% | STLock=%.1f%% | Clean=%.1fs", totalRuns, avgDet, avgTrack, avgSTLock, avgCleanLock)

        // 批量模式默认不自动弹报表，避免出现第二个窗口/弹层
        showReport = CommandLine.arguments.contains("--batch-show-report")

        if BatchSimRunner.isBatchAutoExit {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                NSApplication.shared.terminate(nil)
            }
        }
    }

    private func persistDynamicResults() {
        if CommandLine.arguments.contains("--public-smoke") { return }
        let isValid = JSONSerialization.isValidJSONObject(allResults)
        if !isValid {
            print("⚠️ persistDynamicResults: isValidJSONObject=false, 尝试转换 UInt64 后重试")
            // Fallback: 将所有 UInt64 转为 Int 以确保 JSON 兼容
            let sanitized: [[String: Any]] = allResults.map { dict in
                var d = [String: Any]()
                for (k, v) in dict {
                    if let u = v as? UInt64 { d[k] = Int(u) }
                    else if let u = v as? UInt { d[k] = Int(u) }
                    else { d[k] = v }
                }
                return d
            }
            guard JSONSerialization.isValidJSONObject(sanitized),
                  let data = try? JSONSerialization.data(withJSONObject: sanitized, options: [])
            else {
                print("❌ persistDynamicResults: sanitized 仍然无法序列化")
                return
            }
            UserDefaults.standard.set(data, forKey: dynamicResultsKey)
            print("✅ persistDynamicResults: fallback 写入成功, \(sanitized.count) 条")
            return
        }
        guard let data = try? JSONSerialization.data(withJSONObject: allResults, options: []) else { return }
        UserDefaults.standard.set(data, forKey: dynamicResultsKey)
    }

    private func loadDynamicResults() {
        if CommandLine.arguments.contains("--public-smoke") { allResults = []; return }
        guard let data = UserDefaults.standard.data(forKey: dynamicResultsKey),
              let obj = try? JSONSerialization.jsonObject(with: data, options: []),
              let rows = obj as? [[String: Any]]
        else { return }
        allResults = rows
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 批量仿真视图 (自动模式下的最小 UI)
// ═══════════════════════════════════════════════════════════════

struct BatchSimView: View {
    @ObservedObject var runner = BatchSimRunner.shared
    @State private var didAutoShowReport = false

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "car.2.fill")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)

            Text("批量仿真运行中")
                .font(.title2.bold())

            Text(runner.statusText)
                .font(.body.monospaced())
                .foregroundColor(.secondary)

            Text(batchDetectionOverviewText)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)

            ProgressView(value: Double(runner.currentRun), total: Double(max(runner.totalRuns, 1)))
                .progressViewStyle(.linear)
                .frame(width: 400)

            Text("\(runner.currentRun) / \(runner.totalRuns)")
                .font(.caption.monospaced())
                .foregroundColor(.secondary)

            if !runner.allResults.isEmpty, let lastResult = runner.allResults.last {
                GroupBox("最近一轮结果") {
                    VStack(spacing: 8) {
                        HStack(spacing: 16) {
                            VStack {
                                Text(String(format: "%.1f%%", lastResult["generalTrackRate"] as? Double ?? 0))
                                    .font(.title3.bold().monospaced())
                                    .foregroundColor(.green)
                                Text("追踪率")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            VStack {
                                let stRate = lastResult["singleTargetLockRate"] as? Double ?? 0
                                Text(String(format: "%.1f%%", stRate))
                                    .font(.title3.bold().monospaced())
                                    .foregroundColor(.purple)
                                Text("单目标锁定")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            VStack {
                                Text("\(lastResult["score"] as? Int ?? 0)")
                                    .font(.title3.bold().monospaced())
                                    .foregroundColor(.blue)
                                Text("评分")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        Text(lastRoundDetectionCaption(lastResult))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(8)
                }
                .frame(width: 440)
            }

            if !runner.allResults.isEmpty {
                GroupBox("单目标锁定追踪 — 动态结果表") {
                    ScrollView([.horizontal, .vertical]) {
                        VStack(alignment: .leading, spacing: 0) {
                            HStack(spacing: 6) {
                                Text("轮次").frame(width: 34, alignment: .trailing)
                                Text("赛道").frame(width: 30, alignment: .center)
                                Text("场景").frame(width: 30, alignment: .center)
                                Text("机位").frame(width: 30, alignment: .center)
                                Text("来源").frame(width: 34, alignment: .center)
                                Text("设定召回").frame(width: 50, alignment: .trailing)
                                Text("设定误检").frame(width: 50, alignment: .trailing)
                                Text("检测%").frame(width: 48, alignment: .trailing)
                                Text("追踪%").frame(width: 48, alignment: .trailing)
                                Text("单目标%").frame(width: 54, alignment: .trailing)
                                Text("评分").frame(width: 36, alignment: .trailing)
                            }
                            .font(.caption2.bold())
                            .foregroundColor(.secondary)
                            Divider()
                            ForEach(Array(runner.allResults.enumerated()), id: \.offset) { idx, r in
                                HStack(spacing: 6) {
                                    Text("\(r["run"] as? Int ?? idx + 1)")
                                        .frame(width: 34, alignment: .trailing)
                                    Text("\(r["trackShape"] as? Int ?? 0)")
                                        .frame(width: 30, alignment: .center)
                                    Text("\(r["sceneType"] as? Int ?? 0)")
                                        .frame(width: 30, alignment: .center)
                                    Text("\(r["cameraPreset"] as? Int ?? 0)")
                                        .frame(width: 30, alignment: .center)
                                    Text(detectionSourceShort(r))
                                        .frame(width: 34, alignment: .center)
                                        .foregroundColor(usesSimulatedYolo(r) == true ? .orange : (usesSimulatedYolo(r) == false ? .green : .secondary))
                                    Text(simRecallDisplay(r))
                                        .frame(width: 50, alignment: .trailing)
                                    Text(simFpDisplay(r))
                                        .frame(width: 50, alignment: .trailing)
                                    Text(String(format: "%.1f", r["detectionRate"] as? Double ?? 0))
                                        .frame(width: 48, alignment: .trailing)
                                    Text(String(format: "%.1f", r["generalTrackRate"] as? Double ?? 0))
                                        .frame(width: 48, alignment: .trailing)
                                    Text(String(format: "%.1f", r["singleTargetLockRate"] as? Double ?? 0))
                                        .frame(width: 54, alignment: .trailing)
                                        .foregroundColor(.purple)
                                    Text("\(r["score"] as? Int ?? 0)")
                                        .frame(width: 36, alignment: .trailing)
                                }
                                .font(.caption2.monospaced())
                                Divider()
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .frame(minHeight: 160, maxHeight: 360)
                }
                .frame(maxWidth: 720)
            }

            if !runner.isRunning && !runner.allResults.isEmpty {
                Button("查看批量统计报表") {
                    runner.showReport = true
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(minWidth: 560, minHeight: 560)
        .padding()
        .onAppear {
            if BatchSimRunner.isBatchMode && !runner.isRunning {
                if CommandLine.arguments.contains("--public-smoke") {
                    runner.startBatch(config: .publicSmoke)
                } else if BatchSimRunner.isSingle1Mode {
                    runner.startBatch(config: .single1)
                } else if BatchSimRunner.isSt10Mode {
                    runner.startBatch(config: .stDegrade10)
                } else if BatchSimRunner.isStUnseen100Mode {
                    runner.startBatch(config: .stUnseen100)
                } else if BatchSimRunner.isNewCam50Mode {
                    runner.startBatch(config: .newCam50)
                } else if BatchSimRunner.isQuick8Mode {
                    runner.startBatch(config: .quick8)
                } else if BatchSimRunner.isRecallTestMode {
                    runner.startBatch(config: .recallTest)
                } else if BatchSimRunner.isStVerify095010Mode {
                    runner.startBatch(config: .stVerify095010)
                } else if BatchSimRunner.isRecallFpTestMode {
                    runner.startBatch(config: .recallFpGrid)
                } else if CommandLine.arguments.contains("--batch-track1-real-yolo-st60") {
                    runner.startBatch(config: .track1RealYoloST60)
                } else if CommandLine.arguments.contains("--batch-run009-real-yolo-st60") {
                    runner.startBatch(config: .run009RealYoloST60)
                } else if CommandLine.arguments.contains("--batch-run009-yolo-calib-pair-st60") {
                    runner.startBatch(config: .run009YoloCalibPairST60)
                } else if CommandLine.arguments.contains("--batch-run009-simyolo-calib-st60") {
                    runner.startBatch(config: .run009SimYoloCalibST60)
                } else if CommandLine.arguments.contains("--batch-recall-fp-env10-st60") {
                    runner.startBatch(config: .recallFpEnv10ST60)
                } else if CommandLine.arguments.contains("--batch-tracker7w-st60") {
                    runner.startBatch(config: .tracker7wST60)
                } else if CommandLine.arguments.contains("--batch-tracker7w-real-yolo-st60") {
                    runner.startBatch(config: .tracker7wRealYoloST60)
                } else if BatchSimRunner.isTrack1ThirtySecondLockThresholdMode {
                    runner.startBatch(config: .track1ThirtySecondLockThreshold)
                } else if BatchSimRunner.isMarkerCVMode {
                    let config: BatchSimConfig = CommandLine.arguments.contains("--batch-marker-cv-video-samples") ? .markerCVVideoSamples : .markerCVMultiscene30
                    runner.startBatch(config: config)
                } else if BatchSimRunner.isFiducialCVMode {
                    let config: BatchSimConfig = CommandLine.arguments.contains("--batch-fiducial-cv-video-samples") ? .fiducialCVVideoSamples : .fiducialCVMultiscene20
                    runner.startBatch(config: config)
                } else if BatchSimRunner.isIDFiducialCVMode {
                    let config: BatchSimConfig = CommandLine.arguments.contains("--batch-id-fiducial-cv-smoke5")
                        ? .idFiducialCVSmoke5
                        : (CommandLine.arguments.contains("--batch-id-fiducial-cv-video-samples") ? .idFiducialCVVideoSamples : .idFiducialCVMultiscene20)
                    runner.startBatch(config: config)
                } else if BatchSimRunner.isMarkerYoloThresholdMode {
                    runner.startBatch(config: .markerYoloThreshold30)
                } else if CommandLine.arguments.contains("--batch-multiscene10-real-yolo-st60") {
                    runner.startBatch(config: .multiscene10RealYoloST60)
                } else if CommandLine.arguments.contains("--batch-multiscene10-simyolo-aligned-st60") {
                    runner.startBatch(config: .multiscene10SimYoloAlignedST60)
                } else if CommandLine.arguments.contains("--batch-multiscene10-yolo-simyolo-pair-st60") {
                    runner.startBatch(config: .multiscene10YoloSimYoloPairST60)
                } else if BatchSimRunner.isRecallFpEnv40ST60Mode {
                    runner.startBatch(config: .recallFpEnv40ST60)
                } else if BatchSimRunner.isSimYoloCalibPairMode {
                    runner.startBatch(config: .simYoloCalibPair)
                } else if BatchSimRunner.isQuick5Mode {
                    runner.startBatch(config: .quick5)
                } else {
                    runner.startBatch()
                }
            }
        }
        .onChange(of: runner.showReport) { _, newValue in
            if !newValue {
                didAutoShowReport = true
            }
        }
        .onChange(of: runner.isRunning) { _, newValue in
            if !newValue && !runner.allResults.isEmpty && !didAutoShowReport {
                if CommandLine.arguments.contains("--batch-show-report") {
                    runner.showReport = true
                }
            }
        }
        .sheet(isPresented: $runner.showReport) {
            BatchSimReportView(results: runner.allResults)
        }
    }

    /// 从持久化 JSON 读回时可能是 Bool / Int / NSNumber
    private func usesSimulatedYolo(_ r: [String: Any]) -> Bool? {
        if let b = r["usesSimulatedYolo"] as? Bool { return b }
        if let i = r["usesSimulatedYolo"] as? Int { return i != 0 }
        if let n = r["usesSimulatedYolo"] as? NSNumber { return n.boolValue }
        return nil
    }

    private var batchDetectionOverviewText: String {
        if let currentSim = runner.currentUsesSimulatedYolo {
            if currentSim {
                return String(
                    format: "当前运行：模拟检测 SimYOLO。不会调用真实模型权重；设定召回 %.1f%%，设定误检 %.1f%%。",
                    Double(runner.currentRecallRate * 100),
                    Double(runner.currentFpRate * 100)
                )
            }
            return "当前运行：真实检测 RealYOLO，使用 \(VSimCfg.modelDisplayName)。"
        }
        if BatchSimRunner.isRecallFpTestMode {
            return "本批为「模拟检测」：不调用随应用打包的真实神经网络。表中「设定召回」「设定误检」为人工参数；「检测%」等为仿真统计。"
        }
        if BatchSimRunner.isRecallTestMode {
            return "本批为「模拟检测」：「设定召回」为当前轮人工参数；不使用真实虚拟赛道模型权重做检测推理。"
        }
        if BatchSimRunner.isSimYoloCalibPairMode {
            return "本批为对照：一轮「真实检测」（\(VSimCfg.modelDisplayName)），一轮「模拟检测」；请看「来源」列。"
        }
        if BatchSimRunner.isTrainingMode {
            return "训练/导出模式：检测链路与常规批量不同，以命令行说明为准。"
        }
        return "本批为「真实检测」：使用 \(VSimCfg.modelDisplayName)。「设定召回」「设定误检」列不适用（显示为横杠）。"
    }

    private func detectionSourceShort(_ r: [String: Any]) -> String {
        guard let sim = usesSimulatedYolo(r) else { return "—" }
        return sim ? "模拟" : "真实"
    }

    private func recallRateDouble(_ r: [String: Any]) -> Double {
        if let d = r["recallRate"] as? Double { return d }
        if let f = r["recallRate"] as? Float { return Double(f) }
        if let n = r["recallRate"] as? NSNumber { return n.doubleValue }
        return 0
    }

    private func fpRateDouble(_ r: [String: Any]) -> Double {
        if let d = r["fpRate"] as? Double { return d }
        if let f = r["fpRate"] as? Float { return Double(f) }
        if let n = r["fpRate"] as? NSNumber { return n.doubleValue }
        return 0
    }

    private func simRecallDisplay(_ r: [String: Any]) -> String {
        guard usesSimulatedYolo(r) == true else { return "—" }
        return String(format: "%.0f%%", recallRateDouble(r) * 100)
    }

    private func simFpDisplay(_ r: [String: Any]) -> String {
        guard usesSimulatedYolo(r) == true else { return "—" }
        let v = fpRateDouble(r)
        if v < 0.001 { return "0%" }
        return String(format: "%.1f%%", v * 100)
    }

    private func lastRoundDetectionCaption(_ r: [String: Any]) -> String {
        if usesSimulatedYolo(r) == nil {
            return "检测来源：历史记录无「usesSimulatedYolo」字段，请重新跑一批以区分真实/模拟。"
        }
        let src = detectionSourceShort(r)
        if usesSimulatedYolo(r) == true {
            return "检测来源：\(src)　设定召回 \(simRecallDisplay(r))　设定误检 \(simFpDisplay(r))"
        }
        return "检测来源：\(src)（模型 \(VSimCfg.modelDisplayName)）　模拟参数不适用"
    }
}

struct BatchSimReportView: View {
    let results: [[String: Any]]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("批量仿真报表")
                .font(.title2.bold())

            Text(summaryLine)
                .font(.subheadline.monospaced())
                .foregroundColor(.secondary)

            GroupBox("按场景 平均追踪率") {
                metricsTable(groupBy: "sceneType")
            }
            GroupBox("按机位 平均追踪率") {
                metricsTable(groupBy: "cameraPreset")
            }
            GroupBox("按赛道 平均追踪率") {
                metricsTable(groupBy: "trackShape")
            }
            GroupBox("按场景 单目标锁定率") {
                metricsTable(groupBy: "sceneType", metric: "singleTargetLockRate")
            }
            GroupBox("按场景 最长干净锁定段") {
                metricsTable(groupBy: "sceneType", metric: "cleanLockSec", suffix: "s", goodThreshold: 10)
            }
            GroupBox("自动结论") {
                Text(conclusion)
                    .font(.callout)
            }

            HStack {
                Spacer()
                Button("关闭") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(minWidth: 760, minHeight: 700)
    }

    private var summaryLine: String {
        let det = avg(of: "detectionRate")
        let track = avg(of: "generalTrackRate")
        let stLock = avg(of: "singleTargetLockRate")
        let clean = avg(of: "cleanLockSec")
        let yolo = avg(of: "yoloDetRate")
        return String(format: "Runs=%d  Det=%.1f%%  Track=%.1f%%  STLock=%.1f%%  Clean=%.1fs  YOLO=%.1f%%",
                      results.count, det, track, stLock, clean, yolo)
    }

    private var conclusion: String {
        let det = avg(of: "detectionRate")
        let track = avg(of: "generalTrackRate")
        let stLock = avg(of: "singleTargetLockRate")

        if track < 30 && det < 50 {
            return "本批次低追踪率主要由检测层造成：检测率偏低（建议优先调检测阈值/模型）。"
        }
        if track < 30 && det >= 60 {
            return "本批次低追踪率并非检测层主因：检测率尚可，但追踪仍低，优先排查匹配锁车、机位构图与PID控制链。"
        }
        if track >= 60 && stLock >= 40 {
            return "整体追踪表现较好，单目标锁定率达标，建议针对低分场景做专项优化。"
        }
        if track >= 60 && stLock < 40 {
            return "一般追踪表现较好，但单目标锁定率偏低，建议优化视觉特征区分力和运动模型精度。"
        }
        return "当前为混合瓶颈：检测层与控制层均有改进空间。建议先看低分样本中的检测率与追踪率分布。"
    }

    @ViewBuilder
    private func metricsTable(groupBy key: String, metric: String = "generalTrackRate", suffix: String = "%", goodThreshold: Double = 60) -> some View {
        let groups = groupedAvg(by: key, metric: metric)
        VStack(alignment: .leading, spacing: 4) {
            ForEach(groups, id: \.name) { row in
                HStack {
                    Text(row.name).font(.caption.monospaced())
                    Spacer()
                    Text(String(format: "%.2f%@  (n=%d)", row.value, suffix, row.count))
                        .font(.caption.monospaced())
                        .foregroundColor(row.value >= goodThreshold ? .green : (row.value >= goodThreshold * 0.5 ? .orange : .red))
                }
            }
        }
        .padding(6)
    }

    private func avg(of key: String) -> Double {
        let vals = results.compactMap { $0[key] as? Double }
        guard !vals.isEmpty else { return 0 }
        return vals.reduce(0, +) / Double(vals.count)
    }

    private func groupedAvg(by key: String, metric: String) -> [(name: String, value: Double, count: Int)] {
        var groups: [String: [Double]] = [:]
        for r in results {
            guard let m = r[metric] as? Double else { continue }
            if let i = r[key] as? Int {
                groups["\(i)"] = (groups["\(i)"] ?? []) + [m]
            } else if let s = r[key] as? String {
                groups[s] = (groups[s] ?? []) + [m]
            }
        }
        var rows: [(name: String, value: Double, count: Int)] = []
        for (k, v) in groups {
            guard !v.isEmpty else { continue }
            let avg = v.reduce(0, +) / Double(v.count)
            rows.append((name: k, value: avg, count: v.count))
        }
        rows.sort { lhs, rhs in
            (Int(lhs.name) ?? .max) < (Int(rhs.name) ?? .max)
        }
        return rows
    }
}
