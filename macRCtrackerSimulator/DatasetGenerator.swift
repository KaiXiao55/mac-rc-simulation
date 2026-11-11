//
//  DatasetGenerator.swift
//  macRCtrackerSimulator
//
//  批量数据集生成器 — 生成 YOLO 训练数据
//
//  ★ 重写版 v3: 使用 Timer 驱动，完全在主线程执行
//
//  核心修复:
//    ① 不再使用 DispatchQueue.global + main.sync (死锁根源)
//    ② 改用 Timer.scheduledTimer 在主线程 RunLoop 中逐帧驱动
//    ③ 每个 Timer tick 处理一帧: 构建场景/放车/渲染/保存
//    ④ 场景组切换时重建 VirtualRaceEngine
//    ⑤ 所有 SceneKit 操作自然在主线程
//
//  Created by Manus AI on 2026/3/22.
//  Rewritten v3: 2026/3/22 — Timer 驱动，修复死锁
//

import SceneKit
import AppKit
import Foundation
import Combine

// ═══════════════════════════════════════════════════════════════
// MARK: - 数据集生成配置
// ═══════════════════════════════════════════════════════════════

struct DatasetConfig {
    var totalImages: Int = 10000
    var imageSize: Int = 640
    var outputDir: String = ""
    var trainRatio: CGFloat = 0.8
    var trackShapes: [Int] = Array(0...9)
    var sceneTypes: [Int] = [0, 1, 2, 3, 4, 5, 6, 7]
    var cameraPresets: [Int] = Array(0...8)
    var carCounts: [Int] = [1, 2, 3, 4, 5]
    var enableRampsChance: CGFloat = 0.3
    var enableOcclusionChance: CGFloat = 0.5
    var enableDriftChance: CGFloat = 0.5
    var className: String = "rc_car"

    /// 多标签模式开关 (默认 false = 兼容单标签)
    /// true = 3 类: 按距离/bbox面积分类 (rc_far / rc_mid / rc_near)
    var useMultiLabel: Bool = false

    /// 多标签类名列表 (3 类: 按距离分)
    static let multiLabelNames: [String] = [
        "rc_far",   // 0: 面积 < 0.002
        "rc_mid",   // 1: 0.002 <= 面积 < 0.01
        "rc_near"   // 2: 面积 >= 0.01
    ]
}

// ═══════════════════════════════════════════════════════════════
// MARK: - YOLO 标注
// ═══════════════════════════════════════════════════════════════

struct YOLOLabel {
    let classId: Int
    let cx: CGFloat
    let cy: CGFloat
    let w: CGFloat
    let h: CGFloat
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 场景组
// ═══════════════════════════════════════════════════════════════

private struct SceneGroup {
    let trackShape: Int
    let sceneType: Int
    let cameraPreset: Int
    let carCount: Int
    let seed: UInt64
    let enableRamps: Bool
    let enableOcclusion: Bool
    let enableDrift: Bool
    let framesPerScene: Int
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 数据集生成引擎 (Timer 驱动)
// ═══════════════════════════════════════════════════════════════

final class DatasetGenerator: ObservableObject {
    @Published var isRunning = false
    @Published var progress: CGFloat = 0
    @Published var currentImage: Int = 0
    @Published var totalImages: Int = 0
    @Published var status: String = "就绪"
    @Published var errors: [String] = []

    private var config = DatasetConfig()
    private var shouldCancel = false

    // Timer 驱动状态
    private var timer: Timer?
    private var groups: [SceneGroup] = []
    private var currentGroupIdx: Int = 0
    private var currentFrameInGroup: Int = 0
    private var globalIdx: Int = 0
    private var totalFrames: Int = 0
    private var trainCount: Int = 0
    private var errorCount: Int = 0
    private var baseDir: String = ""
    private var currentEngine: VirtualRaceEngine?

    // 新增：每帧 manifest 条目（供外部切分脚本按 group 切）
    private var manifestEntries: [[String: Any]] = []

    // 新增：完成回调（供多阶段编排使用，如 Split Ablation pool + holdout）
    var onComplete: ((Bool) -> Void)?

    // ═══════════════════════════════════════════════════════════
    // MARK: 启动生成
    // ═══════════════════════════════════════════════════════════

    func startGeneration(config: DatasetConfig) {
        guard !isRunning else {
            print("[DatasetGen] Already running, ignoring startGeneration call")
            return
        }
        self.config = config
        self.shouldCancel = false
        self.isRunning = true
        self.totalImages = config.totalImages
        self.currentImage = 0
        self.progress = 0
        self.errors = []
        self.errorCount = 0
        self.globalIdx = 0
        self.currentGroupIdx = 0
        self.currentFrameInGroup = 0
        self.currentEngine = nil
        self.manifestEntries = []

        print("═══════════════════════════════════════════════════")
        print("  DatasetGenerator.startGeneration() [v3 Timer]")
        print("  totalImages = \(config.totalImages)")
        print("  imageSize   = \(config.imageSize)")
        print("  outputDir   = \(config.outputDir.isEmpty ? "~/Desktop/rc_yolo_dataset" : config.outputDir)")
        print("═══════════════════════════════════════════════════")

        baseDir = config.outputDir.isEmpty
            ? NSHomeDirectory() + "/Desktop/rc_yolo_dataset"
            : config.outputDir

        // 1. 创建目录
        let dirs = [
            "\(baseDir)/images/train",
            "\(baseDir)/images/val",
            "\(baseDir)/labels/train",
            "\(baseDir)/labels/val",
        ]
        for dir in dirs {
            do {
                try FileManager.default.createDirectory(
                    atPath: dir,
                    withIntermediateDirectories: true
                )
                print("[DatasetGen] Created directory: \(dir)")
            } catch {
                print("[DatasetGen] ERROR creating directory \(dir): \(error)")
                status = "错误: 无法创建目录 \(dir)"
                isRunning = false
                return
            }
        }

        // 2. 生成场景组
        groups = generateSceneGroups()
        totalFrames = groups.reduce(0) { $0 + $1.framesPerScene }
        trainCount = Int(CGFloat(totalFrames) * config.trainRatio)

        print("[DatasetGen] Generated \(groups.count) scene groups, \(totalFrames) total frames")
        print("[DatasetGen] Train: \(trainCount), Val: \(totalFrames - trainCount)")

        status = "开始生成 \(totalFrames) 张图片 (\(groups.count) 个场景组)..."

        // 3. 启动 Timer (在主线程 RunLoop 中)
        //    每 0.05 秒处理一帧 (20 fps 节奏)
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            self?.processNextFrame()
        }
        // 确保 Timer 在 common 模式下运行 (避免被 UI 事件阻塞)
        if let t = timer {
            RunLoop.main.add(t, forMode: .common)
        }

        print("[DatasetGen] Timer started")
    }

    func cancel() {
        shouldCancel = true
        print("[DatasetGen] Cancel requested")
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: Timer 回调 — 逐帧处理
    // ═══════════════════════════════════════════════════════════

    private func processNextFrame() {
        // 检查是否完成或取消
        if shouldCancel || currentGroupIdx >= groups.count {
            finishGeneration()
            return
        }

        let group = groups[currentGroupIdx]

        // 如果是新场景组的第一帧，构建场景
        if currentFrameInGroup == 0 {
            let groupStatus = "场景 \(currentGroupIdx + 1)/\(groups.count): Track\(group.trackShape + 1) 场景\(group.sceneType) 机位\(group.cameraPreset) 车\(group.carCount)辆"
            status = groupStatus
            print("[DatasetGen] \(groupStatus)")

            let eng = VirtualRaceEngine()
            eng.randomSeed = group.seed
            eng.carCount = group.carCount
            eng.sceneType = group.sceneType
            eng.trackShape = group.trackShape
            eng.cameraPreset = group.cameraPreset
            eng.enableRamps = group.enableRamps
            eng.enableOcclusion = group.enableOcclusion
            eng.enableLightVariation = true
            eng.enableDrift = group.enableDrift
            eng.enableLensDistortion = false
            eng.useSimulatedYOLO = true

            eng.datasetBuildScene()

            if eng.exposedScene == nil {
                print("[DatasetGen] ERROR: 场景构建失败 for group \(currentGroupIdx)")
                errorCount += 1
                errors.append("场景组 \(currentGroupIdx): 场景构建失败")
                // 跳过整个场景组
                globalIdx += group.framesPerScene
                currentGroupIdx += 1
                currentFrameInGroup = 0
                currentEngine = nil
                return
            }

            currentEngine = eng
        }

        guard let eng = currentEngine else {
            // 不应该到这里
            currentGroupIdx += 1
            currentFrameInGroup = 0
            return
        }

        // 处理当前帧
        let fi = currentFrameInGroup
        let frameIdx = globalIdx

        // 车辆位置参数
        let baseProg = CGFloat(fi) / CGFloat(max(group.framesPerScene, 1))
        var progValues: [CGFloat] = []
        var latOffsets: [CGFloat] = []
        var slipAngles: [CGFloat] = []
        for ci in 0..<group.carCount {
            let prog = (baseProg + CGFloat(ci) / CGFloat(group.carCount))
                .truncatingRemainder(dividingBy: 1.0)
            progValues.append(prog)
            latOffsets.append(CGFloat.random(in: -0.6...0.6))
            slipAngles.append(group.enableDrift
                ? CGFloat.random(in: -0.3...0.3)
                : CGFloat.random(in: -0.05...0.05))
        }

        // 相机朝向参数
        let lookAtCar: Int
        if Double.random(in: 0...1) < 0.7 && group.carCount > 0 {
            lookAtCar = Int.random(in: 0..<group.carCount)
        } else {
            lookAtCar = -1
        }
        let panJitter = CGFloat.random(in: -0.25...0.25)
        let tiltJitter = CGFloat.random(in: -0.12...0.12)

        // 放置车辆
        eng.datasetPlaceCars(
            progressValues: progValues,
            lateralOffsets: latOffsets,
            slipAngles: slipAngles
        )

        // 设置相机朝向
        if lookAtCar >= 0 && lookAtCar < group.carCount {
            let carPos = eng.datasetCarPosition(lookAtCar)
            let target = SCNVector3(
                carPos.x,
                carPos.y + VSimCfg.carHeight * 0.3,
                carPos.z
            )
            eng.datasetSetCameraLookAt(target)
        } else {
            eng.datasetSetCameraLookAt(eng.datasetTrackCenter)
        }

        // 添加相机抖动
        eng.datasetAddCameraJitter(
            panJitter: panJitter,
            tiltJitter: tiltJitter
        )

        // 渲染 + 标注
        let imageSize = config.imageSize
        if let (cgImage, rawLabels) = eng.datasetRenderFrame(imageSize: imageSize, useMultiLabel: config.useMultiLabel) {
            let labels = rawLabels.map {
                YOLOLabel(classId: $0.classId, cx: $0.cx, cy: $0.cy, w: $0.w, h: $0.h)
            }

            let isTrain = frameIdx < trainCount
            let split = isTrain ? "train" : "val"
            let imgPath = "\(baseDir)/images/\(split)/\(String(format: "%06d", frameIdx)).jpg"
            let lblPath = "\(baseDir)/labels/\(split)/\(String(format: "%06d", frameIdx)).txt"

            saveJPEG(image: cgImage, path: imgPath, quality: 0.92)

            let labelLines = labels.map { label in
                "\(label.classId) \(String(format: "%.6f", label.cx)) \(String(format: "%.6f", label.cy)) \(String(format: "%.6f", label.w)) \(String(format: "%.6f", label.h))"
            }
            let labelContent = labelLines.joined(separator: "\n")
            do {
                try labelContent.write(
                    toFile: lblPath,
                    atomically: true,
                    encoding: .utf8
                )
            } catch {
                print("[DatasetGen] ERROR writing label \(lblPath): \(error)")
                errorCount += 1
                errors.append("帧 \(frameIdx): 写入标注失败")
            }

            if frameIdx % 10 == 0 {
                print("[DatasetGen] Frame \(frameIdx)/\(totalFrames) saved: \(labels.count) cars, split=\(split)")
            }

            // 记录 manifest 条目（供 Split Ablation 等外部切分脚本按 group 切）
            let entry: [String: Any] = [
                "frame_idx": frameIdx,
                "group_idx": currentGroupIdx,
                "frame_in_group": fi,
                "track_shape": group.trackShape,
                "scene_type": group.sceneType,
                "camera_preset": group.cameraPreset,
                "car_count": group.carCount,
                "seed": String(group.seed),
                "enable_ramps": group.enableRamps,
                "enable_occlusion": group.enableOcclusion,
                "enable_drift": group.enableDrift,
                "split": split,
                "image_relpath": "images/\(split)/\(String(format: "%06d", frameIdx)).jpg",
                "label_relpath": "labels/\(split)/\(String(format: "%06d", frameIdx)).txt",
                "num_boxes": labels.count
            ]
            manifestEntries.append(entry)
        } else {
            print("[DatasetGen] Frame \(frameIdx) render failed")
            errorCount += 1
            errors.append("帧 \(frameIdx): 渲染失败")
        }

        globalIdx += 1
        currentFrameInGroup += 1

        // 更新进度
        currentImage = globalIdx
        progress = CGFloat(globalIdx) / CGFloat(max(totalFrames, 1))

        // 检查是否需要切换到下一个场景组
        if currentFrameInGroup >= group.framesPerScene {
            currentGroupIdx += 1
            currentFrameInGroup = 0
            currentEngine = nil  // 释放引擎
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 完成
    // ═══════════════════════════════════════════════════════════

    private func finishGeneration() {
        timer?.invalidate()
        timer = nil

        // 生成 data.yaml
        writeDataYaml(baseDir: baseDir)
        print("[DatasetGen] Written data.yaml")

        // 生成统计报告
        writeReport(
            baseDir: baseDir,
            groups: groups,
            totalFrames: globalIdx,
            trainCount: trainCount,
            errorCount: errorCount
        )
        print("[DatasetGen] Written README.md")

        // 新增：写 manifest.json（每帧对应的 group_idx 等元数据）
        writeManifest(baseDir: baseDir)
        print("[DatasetGen] Written manifest.json (\(manifestEntries.count) entries)")

        let finalStatus: String
        if shouldCancel {
            finalStatus = "已取消 (已生成 \(globalIdx) 张)"
        } else {
            finalStatus = "完成! 共 \(globalIdx) 张 (\(errorCount) 个错误), 保存至 \(baseDir)"
        }

        print("═══════════════════════════════════════════════════")
        print("  DatasetGenerator FINISHED")
        print("  \(finalStatus)")
        print("═══════════════════════════════════════════════════")

        isRunning = false
        status = finalStatus

        // 新增：通知编排器（如 Split Ablation runner）本阶段已完成
        let wasCancelled = shouldCancel
        onComplete?(!wasCancelled)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 场景组生成
    // ═══════════════════════════════════════════════════════════

    private func generateSceneGroups() -> [SceneGroup] {
        var groups: [SceneGroup] = []
        let total = max(1, config.totalImages)

        let numTracks = config.trackShapes.count
        let numScenes = config.sceneTypes.count
        let numCameras = config.cameraPresets.count
        let numCombinations = numTracks * numScenes * numCameras

        guard numCombinations > 0 else {
            print("[DatasetGen] WARNING: numCombinations = 0")
            return []
        }

        func appendGroup(
            track: Int, scene: Int, cam: Int,
            frames: Int,
            seedCounter: inout UInt64
        ) {
            let carCount = config.carCounts.randomElement() ?? 3
            let enableRamps = CGFloat.random(in: 0...1) < config.enableRampsChance || scene == 3
            let enableOcclusion = CGFloat.random(in: 0...1) < config.enableOcclusionChance
            let enableDrift = CGFloat.random(in: 0...1) < config.enableDriftChance

            groups.append(SceneGroup(
                trackShape: track,
                sceneType: scene,
                cameraPreset: cam,
                carCount: carCount,
                seed: seedCounter,
                enableRamps: enableRamps,
                enableOcclusion: enableOcclusion,
                enableDrift: enableDrift,
                framesPerScene: frames
            ))
            seedCounter += 7
        }

        var seedCounter: UInt64 = 42

        if total <= numCombinations {
            var combos: [(Int, Int, Int)] = []
            for trackIdx in 0..<numTracks {
                for sceneIdx in 0..<numScenes {
                    for camIdx in 0..<numCameras {
                        combos.append((
                            config.trackShapes[trackIdx],
                            config.sceneTypes[sceneIdx],
                            config.cameraPresets[camIdx]
                        ))
                    }
                }
            }
            combos.shuffle()
            for i in 0..<total {
                let (track, scene, cam) = combos[i]
                appendGroup(track: track, scene: scene, cam: cam, frames: 1, seedCounter: &seedCounter)
            }
            groups.shuffle()
            return groups
        }

        let baseFrames = total / numCombinations
        var extra = total % numCombinations

        for trackIdx in 0..<numTracks {
            for sceneIdx in 0..<numScenes {
                for camIdx in 0..<numCameras {
                    let track = config.trackShapes[trackIdx]
                    let scene = config.sceneTypes[sceneIdx]
                    let cam = config.cameraPresets[camIdx]

                    var frames = baseFrames
                    if extra > 0 {
                        frames += 1
                        extra -= 1
                    }

                    appendGroup(track: track, scene: scene, cam: cam, frames: frames, seedCounter: &seedCounter)
                }
            }
        }

        groups.shuffle()
        return groups
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 文件 I/O
    // ═══════════════════════════════════════════════════════════

    private func saveJPEG(image: CGImage, path: String, quality: CGFloat) {
        let rep = NSBitmapImageRep(cgImage: image)
        if let jpegData = rep.representation(
            using: .jpeg,
            properties: [.compressionFactor: quality]
        ) {
            do {
                try jpegData.write(to: URL(fileURLWithPath: path))
            } catch {
                print("[DatasetGen] ERROR saving JPEG \(path): \(error)")
            }
        } else {
            print("[DatasetGen] ERROR: Failed to create JPEG data for \(path)")
        }
    }

    private func writeDataYaml(baseDir: String) {
        let yaml: String
        if config.useMultiLabel {
            // 多标签模式: 9 类 = 颜色大类(暖/冷/中性) × 距离(远/中/近)
            var namesBlock = ""
            for (idx, name) in DatasetConfig.multiLabelNames.enumerated() {
                namesBlock += "  \(idx): \(name)\n"
            }
            yaml = """
            # YOLO11n 多标签训练配置 (3 类: 按距离/面积分类)
            # 由 macRCSimulator DatasetGenerator 自动生成
            # \(Date())
            # 标签策略: rc_far(面积<0.002) / rc_mid(0.002~0.01) / rc_near(面积>=0.01)

            path: \(baseDir)
            train: images/train
            val: images/val

            nc: \(DatasetConfig.multiLabelNames.count)
            names:
            \(namesBlock)
            """
        } else {
            // 兼容模式: 单标签
            yaml = """
            # YOLO11n 训练配置
            # 由 macRCSimulator DatasetGenerator 自动生成
            # \(Date())

            path: \(baseDir)
            train: images/train
            val: images/val

            nc: 1
            names:
              0: \(config.className)
            """
        }
        do {
            try yaml.write(
                toFile: "\(baseDir)/data.yaml",
                atomically: true,
                encoding: .utf8
            )
        } catch {
            print("[DatasetGen] ERROR writing data.yaml: \(error)")
        }
    }

    private func writeReport(baseDir: String, groups: [SceneGroup],
                             totalFrames: Int, trainCount: Int, errorCount: Int) {
        let labelModeDesc = config.useMultiLabel
            ? "多标签 (9类: 颜色大类×距离)"
            : "单标签 (\(config.className))"
        var report = """
        # RC YOLO 数据集生成报告
        生成时间: \(Date())
        总图片数: \(totalFrames)
        训练集: \(trainCount)
        验证集: \(totalFrames - trainCount)
        图片尺寸: \(config.imageSize)x\(config.imageSize)
        标签模式: \(labelModeDesc)
        场景组数: \(groups.count)
        错误数: \(errorCount)

        ## 核心特性 (v3 Timer 驱动)
        - Timer 驱动主线程执行 (修复 v2 死锁)
        - VirtualRaceEngine 完整场景构建逻辑
        - 10 组手绘赛道 (Track1-10)
        - 8 种场景类型 (室内/草地/柏油/娱乐/沙地/专业/公园/开放草地)
        - 9 个相机点位
        - 精确 8 角点投影标注
        - 完整车辆模型 + 环境物体 + 赛道护栏

        ## 参数分布

        ### 赛道形状
        """
        for shape in config.trackShapes {
            let count = groups.filter { $0.trackShape == shape }
                .reduce(0) { $0 + $1.framesPerScene }
            let name = TrackShape(rawValue: shape)?.displayName ?? "Track \(shape + 1)"
            report += "\n- \(name): ~\(count) 张"
        }

        report += "\n\n### 场景类型"
        let sceneNames = [
            "室内场地", "室外草地", "室外柏油", "娱乐场地",
            "沙地越野", "专业赛道", "公园小树林", "开放草地"
        ]
        for st in config.sceneTypes {
            let count = groups.filter { $0.sceneType == st }
                .reduce(0) { $0 + $1.framesPerScene }
            let name = st < sceneNames.count ? sceneNames[st] : "场景\(st)"
            report += "\n- \(name): ~\(count) 张"
        }

        report += "\n\n### 拍摄点位"
        for cp in config.cameraPresets {
            let count = groups.filter { $0.cameraPreset == cp }
                .reduce(0) { $0 + $1.framesPerScene }
            let name = cp < VSimCfg.cameraPresets.count
                ? VSimCfg.cameraPresets[cp].name
                : "点位\(cp)"
            report += "\n- \(name): ~\(count) 张"
        }

        report += "\n\n## 使用方法\n"
        report += """
        ```bash
        # 安装 ultralytics
        pip install ultralytics

        # 训练 YOLO11n
        yolo detect train data=\(baseDir)/data.yaml model=yolo11n.pt epochs=100 imgsz=640

        # 验证
        yolo detect val data=\(baseDir)/data.yaml model=runs/detect/train/weights/best.pt
        ```
        """

        do {
            try report.write(
                toFile: "\(baseDir)/README.md",
                atomically: true,
                encoding: .utf8
            )
        } catch {
            print("[DatasetGen] ERROR writing README.md: \(error)")
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 新增 — manifest.json 写入
    // ═══════════════════════════════════════════════════════════

    private func writeManifest(baseDir: String) {
        let payload: [String: Any] = [
            "schema_version": 2,
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "base_dir": baseDir,
            "image_size": config.imageSize,
            "class_name": config.className,
            "label_mode": config.useMultiLabel ? "multi_label_\(DatasetConfig.multiLabelNames.count)" : "single_label",
            "num_classes": config.useMultiLabel ? DatasetConfig.multiLabelNames.count : 1,
            "class_names": config.useMultiLabel ? DatasetConfig.multiLabelNames : [config.className],
            "total_frames": manifestEntries.count,
            "num_groups": Set(manifestEntries.compactMap { $0["group_idx"] as? Int }).count,
            "frames": manifestEntries
        ]
        do {
            let data = try JSONSerialization.data(
                withJSONObject: payload,
                options: [.prettyPrinted, .sortedKeys]
            )
            try data.write(to: URL(fileURLWithPath: "\(baseDir)/manifest.json"))
        } catch {
            print("[DatasetGen] ERROR writing manifest.json: \(error)")
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 新增 — Split Ablation 一键编排
    //
    // 任务对应：vsim_切分方法论验证_任务书.md
    // 顺序生成 pool + holdout，两批的 (track, scene, cam) 三元组完全互斥
    // trainRatio=1.0 让所有帧落到 images/train/，由后续 Python 脚本按 manifest 重新切分
    //
    // lite=false (默认): pool = 25 group × 200 帧 = 5000, holdout = 30 group × 30 帧 = 900
    // lite=true         : pool = 25 group × 24  帧 = 600,  holdout = 30 group × 8  帧 = 240
    //                      (精简版，供快速验证用，总帧数约为 full 的 1/7)
    // ═══════════════════════════════════════════════════════════

    func runSplitAblation(baseDir: String, lite: Bool = false, useMultiLabel: Bool = false, onAllDone: (() -> Void)? = nil) {
        guard !isRunning else {
            print("[SplitAblation] 已有任务在运行，跳过")
            return
        }

        // 创建根输出目录
        do {
            try FileManager.default.createDirectory(
                atPath: baseDir,
                withIntermediateDirectories: true
            )
        } catch {
            print("[SplitAblation] ERROR 创建根目录失败: \(error)")
            return
        }

        // ── Pool 配置（5 tracks × 5 scenes × 1 cam = 25 group）──
        // lite=false: 200 帧/组, total 5000; lite=true: 24 帧/组, total 600
        var poolCfg = DatasetConfig()
        poolCfg.totalImages = lite ? 600 : 5000
        poolCfg.imageSize = 640
        poolCfg.trainRatio = 1.0
        poolCfg.outputDir = "\(baseDir)/pool"
        poolCfg.trackShapes = [0, 2, 4, 6, 8]
        poolCfg.sceneTypes = [0, 1, 2, 5, 6]
        poolCfg.cameraPresets = [0]
        poolCfg.carCounts = [1, 2, 3]
        poolCfg.enableRampsChance = 0.3
        poolCfg.enableOcclusionChance = 0.5
        poolCfg.enableDriftChance = 0.5
        poolCfg.className = "rc_car"
        poolCfg.useMultiLabel = useMultiLabel

        // ── Holdout 配置（25 tracks × 3 scenes × 2 cams = 30 group）──
        // 与 pool 在 trackShape / sceneType / cameraPreset 三个维度上全部互斥
        // lite=false: 30 帧/组, total 900; lite=true: 8 帧/组, total 240
        var holdoutCfg = DatasetConfig()
        holdoutCfg.totalImages = lite ? 240 : 900
        holdoutCfg.imageSize = 640
        holdoutCfg.trainRatio = 1.0
        holdoutCfg.outputDir = "\(baseDir)/holdout"
        holdoutCfg.trackShapes = [1, 3, 5, 7, 9]
        holdoutCfg.sceneTypes = [3, 4, 7]
        holdoutCfg.cameraPresets = [3, 6]
        holdoutCfg.carCounts = [1, 2, 3]
        holdoutCfg.enableRampsChance = 0.3
        holdoutCfg.enableOcclusionChance = 0.5
        holdoutCfg.enableDriftChance = 0.5
        holdoutCfg.className = "rc_car"
        holdoutCfg.useMultiLabel = useMultiLabel

        // 动态计算每组帧数用于日志
        let poolGroups = poolCfg.trackShapes.count * poolCfg.sceneTypes.count * poolCfg.cameraPresets.count
        let poolFramesPerGroup = poolCfg.totalImages / max(1, poolGroups)
        let holdoutGroups = holdoutCfg.trackShapes.count * holdoutCfg.sceneTypes.count * holdoutCfg.cameraPresets.count
        let holdoutFramesPerGroup = holdoutCfg.totalImages / max(1, holdoutGroups)

        // 打印计划
        print("═══════════════════════════════════════════════════")
        print("  vsim Split Ablation — 一键生成\(lite ? "（LITE 精简模式）" : "")")
        print("  根输出: \(baseDir)")
        print("  Phase 1/2: Pool   → \(poolGroups) group × \(poolFramesPerGroup) frame = \(poolCfg.totalImages)")
        print("                tracks=\(poolCfg.trackShapes) scenes=\(poolCfg.sceneTypes) cams=\(poolCfg.cameraPresets)")
        print("  Phase 2/2: Holdout→ \(holdoutGroups) group × \(holdoutFramesPerGroup) frame = \(holdoutCfg.totalImages)")
        print("                tracks=\(holdoutCfg.trackShapes) scenes=\(holdoutCfg.sceneTypes) cams=\(holdoutCfg.cameraPresets)")
        print("═══════════════════════════════════════════════════")

        // 链式回调：pool 完成后启动 holdout
        self.onComplete = { [weak self] success in
            guard let self = self else { return }
            if !success {
                print("[SplitAblation] Pool 生成失败/取消，跳过 Holdout")
                self.onComplete = nil
                return
            }
            print("[SplitAblation] ─────────────────────────────────────")
            print("[SplitAblation] Pool 完成，2 秒后启动 Holdout...")
            print("[SplitAblation] ─────────────────────────────────────")
            // 设置第二阶段完成回调
            self.onComplete = { _ in
                print("═══════════════════════════════════════════════════")
                print("  vsim Split Ablation — 全部完成")
                print("  下一步：把 \(baseDir) 整个目录打包传给 Manus")
                print("  目录结构应包含：pool/{images,labels,manifest.json},")
                print("                  holdout/{images,labels,manifest.json}")
                print("═══════════════════════════════════════════════════")
                self.onComplete = nil
                onAllDone?()
            }
            // 延迟 2 秒启动，给 SwiftUI/RunLoop 缓冲
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.startGeneration(config: holdoutCfg)
            }
        }

        // 启动第一阶段
        startGeneration(config: poolCfg)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: Scale Effect 数据集编排
    //
    // 生成 train_pool / val / eval_pos / eval_neg 四批。eval 在 track、scene、
    // camera 三个维度上均与 train/val 互斥，供 YOLO11L 数据量缩放实验使用。
    // ═══════════════════════════════════════════════════════════

    func runScaleEffectDataset(baseDir: String, onAllDone: (() -> Void)? = nil) {
        guard !isRunning else {
            print("[ScaleEffect] 已有任务在运行，跳过")
            return
        }

        do {
            try FileManager.default.createDirectory(
                atPath: baseDir,
                withIntermediateDirectories: true
            )
        } catch {
            print("[ScaleEffect] ERROR 创建根目录失败: \(error)")
            return
        }

        func cfg(
            name: String,
            total: Int,
            tracks: [Int],
            scenes: [Int],
            cameras: [Int],
            carCounts: [Int],
            rampsChance: CGFloat
        ) -> DatasetConfig {
            var c = DatasetConfig()
            c.totalImages = total
            c.imageSize = 640
            c.trainRatio = 1.0
            c.outputDir = "\(baseDir)/\(name)"
            c.trackShapes = tracks
            c.sceneTypes = scenes
            c.cameraPresets = cameras
            c.carCounts = carCounts
            c.enableRampsChance = rampsChance
            c.enableOcclusionChance = 0.55
            c.enableDriftChance = 0.55
            c.className = "rc_car"
            c.useMultiLabel = false
            return c
        }

        let trainScenes = [0, 1, 2, 5, 6]
        var phases: [(name: String, config: DatasetConfig)] = [
            ("train_pool", cfg(
                name: "train_pool",
                total: 9000,
                tracks: [0, 1, 2, 3, 4, 5],
                scenes: trainScenes,
                cameras: [0, 1, 2, 3, 4],
                carCounts: [0, 2, 3, 4, 5, 6, 7],
                rampsChance: 0.35
            )),
            ("val", cfg(
                name: "val",
                total: 1500,
                tracks: [6, 7],
                scenes: trainScenes,
                cameras: [5, 6],
                carCounts: [0, 2, 3, 4, 5, 6, 7],
                rampsChance: 0.35
            )),
            ("eval_pos", cfg(
                name: "eval_pos",
                total: 3600,
                tracks: [8, 9],
                scenes: [3, 4, 7],
                cameras: [7, 8],
                carCounts: [2, 3, 4, 5, 6, 7],
                rampsChance: 0.75
            )),
            ("eval_neg", cfg(
                name: "eval_neg",
                total: 600,
                tracks: [8, 9],
                scenes: [3, 4, 7],
                cameras: [7, 8],
                carCounts: [0],
                rampsChance: 0.75
            )),
        ]

        print("═══════════════════════════════════════════════════")
        print("  vsim YOLO Scale Effect Dataset")
        print("  根输出: \(baseDir)")
        for (idx, phase) in phases.enumerated() {
            let c = phase.config
            let groups = c.trackShapes.count * c.sceneTypes.count * c.cameraPresets.count
            let framesPerGroup = c.totalImages / max(1, groups)
            print("  Phase \(idx + 1)/\(phases.count): \(phase.name) → \(groups) group × ~\(framesPerGroup) frame = \(c.totalImages)")
            print("    tracks=\(c.trackShapes) scenes=\(c.sceneTypes) cams=\(c.cameraPresets) carCounts=\(c.carCounts)")
        }
        print("  OOD eval 与 train/val 在 track、scene、camera 维度均互斥")
        print("═══════════════════════════════════════════════════")

        func runNextPhase() {
            guard !phases.isEmpty else {
                print("═══════════════════════════════════════════════════")
                print("  vsim YOLO Scale Effect Dataset — 全部完成")
                print("  下一步：运行 build_vsim_scale_splits.py 生成 1k/2k/.../9k YOLO 清单")
                print("═══════════════════════════════════════════════════")
                self.onComplete = nil
                onAllDone?()
                return
            }

            let phase = phases.removeFirst()
            print("[ScaleEffect] 启动 \(phase.name)，剩余阶段: \(phases.map { $0.name })")
            self.onComplete = { [weak self] success in
                guard let self = self else { return }
                if !success {
                    print("[ScaleEffect] \(phase.name) 生成失败/取消，停止后续阶段")
                    self.onComplete = nil
                    return
                }
                print("[ScaleEffect] \(phase.name) 完成")
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    runNextPhase()
                }
            }
            self.startGeneration(config: phase.config)
        }

        runNextPhase()
    }
}
