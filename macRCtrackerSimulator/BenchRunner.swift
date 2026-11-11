//
//  BenchRunner.swift
//  macRCtrackerSimulator
//
//  一键 YOLO 性能评测编排器 — CLI 入口 + 批量调度
//
//  功能:
//    ① 流水线 A — 真实视频: 遍历所有 SAM3 标注场景 × 多个 YOLO 模型, 批量评测
//    ② 流水线 B — 虚拟赛道: 用 vsim_yolo11n 对合成场景做 GT 检测评测
//    ③ 可选: 跨域测试 (vsim 模型跑真实视频 / real 模型跑虚拟场景)
//    ④ 汇总输出: CSV 总表 + Markdown 报告 + 终端摘要
//
//  用法:
//    ./macRCtrackerSimulator --bench                    # 全量评测
//    ./macRCtrackerSimulator --bench --models real      # 仅真实视频模型
//    ./macRCtrackerSimulator --bench --models vsim      # 仅虚拟赛道模型
//    ./macRCtrackerSimulator --bench --scenes 草地       # 仅指定场景
//    ./macRCtrackerSimulator --bench --cross-domain     # 加入跨域测试
//    ./macRCtrackerSimulator --bench --output ~/Desktop/bench_results
//
//  Created on 2026/4/12.
//  macOS 14.0+
//

import SwiftUI
import Foundation

// ═══════════════════════════════════════════════════════════════
// MARK: - Bench 配置
// ═══════════════════════════════════════════════════════════════

struct BenchCfg {
    /// 真实视频模型列表 (会自动检测 Bundle 中是否存在)
    static let realModels = ["yolo11_640imgsz", "RC_YOLO11n_320", "Detector"]
    /// 虚拟赛道模型
    static let vsimModels = ["vsim_yolo11n"]
    /// 默认置信度阈值
    static let defaultConf: Float = 0.25
    /// 单任务超时 (秒)
    static let taskTimeout: TimeInterval = 300  // 5 分钟
    /// CSV 分隔符
    static let csvSep = ","
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 单条评测结果
// ═══════════════════════════════════════════════════════════════

struct BenchResult {
    let pipeline: String     // "real" | "vsim"
    let model: String
    let scene: String
    let precision: Double
    let recall: Double
    let f1: Double
    let mapAt50: Double
    let detectionRate: Double
    let totalTP: Int
    let totalFP: Int
    let totalFN: Int
    let totalGT: Int
    let totalFrames: Int
    let elapsed: TimeInterval
    let error: String?       // nil = 成功

    var csvLine: String {
        let fields: [String] = [
            pipeline, model, scene,
            String(format: "%.4f", precision),
            String(format: "%.4f", recall),
            String(format: "%.4f", f1),
            String(format: "%.4f", mapAt50),
            String(format: "%.4f", detectionRate),
            "\(totalTP)", "\(totalFP)", "\(totalFN)",
            "\(totalGT)", "\(totalFrames)",
            String(format: "%.1f", elapsed),
            error ?? ""
        ]
        return fields.joined(separator: BenchCfg.csvSep)
    }

    static var csvHeader: String {
        ["pipeline", "model", "scene",
         "precision", "recall", "f1", "mAP50", "det_rate",
         "TP", "FP", "FN", "GT_total", "frames",
         "elapsed_sec", "error"
        ].joined(separator: BenchCfg.csvSep)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 评测任务定义
// ═══════════════════════════════════════════════════════════════

enum BenchPipeline: String {
    case real = "real"
    case vsim = "vsim"
}

struct BenchTask {
    let pipeline: BenchPipeline
    let modelName: String
    let sceneName: String
    let videoURL: URL?         // real 模式
    let annotationURL: URL?    // real 模式
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Bench CLI 入口检测
// ═══════════════════════════════════════════════════════════════

extension RCSimApp {
    static var isBenchMode: Bool {
        CommandLine.arguments.contains("--bench")
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Bench CLI View (最小化 GUI 占位)
// ═══════════════════════════════════════════════════════════════

struct BenchCLIView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 60))
                .foregroundColor(.orange)
            Text("YOLO 一键性能评测 (Bench 模式)")
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

// ═══════════════════════════════════════════════════════════════
// MARK: - BenchRunner 核心编排器
// ═══════════════════════════════════════════════════════════════

final class BenchRunner {

    // ── 参数 ──
    private let modelFilter: String   // "all" | "real" | "vsim"
    private let sceneFilter: String   // "all" | 具体场景名
    private let crossDomain: Bool
    private let outputDir: URL
    private let confThreshold: Float

    // ── 结果 ──
    private var results: [BenchResult] = []
    private var csvURL: URL { outputDir.appendingPathComponent("bench_summary.csv") }
    private var reportURL: URL { outputDir.appendingPathComponent("bench_report.md") }

    // ═══════════════════════════════════════════════════════════
    // MARK: 初始化 (解析 CLI 参数)
    // ═══════════════════════════════════════════════════════════

    init() {
        let args = CommandLine.arguments

        self.modelFilter = Self.argValue(args, key: "--models") ?? "all"
        self.sceneFilter = Self.argValue(args, key: "--scenes") ?? "all"
        self.crossDomain = args.contains("--cross-domain")
        self.confThreshold = Self.argValue(args, key: "--conf").flatMap { Float($0) } ?? BenchCfg.defaultConf

        let outPath = Self.argValue(args, key: "--output") ?? (NSHomeDirectory() + "/Desktop/bench_results")
        self.outputDir = URL(fileURLWithPath: outPath)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 主入口
    // ═══════════════════════════════════════════════════════════

    static func run() {
        let runner = BenchRunner()
        runner.execute()
    }

    private func execute() {
        printBanner()

        // 创建输出目录
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        // ── 1. 枚举所有任务 ──
        let tasks = buildTaskList()
        guard !tasks.isEmpty else {
            print("  ⚠️  未找到可执行的评测任务")
            print("  请确认:")
            print("    - SAM3_Annotations 目录下有标注文件")
            print("    - Bundle 中包含 YOLO 模型 (.mlmodelc/.mlpackage)")
            exit(1)
        }

        print("  任务总数: \(tasks.count)")
        print("  模型筛选: \(modelFilter)")
        print("  场景筛选: \(sceneFilter)")
        print("  跨域测试: \(crossDomain ? "开启" : "关闭")")
        print("  置信度阈值: \(confThreshold)")
        print("  输出目录: \(outputDir.path)")
        print("═══════════════════════════════════════════════════\n")

        // 写入 CSV 头
        writeCSVHeader()

        // ── 2. 逐任务执行 ──
        for (idx, task) in tasks.enumerated() {
            let label = "[\(idx + 1)/\(tasks.count)]"
            print("\(label) \(task.pipeline.rawValue) | \(task.modelName) | \(task.sceneName)")

            let startTime = CFAbsoluteTimeGetCurrent()
            let result = executeTask(task)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            // 填入耗时
            let finalResult = BenchResult(
                pipeline: result.pipeline, model: result.model, scene: result.scene,
                precision: result.precision, recall: result.recall,
                f1: result.f1, mapAt50: result.mapAt50, detectionRate: result.detectionRate,
                totalTP: result.totalTP, totalFP: result.totalFP, totalFN: result.totalFN,
                totalGT: result.totalGT, totalFrames: result.totalFrames,
                elapsed: elapsed, error: result.error
            )

            results.append(finalResult)
            appendCSVLine(finalResult)

            if let err = finalResult.error {
                print("       ❌ 错误: \(err)  (\(String(format: "%.1f", elapsed))s)")
            } else {
                print(String(format: "       ✅ Recall=%.1f%%  Precision=%.1f%%  F1=%.3f  mAP50=%.3f  (%0.1fs)",
                             finalResult.recall * 100, finalResult.precision * 100,
                             finalResult.f1, finalResult.mapAt50, elapsed))
            }
        }

        // ── 3. 生成汇总 ──
        printSummary()
        writeMarkdownReport()

        print("\n  📄 CSV 报告: \(csvURL.path)")
        print("  📋 Markdown 报告: \(reportURL.path)")

        exit(0)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 构建任务列表
    // ═══════════════════════════════════════════════════════════

    private func buildTaskList() -> [BenchTask] {
        var tasks: [BenchTask] = []

        let availableModels = YOLOGTTestEngine.availableModels().map { $0.name }

        // ── 流水线 A: 真实视频 ──
        if modelFilter == "all" || modelFilter == "real" {
            let realModels = BenchCfg.realModels.filter { availableModels.contains($0) }
            let scenes = discoverRealScenes()

            for model in realModels {
                for scene in scenes {
                    if sceneFilter != "all" && scene.name != sceneFilter { continue }
                    tasks.append(BenchTask(
                        pipeline: .real, modelName: model, sceneName: scene.name,
                        videoURL: scene.videoURL, annotationURL: scene.annotationURL
                    ))
                }
            }
        }

        // ── 流水线 B: 虚拟赛道 ──
        if modelFilter == "all" || modelFilter == "vsim" {
            let vsimModels = BenchCfg.vsimModels.filter { availableModels.contains($0) }

            for model in vsimModels {
                tasks.append(BenchTask(
                    pipeline: .vsim, modelName: model, sceneName: "vsim_all_tracks",
                    videoURL: nil, annotationURL: nil
                ))
            }
        }

        // ── 跨域测试: vsim 模型 → 真实视频 ──
        if crossDomain {
            let vsimModels = BenchCfg.vsimModels.filter { availableModels.contains($0) }
            let scenes = discoverRealScenes()

            for model in vsimModels {
                for scene in scenes {
                    if sceneFilter != "all" && scene.name != sceneFilter { continue }
                    // 避免重复 (如果 modelFilter 已包含)
                    tasks.append(BenchTask(
                        pipeline: .real, modelName: model, sceneName: "cross_\(scene.name)",
                        videoURL: scene.videoURL, annotationURL: scene.annotationURL
                    ))
                }
            }
        }

        return tasks
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 发现真实视频场景
    // ═══════════════════════════════════════════════════════════

    private struct RealScene {
        let name: String
        let videoURL: URL
        let annotationURL: URL
    }

    private func discoverRealScenes() -> [RealScene] {
        var scenes: [RealScene] = []
        let fm = FileManager.default

        // 扫描 SAM3 标注根目录
        for root in GTTestCfg.sam3AnnotationRoots {
            let expandedRoot = NSString(string: root).expandingTildeInPath
            guard fm.fileExists(atPath: expandedRoot) else { continue }

            guard let subdirs = try? fm.contentsOfDirectory(atPath: expandedRoot) else { continue }

            for subdir in subdirs {
                if subdir.hasPrefix(".") { continue }
                let scenePath = (expandedRoot as NSString).appendingPathComponent(subdir)

                // 查找标注 JSON
                let correctedAnn = (scenePath as NSString).appendingPathComponent("\(subdir).corrected.annotations.json")
                let normalAnn = (scenePath as NSString).appendingPathComponent("\(subdir)_annotations.json")

                var annPath: String?
                if fm.fileExists(atPath: correctedAnn) {
                    annPath = correctedAnn
                } else if fm.fileExists(atPath: normalAnn) {
                    annPath = normalAnn
                }

                guard let finalAnnPath = annPath else { continue }

                // 查找对应视频 (在多个候选目录中搜索)
                let videoSearchDirs = [
                    NSHomeDirectory() + "/Documents/MacRCSimulator/云台仿真器测试视频0308_副本",
                    NSHomeDirectory() + "/Desktop/云台仿真器测试视频0308_副本",
                    (expandedRoot as NSString).deletingLastPathComponent
                ]

                var videoURL: URL?
                let videoExts = ["mp4", "mov", "MP4", "MOV"]
                for dir in videoSearchDirs {
                    for ext in videoExts {
                        let candidate = (dir as NSString).appendingPathComponent("\(subdir).\(ext)")
                        if fm.fileExists(atPath: candidate) {
                            videoURL = URL(fileURLWithPath: candidate)
                            break
                        }
                    }
                    if videoURL != nil { break }
                }

                guard let vURL = videoURL else {
                    print("  ⚠️  场景 \(subdir): 找到标注但未找到视频, 跳过")
                    continue
                }

                scenes.append(RealScene(
                    name: subdir,
                    videoURL: vURL,
                    annotationURL: URL(fileURLWithPath: finalAnnPath)
                ))
            }
        }

        return scenes.sorted { $0.name < $1.name }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 执行单条任务
    // ═══════════════════════════════════════════════════════════

    private func executeTask(_ task: BenchTask) -> BenchResult {
        let engine = YOLOGTTestEngine()
        engine.confThreshold = confThreshold

        // 加载模型
        guard engine.loadBundledModel(name: task.modelName) else {
            return errorResult(task, msg: "模型加载失败: \(task.modelName)")
        }

        // 预热等待
        Thread.sleep(forTimeInterval: 0.5)

        let semaphore = DispatchSemaphore(value: 0)
        var timedOut = false

        // ── 启动对应流水线 ──
        switch task.pipeline {
        case .real:
            guard let vURL = task.videoURL, let aURL = task.annotationURL else {
                return errorResult(task, msg: "视频或标注 URL 缺失")
            }
            engine.runVideoGTTest(videoURL: vURL, annotationURL: aURL, keepImages: false)

        case .vsim:
            engine.runVSimGTTest(keepImages: false)
        }

        // 等待完成 (带超时)
        DispatchQueue.global().async {
            while engine.running {
                Thread.sleep(forTimeInterval: 0.5)
            }
            semaphore.signal()
        }

        // RunLoop 驱动 (某些 Vision / SceneKit 操作需要 RunLoop)
        let deadline = Date(timeIntervalSinceNow: BenchCfg.taskTimeout)
        while semaphore.wait(timeout: .now()) == .timedOut {
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
            if Date() > deadline {
                timedOut = true
                break
            }
        }

        if timedOut {
            return errorResult(task, msg: "TIMEOUT (超过 \(Int(BenchCfg.taskTimeout))s)")
        }

        // ── 提取结果 ──
        guard let s = engine.summary else {
            return errorResult(task, msg: "未产生结果")
        }

        return BenchResult(
            pipeline: task.pipeline.rawValue,
            model: task.modelName,
            scene: task.sceneName,
            precision: s.precision,
            recall: s.recall,
            f1: s.f1,
            mapAt50: s.mapAt50,
            detectionRate: s.detectionRate,
            totalTP: s.totalTP,
            totalFP: s.totalFP,
            totalFN: s.totalFN,
            totalGT: s.totalGT,
            totalFrames: s.totalFrames,
            elapsed: 0, // 外层会填入
            error: nil
        )
    }

    private func errorResult(_ task: BenchTask, msg: String) -> BenchResult {
        BenchResult(
            pipeline: task.pipeline.rawValue,
            model: task.modelName,
            scene: task.sceneName,
            precision: 0, recall: 0, f1: 0, mapAt50: 0, detectionRate: 0,
            totalTP: 0, totalFP: 0, totalFN: 0,
            totalGT: 0, totalFrames: 0, elapsed: 0, error: msg
        )
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: CSV 输出
    // ═══════════════════════════════════════════════════════════

    private func writeCSVHeader() {
        let header = BenchResult.csvHeader + "\n"
        try? header.write(to: csvURL, atomically: true, encoding: .utf8)
    }

    private func appendCSVLine(_ result: BenchResult) {
        let line = result.csvLine + "\n"
        if let handle = try? FileHandle(forWritingTo: csvURL) {
            handle.seekToEndOfFile()
            if let data = line.data(using: .utf8) {
                handle.write(data)
            }
            handle.closeFile()
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 终端汇总
    // ═══════════════════════════════════════════════════════════

    private func printSummary() {
        let ok = results.filter { $0.error == nil }
        let failed = results.filter { $0.error != nil }

        print("\n═══════════════════════════════════════════════════")
        print("  Bench 完成 — 共 \(results.count) 项, 成功 \(ok.count), 失败 \(failed.count)")
        print("═══════════════════════════════════════════════════\n")

        // 按 pipeline + model 分组统计
        let grouped = Dictionary(grouping: ok) { "\($0.pipeline)|\($0.model)" }

        for key in grouped.keys.sorted() {
            let items = grouped[key]!
            let parts = key.split(separator: "|")
            let pipeline = parts[0]
            let model = parts[1]

            let avgRecall = items.map(\.recall).reduce(0, +) / Double(items.count)
            let avgPrecision = items.map(\.precision).reduce(0, +) / Double(items.count)
            let avgF1 = items.map(\.f1).reduce(0, +) / Double(items.count)
            let avgMAP = items.map(\.mapAt50).reduce(0, +) / Double(items.count)
            let avgDetRate = items.map(\.detectionRate).reduce(0, +) / Double(items.count)

            print(String(format: "  [%@] %@ (%d 场景)", String(pipeline), String(model), items.count))
            print(String(format: "    Recall=%.1f%%  Precision=%.1f%%  F1=%.3f  mAP50=%.3f  DetRate=%.1f%%",
                         avgRecall * 100, avgPrecision * 100, avgF1, avgMAP, avgDetRate * 100))

            // 最佳 / 最差场景
            if items.count > 1 {
                let best = items.max(by: { $0.recall < $1.recall })!
                let worst = items.min(by: { $0.recall < $1.recall })!
                print(String(format: "    最佳: %@ (Recall=%.1f%%)", best.scene, best.recall * 100))
                print(String(format: "    最差: %@ (Recall=%.1f%%)", worst.scene, worst.recall * 100))
            }
            print()
        }

        if !failed.isEmpty {
            print("  失败任务:")
            for f in failed {
                print("    - \(f.pipeline)|\(f.model)|\(f.scene): \(f.error ?? "unknown")")
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: Markdown 报告
    // ═══════════════════════════════════════════════════════════

    private func writeMarkdownReport() {
        var md: [String] = []
        let ok = results.filter { $0.error == nil }

        md.append("# YOLO 一键性能评测报告")
        md.append("")
        md.append("- 评测时间: \(ISO8601DateFormatter().string(from: Date()))")
        md.append("- 置信度阈值: \(String(format: "%.2f", confThreshold))")
        md.append("- 任务总数: \(results.count) (成功: \(ok.count))")
        md.append("")

        // 按模型分组
        md.append("## 模型汇总")
        md.append("")
        md.append("| Pipeline | Model | Scenes | Avg Recall | Avg Precision | Avg F1 | Avg mAP50 | Avg DetRate |")
        md.append("|----------|-------|--------|------------|---------------|--------|-----------|-------------|")

        let grouped = Dictionary(grouping: ok) { "\($0.pipeline)|\($0.model)" }
        for key in grouped.keys.sorted() {
            let items = grouped[key]!
            let parts = key.split(separator: "|")
            let avgR = items.map(\.recall).reduce(0, +) / Double(items.count)
            let avgP = items.map(\.precision).reduce(0, +) / Double(items.count)
            let avgF = items.map(\.f1).reduce(0, +) / Double(items.count)
            let avgM = items.map(\.mapAt50).reduce(0, +) / Double(items.count)
            let avgD = items.map(\.detectionRate).reduce(0, +) / Double(items.count)

            md.append(String(format: "| %@ | %@ | %d | %.1f%% | %.1f%% | %.3f | %.3f | %.1f%% |",
                             String(parts[0]), String(parts[1]), items.count,
                             avgR * 100, avgP * 100, avgF, avgM, avgD * 100))
        }

        // 逐任务明细
        md.append("")
        md.append("## 逐任务明细")
        md.append("")
        md.append("| Pipeline | Model | Scene | Precision | Recall | F1 | mAP50 | DetRate | TP | FP | FN | GT | Frames | Time |")
        md.append("|----------|-------|-------|-----------|--------|----|-------|---------|----|----|----|----|--------|------|")

        for r in results {
            if let err = r.error {
                md.append("| \(r.pipeline) | \(r.model) | \(r.scene) | - | - | - | - | - | - | - | - | - | - | ❌ \(err) |")
            } else {
                md.append(String(format: "| %@ | %@ | %@ | %.3f | %.3f | %.3f | %.3f | %.1f%% | %d | %d | %d | %d | %d | %.1fs |",
                                 r.pipeline, r.model, r.scene,
                                 r.precision, r.recall, r.f1, r.mapAt50, r.detectionRate * 100,
                                 r.totalTP, r.totalFP, r.totalFN, r.totalGT, r.totalFrames, r.elapsed))
            }
        }

        // 场景排名
        let realResults = ok.filter { $0.pipeline == "real" }
        if !realResults.isEmpty {
            md.append("")
            md.append("## 场景排名 (按 Recall)")
            md.append("")
            let sorted = realResults.sorted { $0.recall > $1.recall }
            md.append("| # | Scene | Model | Recall | F1 | mAP50 |")
            md.append("|---|-------|-------|--------|----|-------|")
            for (i, r) in sorted.enumerated() {
                md.append(String(format: "| %d | %@ | %@ | %.1f%% | %.3f | %.3f |",
                                 i + 1, r.scene, r.model, r.recall * 100, r.f1, r.mapAt50))
            }
        }

        let text = md.joined(separator: "\n")
        try? text.write(to: reportURL, atomically: true, encoding: .utf8)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: Banner
    // ═══════════════════════════════════════════════════════════

    private func printBanner() {
        print("")
        print("═══════════════════════════════════════════════════")
        print("  🏎️  YOLO 一键性能评测 (Bench Mode)")
        print("═══════════════════════════════════════════════════")
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 工具
    // ═══════════════════════════════════════════════════════════

    private static func argValue(_ args: [String], key: String) -> String? {
        guard let idx = args.firstIndex(of: key), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }
}
