//
//  DatasetGeneratorUI.swift
//  macRCtrackerSimulator
//
//  数据集生成器 UI — 配置参数 + 进度显示 + 一键生成
//
//  Created by Manus AI on 2026/3/22.
//

import SwiftUI

struct DatasetGeneratorView: View {
    @StateObject private var generator = DatasetGenerator()

    // ── 配置参数 ──
    @State private var totalImages: Double = 100
    @State private var imageSize: Int = 640
    @State private var trainRatio: Double = 0.8
    @State private var outputDir: String = ""
    @State private var enableAllTracks = true
    @State private var enableAllScenes = true
    @State private var enableAllPresets = true
    @State private var useMultiLabel = false

    var body: some View {
        VStack(spacing: 0) {
            // ── 标题栏 ──
            headerBar

            HSplitView {
                // ── 左侧: 配置面板 ──
                configPanel
                    .frame(minWidth: 280, maxWidth: 350)

                // ── 右侧: 进度和状态 ──
                progressPanel
                    .frame(minWidth: 400)
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 标题栏
    // ═══════════════════════════════════════════════════════════

    private var headerBar: some View {
        HStack {
            Image(systemName: "photo.stack.fill")
                .foregroundColor(.blue)
            Text("YOLO 数据集生成器")
                .font(.headline)
            Spacer()
            Text("RC 赛车检测训练数据")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 配置面板
    // ═══════════════════════════════════════════════════════════

    private var configPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // 数量
                GroupBox("生成数量") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("总图片数")
                                .font(.caption)
                            Spacer()
                            Text("\(Int(totalImages))")
                                .font(.caption.monospaced())
                                .foregroundColor(.blue)
                        }
                        Slider(value: $totalImages, in: 100...50000, step: 100)
                        HStack {
                            Text("训练/验证比例")
                                .font(.caption)
                            Spacer()
                            Text("\(Int(trainRatio * 100))% / \(Int((1 - trainRatio) * 100))%")
                                .font(.caption.monospaced())
                        }
                        Slider(value: $trainRatio, in: 0.5...0.95, step: 0.05)
                    }
                    .padding(4)
                }

                // 图片尺寸
                GroupBox("图片参数") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("图片尺寸").font(.caption)
                            Spacer()
                            Picker("", selection: $imageSize) {
                                Text("320x320").tag(320)
                                Text("416x416").tag(416)
                                Text("640x640").tag(640)
                                Text("1280x1280").tag(1280)
                            }
                            .pickerStyle(.menu)
                            .frame(width: 120)
                        }
                    }
                    .padding(4)
                }

                // 赛道
                GroupBox("赛道布局") {
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("全部10组赛道", isOn: $enableAllTracks)
                            .font(.caption)
                        if enableAllTracks {
                            Text("Track 1~10 (各种弯道布局)")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .lineLimit(3)
                        }
                    }
                    .padding(4)
                }

                // 场景
                GroupBox("场景类型") {
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("全部8种场景", isOn: $enableAllScenes)
                            .font(.caption)
                        Text("室内场地 / 室外草地 / 室外柏油 / 娱乐场地 / 沙地越野 / 专业赛道 / 公园小树林 / 开放草地")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(3)
                    }
                    .padding(4)
                }

                // 拍摄点位
                GroupBox("拍摄点位") {
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("全部9种点位", isOn: $enableAllPresets)
                            .font(.caption)
                        Text("包络外-低位近 / 中位 / 高位看台 / 中位看台 / 弯道角 / 直道侧 / 手持低 / 手持中 / 赛道中间俯瞰")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(3)
                    }
                    .padding(4)
                }

                // 标签模式
                GroupBox("标签模式") {
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("多标签模式 (3类)", isOn: $useMultiLabel)
                            .font(.caption)
                        if useMultiLabel {
                            Text("按距离分类: rc_far / rc_mid / rc_near")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        } else {
                            Text("单标签: rc_car (兼容模式)")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(4)
                }

                // 输出目录
                GroupBox("输出目录") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(outputDir.isEmpty ? "~/Desktop/rc_yolo_dataset" : outputDir)
                            .font(.caption.monospaced())
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Button("选择目录...") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = false
                            panel.canChooseDirectories = true
                            panel.canCreateDirectories = true
                            panel.message = "选择数据集输出目录"
                            if panel.runModal() == .OK, let url = panel.url {
                                outputDir = url.path
                            }
                        }
                        .font(.caption)
                    }
                    .padding(4)
                }

                Spacer()
            }
            .padding()
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 进度面板
    // ═══════════════════════════════════════════════════════════

    private var progressPanel: some View {
        VStack(spacing: 16) {
            Spacer()

            // 大数字显示
            VStack(spacing: 4) {
                Text("\(generator.currentImage)")
                    .font(.system(size: 64, weight: .bold, design: .monospaced))
                    .foregroundColor(.blue)
                Text("/ \(generator.totalImages)")
                    .font(.title3.monospaced())
                    .foregroundColor(.secondary)
            }

            // 进度条
            ProgressView(value: generator.progress)
                .progressViewStyle(.linear)
                .frame(maxWidth: 300)

            // 百分比
            Text(String(format: "%.1f%%", generator.progress * 100))
                .font(.headline.monospaced())

            // 状态
            Text(generator.status)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)
                .multilineTextAlignment(.center)

            // 错误
            if !generator.errors.isEmpty {
                Text("错误: \(generator.errors.count) 个")
                    .font(.caption)
                    .foregroundColor(.red)
            }

            Spacer()

            // 操作按钮
            HStack(spacing: 16) {
                if generator.isRunning {
                    Button(action: { generator.cancel() }) {
                        Label("取消", systemImage: "xmark.circle")
                            .frame(width: 120)
                    }
                    .controlSize(.large)
                    .buttonStyle(.bordered)
                    .tint(.red)
                } else {
                    Button(action: startGeneration) {
                        Label("开始生成", systemImage: "play.fill")
                            .frame(width: 160)
                    }
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)

                    if generator.currentImage > 0 {
                        Button(action: openOutputDir) {
                            Label("打开目录", systemImage: "folder")
                        }
                        .controlSize(.large)
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding(.bottom, 20)
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 操作
    // ═══════════════════════════════════════════════════════════

    private func startGeneration() {
        var cfg = DatasetConfig()
        cfg.totalImages = Int(totalImages)
        cfg.imageSize = imageSize
        cfg.trainRatio = trainRatio
        cfg.outputDir = outputDir
        cfg.trackShapes = enableAllTracks ? Array(0...9) : [0, 1, 2]
        cfg.sceneTypes = enableAllScenes ? [0, 1, 2, 3, 4, 5, 6, 7] : [0, 1, 2]
        cfg.cameraPresets = enableAllPresets ? Array(0...8) : [0, 1, 2]
        cfg.useMultiLabel = useMultiLabel
        generator.startGeneration(config: cfg)
    }

    private func openOutputDir() {
        let dir = outputDir.isEmpty ? NSHomeDirectory() + "/Desktop/rc_yolo_dataset" : outputDir
        NSWorkspace.shared.open(URL(fileURLWithPath: dir))
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 命令行模式视图 (--generate-dataset)
// ═══════════════════════════════════════════════════════════════

struct DatasetCLIView: View {
    // ★ 修复: 使用自己的 StateObject，通过 onAppear 从 AppDelegate 获取
    @StateObject private var generator = DatasetGenerator()
    @State private var linkedToDelegate = false

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "photo.stack.fill")
                .font(.system(size: 48))
                .foregroundColor(.blue)

            Text(generator.isRunning ? "数据集生成中..." : (generator.currentImage > 0 ? "生成完成!" : "准备中..."))
                .font(.title2.bold())

            Text(generator.status)
                .font(.body.monospaced())
                .foregroundColor(.secondary)
                .lineLimit(3)

            ProgressView(value: generator.progress)
                .progressViewStyle(.linear)
                .frame(width: 400)

            Text("\(generator.currentImage) / \(generator.totalImages)")
                .font(.headline.monospaced())
                .foregroundColor(.blue)

            if !generator.errors.isEmpty {
                Text("错误: \(generator.errors.count) 个")
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
        .frame(width: 600, height: 400)
        .padding()
        .onAppear {
            // ★ 修复: 在 onAppear 中启动生成，确保 RunLoop 已就绪
            if !linkedToDelegate {
                linkedToDelegate = true
                print("[DatasetCLIView] onAppear — scheduling generation start")
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    startCLIGeneration()
                }
            }
        }
        .onChange(of: generator.isRunning) { running in
            if !running && generator.currentImage > 0 {
                print("═══════════════════════════════════════════════════")
                print("数据集生成完成: \(generator.currentImage) 张图片")
                print("═══════════════════════════════════════════════════")
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    NSApplication.shared.terminate(nil)
                }
            }
        }
    }

    private func startCLIGeneration() {
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

        // 默认启用全部赛道/场景/点位
        cfg.trackShapes = Array(0...9)
        cfg.sceneTypes = [0, 1, 2, 3, 4, 5, 6, 7]
        cfg.cameraPresets = Array(0...8)

        print("[DatasetCLIView] Starting generation: \(cfg.totalImages) images, size=\(cfg.imageSize), output=\(cfg.outputDir), multiLabel=\(cfg.useMultiLabel)")
        generator.startGeneration(config: cfg)
    }
}
