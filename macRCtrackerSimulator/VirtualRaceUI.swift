//
//  VirtualRaceUI.swift
//  macRCtrackerSimulator
//
//  虚拟 3D 仿真环境 — SwiftUI 界面
//
//  功能:
//    ① 参数配置面板 (车辆数、仿真时长、相机高度、随机种子等)
//    ② 实时渲染画面 + GT 叠加层 + 检测框叠加层
//    ③ 统计指标面板 (追踪率、正确追踪率、遮挡率、IoU、评分)
//    ④ 仿真结论弹窗
//    ⑤ CSV 导出
//
//  Created by Manus AI on 2026/3/17.
//

import SwiftUI
import AppKit
import SceneKit

// ═══════════════════════════════════════════════════════════════
// MARK: - 虚拟仿真主视图
// ═══════════════════════════════════════════════════════════════

struct VirtualRaceView: View {
    @StateObject private var engine = VirtualRaceEngine()
    @State private var showResult = false
    @State private var showCSVExport = false
    @State private var simKnobHelpExpanded = true

    var body: some View {
        HSplitView {
            // ═══ 左: 画面 + 统计 ═══
            VStack(spacing: 0) {
                // 渲染画面
                ZStack {
                    if engine.freeCamera, let sc = engine.exposedScene {
                        // ═══ 3D 自由相机模式: SCNView 支持旋转/缩放/平移 ═══
                        FreeCameraSceneView(scene: sc,
                                            centerX: engine.freeCamCenterX,
                                            centerZ: engine.freeCamCenterZ,
                                            radius: engine.freeCamRadius)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else if let snap = engine.snap, let img = snap.renderedImage {
                        GeometryReader { geo in
                            let imgSize = CGSize(width: img.width, height: img.height)
                            let fitRect = fitRect(imgSize: imgSize, viewSize: geo.size)

                            // 渲染图
                            Image(decorative: img, scale: 1)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: fitRect.width, height: fitRect.height)
                                .position(x: fitRect.midX, y: fitRect.midY)

                            // GT 叠加层
                            ForEach(0..<snap.gtRects.count, id: \.self) { i in
                                let r = snap.gtRects[i]
                                let screenR = CGRect(
                                    x: fitRect.minX + r.minX * fitRect.width,
                                    y: fitRect.minY + r.minY * fitRect.height,
                                    width: r.width * fitRect.width,
                                    height: r.height * fitRect.height
                                )
                                Rectangle()
                                    .stroke(Color.green.opacity(0.5), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                                    .frame(width: screenR.width, height: screenR.height)
                                    .position(x: screenR.midX, y: screenR.midY)
                            }

                            // 主目标 GT
                            if let gt = snap.gtPrimaryRect {
                                let screenR = CGRect(
                                    x: fitRect.minX + gt.minX * fitRect.width,
                                    y: fitRect.minY + gt.minY * fitRect.height,
                                    width: gt.width * fitRect.width,
                                    height: gt.height * fitRect.height
                                )
                                Rectangle()
                                    .stroke(snap.gtPrimaryOccluded ? Color.orange : Color.green,
                                            lineWidth: 2)
                                    .frame(width: screenR.width, height: screenR.height)
                                    .position(x: screenR.midX, y: screenR.midY)
                                Text("GT" + (snap.gtPrimaryOccluded ? " (遮挡)" : ""))
                                    .font(.caption2.bold())
                                    .foregroundColor(.green)
                                    .position(x: screenR.midX, y: screenR.minY - 8)
                            }

                            // 检测框
                            if let det = snap.detectedRect {
                                let screenR = CGRect(
                                    x: fitRect.minX + det.minX * fitRect.width,
                                    y: fitRect.minY + det.minY * fitRect.height,
                                    width: det.width * fitRect.width,
                                    height: det.height * fitRect.height
                                )
                                Rectangle()
                                    .stroke(Color.cyan, lineWidth: 2)
                                    .frame(width: screenR.width, height: screenR.height)
                                    .position(x: screenR.midX, y: screenR.midY)
                                Text(snap.detSource)
                                    .font(.caption2.bold())
                                    .foregroundColor(.cyan)
                                    .position(x: screenR.midX, y: screenR.maxY + 8)
                            }

                            // 模拟 YOLO 误检框（FP）：橙虚线，与主追踪框区分
                            ForEach(Array(snap.simFalsePositiveRects.enumerated()), id: \.offset) { _, fp in
                                let screenR = CGRect(
                                    x: fitRect.minX + fp.minX * fitRect.width,
                                    y: fitRect.minY + fp.minY * fitRect.height,
                                    width: fp.width * fitRect.width,
                                    height: fp.height * fitRect.height
                                )
                                Rectangle()
                                    .strokeBorder(Color.orange, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                                    .frame(width: screenR.width, height: screenR.height)
                                    .position(x: screenR.midX, y: screenR.midY)
                                Text("FP误检")
                                    .font(.caption2.bold())
                                    .foregroundColor(.orange)
                                    .position(x: screenR.midX, y: screenR.maxY + 8)
                            }

                            // 状态标签
                            HStack(spacing: 8) {
                                Text(snap.state)
                                    .font(.caption.bold().monospaced())
                                    .foregroundColor(stateColor(snap.state))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.black.opacity(0.7))
                                    .cornerRadius(4)

                                Text(String(format: "F%d  %.1fs", snap.frameIdx, snap.time))
                                    .font(.caption2.monospaced())
                                    .foregroundColor(.white.opacity(0.7))
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 2)
                                    .background(Color.black.opacity(0.5))
                                    .cornerRadius(3)
                            }
                            .position(x: fitRect.minX + 80, y: fitRect.minY + 15)
                        }
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "car.2.fill")
                                .font(.system(size: 48))
                                .foregroundColor(.gray)
                            Text("虚拟 3D 赛车仿真")
                                .font(.title3)
                                .foregroundColor(.gray)
                            Text("点击右侧 Generate & Run 开始")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // 3D 浏览模式切换按钮 (右上角悬浮)
                    if engine.exposedScene != nil {
                        VStack {
                            HStack {
                                Spacer()
                                Button(action: {
                                    engine.freeCamera.toggle()
                                }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: engine.freeCamera ? "video.fill" : "rotate.3d")
                                        Text(engine.freeCamera ? "仿真视角" : "3D 浏览")
                                    }
                                    .font(.caption.bold())
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 5)
                                    .background(engine.freeCamera ? Color.orange.opacity(0.85) : Color.accentColor.opacity(0.85))
                                    .foregroundColor(.white)
                                    .cornerRadius(6)
                                }
                                .buttonStyle(.plain)
                                .padding(8)
                            }
                            Spacer()
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
                .clipped()

                // 进度条
                VStack(spacing: 2) {
                    ProgressView(value: engine.progress)
                        .progressViewStyle(.linear)
                    HStack {
                        Text(engine.status)
                            .font(.caption2.monospaced())
                            .lineLimit(1)
                        Spacer()
                        if engine.running {
                            Circle()
                                .fill(engine.paused ? Color.yellow : Color.red)
                                .frame(width: 8, height: 8)
                            Text(engine.paused ? "PAUSED" : "RUN")
                                .font(.caption.bold())
                        }
                    }
                    .foregroundColor(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(Color(white: 0.08))

                // 统计指标
                VSimStatsBar(stats: engine.stats)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background(Color(nsColor: .controlBackgroundColor))
            }
            .frame(minWidth: 700)

            // ═══ 右: 控制面板 ═══
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    // ── 场景配置 ──
                                        // ══ 策略切换 ══
                    GroupBox("★ 控制策略 Strategy") {
                        VStack(alignment: .leading, spacing: 8) {
                            Picker("策略", selection: $engine.useNeuralNetwork) {
                                Text("MLP 神经网络").tag(true)
                                Text("PID 规则控制").tag(false)
                            }
                            .pickerStyle(.segmented)
                            HStack {
                                Circle()
                                    .fill(engine.useNeuralNetwork ? Color.blue : Color.orange)
                                    .frame(width: 8, height: 8)
                                Text(engine.useNeuralNetwork ? "MLP: GimbalMLP.mlpackage 推理" : "PID: 规则控制 (Kp/Ki/Kd)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(4)
                    }

                    GroupBox("场景配置 Scene") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("车辆数量").font(.caption)
                                Spacer()
                                Stepper("\(engine.carCount)", value: $engine.carCount, in: 1...10)
                                    .font(.caption.monospaced())
                            }
                            HStack {
                                Text("仿真时长(秒)").font(.caption)
                                Spacer()
                                TextField("", value: $engine.simDuration, format: .number)
                                    .frame(width: 60)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.caption.monospaced())
                            }
                            HStack {
                                Text("相机高度(米)").font(.caption)
                                Spacer()
                                TextField("", value: Binding(
                                    get: { Double(engine.cameraHeight) },
                                    set: { engine.cameraHeight = CGFloat($0) }
                                ), format: .number)
                                    .frame(width: 60)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.caption.monospaced())
                            }
                            HStack {
                                Text("随机种子").font(.caption)
                                Spacer()
                                TextField("", value: $engine.randomSeed, format: .number)
                                    .frame(width: 80)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.caption.monospaced())
                                Button("🎲") {
                                    engine.randomSeed = UInt64.random(in: 0...9999)
                                }
                                .font(.caption)
                            }
                            // 场景类型选择
                            HStack {
                                Text("场景类型").font(.caption)
                                Spacer()
                                Picker("", selection: $engine.sceneType) {
                                    Text("室内场地").tag(0)
                                    Text("室外草地").tag(1)
                                    Text("室外柏油").tag(2)
                                    Text("娱乐场地").tag(3)
                                    Text("沙地越野").tag(4)
                                    Text("专业赛道").tag(5)
                                    Text("公园小树林").tag(6)
                                    Text("开放草地").tag(7)
                                }
                                .pickerStyle(.menu)
                                .frame(width: 120)
                            }
                            // 无赛道场景提示
                            if engine.sceneType == 6 || engine.sceneType == 7 {
                                Text("⚠ 无赛道场景: 车辆沿虚拟路径行驶, 赛道不可见")
                                    .font(.caption2)
                                    .foregroundColor(.orange)
                            }
                            // ── RC赛道选择 (10组预设紧凑赛道) ──
                            HStack {
                                Text("赛道布局").font(.caption)
                                Spacer()
                                Picker("", selection: $engine.trackShape) {
                                    Text("Track 1 (24x20m)").tag(0)
                                    Text("Track 2 (24x20m)").tag(1)
                                    Text("Track 3 (24x20m)").tag(2)
                                    Text("Track 4 (24x20m)").tag(3)
                                    Text("Track 5 (24x20m)").tag(4)
                                    Text("Track 6 (24x20m)").tag(5)
                                    Text("Track 7 (24x20m)").tag(6)
                                    Text("Track 8 (24x20m)").tag(7)
                                    Text("Track 9 (24x20m)").tag(8)
                                    Text("Track 10 (24x20m)").tag(9)
                                }
                                .pickerStyle(.menu)
                                .frame(width: 180)
                            }
                            if let shape = TrackShape(rawValue: engine.trackShape) {
                                let fs = shape.fieldSize
                                Text("\(shape.displayName): \(String(format:"%.0f",fs.0))x\(String(format:"%.0f",fs.1))m, 路宽\(String(format:"%.1f",shape.recommendedWidth))m")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            Divider()
                            // 拍摄点位预设
                            HStack {
                                Text("拍摄点位").font(.caption)
                                Spacer()
                                Picker("", selection: $engine.cameraPreset) {
                                    Text("赛道边低位").tag(0)
                                    Text("赛道边中位").tag(1)
                                    Text("SRC看台高位").tag(2)
                                    Text("SRC看台中位").tag(3)
                                    Text("弯道外侧").tag(4)
                                    Text("直道侧面").tag(5)
                                    Text("手持跟拍低位").tag(6)
                                    Text("手持跟拍中位").tag(7)
                                    Text("高台俯瞰").tag(8)
                                }
                                .pickerStyle(.menu)
                                .frame(width: 120)
                            }
                            // 点位参数显示
                            if engine.cameraPreset < VSimCfg.cameraPresets.count {
                                let p = VSimCfg.cameraPresets[engine.cameraPreset]
                                Text("\(p.name): H=\(String(format:"%.1f", p.height))m, 俯角=\(String(format:"%.0f", p.tiltDeg))°")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            Toggle("启用飞坡", isOn: $engine.enableRamps)
                                .toggleStyle(.switch)
                                .font(.caption)
                                .help("赛道上生成飞坡坡道，车辆可飞起和着陆")
                            Toggle("启用遮挡物", isOn: $engine.enableOcclusion)
                                .toggleStyle(.switch)
                                .font(.caption)
                            Toggle("光照随机化", isOn: $engine.enableLightVariation)
                                .toggleStyle(.switch)
                                .font(.caption)
                            Toggle("赛道模式 PID", isOn: $engine.racingMode)
                                .toggleStyle(.switch)
                                .font(.caption)
                            Toggle("镜头畸变 (Barrel)", isOn: $engine.enableLensDistortion)
                                .toggleStyle(.switch)
                                .font(.caption)
                                .help("模拟运动相机的桶形畸变效果")
                            Toggle("车辆漂移 (Drift)", isOn: $engine.enableDrift)
                                .toggleStyle(.switch)
                                .font(.caption)
                                .help("过弯时模拟RC赛车侧滑漂移")
                            Toggle("自动变焦 (Zoom)", isOn: $engine.enableZoom)
                                .toggleStyle(.switch)
                                .font(.caption)
                                .help("对齐 iOS AutoZoomEngine，动态调整光学变焦保持目标占比")
                            Divider()
                            Toggle("模拟YOLO (SimYOLO)", isOn: $engine.useSimulatedYOLO)
                                .toggleStyle(.switch)
                                .font(.caption)
                                .help("用GT真实位置+噪声/置信度变化代替真实YOLO推理，无需加载模型")
                            if engine.useSimulatedYOLO {
                                Text("⚠ 模拟YOLO: 用GT+噪声代替真实检测, 距离/遮挡/尺寸影响置信度")
                                    .font(.caption2)
                                    .foregroundColor(.orange)
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text("设定召回率")
                                            .font(.caption)
                                        Slider(value: Binding(
                                            get: { Double(engine.simulatedRecallRate) },
                                            set: { engine.simulatedRecallRate = Float($0) }
                                        ), in: 0...1.0, step: 0.01)
                                        Text(String(format: "%.0f%%", engine.simulatedRecallRate * 100))
                                            .font(.caption.monospaced())
                                            .frame(width: 40, alignment: .trailing)
                                    }
                                    HStack {
                                        Text("设定误检率")
                                            .font(.caption)
                                        Slider(value: Binding(
                                            get: { Double(engine.simulatedFPRate) },
                                            set: { engine.simulatedFPRate = Float($0) }
                                        ), in: 0...1.0, step: 0.01)
                                        Text(String(format: "%.1f%%", engine.simulatedFPRate * 100))
                                            .font(.caption.monospaced())
                                            .frame(width: 40, alignment: .trailing)
                                    }
                                    HStack {
                                        Text("周期波动")
                                            .font(.caption)
                                        Slider(value: Binding(
                                            get: { Double(engine.simYoloPeriodicVariance) },
                                            set: { engine.simYoloPeriodicVariance = Float($0) }
                                        ), in: 0...0.60, step: 0.05)
                                        Text(String(format: "%.0f%%", engine.simYoloPeriodicVariance * 100))
                                            .font(.caption.monospaced())
                                            .frame(width: 40, alignment: .trailing)
                                    }
                                    DisclosureGroup(isExpanded: $simKnobHelpExpanded) {
                                        VStack(alignment: .leading, spacing: 8) {
                                            Text("这两个滑块是「仿真旋钮」：用很少的参数去模仿「检测器有多准、有多爱乱报」的大致手感，不是临床或论文里那种逐帧逐像素完整填完的混淆矩阵。")
                                            Text("下面用「每一帧、相对真值车框」的说法，并解释英文缩写：")
                                            Text("• TP（True Positive，真阳性）：真车在画面里，算法也给出了对上那辆车的检测框（通常用与真值框的 IoU 是否够大来判断是否对上）。")
                                            Text("• FN（False Negative，假阴性／漏检）：真值里本该检出这辆车，但这一帧没有合格检测框对上。设定召回率调低时，漏检更容易变多，也就是在「有真车」这件事上 FN 相对变多。")
                                            Text("• FP（False Positive，假阳性／误检）：多出来的、容易干扰跟踪的假框（例如背景纹理被当成车）。设定误检率调高时，会以一定概率往画面里塞这类假框，相当于 FP 变多。")
                                            Text("• TN（True Negative，真阴性）：「正确地没有把某块背景当成车」。在整张图里这类次数极大，单独做一个旋钮既不直观也难和真实 YOLO 对齐，所以本仿真不单独提供 TN 旋钮；我们通过误检率来控制「多报假目标」的强度。")
                                            Text("和旋钮的对应关系（好记版）：召回率主要管漏检（FN 侧）；误检率主要管多出来的假框（FP 侧）。二者同时变差时，追踪和单目标锁定往往会更难，因为既要补漏检又要抗假框。")
                                            Text("周期波动会让检测质量按时间起伏：好相位召回上升、误检下降；坏相位召回下降、误检上升。默认 30%。")
                                            Text("对齐批量结果：默认已开「单目标锁定」，与 batch_summary 里 Pass2 的单目标锁定% 一致条件时，请设随机种子 42、Track1、室内、赛道边中位、5 车、60s。若要看表里 Pass1 的检测%、追踪%，请临时关闭「单目标锁定」再跑 60s。")
                                        }
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                        .multilineTextAlignment(.leading)
                                        .fixedSize(horizontal: false, vertical: true)
                                        .textSelection(.enabled)
                                    } label: {
                                        Text("仿真旋钮说明（与 TP / FN / FP / TN）")
                                            .font(.caption.bold())
                                            .foregroundColor(.primary)
                                    }
                                }
                            }
                        }
                        .padding(4)
                    }

                    // ── 运行控制 ──
                    GroupBox("运行 Run") {
                        VStack(spacing: 8) {
                            Toggle("逐帧 Debug 模式", isOn: $engine.debugMode)
                                .toggleStyle(.switch)
                                .font(.caption)

                            if engine.debugMode {
                                GroupBox("Debug — 参数面板") {
                                    ScrollView {
                                        VStack(alignment: .leading, spacing: 6) {
                                            Text("仿真与场景（当前滑块／开关值）")
                                                .font(.caption2.bold())
                                            VSimDataRow(label: "模拟 YOLO", value: engine.useSimulatedYOLO ? "开" : "关")
                                            VSimDataRow(label: "设定召回率", value: String(format: "%.0f%% (Recall knob)", engine.simulatedRecallRate * 100))
                                            VSimDataRow(label: "设定误检率", value: String(format: "%.1f%% (FP 注入概率/帧)", engine.simulatedFPRate * 100))
                                            VSimDataRow(label: "周期波动", value: String(format: "±%.0f%% / %.0fs", engine.simYoloPeriodicVariance * 100, engine.simYoloPeriodicPeriodSec))
                                            VSimDataRow(label: "单目标锁定 ST", value: engine.singleTargetMode ? "开" : "关")
                                            VSimDataRow(label: "随机种子", value: "\(engine.randomSeed)")
                                            VSimDataRow(label: "仿真时长", value: String(format: "%.0f s", engine.simDuration))
                                            VSimDataRow(label: "车辆数", value: "\(engine.carCount)")
                                            VSimDataRow(label: "赛道 / 场景 / 机位", value: "T\(engine.trackShape + 1) · 场景\(engine.sceneType) · 预设\(engine.cameraPreset)")
                                            VSimDataRow(label: "相机高度", value: String(format: "%.1f m", Double(engine.cameraHeight)))
                                            Divider().padding(.vertical, 2)
                                            Text("单目标锁定率（引擎定义：主车可见帧中 IoU>0.15）")
                                                .font(.caption2.bold())
                                            let st = engine.stats
                                            let stPct = st.stTotalFrames > 0
                                                ? Double(st.stLockFrames) / Double(st.stTotalFrames) * 100
                                                : 0
                                            VSimDataRow(
                                                label: "ST 累计",
                                                value: st.stTotalFrames > 0
                                                    ? String(format: "%d / %d 帧  (%.2f%%)", st.stLockFrames, st.stTotalFrames, stPct)
                                                    : "尚无数据（未跑或主车未可见）"
                                            )
                                            Text("误检框叠加：须「模拟 YOLO」为开；橙框为「钉在画面上的假物体」（归一化坐标固定），视角（舵机）相对误检出现时刻累计变化足够大才会消失。")
                                                .font(.caption2)
                                                .foregroundColor(.secondary)
                                            if let s = engine.snap {
                                                Divider().padding(.vertical, 2)
                                                Text("当前帧中间量（snap）")
                                                    .font(.caption2.bold())
                                                VSimDataRow(label: "帧 / 时间", value: "F\(s.frameIdx)  \(String(format: "%.2fs", s.time))")
                                                VSimDataRow(label: "本帧跑 YOLO", value: s.didRunYoloThisFrame ? "是" : "否")
                                                VSimDataRow(label: "检测源", value: s.detSource)
                                                VSimDataRow(label: "状态", value: s.state)
                                                VSimDataRow(label: "YOLO 框数", value: "\(s.yoloDetCount)")
                                                VSimDataRow(label: "本帧 FP 框数", value: "\(s.simFalsePositiveRects.count)")
                                                VSimDataRow(label: "选中框置信度", value: String(format: "%.3f", s.yoloConf))
                                                VSimDataRow(label: "平滑框 vs 主车 GT IoU", value: String(format: "%.3f (ST 用>0.15)", s.primaryVsGtIoU))
                                                VSimDataRow(label: "平滑框 vs 任一 GT IoU(max)", value: String(format: "%.3f", s.detGtIoU))
                                                VSimDataRow(label: "追踪在车上(>0.2)", value: s.isTrackOnCar ? "是" : "否")
                                                VSimDataRow(label: "连续丢失计数", value: "\(s.targetLostCount)")
                                            }
                                        }
                                        .font(.caption2)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                    }
                                    .frame(maxHeight: 340)
                                }
                            }

                            Toggle("单目标锁定 (ST Lock)", isOn: $engine.singleTargetMode)
                                .toggleStyle(.switch)
                                .font(.caption)
                                .help("启用 Pass 2 单目标追踪：ByteTrack + VNTracker + ReID 持续锁定同一车辆")
                                .disabled(engine.running)

                            Toggle("帧降质 (Sim→Real)", isOn: $engine.enableFrameDegrade)
                                .toggleStyle(.switch)
                                .font(.caption)
                                .help("给渲染帧加运动模糊、JPEG伪影、曝光跳变，逼近真实视频")
                                .disabled(engine.running)

                            if engine.enableFrameDegrade {
                                HStack {
                                    Text("降质强度")
                                    Slider(value: $engine.degradeIntensity, in: 0...1, step: 0.1)
                                    Text(String(format: "%.0f%%", engine.degradeIntensity * 100))
                                        .frame(width: 40)
                                }
                                .font(.caption)
                            }

                            Button(action: {
                                if engine.running {
                                    engine.stop()
                                } else {
                                    engine.start()
                                }
                            }) {
                                HStack {
                                    Image(systemName: engine.running ? "stop.fill" : "play.fill")
                                    Text(engine.running ? "Stop" : "Generate & Run")
                                }
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(engine.running ? .red : .accentColor)
                            .disabled(false)

                            if engine.running {
                                HStack(spacing: 8) {
                                    Button(action: { engine.stepForward() }) {
                                        Label("Step", systemImage: "forward.frame")
                                            .frame(maxWidth: .infinity)
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(.orange)
                                    .disabled(!engine.debugMode)

                                    Button(action: { engine.togglePause() }) {
                                        Label(engine.paused ? "Resume" : "Pause",
                                              systemImage: engine.paused ? "play" : "pause")
                                            .frame(maxWidth: .infinity)
                                    }
                                    .buttonStyle(.bordered)
                                }

                                if engine.debugMode, engine.historyCount > 1 {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("历史帧回看: \(engine.debugFrameIdx + 1)/\(engine.historyCount)")
                                            .font(.caption2.monospaced())
                                            .foregroundColor(.secondary)
                                        Slider(
                                            value: Binding(
                                                get: { Double(engine.debugFrameIdx) },
                                                set: { engine.jumpToFrame(Int($0)) }
                                            ),
                                            in: 0...Double(max(engine.historyCount - 1, 1)),
                                            step: 1
                                        )
                                    }
                                }
                            }

                            if engine.stats.totalFrames > 0 && !engine.running {
                                Button("查看仿真结论") {
                                    showResult = true
                                }
                                .frame(maxWidth: .infinity)

                                Button("导出 CSV") {
                                    showCSVExport = true
                                }
                                .frame(maxWidth: .infinity)
                            }
                        }
                        .padding(4)
                    }

                    // ── 实时数据 ──
                    if let snap = engine.snap {
                        GroupBox("实时数据 Live") {
                            VStack(alignment: .leading, spacing: 4) {
                                VSimDataRow(label: "主车车速", value: String(format: "%.1f km/h", snap.primarySpeedKmh))
                                if let p = snap.gimbalBaseWorldPos {
                                    VSimDataRow(label: "云台位置", value: String(format: "x=%.2f y=%.2f z=%.2f", p.x, p.y, p.z))
                                }
                                VSimDataRow(label: "舵机 H", value: String(format: "cmd=%.1f° act=%.1f°", snap.cmdH, snap.actH))
                                VSimDataRow(label: "舵机 V", value: String(format: "cmd=%.1f° act=%.1f°", snap.cmdV, snap.actV))
                                VSimDataRow(label: "误差", value: String(format: "H=%.1f° V=%.1f°", snap.errDegX, snap.errDegY))
                                VSimDataRow(label: "检测源", value: snap.detSource)
                                VSimDataRow(label: "GT可见", value: snap.gtPrimaryVisible ? "是" : "否")
                                VSimDataRow(label: "GT遮挡", value: snap.gtPrimaryOccluded ? "是" : "否")
                                if let det = snap.detectedRect, let gt = snap.gtPrimaryRect {
                                    let iou = VirtualRaceEngine.iou(det, gt)
                                    VSimDataRow(label: "IoU", value: String(format: "%.3f", iou))
                                }
                            }
                            .padding(4)
                        }
                    }

                    // ── 说明 ──
                                        // ══ 历史成绩 ══
                    GroupBox("历史成绩 History") {
                        SimHistoryPanel(engine: engine)
                    }

                    GroupBox("说明 Info") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("• 绿色框 = Ground Truth (主目标)")
                            Text("• 绿色虚线框 = 其他车辆 GT")
                            Text("• 橙色框 = GT 被遮挡")
                            Text("• 青色框 = 检测/追踪结果")
                            Text("• 自动生成 3D 赛道和多辆 RC 赛车")
                            Text("• 标定赛道: 在标定Tab手动标记→导出JSON→加载仿真")
                            Text("• 复用现有 YOLO + VNTracker + PID")
                        }
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(4)
                    }

                    Spacer()
                }
                .padding(12)
            }
            .frame(width: 280)
        }
        .sheet(isPresented: $showResult) {
            VSimResultSheet(stats: engine.stats, isPresented: $showResult)
        }
        .fileExporter(
            isPresented: $showCSVExport,
            document: VSimCSVDocument(engine: engine),
            contentType: .commaSeparatedText,
            defaultFilename: "virtual_sim_results.csv"
        ) { _ in }
    }

    private func fitRect(imgSize: CGSize, viewSize: CGSize) -> CGRect {
        let scale = min(viewSize.width / imgSize.width, viewSize.height / imgSize.height)
        let w = imgSize.width * scale
        let h = imgSize.height * scale
        return CGRect(x: (viewSize.width - w) / 2, y: (viewSize.height - h) / 2, width: w, height: h)
    }

    private func stateColor(_ state: String) -> Color {
        switch state {
        case "TRACK": return .green
        case "HOLD": return .yellow
        case "SCAN": return .orange
        case "RETURN": return .red
        default: return .gray
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 3D 自由相机视图 (NSViewRepresentable)
// ═══════════════════════════════════════════════════════════════

/// 封装 SCNView 供 SwiftUI 使用，启用 allowsCameraControl 实现自由旋转/缩放/平移
///
/// 操作方式:
///   - 左键拖拽: 旋转 (Orbit)
///   - 右键拖拽 / 两指平移: 平移 (Pan)
///   - 滚轮 / 两指缩放: 缩放 (Zoom)
struct FreeCameraSceneView: NSViewRepresentable {
    let scene: SCNScene
    let centerX: CGFloat
    let centerZ: CGFloat
    let radius: CGFloat

    func makeNSView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = scene
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = .black
        scnView.antialiasingMode = .multisampling4X
        scnView.showsStatistics = false

        let cam = SCNCamera()
        cam.fieldOfView = 60
        cam.zNear = 0.1
        cam.zFar = 500
        let camNode = SCNNode()
        camNode.camera = cam

        // 用引擎传入的赛道包围盒定位（不用 scene.rootNode.boundingBox — SCNFloor 会破坏它）
        let maxSpan = max(radius * 2, 10)
        let eyeH = maxSpan * 0.7
        let eyeZ = centerZ + maxSpan * 0.7
        camNode.position = SCNVector3(Float(centerX), Float(eyeH), Float(eyeZ))
        camNode.look(at: SCNVector3(Float(centerX), 0, Float(centerZ)))
        camNode.name = "__freeCamera__"
        scene.rootNode.addChildNode(camNode)
        scnView.pointOfView = camNode

        return scnView
    }

    func updateNSView(_ scnView: SCNView, context: Context) {
        // scene 已经是同一实例，不需要重新赋值
        // 确保自由相机控制始终开启
        scnView.allowsCameraControl = true
    }

    static func dismantleNSView(_ scnView: SCNView, coordinator: ()) {
        // 移除自由相机节点，避免污染仿真场景
        if let freeNode = scnView.scene?.rootNode.childNode(withName: "__freeCamera__", recursively: false) {
            freeNode.removeFromParentNode()
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 统计指标条
// ═══════════════════════════════════════════════════════════════

struct VSimStatsBar: View {
    let stats: VSimStats

    var body: some View {
        HStack(spacing: 8) {
            // ═══ ★ 核心指标 ═══
            VSimMetric(label: "检测率", value: String(format: "%.1f%%", stats.detectionRate),
                       color: stats.detectionRate > 75 ? .green : (stats.detectionRate > 50 ? .yellow : .red))
            VSimMetric(label: "追踪率", value: String(format: "%.1f%%", stats.generalTrackRate),
                       color: stats.generalTrackRate > 70 ? .green : (stats.generalTrackRate > 50 ? .yellow : .red))
            VSimMetric(label: "单目标锁定", value: String(format: "%.1f%%", stats.singleTargetLockRate),
                       color: stats.singleTargetLockRate > 70 ? .green : (stats.singleTargetLockRate > 50 ? .yellow : .red))
            VSimMetric(label: "最长丢失", value: "\(stats.maxConsecutiveLost)f",
                       color: stats.maxConsecutiveLost < 15 ? .green : (stats.maxConsecutiveLost < 45 ? .yellow : .red))
            Divider().frame(height: 24)
            // ═══ 辅助指标 ═══
            VSimMetric(label: "变焦", value: String(format: "%.1fx", stats.avgZoom),
                       color: .blue)
            VSimMetric(label: "误差H", value: String(format: "%.1f°", stats.avgErrH),
                       color: stats.avgErrH < 3 ? .green : (stats.avgErrH < 6 ? .yellow : .red))
            VSimMetric(label: "抽动H", value: String(format: "%.1f%%", stats.jitterH),
                       color: stats.jitterH < 15 ? .green : (stats.jitterH < 30 ? .yellow : .red))
            Divider().frame(height: 24)
            VSimMetric(label: "评分", value: "\(stats.score)",
                       color: stats.score >= 70 ? .green : (stats.score >= 50 ? .yellow : .red))
        }
    }
}

struct VSimMetric: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .bold()
                .foregroundColor(color)
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 数据行
// ═══════════════════════════════════════════════════════════════

struct VSimDataRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(width: 50, alignment: .leading)
            Text(value)
                .font(.caption2.monospaced())
            Spacer()
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 仿真结论弹窗
// ═══════════════════════════════════════════════════════════════

struct VSimResultSheet: View {
    let stats: VSimStats
    @Binding var isPresented: Bool

    var body: some View {
        VStack(spacing: 0) {
            // 标题
            HStack {
                Image(systemName: "flag.checkered")
                    .font(.title2)
                Text("虚拟仿真结论")
                    .font(.title2.bold())
                Spacer()
                Text("\(stats.score) 分")
                    .font(.title.bold())
                    .foregroundColor(scoreColor(stats.score))
                Text(stats.grade)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            ScrollView {
                VStack(spacing: 12) {
                    // ── ★ 核心指标 ──
                    VSimResultSection(title: "核心指标", icon: "viewfinder") {
                        VSimGaugeRow(label: "检测率 = YOLO判断正确帧/YOLO运行帧",
                                     value: stats.detectionRate / 100,
                                     text: String(format: "%.1f%% (%d/%d)", stats.detectionRate, stats.detCorrectFrames, stats.yoloRunFrames),
                                     good: stats.detectionRate > 75)
                        VSimGaugeRow(label: "一般追踪率 = 输出框在车上的帧/总帧",
                                     value: stats.generalTrackRate / 100,
                                     text: String(format: "%.1f%% (%d/%d)", stats.generalTrackRate, stats.trackOnCarFrames, stats.totalFrames),
                                     good: stats.generalTrackRate > 70)
                        VSimGaugeRow(label: "单目标锁定率 = ByteTrack+ReID持续追踪同一目标",
                                     value: stats.singleTargetLockRate / 100,
                                     text: String(format: "%.1f%% (%d/%d)", stats.singleTargetLockRate, stats.stLockFrames, stats.stTotalFrames),
                                     good: stats.singleTargetLockRate > 70)
                        HStack {
                            Text("最长连续丢失").font(.caption).foregroundColor(.secondary)
                            Spacer()
                            Text("\(stats.maxConsecutiveLost) 帧")
                                .font(.caption.monospaced())
                                .foregroundColor(stats.maxConsecutiveLost < 15 ? .green : .red)
                        }
                    }

                    // ── 检测层统计 ──
                    VSimResultSection(title: "检测层统计", icon: "scope") {
                        VSimGaugeRow(label: "控制可用率 = pidRect非空帧/总帧",
                                     value: stats.generalTrackRate / 100,
                                     text: String(format: "%.1f%% (%d/%d)", stats.generalTrackRate, stats.trackingFrames, stats.totalFrames),
                                     good: stats.generalTrackRate > 75)
                        VSimGaugeRow(label: "YOLO出框率 = YOLO匹配帧/总帧",
                                     value: stats.yoloDetRate / 100,
                                     text: String(format: "%.1f%% (%d/%d)", stats.yoloDetRate, stats.yoloMatchFrames, stats.totalFrames),
                                     good: stats.yoloDetRate > 45)
                    }

                    // ── 变焦分析 ──
                    VSimResultSection(title: "自动变焦", icon: "magnifyingglass") {
                        HStack {
                            Text("平均光学变焦").font(.caption).foregroundColor(.secondary)
                            Spacer()
                            Text(String(format: "%.2fx", stats.avgZoom))
                                .font(.caption.monospaced())
                                .foregroundColor(.blue)
                        }
                    }

                    // ── 舶机精度 + 抽动 ──
                    VSimResultSection(title: "舶机精度", icon: "gauge") {
                        VSimGaugeRow(label: "平均水平误差",
                                     value: min(stats.avgErrH / 15, 1),
                                     text: String(format: "%.1f° (%d帧)", stats.avgErrH, stats.errCount),
                                     good: stats.avgErrH < 3)
                        VSimGaugeRow(label: "平均垂直误差",
                                     value: min(stats.avgErrV / 15, 1),
                                     text: String(format: "%.1f°", stats.avgErrV),
                                     good: stats.avgErrV < 3)
                        HStack {
                            Text("最大误差 H/V").font(.caption).foregroundColor(.secondary)
                            Spacer()
                            Text(String(format: "%.1f° / %.1f°", stats.maxErrH, stats.maxErrV))
                                .font(.caption.monospaced())
                        }
                        Divider()
                        VSimGaugeRow(label: "水平抽动率",
                                     value: min(stats.jitterH / 50, 1),
                                     text: String(format: "%.1f%% (%d次)", stats.jitterH, stats.dirChangeH),
                                     good: stats.jitterH < 15)
                        VSimGaugeRow(label: "俯仰抽动率",
                                     value: min(stats.jitterV / 50, 1),
                                     text: String(format: "%.1f%% (%d次)", stats.jitterV, stats.dirChangeV),
                                     good: stats.jitterV < 15)
                    }

                    // ── YOLO 推理 ──
                    VSimResultSection(title: "YOLO 推理", icon: "cpu") {
                        VSimGaugeRow(label: "平均推理耗时",
                                     value: min(stats.avgYoloMs / 60, 1),
                                     text: String(format: "%.1f ms (%d次)", stats.avgYoloMs, stats.yoloMsCount),
                                     good: stats.avgYoloMs < 30)
                    }

                    // ── 综合建议 ──
                    VSimResultSection(title: "综合建议", icon: "lightbulb") {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(suggestions(), id: \.self) { s in
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

    private func suggestions() -> [String] {
        var list: [String] = []
        // ★ 新指标体系诊断
        if stats.detectionRate < 60 {
            list.append("检测率偏低（\(String(format: "%.0f", stats.detectionRate))%），YOLO 对有无车判断不准确，可能需要调整置信度阈值或重新训练模型。")
        }
        if stats.generalTrackRate < 50 {
            list.append("一般追踪率偏低（\(String(format: "%.0f", stats.generalTrackRate))%），输出框大部分时间不在实际车上，可能是追踪丢失或跟错目标。")
        }
        if stats.singleTargetLockRate > 0 && stats.singleTargetLockRate < 50 {
            list.append("单目标锁定率偏低（\(String(format: "%.0f", stats.singleTargetLockRate))%），ByteTrack+ReID 在遮挡后重关联失败较多，建议优化外观特征或降低重关联阈值。")
        } else if stats.singleTargetLockRate >= 80 {
            list.append("单目标锁定表现优秀（\(String(format: "%.0f", stats.singleTargetLockRate))%），ByteTrack+ReID 能有效应对遮挡和目标丢失。")
        }
        if stats.maxConsecutiveLost > 30 {
            list.append("最长连续丢失 \(stats.maxConsecutiveLost) 帧，建议优化惯性追踪和扫描策略。")
        }
        if stats.avgErrH > 5 {
            list.append("水平误差偏大（\(String(format: "%.1f", stats.avgErrH))°），建议增大 KpH 或减小 BLE 延迟。")
        }
        if stats.jitterH > 25 || stats.jitterV > 25 {
            list.append("抽动较高（H=\(String(format: "%.0f", stats.jitterH))% V=\(String(format: "%.0f", stats.jitterV))%），建议降低 PID 增益或增大死区。")
        }
        if stats.avgYoloMs > 35 {
            list.append("YOLO 推理较慢（\(String(format: "%.0f", stats.avgYoloMs))ms），可尝试更小的模型。")
        }
        if list.isEmpty {
            list.append("各项指标均在优秀范围，当前算法配置在虚拟环境中表现良好。")
        }
        return list
    }
}

struct VSimResultSection<Content: View>: View {
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

struct VSimGaugeRow: View {
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
                        .frame(width: g.size.width * CGFloat(max(0, min(1, value))))
                }
            }
            .frame(height: 6)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - CSV 文档
// ═══════════════════════════════════════════════════════════════

import UniformTypeIdentifiers

struct VSimCSVDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.commaSeparatedText] }

    let csvData: Data

    init(engine: VirtualRaceEngine) {
        var csv = "=== Virtual Race Simulation Summary ===\n"
        csv += "totalFrames,\(engine.stats.totalFrames)\n"
        csv += "detectionRate,\(String(format: "%.1f", engine.stats.detectionRate))\n"
        csv += "generalTrackRate,\(String(format: "%.1f", engine.stats.generalTrackRate))\n"

        csv += "trackingFrames,\(engine.stats.trackingFrames)\n"
        csv += "trackingRate,\(String(format: "%.1f", engine.stats.generalTrackRate))\n"
        csv += "yoloMatchFrames,\(engine.stats.yoloMatchFrames)\n"
        csv += "yoloRunFrames,\(engine.stats.yoloRunFrames)\n"
        csv += "yoloDetRate,\(String(format: "%.1f", engine.stats.yoloDetRate))\n"
        csv += "pureDetFrames,\(engine.stats.pureDetFrames)\n"
        csv += "detCorrectFrames,\(engine.stats.detCorrectFrames)\n"
        csv += "trackOnCarFrames,\(engine.stats.trackOnCarFrames)\n"

        csv += "maxConsecutiveLost,\(engine.stats.maxConsecutiveLost)\n"
        csv += "avgErrH,\(String(format: "%.2f", engine.stats.avgErrH))\n"
        csv += "avgErrV,\(String(format: "%.2f", engine.stats.avgErrV))\n"
        csv += "jitterH,\(String(format: "%.1f", engine.stats.jitterH))\n"
        csv += "jitterV,\(String(format: "%.1f", engine.stats.jitterV))\n"
        csv += "avgZoom,\(String(format: "%.2f", engine.stats.avgZoom))\n"
        csv += "avgYoloMs,\(String(format: "%.1f", engine.stats.avgYoloMs))\n"
        csv += "score,\(engine.stats.score)\n"
        csv += "grade,\(engine.stats.grade)\n"
        self.csvData = csv.data(using: .utf8) ?? Data()
    }

    init(configuration: ReadConfiguration) throws {
        csvData = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        return FileWrapper(regularFileWithContents: csvData)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 仿真历史成绩面板
// ═══════════════════════════════════════════════════════════════

private enum SimHistoryViewTab: String, CaseIterable {
    case runs = "单次"
    case matrix = "组合表"
}

/// 场景 × 赛道 × 机位 汇总行
private struct SimComboAggregate: Identifiable {
    let sceneType: Int
    let trackShape: Int
    let cameraPreset: Int
    let runCount: Int
    let avgGeneralTrack: Double
    let avgDetection: Double
    let bestScore: Int
    /// 用于重现：该组合下得分最高的一次完整参数
    let replayRecord: SimHistoryRecord

    var id: String { "\(sceneType)-\(trackShape)-\(cameraPreset)" }
}

struct SimHistoryPanel: View {
    @ObservedObject private var store = SimHistoryStore.shared
    @ObservedObject var engine: VirtualRaceEngine
    @State private var filterStrategy: String = "All"
    @State private var showAll = false
    @State private var tab: SimHistoryViewTab = .matrix

    private static let sceneNames = ["室内", "草地", "柏油", "娱乐", "沙地", "专业", "树林", "开放"]
    private static let presetShort = ["边低", "边中", "台高", "台中", "弯外", "直侧", "跟低", "跟中", "俯瞰"]

    private var filtered: [SimHistoryRecord] {
        store.records.filter { r in
            filterStrategy == "All" || r.strategy == filterStrategy
        }
    }

    private var comboAggregates: [SimComboAggregate] {
        let groups = Dictionary(grouping: filtered) { r -> String in
            "\(r.sceneType)-\(r.trackShape)-\(r.cameraPreset)"
        }
        return groups.compactMap { _, list -> SimComboAggregate? in
            guard !list.isEmpty else { return nil }
            let best = list.max(by: { a, b in
                if a.score != b.score { return a.score < b.score }
                return a.timestamp < b.timestamp
            })!
            let avgT = list.map(\.generalTrackRate).reduce(0, +) / Double(list.count)
            let avgD = list.map(\.detectionRate).reduce(0, +) / Double(list.count)
            return SimComboAggregate(
                sceneType: best.sceneType,
                trackShape: best.trackShape,
                cameraPreset: best.cameraPreset,
                runCount: list.count,
                avgGeneralTrack: avgT,
                avgDetection: avgD,
                bestScore: list.map(\.score).max() ?? 0,
                replayRecord: best
            )
        }
        .sorted { a, b in
            if a.bestScore != b.bestScore { return a.bestScore > b.bestScore }
            if a.sceneType != b.sceneType { return a.sceneType < b.sceneType }
            if a.trackShape != b.trackShape { return a.trackShape < b.trackShape }
            return a.cameraPreset < b.cameraPreset
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("", selection: $tab) {
                ForEach(SimHistoryViewTab.allCases, id: \.self) { t in
                    Text(t.rawValue).tag(t)
                }
            }
            .pickerStyle(.segmented)

            HStack(spacing: 4) {
                Picker("", selection: $filterStrategy) {
                    Text("全部").tag("All")
                    Text("MLP").tag("MLP")
                    Text("PID").tag("PID")
                }
                .pickerStyle(.segmented)
                .frame(width: 160)

                Spacer()

                Text(tab == .matrix ? "\(comboAggregates.count) 组" : "\(filtered.count) 条")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                if !store.records.isEmpty {
                    Button(action: { store.clearAll() }) {
                        Image(systemName: "trash")
                            .font(.caption2)
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.red.opacity(0.7))
                    .help("清除所有历史记录")
                }
            }

            if filtered.isEmpty {
                Text("暂无历史记录，运行仿真后自动保存")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            } else {
                let avgTrack = filtered.map(\.generalTrackRate).reduce(0, +) / Double(max(filtered.count, 1))
                let avgScore = filtered.map { Double($0.score) }.reduce(0, +) / Double(max(filtered.count, 1))
                let bestScore = filtered.map(\.score).max() ?? 0

                HStack(spacing: 12) {
                    VStack(spacing: 1) {
                        Text(String(format: "%.1f%%", avgTrack))
                            .font(.caption.bold().monospaced())
                            .foregroundColor(avgTrack > 70 ? .green : (avgTrack > 50 ? .yellow : .red))
                        Text("平均追踪")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    }
                    VStack(spacing: 1) {
                        Text(String(format: "%.0f", avgScore))
                            .font(.caption.bold().monospaced())
                            .foregroundColor(avgScore >= 70 ? .green : (avgScore >= 50 ? .yellow : .red))
                        Text("平均分")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    }
                    VStack(spacing: 1) {
                        Text("\(bestScore)")
                            .font(.caption.bold().monospaced())
                            .foregroundColor(.green)
                        Text("最高分")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)

                Divider()

                if tab == .matrix {
                    Text("按 场景·赛道·机位 汇总；重现使用该组合下得分最高的一次参数")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(comboAggregates) { agg in
                                SimComboRowView(engine: engine, aggregate: agg)
                            }
                        }
                    }
                    .frame(maxHeight: 220)
                } else {
                    let displayRecords = showAll ? filtered : Array(filtered.prefix(5))
                    ForEach(displayRecords) { r in
                        SimHistoryRow(engine: engine, record: r)
                    }

                    if filtered.count > 5 {
                        Button(showAll ? "收起" : "查看全部 (\(filtered.count) 条)") {
                            showAll.toggle()
                        }
                        .font(.caption2)
                        .frame(maxWidth: .infinity)
                    }
                }
            }
        }
        .padding(4)
    }
}

private struct SimComboRowView: View {
    @ObservedObject var engine: VirtualRaceEngine
    let aggregate: SimComboAggregate

    private static let sceneNames = ["室内", "草地", "柏油", "娱乐", "沙地", "专业", "树林", "开放"]
    private static let presetShort = ["边低", "边中", "台高", "台中", "弯外", "直侧", "跟低", "跟中", "俯瞰"]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                let sn = aggregate.sceneType < Self.sceneNames.count ? Self.sceneNames[aggregate.sceneType] : "?"
                let pn = aggregate.cameraPreset < Self.presetShort.count ? Self.presetShort[aggregate.cameraPreset] : "?"
                Text("\(sn) · T\(aggregate.trackShape + 1) · \(pn)")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .lineLimit(1)
                Spacer(minLength: 0)
                Button(action: {
                    engine.replayFromHistory(record: aggregate.replayRecord)
                }) {
                    Text("重现")
                        .font(.system(size: 9, weight: .bold))
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(engine.running)
            }
            HStack(spacing: 8) {
                Text("\(aggregate.runCount)次")
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
                Text(String(format: "追踪均%.1f%%", aggregate.avgGeneralTrack))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(aggregate.avgGeneralTrack > 70 ? .green : (aggregate.avgGeneralTrack > 50 ? .yellow : .red))
                Text(String(format: "检测均%.0f%%", aggregate.avgDetection))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
                Spacer()
                Text("最佳\(aggregate.bestScore)分")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(aggregate.bestScore >= 70 ? .green : .orange)
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 4)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
        .cornerRadius(4)
    }
}

struct SimHistoryRow: View {
    @ObservedObject var engine: VirtualRaceEngine
    let record: SimHistoryRecord

    private static let sceneNames = ["室内", "草地", "柏油", "娱乐", "沙地", "专业", "树林", "开放"]
    private static let presetShort = ["边低", "边中", "台高", "台中", "弯外", "直侧", "跟低", "跟中", "俯瞰"]

    private static let timeFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MM/dd HH:mm"
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Text(record.strategy)
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(record.strategy == "MLP" ? Color.blue : Color.orange)
                    .cornerRadius(3)

                let sceneName = record.sceneType < Self.sceneNames.count ? Self.sceneNames[record.sceneType] : "?"
                let presetName = record.cameraPreset < Self.presetShort.count ? Self.presetShort[record.cameraPreset] : "?"
                Text("T\(record.trackShape+1) \(sceneName) \(presetName)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(1)

                Spacer(minLength: 0)

                Button(action: { engine.replayFromHistory(record: record) }) {
                    Text("重现")
                        .font(.system(size: 9, weight: .bold))
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(engine.running)

                Text(Self.timeFmt.string(from: record.timestamp))
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 8) {
                HStack(spacing: 2) {
                    Text("Track")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f%%", record.generalTrackRate))
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundColor(record.generalTrackRate > 70 ? .green : (record.generalTrackRate > 50 ? .yellow : .red))
                }
                HStack(spacing: 2) {
                    Text("STLock")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                    Text(String(format: "%.0f%%", record.singleTargetLockRate))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(record.singleTargetLockRate > 70 ? .green : .yellow)
                }
                HStack(spacing: 2) {
                    Text("Det")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                    Text(String(format: "%.0f%%", record.detectionRate))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(record.detectionRate > 75 ? .green : .red)
                }
                Spacer()
                Text("\(record.score)分")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(record.score >= 70 ? .green : (record.score >= 55 ? .yellow : .red))
            }
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 4)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .cornerRadius(4)
    }
}
