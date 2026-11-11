//
//  VideoAnnotationModule.swift
//  macRCtrackerSimulator
//
//  视频标注模块 — 手动 + SAM3 远程自动标注 RC 遥控车
//
//  功能:
//    ① 加载视频并逐帧浏览（支持拖放 / 文件选择器）
//    ② 手动拖拽画框标注 RC 遥控车位置
//    ③ 远程调用 Replicate SAM3 API 自动分割 → 提取 bbox
//    ④ 手动修正自动标注结果（删除/添加/调整/复制上帧）
//    ⑤ 导出 YOLO 格式标注（与 DatasetGenerator 兼容）
//    ⑥ 导入/导出标注项目 JSON（可中断续标）
//    ⑦ 帧级别插值（两个关键帧之间线性插值 bbox）
//
//  集成方法:
//    1. 将本文件添加到 Xcode 项目
//    2. 在 macRCtrackerSimulatorApp.swift 的 TabView 中添加:
//         VideoAnnotationView()
//             .tabItem { Label("标注 Annotator", systemImage: "pencil.and.ruler") }
//             .tag(4)
//
//  Created by Manus AI on 2026/4/7.
//  macOS 14.0+
//

import SwiftUI
import Combine
import AVFoundation
import AppKit
import UniformTypeIdentifiers

// ═══════════════════════════════════════════════════════════════
// MARK: - 标注数据模型
// ═══════════════════════════════════════════════════════════════

/// 单个标注框 — 归一化坐标 [0,1]，与 ByteTracker / YOLOLabel 一致
struct AnnBox: Identifiable, Codable, Equatable {
    let id: UUID
    var classId: Int           // 0 = rc_car
    var cx: CGFloat            // 中心 x
    var cy: CGFloat            // 中心 y
    var w: CGFloat             // 宽
    var h: CGFloat             // 高
    var confidence: Float      // 手动 = 1.0
    var source: BoxSource

    enum BoxSource: String, Codable, CaseIterable { case manual, sam3 }

    private enum CodingKeys: String, CodingKey {
        case id, classId, cx, cy, w, h, confidence, source
    }

    init(classId: Int = 0, cx: CGFloat, cy: CGFloat, w: CGFloat, h: CGFloat,
         confidence: Float = 1.0, source: BoxSource = .manual) {
        self.id = UUID()
        self.classId = classId; self.cx = cx; self.cy = cy
        self.w = w; self.h = h
        self.confidence = confidence; self.source = source
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // 兼容 SAM3 生成的非 UUID 格式 id（如 "SAM3-1"）：
        // 先尝试按标准 UUID 解码，失败则用字符串的确定性哈希生成 UUID
        if let uuid = try? c.decode(UUID.self, forKey: .id) {
            self.id = uuid
        } else {
            let raw = (try? c.decode(String.self, forKey: .id)) ?? UUID().uuidString
            // 用 UUID v5 风格：基于字符串内容生成稳定的 UUID，保证同一 id 字符串始终映射到同一 UUID
            self.id = UUID(uuidString: AnnBox.deterministicUUID(from: raw)) ?? UUID()
        }
        self.classId    = try c.decode(Int.self, forKey: .classId)
        self.cx         = try c.decode(CGFloat.self, forKey: .cx)
        self.cy         = try c.decode(CGFloat.self, forKey: .cy)
        self.w          = try c.decode(CGFloat.self, forKey: .w)
        self.h          = try c.decode(CGFloat.self, forKey: .h)
        self.confidence = try c.decode(Float.self, forKey: .confidence)
        self.source     = try c.decode(BoxSource.self, forKey: .source)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(classId, forKey: .classId)
        try c.encode(cx, forKey: .cx)
        try c.encode(cy, forKey: .cy)
        try c.encode(w, forKey: .w)
        try c.encode(h, forKey: .h)
        try c.encode(confidence, forKey: .confidence)
        try c.encode(source, forKey: .source)
    }

    /// 从任意字符串生成确定性 UUID 字符串（基于简单哈希，保证同一输入始终得到同一输出）
    private static func deterministicUUID(from string: String) -> String {
        var hash = string.utf8.reduce(UInt64(0)) { acc, byte in
            var h = acc &+ UInt64(byte)
            h = h &+ (h << 10)
            h ^= (h >> 6)
            return h
        }
        hash = hash &+ (hash << 3)
        hash ^= (hash >> 11)
        hash = hash &+ (hash << 15)
        let a = UInt32(hash & 0xFFFFFFFF)
        let b = UInt32((hash >> 32) & 0xFFFFFFFF)
        return String(format: "%08X-%04X-4%03X-8%03X-%08X%04X",
                      a,
                      UInt16(b & 0xFFFF),
                      UInt16((b >> 16) & 0x0FFF),
                      UInt16(a & 0x0FFF),
                      b,
                      UInt16(a >> 16))
    }

    /// 转为像素 CGRect
    func pixelRect(imgW: CGFloat, imgH: CGFloat) -> CGRect {
        CGRect(x: (cx - w/2) * imgW, y: (cy - h/2) * imgH,
               width: w * imgW, height: h * imgH)
    }

    /// YOLO 格式行
    var yoloLine: String {
        String(format: "%d %.6f %.6f %.6f %.6f", classId, cx, cy, w, h)
    }
}

/// 整个视频的标注项目
struct AnnProject: Codable {
    var videoPath: String
    var totalFrames: Int
    var fps: Double
    var width: Int
    var height: Int
    var className: String = "rc_car"
    var frames: [Int: [AnnBox]]          // frameIndex → boxes
    var derivedFromPath: String?
    var derivedFromKind: String?
    /// SAM3 / 云端 mask 帧号相对「本机解码帧序号」的偏移：第 L 帧画面读 `frames[L + sam3MaskOffset]`。对齐后可用标注器「重映射」写回连续键并归零。
    var sam3MaskOffset: Int = 0

    enum CodingKeys: String, CodingKey {
        case videoPath, totalFrames, fps, width, height, className, frames
        case derivedFromPath, derivedFromKind, sam3MaskOffset
    }

    init(videoPath: String, totalFrames: Int, fps: Double, width: Int, height: Int,
         derivedFromPath: String? = nil, derivedFromKind: String? = nil,
         sam3MaskOffset: Int = 0) {
        self.videoPath = videoPath
        self.totalFrames = totalFrames
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = [:]
        self.derivedFromPath = derivedFromPath
        self.derivedFromKind = derivedFromKind
        self.sam3MaskOffset = sam3MaskOffset
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        videoPath = try c.decode(String.self, forKey: .videoPath)
        totalFrames = try c.decode(Int.self, forKey: .totalFrames)
        fps = try c.decode(Double.self, forKey: .fps)
        width = try c.decode(Int.self, forKey: .width)
        height = try c.decode(Int.self, forKey: .height)
        className = try c.decodeIfPresent(String.self, forKey: .className) ?? "rc_car"
        frames = try c.decode([Int: [AnnBox]].self, forKey: .frames)
        derivedFromPath = try c.decodeIfPresent(String.self, forKey: .derivedFromPath)
        derivedFromKind = try c.decodeIfPresent(String.self, forKey: .derivedFromKind)
        sam3MaskOffset = try c.decodeIfPresent(Int.self, forKey: .sam3MaskOffset) ?? 0
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(videoPath, forKey: .videoPath)
        try c.encode(totalFrames, forKey: .totalFrames)
        try c.encode(fps, forKey: .fps)
        try c.encode(width, forKey: .width)
        try c.encode(height, forKey: .height)
        try c.encode(className, forKey: .className)
        try c.encode(frames, forKey: .frames)
        try c.encodeIfPresent(derivedFromPath, forKey: .derivedFromPath)
        try c.encodeIfPresent(derivedFromKind, forKey: .derivedFromKind)
        try c.encode(sam3MaskOffset, forKey: .sam3MaskOffset)
    }

    /// 本机仿真 / 播放第 `idx` 帧时应对应的标注（已含 SAM3 偏移）
    func boxes(forLocalFrame idx: Int) -> [AnnBox] {
        let k = idx + sam3MaskOffset
        guard k >= 0 else { return [] }
        return frames[k] ?? []
    }
}

private enum AnnotationVariant {
    case originalSAM3
    case corrected
}

/// 扫描视频轨道的每一帧 PTS（不解码到像素）。
/// ★ FIX: AVAssetReader 按解码顺序 (DTS) 输出 sample buffer，
///   对于含 B 帧的视频，DTS 与 PTS 不同，导致收集到的 PTS 数组不是递增的。
///   必须按 PTS 升序排序，才能让帧索引 i 对应第 i 个展示帧。
private enum VideoFrameTimeline {
    static func collectPresentationTimes(asset: AVAsset, track: AVAssetTrack) async -> [CMTime] {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let reader = try AVAssetReader(asset: asset)
                    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
                    output.alwaysCopiesSampleData = false
                    guard reader.canAdd(output) else {
                        cont.resume(returning: [])
                        return
                    }
                    reader.add(output)
                    guard reader.startReading() else {
                        reader.cancelReading()
                        cont.resume(returning: [])
                        return
                    }
                    var times: [CMTime] = []
                    while let sb = output.copyNextSampleBuffer() {
                        times.append(CMSampleBufferGetPresentationTimeStamp(sb))
                    }
                    reader.cancelReading()
                    // ★ FIX: 按 PTS 升序排序，消除 B 帧导致的 DTS/PTS 乱序
                    times.sort { CMTimeCompare($0, $1) < 0 }
                    cont.resume(returning: times)
                } catch {
                    cont.resume(returning: [])
                }
            }
        }
    }
}

struct AnnotatorStorageConfig: Decodable {
    var video_dir: String?
    var output_dir: String?
    var token: String?
    var prompt: String?
    var video_urls: [String: String]?

    var videoDirURL: URL? {
        guard let video_dir, !video_dir.isEmpty else { return nil }
        return URL(fileURLWithPath: video_dir, isDirectory: true)
    }

    var outputDirURL: URL? {
        guard let output_dir, !output_dir.isEmpty else { return nil }
        return URL(fileURLWithPath: output_dir, isDirectory: true)
    }

    var configuredVideoNames: [String] {
        (video_urls ?? [:]).keys.sorted()
    }

    static func load() -> AnnotatorStorageConfig {
        let fm = FileManager.default
        var candidates: [URL] = []
        if let bundled = Bundle.main.url(forResource: "annotate_config", withExtension: "json") {
            candidates.append(bundled)
        }
        candidates.append(URL(fileURLWithPath: NSHomeDirectory() + "/Documents/MacRCSimulator/annotate_config.json"))
        candidates.append(URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("annotate_config.json"))
        candidates.append(URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("macRCtrackerSimulator/annotate_config.json"))

        for url in candidates {
            guard fm.fileExists(atPath: url.path),
                  let data = try? Data(contentsOf: url),
                  let cfg = try? JSONDecoder().decode(AnnotatorStorageConfig.self, from: data) else { continue }
            return cfg
        }

        return AnnotatorStorageConfig(
            video_dir: NSHomeDirectory() + "/Documents/MacRCSimulator/云台仿真器测试视频0308_副本",
            output_dir: NSHomeDirectory() + "/Documents/MacRCSimulator/SAM3_Annotations",
            token: nil,
            prompt: "RC car",
            video_urls: [:]
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - SAM3 远程推理客户端 (Replicate)
// ═══════════════════════════════════════════════════════════════

struct SAM3Cfg: Equatable {
    var apiToken: String = EnvLoader.replicateToken  // ★ 默认从 .env 加载
    var modelVersion: String = "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269"
    var prompt: String = "RC car"
    var negativePrompt: String = ""
    var minMaskArea: Int = 30
}

enum SAM3Status: Equatable {
    case idle
    case uploading
    case processing(String)
    case extracting(Int, Int)
    case done(Int)
    case failed(String)
}

final class SAM3Client: ObservableObject {
    @Published var status: SAM3Status = .idle
    @Published var progress: CGFloat = 0

    private var cfg = SAM3Cfg()
    private var currentTask: URLSessionTask?

    func updateCfg(_ c: SAM3Cfg) { cfg = c }
    func cancel() { currentTask?.cancel(); currentTask = nil; DispatchQueue.main.async { self.status = .idle } }

    // ── 上传专用 URLSession（超时更长，适合大文件） ──
    private lazy var uploadSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300   // 单次请求 5 分钟
        config.timeoutIntervalForResource = 600  // 整体资源 10 分钟
        config.waitsForConnectivity = true        // 等待网络恢复
        return URLSession(configuration: config)
    }()

    // ── 入口: 启动视频标注 ──
    func annotateVideo(localURL: URL, completion: @escaping (Result<URL, Error>) -> Void) {
        guard !cfg.apiToken.isEmpty else {
            main { self.status = .failed("请先设置 Replicate API Token") }; return
        }
        main { self.status = .uploading; self.progress = 0 }

        // 1) 上传本地文件到 Replicate Files API（带重试）
        uploadFileWithRetry(localURL, maxRetries: 3, attempt: 1) { [weak self] res in
            switch res {
            case .success(let remoteStr):
                self?.createPrediction(videoStr: remoteStr, completion: completion)
            case .failure(let e):
                self?.main { self?.status = .failed("上传失败(已重试3次): \(e.localizedDescription)") }
                completion(.failure(e))
            }
        }
    }

    // ── 入口: 用远程 URL 直接启动视频标注（跳过上传） ──
    func annotateVideoWithURL(remoteURL: String, localURL: URL, completion: @escaping (Result<URL, Error>) -> Void) {
        guard !cfg.apiToken.isEmpty else {
            main { self.status = .failed("请先设置 Replicate API Token") }; return
        }
        print("[SAM3] 使用远程 URL 模式，跳过上传: \(remoteURL)")
        main { self.status = .processing("使用预上传 URL，直接创建推理任务..."); self.progress = 0.1 }
        createPrediction(videoStr: remoteURL, completion: completion)
    }

    // ── 上传（带重试 + 指数退避） ──
    private func uploadFileWithRetry(_ url: URL, maxRetries: Int, attempt: Int, completion: @escaping (Result<String, Error>) -> Void) {
        print("[SAM3] 上传尝试 \(attempt)/\(maxRetries) — \(url.lastPathComponent)")
        main { self.status = .uploading; self.progress = CGFloat(attempt - 1) * 0.05 }

        uploadFileSingle(url) { [weak self] res in
            switch res {
            case .success:
                completion(res)
            case .failure(let error):
                print("[SAM3] 上传失败(\(attempt)/\(maxRetries)): \(error.localizedDescription)")
                if attempt < maxRetries {
                    // 指数退避: 5s, 10s, 20s
                    let delay = Double(5 * (1 << (attempt - 1)))
                    print("[SAM3] \(delay)秒后重试...")
                    self?.main { self?.status = .processing("上传失败，\(Int(delay))秒后第\(attempt+1)次重试...") }
                    DispatchQueue.global().asyncAfter(deadline: .now() + delay) {
                        self?.uploadFileWithRetry(url, maxRetries: maxRetries, attempt: attempt + 1, completion: completion)
                    }
                } else {
                    completion(.failure(error))
                }
            }
        }
    }

    // ── 单次上传 ──
    private func uploadFileSingle(_ url: URL, completion: @escaping (Result<String, Error>) -> Void) {
        guard let fileData = try? Data(contentsOf: url) else {
            completion(.failure(err("无法读取文件"))); return
        }
        let fileSizeMB = Double(fileData.count) / 1_048_576.0
        print("[SAM3] 文件大小: \(String(format: "%.1f", fileSizeMB)) MB")

        var req = URLRequest(url: URL(string: "https://api.replicate.com/v1/files")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(cfg.apiToken)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 300  // 5 分钟超时
        let boundary = UUID().uuidString
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"content\"; filename=\"\(url.lastPathComponent)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        req.httpBody = body

        let task = uploadSession.dataTask(with: req) { data, response, error in
            if let error {
                print("[SAM3] 网络错误: \(error.localizedDescription)")
                completion(.failure(error)); return
            }
            if let httpResp = response as? HTTPURLResponse {
                print("[SAM3] 上传响应状态码: \(httpResp.statusCode)")
            }
            guard let data, let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                let respStr = String(data: data ?? Data(), encoding: .utf8) ?? "nil"
                print("[SAM3] 上传响应解析失败: \(respStr)")
                completion(.failure(self.err("上传响应解析失败: \(respStr)"))); return
            }
            // Replicate Files API 返回 { "urls": { "get": "..." } }
            if let urls = json["urls"] as? [String: Any], let get = urls["get"] as? String {
                print("[SAM3] ✓ 上传成功: \(get)")
                completion(.success(get))
            } else if let directURL = json["url"] as? String {
                print("[SAM3] ✓ 上传成功(direct): \(directURL)")
                completion(.success(directURL))
            } else {
                let respStr = String(data: data, encoding: .utf8) ?? "nil"
                print("[SAM3] 上传响应缺少 URL: \(respStr)")
                completion(.failure(self.err("上传响应缺少 URL: \(respStr)")))
            }
        }
        task.resume()
        currentTask = task
    }

    // ── 创建预测 ──
    private func createPrediction(videoStr: String, completion: @escaping (Result<URL, Error>) -> Void) {
        main { self.status = .processing("创建 SAM3 推理任务...") }

        var req = URLRequest(url: URL(string: "https://api.replicate.com/v1/predictions")!)
        req.httpMethod = "POST"
        req.setValue("Bearer \(cfg.apiToken)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var input: [String: Any] = [
            "video": videoStr,
            "prompt": cfg.prompt,
            "return_zip": true,
            "mask_only": true,
            "mask_opacity": 0.6,
            "mask_color": "green"
        ]
        if !cfg.negativePrompt.isEmpty { input["negative_prompt"] = cfg.negativePrompt }

        let body: [String: Any] = ["version": cfg.modelVersion, "input": input]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: req) { [weak self] data, _, error in
            if let error { self?.main { self?.status = .failed(error.localizedDescription) }; completion(.failure(error)); return }
            guard let data, let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let predId = json["id"] as? String else {
                let msg = String(data: data ?? Data(), encoding: .utf8) ?? "Unknown"
                self?.main { self?.status = .failed("创建预测失败: \(msg)") }; return
            }
            self?.poll(predId: predId, completion: completion)
        }.resume()
    }

    // ── 轮询 ──
    private func poll(predId: String, completion: @escaping (Result<URL, Error>) -> Void) {
        let url = URL(string: "https://api.replicate.com/v1/predictions/\(predId)")!
        func tick() {
            var req = URLRequest(url: url)
            req.setValue("Bearer \(cfg.apiToken)", forHTTPHeaderField: "Authorization")
            URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
                guard let data, let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let st = json["status"] as? String else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) { tick() }; return
                }
                switch st {
                case "succeeded":
                    if let out = json["output"] as? String, let u = URL(string: out) {
                        self?.main { self?.status = .processing("推理完成，下载结果..."); self?.progress = 0.8 }
                        completion(.success(u))
                    }
                case "failed", "canceled":
                    let msg = (json["error"] as? String) ?? "推理失败"
                    self?.main { self?.status = .failed(msg) }
                    completion(.failure(self?.err(msg) ?? NSError()))
                default:
                    self?.main {
                        self?.status = .processing("SAM3 推理中 (\(st))...")
                        self?.progress = min((self?.progress ?? 0) + 0.02, 0.75)
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) { tick() }
                }
            }.resume()
        }
        tick()
    }

    // ── Mask → BBox 提取 (静态方法，后台线程安全) ──
    static func bboxesFromMask(_ nsImage: NSImage, minArea: Int = 30) -> [AnnBox] {
        guard let cg = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return [] }
        let W = cg.width, H = cg.height
        guard let ctx = CGContext(data: nil, width: W, height: H,
                                   bitsPerComponent: 8, bytesPerRow: W,
                                   space: CGColorSpaceCreateDeviceGray(),
                                   bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return [] }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
        guard let ptr = ctx.data else { return [] }
        let px = ptr.bindMemory(to: UInt8.self, capacity: W * H)

        // 二值化
        var bin = [Bool](repeating: false, count: W * H)
        for i in 0..<(W*H) { bin[i] = px[i] > 128 }

        // 连通域 BFS
        var labels = [Int](repeating: 0, count: W * H)
        var labelId = 0
        var comps: [(x0: Int, y0: Int, x1: Int, y1: Int, area: Int)] = []

        for y in 0..<H {
            for x in 0..<W {
                let i = y * W + x
                guard bin[i] && labels[i] == 0 else { continue }
                labelId += 1
                var x0 = x, y0 = y, x1 = x, y1 = y, area = 0
                var q = [(x, y)]; labels[i] = labelId
                while !q.isEmpty {
                    let (cx, cy) = q.removeFirst(); area += 1
                    x0 = min(x0, cx); y0 = min(y0, cy)
                    x1 = max(x1, cx); y1 = max(y1, cy)
                    for (dx, dy) in [(-1,0),(1,0),(0,-1),(0,1)] {
                        let nx = cx+dx, ny = cy+dy
                        guard nx >= 0, nx < W, ny >= 0, ny < H else { continue }
                        let ni = ny * W + nx
                        guard bin[ni], labels[ni] == 0 else { continue }
                        labels[ni] = labelId; q.append((nx, ny))
                    }
                }
                if area >= minArea { comps.append((x0, y0, x1, y1, area)) }
            }
        }

        let fW = CGFloat(W), fH = CGFloat(H)
        return comps.map { c in
            AnnBox(cx: CGFloat(c.x0 + c.x1) / 2 / fW,
                   cy: CGFloat(c.y0 + c.y1) / 2 / fH,
                   w:  CGFloat(c.x1 - c.x0) / fW,
                   h:  CGFloat(c.y1 - c.y0) / fH,
                   confidence: 0.9, source: .sam3)
        }
    }

    // helpers
    private func main(_ block: @escaping () -> Void) { DispatchQueue.main.async(execute: block) }
    private func err(_ msg: String) -> NSError { NSError(domain: "SAM3", code: -1, userInfo: [NSLocalizedDescriptionKey: msg]) }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 视频标注引擎
// ═══════════════════════════════════════════════════════════════

final class VideoAnnotationEngine: ObservableObject {
    // ── 视频 ──
    @Published var isLoaded = false
    @Published var currentFrame: Int = 0
    @Published var totalFrames: Int = 0
    @Published var fps: Double = 30
    @Published var videoSize: CGSize = .zero
    @Published var currentImage: NSImage?
    @Published var isPlaying = false

    // ── 标注 ──
    @Published var project: AnnProject?
    @Published var currentBoxes: [AnnBox] = []
    @Published var selectedBoxId: UUID?
    @Published var hasUnsaved = false
    @Published var projectFileURL: URL?
    @Published var storageConfig = AnnotatorStorageConfig.load()

    // ── 保存提示 ──
    @Published var toastMessage: String = ""
    @Published var showToast: Bool = false
    @Published var annotationSource: String = ""   // 标注来源描述
    private var toastTimer: Timer?

    // ── SAM3 ──
    @Published var sam3 = SAM3Client()
    @Published var sam3Cfg = SAM3Cfg()
    /// 已用逐帧 PTS 寻址（与主仿真 AVAssetReader 一致），在帧导航旁显示提示。
    @Published private(set) var timelineAlignedWithReader: Bool = false

    // ── 内部 ──
    private var asset: AVAsset?
    private var generator: AVAssetImageGenerator?
    private var videoURL: URL?
    private var playTimer: Timer?
    private var cache: [Int: NSImage] = [:]
    private let maxCache = 120
    /// 与 `AVAssetReader` 一致的每帧 PTS；非空时 `seekTo` 用真实时间而非 `i/fps`。
    private var presentationTimes: [CMTime] = []

    // ═══════════════════════════════════════════════════════════
    // MARK: 保存提示
    // ═══════════════════════════════════════════════════════════

    func showToastMessage(_ msg: String) {
        toastMessage = msg
        showToast = true
        toastTimer?.invalidate()
        toastTimer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: false) { [weak self] _ in
            DispatchQueue.main.async { self?.showToast = false }
        }
    }

    /// 按文件名匹配（忽略目录路径差异）
    private func videoNameMatches(_ savedPath: String, _ currentURL: URL) -> Bool {
        let savedName = (savedPath as NSString).lastPathComponent
        let currentName = currentURL.lastPathComponent
        // 完全匹配文件名
        if savedName == currentName { return true }
        // 去掉扩展名后匹配（处理 .mp4 vs .mov 等情况）
        let savedStem = (savedName as NSString).deletingPathExtension
        let currentStem = currentURL.deletingPathExtension().lastPathComponent
        return savedStem == currentStem
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 视频加载
    // ═══════════════════════════════════════════════════════════

    func loadVideo(url: URL) {
        videoURL = url
        let a = AVAsset(url: url); asset = a

        Task { @MainActor in
            do {
                let tracks = try await a.loadTracks(withMediaType: .video)
                guard let t = tracks.first else { return }
                let dur = try await a.load(.duration)
                let sz  = try await t.load(.naturalSize)
                let fr  = try await t.load(.nominalFrameRate)

                let totalSec = CMTimeGetSeconds(dur)
                let f = Double(fr)
                let n = Int(totalSec * f)

                let pts = await VideoFrameTimeline.collectPresentationTimes(asset: a, track: t)
                let frameCount = pts.isEmpty ? n : pts.count
                presentationTimes = pts
                timelineAlignedWithReader = !pts.isEmpty
                cache = [:]
                fps = f
                totalFrames = frameCount
                videoSize = sz

                let gen = AVAssetImageGenerator(asset: a)
                gen.appliesPreferredTrackTransform = true
                if pts.isEmpty {
                    gen.requestedTimeToleranceBefore = CMTime(value: 1, timescale: Int32(f * 2))
                    gen.requestedTimeToleranceAfter  = CMTime(value: 1, timescale: Int32(f * 2))
                } else {
                    gen.requestedTimeToleranceBefore = .zero
                    gen.requestedTimeToleranceAfter = .zero
                }
                gen.maximumSize = sz
                generator = gen

                let fallback = makeEmptyProject(for: url, totalFrames: frameCount, fps: f, size: sz)
                let correctedURL = defaultProjectURL(for: url)
                let originalURL = originalSAM3ProjectURL(for: url)

                // ── 尝试加载已有标注（优先 corrected，其次 original SAM3） ──
                // 使用文件名匹配而非绝对路径匹配，避免因目录不同导致加载失败
                var loaded = false

                // 1) 尝试加载 corrected 版本
                if !loaded,
                   FileManager.default.fileExists(atPath: correctedURL.path),
                   let data = try? Data(contentsOf: correctedURL),
                   var saved = try? JSONDecoder().decode(AnnProject.self, from: data),
                   videoNameMatches(saved.videoPath, url) {
                    // 更新 videoPath 为当前实际路径，确保后续保存一致
                    saved.videoPath = url.path
                    project = saved
                    projectFileURL = correctedURL
                    hasUnsaved = false
                    loaded = true
                    let frameCount = saved.frames.filter { !$0.value.isEmpty }.count
                    let boxCount = saved.frames.values.reduce(0) { $0 + $1.count }
                    annotationSource = "已自动加载修正版标注（\(frameCount) 帧, \(boxCount) 框）"
                    print("[Annotator] ✓ 自动加载修正版标注: \(correctedURL.lastPathComponent)")
                    showToastMessage("✓ 已自动加载标注（修正版，\(frameCount) 帧 \(boxCount) 框）")
                }

                // 2) 尝试加载 original SAM3 版本
                if !loaded,
                   FileManager.default.fileExists(atPath: originalURL.path),
                   let data = try? Data(contentsOf: originalURL),
                   var saved = try? JSONDecoder().decode(AnnProject.self, from: data),
                   videoNameMatches(saved.videoPath, url) {
                    saved.videoPath = url.path
                    saved.derivedFromPath = saved.derivedFromPath ?? originalURL.path
                    saved.derivedFromKind = saved.derivedFromKind ?? "sam3_original"

                    // ── 自动帧号重映射：SAM3 帧空间 → 标注器 PTS 帧空间 ──
                    let sam3Total = saved.totalFrames   // SAM3 标注时的帧数
                    let ptsTotal  = frameCount          // 标注器 PTS 收集的帧数
                    if sam3Total > 0 && ptsTotal > 0 && sam3Total != ptsTotal {
                        var remapped: [Int: [AnnBox]] = [:]
                        for (oldKey, boxes) in saved.frames {
                            // 线性映射: newKey = round(oldKey * ptsTotal / sam3Total)
                            let newKey = Int(round(Double(oldKey) * Double(ptsTotal) / Double(sam3Total)))
                            let clampedKey = max(0, min(newKey, ptsTotal - 1))
                            // 如果目标帧已有标注（两个 SAM3 帧映射到同一 PTS 帧），合并
                            if var existing = remapped[clampedKey] {
                                existing.append(contentsOf: boxes)
                                remapped[clampedKey] = existing
                            } else {
                                remapped[clampedKey] = boxes
                            }
                        }
                        saved.frames = remapped
                        print("[Annotator] ⚡ SAM3 帧号重映射: sam3Total=\(sam3Total) → ptsTotal=\(ptsTotal), 映射 \(remapped.count) 帧")
                    }

                    project = saved
                    projectFileURL = correctedURL
                    hasUnsaved = false
                    loaded = true
                    let frameCount2 = saved.frames.filter { !$0.value.isEmpty }.count
                    let boxCount = saved.frames.values.reduce(0) { $0 + $1.count }
                    annotationSource = "已自动加载 SAM3 原始标注（\(frameCount2) 帧, \(boxCount) 框）"
                    print("[Annotator] ✓ 自动加载 SAM3 原始标注: \(originalURL.lastPathComponent)")
                    showToastMessage("✓ 已自动加载标注（SAM3 原始版，\(frameCount2) 帧 \(boxCount) 框）")
                }

                // 3) 尝试在 output_dir 下按文件名搜索任何匹配的标注文件
                if !loaded, let outputRoot = storageConfig.outputDirURL {
                    let stem = url.deletingPathExtension().lastPathComponent
                    let stemDir = outputRoot.appendingPathComponent(stem)
                    if FileManager.default.fileExists(atPath: stemDir.path),
                       let contents = try? FileManager.default.contentsOfDirectory(at: stemDir, includingPropertiesForKeys: nil) {
                        // 优先找 corrected，再找普通 annotations
                        let sorted = contents.filter { $0.pathExtension == "json" }
                            .sorted { ($0.lastPathComponent.contains("corrected") ? 0 : 1) < ($1.lastPathComponent.contains("corrected") ? 0 : 1) }
                        for jsonFile in sorted {
                            if let data = try? Data(contentsOf: jsonFile),
                               var saved = try? JSONDecoder().decode(AnnProject.self, from: data),
                               videoNameMatches(saved.videoPath, url) {
                                saved.videoPath = url.path
                                if !jsonFile.lastPathComponent.contains("corrected") {
                                    saved.derivedFromPath = saved.derivedFromPath ?? jsonFile.path
                                    saved.derivedFromKind = saved.derivedFromKind ?? "auto_discovered"
                                }
                                project = saved
                                projectFileURL = correctedURL
                                hasUnsaved = false
                                loaded = true
                                let frameCount = saved.frames.filter { !$0.value.isEmpty }.count
                                let boxCount = saved.frames.values.reduce(0) { $0 + $1.count }
                                annotationSource = "已自动发现并加载标注（\(frameCount) 帧, \(boxCount) 框）"
                                print("[Annotator] ✓ 自动发现标注: \(jsonFile.lastPathComponent)")
                                showToastMessage("✓ 已自动加载标注（\(frameCount) 帧 \(boxCount) 框）")
                                break
                            }
                        }
                    }
                }

                // 4) 没有找到任何标注 → 空项目
                if !loaded {
                    project = fallback
                    projectFileURL = correctedURL
                    hasUnsaved = false
                    annotationSource = "未找到已有标注，已创建空白项目"
                    print("[Annotator] 未找到标注文件，创建空白项目")
                }

                if var p = project, p.totalFrames != totalFrames {
                    p.totalFrames = totalFrames
                    project = p
                }

                isLoaded = true
                seekTo(0)
            } catch {
                print("[Annotator] 加载视频失败: \(error)")
            }
        }
    }

    private func projectURL(for videoURL: URL, variant: AnnotationVariant) -> URL {
        let stem = videoURL.deletingPathExtension().lastPathComponent
        let fallbackName: String
        switch variant {
        case .originalSAM3:
            fallbackName = "\(stem)_annotations.json"
        case .corrected:
            fallbackName = "\(stem).corrected.annotations.json"
        }
        let fallback = videoURL.deletingPathExtension().deletingLastPathComponent().appendingPathComponent(fallbackName)
        guard let root = storageConfig.outputDirURL else { return fallback }

        let dir = root.appendingPathComponent(stem, isDirectory: true)
        switch variant {
        case .originalSAM3:
            return dir.appendingPathComponent("\(stem)_annotations.json")
        case .corrected:
            return dir.appendingPathComponent("\(stem).corrected.annotations.json")
        }
    }

    private func defaultProjectURL(for videoURL: URL) -> URL {
        projectURL(for: videoURL, variant: .corrected)
    }

    private func originalSAM3ProjectURL(for videoURL: URL) -> URL {
        projectURL(for: videoURL, variant: .originalSAM3)
    }

    private func makeEmptyProject(for videoURL: URL, totalFrames: Int, fps: Double, size: CGSize) -> AnnProject {
        AnnProject(videoPath: videoURL.path,
                   totalFrames: totalFrames,
                   fps: fps,
                   width: Int(size.width),
                   height: Int(size.height),
                   derivedFromPath: nil,
                   derivedFromKind: nil)
    }

    @discardableResult
    func saveProject(to url: URL? = nil, silent: Bool = false) -> URL? {
        guard let proj = project else { return nil }
        let target = url ?? projectFileURL ?? videoURL.map { defaultProjectURL(for: $0) }
        guard let target else { return nil }

        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try enc.encode(proj)
            try data.write(to: target, options: .atomic)
            projectFileURL = target
            hasUnsaved = false
            if !silent {
                showToastMessage("✓ 已保存标注")
            }
            return target
        } catch {
            print("[Annotator] 保存项目失败: \(error)")
            showToastMessage("⚠ 保存失败: \(error.localizedDescription)")
            return nil
        }
    }

    var configuredVideoNames: [String] {
        storageConfig.configuredVideoNames
    }

    var standardProjectPathText: String {
        if let current = projectFileURL {
            return current.path
        }
        if let videoURL {
            return defaultProjectURL(for: videoURL).path
        }
        return storageConfig.outputDirURL?.path ?? "未配置标注输出目录"
    }

    var baseAnnotationPathText: String? {
        project?.derivedFromPath
    }

    @discardableResult
    func quickLoadConfiguredVideo(named fileName: String) -> Bool {
        guard let videoDir = storageConfig.videoDirURL else { return false }
        let url = videoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        loadVideo(url: url)
        return true
    }

    func revealProjectInFinder() {
        let target = projectFileURL ?? videoURL.map { defaultProjectURL(for: $0) }
        guard let target else { return }
        try? FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([target])
    }

    var selectedBox: AnnBox? {
        guard let selectedBoxId else { return nil }
        return currentBoxes.first(where: { $0.id == selectedBoxId })
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 帧导航
    // ═══════════════════════════════════════════════════════════

    func seekTo(_ f: Int) {
        let c = max(0, min(f, totalFrames - 1))
        currentFrame = c
        if let img = cache[c] { currentImage = img; syncBoxes(); return }
        guard let gen = generator else { return }
        let time: CMTime
        if c < presentationTimes.count {
            time = presentationTimes[c]
        } else {
            time = CMTime(value: CMTimeValue(c), timescale: Int32(max(fps, 1.0)))
        }
        Task {
            do {
                let (cg, _) = try await gen.image(at: time)
                let ns = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
                await MainActor.run {
                    if self.cache.count > self.maxCache {
                        let drop = self.cache.keys.sorted().prefix(self.maxCache / 2)
                        drop.forEach { self.cache.removeValue(forKey: $0) }
                    }
                    self.cache[c] = ns
                    if self.currentFrame == c {
                        self.currentImage = ns
                        self.syncBoxes()
                    }
                }
            } catch { print("[Annotator] 提取帧 \(c) 失败") }
        }
    }

    func next()  { seekTo(currentFrame + 1) }
    func prev()  { seekTo(currentFrame - 1) }
    func jump(_ n: Int) { seekTo(currentFrame + n) }

    /// 本机第 `f` 帧在 JSON `frames` 字典中的键（已含 SAM3 偏移）
    private func annotationKey(forLocalFrame f: Int) -> Int? {
        let off = project?.sam3MaskOffset ?? 0
        let k = f + off
        guard k >= 0, k < totalFrames else { return nil }
        return k
    }

    /// 调整 SAM3 mask 帧键相对本机解码帧的偏移，直到画面对齐后可用「重映射」写入 JSON。
    func setSam3MaskOffset(_ v: Int) {
        guard var p = project else { return }
        p.sam3MaskOffset = v
        project = p
        hasUnsaved = true
        syncBoxes()
    }

    /// 按当前 `sam3MaskOffset` 将 `frames` 重写为本机帧键：`new[k] = old[k + offset]`，然后 offset 归零并保存。
    func applySam3RemapToLocalFrames() {
        guard var proj = project else { return }
        let d = proj.sam3MaskOffset
        guard d != 0 else { return }
        var newFrames: [Int: [AnnBox]] = [:]
        for (k, v) in proj.frames {
            let nk = k - d
            if nk >= 0 && nk < totalFrames {
                newFrames[nk] = v
            }
        }
        proj.frames = newFrames
        proj.sam3MaskOffset = 0
        project = proj
        hasUnsaved = true
        _ = saveProject()
        syncBoxes()
        showToastMessage("已按 SAM3 偏移 \(d) 重映射帧键并保存")
    }

    func togglePlay() {
        if isPlaying { playTimer?.invalidate(); playTimer = nil; isPlaying = false; return }
        isPlaying = true
        playTimer = Timer.scheduledTimer(withTimeInterval: 1.0/fps, repeats: true) { [weak self] _ in
            guard let s = self else { return }
            if s.currentFrame >= s.totalFrames - 1 { s.togglePlay(); return }
            s.next()
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 标注操作
    // ═══════════════════════════════════════════════════════════

    private func syncBoxes() {
        if let k = annotationKey(forLocalFrame: currentFrame) {
            currentBoxes = project?.frames[k] ?? []
        } else {
            currentBoxes = []
        }
        if let selectedBoxId, !currentBoxes.contains(where: { $0.id == selectedBoxId }) {
            self.selectedBoxId = currentBoxes.first?.id
        }
    }

    private func clampBox(_ box: inout AnnBox) {
        box.w = min(max(box.w, 0.001), 1)
        box.h = min(max(box.h, 0.001), 1)
        box.cx = min(max(box.cx, box.w / 2), 1 - box.w / 2)
        box.cy = min(max(box.cy, box.h / 2), 1 - box.h / 2)
        box.confidence = min(max(box.confidence, 0), 1)
        box.classId = max(box.classId, 0)
    }

    func addBox(_ b: AnnBox) {
        var box = b
        clampBox(&box)
        currentBoxes.append(box)
        selectedBoxId = box.id
        save()
    }

    func removeBox(_ id: UUID) {
        currentBoxes.removeAll { $0.id == id }
        if selectedBoxId == id { selectedBoxId = currentBoxes.first?.id }
        save()
    }

    func updateBox(_ id: UUID,
                   classId: Int? = nil,
                   cx: CGFloat? = nil,
                   cy: CGFloat? = nil,
                   w: CGFloat? = nil,
                   h: CGFloat? = nil,
                   confidence: Float? = nil,
                   source: AnnBox.BoxSource? = nil) {
        guard let i = currentBoxes.firstIndex(where: { $0.id == id }) else { return }
        var box = currentBoxes[i]
        if let classId { box.classId = classId }
        if let confidence { box.confidence = confidence }
        if let source {
            box.source = source
        } else if classId != nil || cx != nil || cy != nil || w != nil || h != nil || confidence != nil {
            box.source = .manual
        }
        if let w { box.w = w }
        if let h { box.h = h }
        if let cx { box.cx = cx }
        if let cy { box.cy = cy }
        clampBox(&box)
        currentBoxes[i] = box
        save()
    }

    func nudgeSelected(dx: CGFloat = 0, dy: CGFloat = 0, dw: CGFloat = 0, dh: CGFloat = 0) {
        guard let id = selectedBoxId, let box = selectedBox else { return }
        updateBox(id,
                  cx: box.cx + dx,
                  cy: box.cy + dy,
                  w: box.w + dw,
                  h: box.h + dh)
    }

    func clearFrame() {
        currentBoxes.removeAll()
        selectedBoxId = nil
        save()
    }

    /// 复制上一帧标注
    func copyPrev() {
        guard currentFrame > 0, let pk = annotationKey(forLocalFrame: currentFrame - 1),
              let prev = project?.frames[pk] else { return }
        currentBoxes = prev.map {
            AnnBox(classId: $0.classId,
                   cx: $0.cx,
                   cy: $0.cy,
                   w: $0.w,
                   h: $0.h,
                   confidence: $0.confidence,
                   source: $0.source)
        }
        selectedBoxId = currentBoxes.first?.id
        save()
    }

    func copyCurrentToNext() {
        guard currentFrame < totalFrames - 1, let nk = annotationKey(forLocalFrame: currentFrame + 1) else { return }
        let duplicated = currentBoxes.map {
            AnnBox(classId: $0.classId,
                   cx: $0.cx,
                   cy: $0.cy,
                   w: $0.w,
                   h: $0.h,
                   confidence: $0.confidence,
                   source: $0.source)
        }
        project?.frames[nk] = duplicated
        hasUnsaved = true
        _ = saveProject()
        seekTo(currentFrame + 1)
        selectedBoxId = duplicated.first?.id
    }

    /// 在两个关键帧之间线性插值
    func interpolate(from a: Int, to b: Int) {
        guard a < b,
              let ka = annotationKey(forLocalFrame: a), let kb = annotationKey(forLocalFrame: b),
              let boxesA = project?.frames[ka], let boxesB = project?.frames[kb],
              boxesA.count == boxesB.count, !boxesA.isEmpty else { return }
        let span = CGFloat(b - a)
        for f in (a+1)..<b {
            guard let kf = annotationKey(forLocalFrame: f) else { continue }
            let t = CGFloat(f - a) / span
            var interp: [AnnBox] = []
            for i in 0..<boxesA.count {
                let ba = boxesA[i], bb = boxesB[i]
                interp.append(AnnBox(
                    classId: ba.classId,
                    cx: ba.cx + (bb.cx - ba.cx) * t,
                    cy: ba.cy + (bb.cy - ba.cy) * t,
                    w:  ba.w  + (bb.w  - ba.w)  * t,
                    h:  ba.h  + (bb.h  - ba.h)  * t,
                    confidence: max(ba.confidence, bb.confidence),
                    source: .manual
                ))
            }
            project?.frames[kf] = interp
        }
        syncBoxes()
        hasUnsaved = true
        _ = saveProject()
    }

    private func save() {
        guard let k = annotationKey(forLocalFrame: currentFrame) else { return }
        if currentBoxes.isEmpty {
            project?.frames.removeValue(forKey: k)
        } else {
            project?.frames[k] = currentBoxes
        }
        hasUnsaved = true
        _ = saveProject()
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: SAM3 自动标注
    // ═══════════════════════════════════════════════════════════

    func runSAM3() {
        guard let url = videoURL else { return }
        sam3.updateCfg(sam3Cfg)
        sam3.annotateVideo(localURL: url) { [weak self] res in
            switch res {
            case .success(let zipURL): self?.downloadSAM3(zipURL)
            case .failure(let e): DispatchQueue.main.async { self?.sam3.status = .failed(e.localizedDescription) }
            }
        }
    }

    /// 用远程 URL 直接启动 SAM3（跳过上传）
    func runSAM3WithURL(_ remoteURL: String) {
        guard let url = videoURL else { return }
        sam3.updateCfg(sam3Cfg)
        sam3.annotateVideoWithURL(remoteURL: remoteURL, localURL: url) { [weak self] res in
            switch res {
            case .success(let zipURL): self?.downloadSAM3(zipURL)
            case .failure(let e): DispatchQueue.main.async { self?.sam3.status = .failed(e.localizedDescription) }
            }
        }
    }

    private func downloadSAM3(_ url: URL) {
        DispatchQueue.main.async { self.sam3.status = .processing("下载 SAM3 结果...") }
        URLSession.shared.dataTask(with: url) { [weak self] data, _, _ in
            guard let self, let data else { return }
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("sam3_\(UUID().uuidString)")
            try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
            let zipPath = tmp.appendingPathComponent("masks.zip")
            try? data.write(to: zipPath)
            self.processMaskZip(zipPath, tmpDir: tmp)
        }.resume()
    }

    private func processMaskZip(_ zip: URL, tmpDir: URL) {
        // macOS 自带 /usr/bin/ditto 或 /usr/bin/unzip
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        proc.arguments = ["-xk", zip.path, tmpDir.path]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError  = FileHandle.nullDevice
        do { try proc.run(); proc.waitUntilExit() } catch {
            DispatchQueue.main.async { self.sam3.status = .failed("解压失败") }; return
        }

        let fm = FileManager.default
        guard let all = try? fm.contentsOfDirectory(at: tmpDir, includingPropertiesForKeys: nil) else { return }
        let masks = all.filter { $0.lastPathComponent.hasPrefix("mask_") && $0.pathExtension == "png" }
                       .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let total = masks.count
        var result: [Int: [AnnBox]] = [:]

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            for (idx, mf) in masks.enumerated() {
                let num = mf.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "mask_", with: "")
                guard let fi = Int(num), let img = NSImage(contentsOf: mf) else { continue }
                let boxes = SAM3Client.bboxesFromMask(img, minArea: self.sam3Cfg.minMaskArea)
                if !boxes.isEmpty { result[fi] = boxes }
                if idx % 50 == 0 {
                    DispatchQueue.main.async { self.sam3.status = .extracting(idx, total); self.sam3.progress = CGFloat(idx)/CGFloat(total) }
                }
            }
            DispatchQueue.main.async {
                var cnt = 0
                for (fi, boxes) in result {
                    var cur = self.project?.frames[fi] ?? []
                    cur.append(contentsOf: boxes)
                    self.project?.frames[fi] = cur
                    cnt += boxes.count
                }
                self.syncBoxes(); self.hasUnsaved = true
                self.sam3.status = .done(cnt); self.sam3.progress = 1
                try? fm.removeItem(at: tmpDir)
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 导出
    // ═══════════════════════════════════════════════════════════

    /// 导出 YOLO 格式 — images/ + labels/ + classes.txt
    func exportYOLO(to dir: URL) {
        guard let proj = project, let gen = generator else { return }
        let imgDir = dir.appendingPathComponent("images")
        let lblDir = dir.appendingPathComponent("labels")
        try? FileManager.default.createDirectory(at: imgDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: lblDir, withIntermediateDirectories: true)
        try? proj.className.write(to: dir.appendingPathComponent("classes.txt"), atomically: true, encoding: .utf8)

        let annotated = proj.frames.filter { !$0.value.isEmpty }.keys.sorted()
        for fi in annotated {
            let boxes = proj.frames[fi] ?? []; if boxes.isEmpty { continue }
            let name = String(format: "frame_%06d", fi)
            let time: CMTime
            if fi < presentationTimes.count {
                time = presentationTimes[fi]
            } else {
                time = CMTime(value: CMTimeValue(fi), timescale: Int32(max(fps, 1.0)))
            }
            if let cg = try? gen.copyCGImage(at: time, actualTime: nil) {
                let ns = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
                if let tiff = ns.tiffRepresentation,
                   let bmp = NSBitmapImageRep(data: tiff),
                   let jpg = bmp.representation(using: .jpeg, properties: [.compressionFactor: 0.95]) {
                    try? jpg.write(to: imgDir.appendingPathComponent("\(name).jpg"))
                }
            }
            let txt = boxes.map(\.yoloLine).joined(separator: "\n")
            try? txt.write(to: lblDir.appendingPathComponent("\(name).txt"), atomically: true, encoding: .utf8)
        }
        showToastMessage("✓ 已导出 YOLO 数据集（\(annotated.count) 帧）")
    }

    /// 导出项目 JSON
    func exportJSON(to url: URL) {
        _ = saveProject(to: url)
    }

    /// 导入项目 JSON
    func importJSON(from url: URL) {
        guard let d = try? Data(contentsOf: url),
              var p = try? JSONDecoder().decode(AnnProject.self, from: d) else {
            showToastMessage("⚠ 导入失败：无法解析标注文件")
            return
        }
        let importedPath = url.path
        let correctedTarget = videoURL.map { defaultProjectURL(for: $0) } ?? url.deletingLastPathComponent().appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".corrected.json")
        let isAlreadyCorrected = url.lastPathComponent.contains(".corrected.")
        if !isAlreadyCorrected {
            p.derivedFromPath = p.derivedFromPath ?? importedPath
            p.derivedFromKind = p.derivedFromKind ?? "imported_base"
        }
        project = p
        projectFileURL = isAlreadyCorrected ? url : correctedTarget
        hasUnsaved = false
        syncBoxes()
        let frameCount = p.frames.filter { !$0.value.isEmpty }.count
        let boxCount = p.frames.values.reduce(0) { $0 + $1.count }
        annotationSource = "已导入标注（\(frameCount) 帧, \(boxCount) 框）"
        showToastMessage("✓ 已导入标注（\(frameCount) 帧 \(boxCount) 框）")
    }

    /// 统计
    var stats: (annotated: Int, boxes: Int, coverage: Double) {
        guard let p = project else { return (0,0,0) }
        let a = p.frames.filter { !$0.value.isEmpty }.count
        let b = p.frames.values.reduce(0) { $0 + $1.count }
        return (a, b, p.totalFrames > 0 ? Double(a)/Double(p.totalFrames) : 0)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 标注画布 (手动画框 + 显示)
// ═══════════════════════════════════════════════════════════════

struct AnnCanvas: View {
    @ObservedObject var engine: VideoAnnotationEngine

    private enum ResizeHandle: CaseIterable, Hashable {
        case topLeft, topRight, bottomLeft, bottomRight
    }

    private enum CanvasDragMode {
        case draw
        case move(UUID)
        case resize(UUID, ResizeHandle)
    }

    @State private var dragStart: CGPoint?
    @State private var dragCurrent: CGPoint?
    @State private var isDragging = false
    @State private var dragMode: CanvasDragMode?
    @State private var boxAtGestureStart: AnnBox?

    private let handleSize: CGFloat = 10
    private let minimumBoxSize: CGFloat = 0.005

    var body: some View {
        GeometryReader { geo in
            let imgSz = engine.videoSize
            let fitR  = fitRect(imgSize: imgSz, viewSize: geo.size)

            ZStack(alignment: .topLeading) {
                Color.black

                // 视频帧
                if let img = engine.currentImage {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: fitR.width, height: fitR.height)
                        .position(x: fitR.midX, y: fitR.midY)
                }

                // 已有标注框 — 支持点击选中、拖动移动、角点缩放
                ForEach(engine.currentBoxes) { box in
                    let sr = screenRect(for: box, fitR: fitR, imgSize: imgSz)

                    Rectangle()
                        .stroke(boxColor(box), lineWidth: engine.selectedBoxId == box.id ? 3 : 2)
                        .background(Rectangle().fill(boxColor(box).opacity(0.08)))
                        .frame(width: sr.width, height: sr.height)
                        .position(x: sr.midX, y: sr.midY)
                        .onTapGesture { engine.selectedBoxId = box.id }

                    if engine.selectedBoxId == box.id {
                        ForEach(ResizeHandle.allCases, id: \.self) { handle in
                            let hp = handlePoint(handle, rect: sr)
                            Rectangle()
                                .fill(Color.white)
                                .overlay(
                                    Rectangle()
                                        .stroke(Color.yellow, lineWidth: 1)
                                )
                                .frame(width: handleSize, height: handleSize)
                                .position(x: hp.x, y: hp.y)
                        }
                    }

                    Text(boxLabel(box))
                        .font(.caption2.bold())
                        .foregroundColor(.white)
                        .padding(.horizontal, 4).padding(.vertical, 1)
                        .background(boxColor(box).opacity(0.85))
                        .cornerRadius(3)
                        .position(x: sr.midX, y: max(sr.minY - 8, fitR.minY + 8))
                }

                HStack(spacing: 6) {
                    Text(String(format: "F%d  %.2fs", engine.currentFrame, Double(engine.currentFrame)/max(engine.fps,1)))
                        .font(.caption2.monospaced())
                        .foregroundColor(.white.opacity(0.8))
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(3)
                    if !engine.currentBoxes.isEmpty {
                        Text("\(engine.currentBoxes.count) 框")
                            .font(.caption2.bold())
                            .foregroundColor(.green)
                            .padding(.horizontal, 5).padding(.vertical, 2)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(3)
                    }
                }
                .position(x: fitR.minX + 60, y: fitR.minY + 14)

                if case .draw = dragMode, isDragging, let s = dragStart, let c = dragCurrent {
                    Rectangle()
                        .stroke(Color.yellow, lineWidth: 2)
                        .background(Rectangle().fill(Color.yellow.opacity(0.12)))
                        .frame(width: abs(c.x - s.x), height: abs(c.y - s.y))
                        .position(x: (s.x + c.x)/2, y: (s.y + c.y)/2)
                }
            }
            .contentShape(Rectangle())
            .highPriorityGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if dragMode == nil {
                            beginInteraction(at: value.startLocation, fitR: fitR, imgSize: imgSz)
                        }
                        updateInteraction(with: value, fitR: fitR)
                    }
                    .onEnded { value in
                        finishInteraction(with: value, fitR: fitR)
                    }
            )
            .clipped()
        }
    }

    private func beginInteraction(at point: CGPoint, fitR: CGRect, imgSize: CGSize) {
        guard fitR.contains(point) else {
            resetInteraction()
            return
        }

        dragStart = clamped(point, to: fitR)
        dragCurrent = dragStart
        isDragging = true

        if let selected = engine.selectedBox {
            let rect = screenRect(for: selected, fitR: fitR, imgSize: imgSize)
            if let handle = handleHit(at: point, in: rect) {
                dragMode = .resize(selected.id, handle)
                boxAtGestureStart = selected
                return
            }
        }

        if let hit = hitTestBox(at: point, fitR: fitR, imgSize: imgSize) {
            engine.selectedBoxId = hit.id
            dragMode = .move(hit.id)
            boxAtGestureStart = hit
            return
        }

        dragMode = .draw
        boxAtGestureStart = nil
    }

    private func updateInteraction(with value: DragGesture.Value, fitR: CGRect) {
        guard fitR.width > 0, fitR.height > 0, let dragMode else { return }

        switch dragMode {
        case .draw:
            dragCurrent = clamped(value.location, to: fitR)

        case .move(let id):
            guard let startBox = boxAtGestureStart else { return }
            let dx = value.translation.width / fitR.width
            let dy = value.translation.height / fitR.height
            engine.updateBox(id,
                             cx: startBox.cx + dx,
                             cy: startBox.cy + dy)

        case .resize(let id, let handle):
            guard let startBox = boxAtGestureStart else { return }
            let dx = value.translation.width / fitR.width
            let dy = value.translation.height / fitR.height
            let resized = resizedBox(startBox, handle: handle, dx: dx, dy: dy)
            engine.updateBox(id,
                             cx: resized.cx,
                             cy: resized.cy,
                             w: resized.w,
                             h: resized.h)
        }
    }

    private func finishInteraction(with value: DragGesture.Value, fitR: CGRect) {
        defer { resetInteraction() }
        guard let dragMode else { return }

        switch dragMode {
        case .draw:
            guard let s = dragStart else { return }
            let e = clamped(value.location, to: fitR)
            let x1 = (min(s.x, e.x) - fitR.minX) / fitR.width
            let y1 = (min(s.y, e.y) - fitR.minY) / fitR.height
            let x2 = (max(s.x, e.x) - fitR.minX) / fitR.width
            let y2 = (max(s.y, e.y) - fitR.minY) / fitR.height
            let cx = clamp01((x1 + x2) / 2)
            let cy = clamp01((y1 + y2) / 2)
            let bw = clamp01(x2 - x1)
            let bh = clamp01(y2 - y1)
            if bw > minimumBoxSize && bh > minimumBoxSize {
                engine.addBox(AnnBox(cx: cx, cy: cy, w: bw, h: bh))
            }

        case .move, .resize:
            break
        }
    }

    private func resetInteraction() {
        isDragging = false
        dragMode = nil
        dragStart = nil
        dragCurrent = nil
        boxAtGestureStart = nil
    }

    private func hitTestBox(at point: CGPoint, fitR: CGRect, imgSize: CGSize) -> AnnBox? {
        for box in engine.currentBoxes.reversed() {
            if screenRect(for: box, fitR: fitR, imgSize: imgSize).contains(point) {
                return box
            }
        }
        return nil
    }

    private func handleHit(at point: CGPoint, in rect: CGRect) -> ResizeHandle? {
        ResizeHandle.allCases.first { handleRect($0, rect: rect).contains(point) }
    }

    private func resizedBox(_ startBox: AnnBox, handle: ResizeHandle, dx: CGFloat, dy: CGFloat) -> AnnBox {
        var left = startBox.cx - startBox.w / 2
        var right = startBox.cx + startBox.w / 2
        var top = startBox.cy - startBox.h / 2
        var bottom = startBox.cy + startBox.h / 2

        switch handle {
        case .topLeft:
            left = clamp(left + dx, min: 0, max: right - minimumBoxSize)
            top = clamp(top + dy, min: 0, max: bottom - minimumBoxSize)
        case .topRight:
            right = clamp(right + dx, min: left + minimumBoxSize, max: 1)
            top = clamp(top + dy, min: 0, max: bottom - minimumBoxSize)
        case .bottomLeft:
            left = clamp(left + dx, min: 0, max: right - minimumBoxSize)
            bottom = clamp(bottom + dy, min: top + minimumBoxSize, max: 1)
        case .bottomRight:
            right = clamp(right + dx, min: left + minimumBoxSize, max: 1)
            bottom = clamp(bottom + dy, min: top + minimumBoxSize, max: 1)
        }

        return AnnBox(classId: startBox.classId,
                      cx: (left + right) / 2,
                      cy: (top + bottom) / 2,
                      w: right - left,
                      h: bottom - top,
                      confidence: startBox.confidence,
                      source: .manual)
    }

    private func screenRect(for box: AnnBox, fitR: CGRect, imgSize: CGSize) -> CGRect {
        guard imgSize.width > 0, imgSize.height > 0 else { return .zero }
        let nr = box.pixelRect(imgW: imgSize.width, imgH: imgSize.height)
        return CGRect(x: fitR.minX + nr.minX / imgSize.width * fitR.width,
                      y: fitR.minY + nr.minY / imgSize.height * fitR.height,
                      width: nr.width / imgSize.width * fitR.width,
                      height: nr.height / imgSize.height * fitR.height)
    }

    private func handlePoint(_ handle: ResizeHandle, rect: CGRect) -> CGPoint {
        switch handle {
        case .topLeft: return CGPoint(x: rect.minX, y: rect.minY)
        case .topRight: return CGPoint(x: rect.maxX, y: rect.minY)
        case .bottomLeft: return CGPoint(x: rect.minX, y: rect.maxY)
        case .bottomRight: return CGPoint(x: rect.maxX, y: rect.maxY)
        }
    }

    private func handleRect(_ handle: ResizeHandle, rect: CGRect) -> CGRect {
        let p = handlePoint(handle, rect: rect)
        return CGRect(x: p.x - handleSize / 2,
                      y: p.y - handleSize / 2,
                      width: handleSize,
                      height: handleSize)
    }

    private func clamped(_ point: CGPoint, to rect: CGRect) -> CGPoint {
        CGPoint(x: clamp(point.x, min: rect.minX, max: rect.maxX),
                y: clamp(point.y, min: rect.minY, max: rect.maxY))
    }

    // ── 与 VirtualRaceUI 相同的 fitRect ──
    private func fitRect(imgSize: CGSize, viewSize: CGSize) -> CGRect {
        guard imgSize.width > 0, imgSize.height > 0 else {
            return CGRect(origin: .zero, size: viewSize)
        }
        let scale = min(viewSize.width / imgSize.width, viewSize.height / imgSize.height)
        let w = imgSize.width * scale, h = imgSize.height * scale
        return CGRect(x: (viewSize.width - w)/2, y: (viewSize.height - h)/2, width: w, height: h)
    }

    private func boxColor(_ b: AnnBox) -> Color {
        if engine.selectedBoxId == b.id { return .yellow }
        return b.source == .sam3 ? .cyan : .green
    }

    private func boxLabel(_ b: AnnBox) -> String {
        let src = b.source == .sam3 ? "SAM3" : "手动"
        return "\(src) \(String(format: "%.0f%%", b.confidence * 100))"
    }

    private func clamp(_ v: CGFloat, min lo: CGFloat, max hi: CGFloat) -> CGFloat {
        Swift.min(Swift.max(v, lo), hi)
    }

    private func clamp01(_ v: CGFloat) -> CGFloat { clamp(v, min: 0, max: 1) }
}


// ═══════════════════════════════════════════════════════════════
// MARK: - 主视图: VideoAnnotationView (Tab 页)
// ═══════════════════════════════════════════════════════════════

struct VideoAnnotationView: View {
    @StateObject private var engine = VideoAnnotationEngine()
    @State private var showOpen = false
    @State private var showSAM3Sheet = false
    @State private var jumpStep: Int = 30
    @State private var interpFrom: String = ""
    @State private var interpTo: String = ""
    @State private var geomStep: Double = 0.002
    @State private var keyboardMonitor: Any?

    var body: some View {
        ZStack(alignment: .top) {
            VStack(spacing: 0) {
                headerBar
                Divider()
                HSplitView {
                    // ═══ 左: 画布 + 帧导航 ═══
                    VStack(spacing: 0) {
                        if engine.isLoaded {
                            // 标注来源信息栏
                            if !engine.annotationSource.isEmpty {
                                HStack(spacing: 6) {
                                    Image(systemName: "info.circle.fill")
                                        .foregroundColor(.blue)
                                        .font(.caption)
                                    Text(engine.annotationSource)
                                        .font(.caption)
                                        .foregroundColor(.primary)
                                    Spacer()
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 4)
                                .background(Color.blue.opacity(0.08))
                            }
                            AnnCanvas(engine: engine)
                                .frame(minWidth: 640, minHeight: 480)
                            frameNav
                        } else {
                            emptyState
                        }
                    }
                    .frame(minWidth: 700)

                    // ═══ 右: 标注列表 + 工具 ═══
                    rightPanel
                        .frame(minWidth: 280, maxWidth: 350)
                }
            }

            // ═══ Toast 保存提示浮层 ═══
            if engine.showToast {
                VStack {
                    Spacer().frame(height: 60)
                    Text(engine.toastMessage)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.black.opacity(0.8))
                                .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
                        )
                        .transition(.move(edge: .top).combined(with: .opacity))
                        .animation(.easeInOut(duration: 0.3), value: engine.showToast)
                        .zIndex(999)
                    Spacer()
                }
                .allowsHitTesting(false)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers {
                p.loadItem(forTypeIdentifier: "public.file-url", options: nil) { data, _ in
                    guard let data = data as? Data, let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                    DispatchQueue.main.async { engine.loadVideo(url: url) }
                }
            }
            return true
        }
        .fileImporter(isPresented: $showOpen,
                      allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .avi],
                      allowsMultipleSelection: false) { res in
            if case .success(let urls) = res, let url = urls.first {
                _ = url.startAccessingSecurityScopedResource()
                engine.loadVideo(url: url)
            }
        }
        .sheet(isPresented: $showSAM3Sheet) {
            SAM3Sheet(cfg: $engine.sam3Cfg, isPresented: $showSAM3Sheet)
        }
        .onAppear { installKeyboardMonitor() }
        .onDisappear {
            removeKeyboardMonitor()
            _ = engine.saveProject(silent: true)
        }
    }

    private func installKeyboardMonitor() {
        guard keyboardMonitor == nil else { return }
        keyboardMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
            handleKeyEvent(event) ? nil : event
        }
    }

    private func removeKeyboardMonitor() {
        guard let keyboardMonitor else { return }
        NSEvent.removeMonitor(keyboardMonitor)
        self.keyboardMonitor = nil
    }

    private func handleKeyEvent(_ event: NSEvent) -> Bool {
        guard engine.isLoaded else { return false }

        if let responder = NSApp.keyWindow?.firstResponder as? NSTextView,
           responder.isEditable {
            return false
        }

        if event.modifierFlags.intersection([.command, .option, .control]).isEmpty == false {
            return false
        }

        switch event.keyCode {
        case 0: // a
            engine.prev()
            return true
        case 2: // d
            engine.next()
            return true
        case 49: // space
            engine.togglePlay()
            return true
        case 51, 117: // delete / forward delete
            guard let id = engine.selectedBoxId else { return false }
            engine.removeBox(id)
            return true
        case 123: // left arrow
            engine.prev()
            return true
        case 124: // right arrow
            engine.next()
            return true
        default:
            return false
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 标题栏
    // ═══════════════════════════════════════════════════════════

    private var headerBar: some View {
        HStack(spacing: 12) {
            Image(systemName: "pencil.and.ruler").foregroundColor(.orange)
            Text("视频标注 Video Annotator").font(.headline)

            Divider().frame(height: 18)

            Button(action: { showOpen = true }) { Label("打开视频", systemImage: "film") }

            if !engine.configuredVideoNames.isEmpty {
                Menu {
                    ForEach(engine.configuredVideoNames, id: \.self) { name in
                        Button(name) { _ = engine.quickLoadConfiguredVideo(named: name) }
                    }
                } label: {
                    Label("快捷打开", systemImage: "list.bullet.rectangle")
                }
            }

            Button(action: { _ = engine.quickLoadConfiguredVideo(named: "上海one 场地 rc.mp4") }) {
                Label("打开上海 onerc", systemImage: "flag.checkered.2.crossed")
            }
            .disabled(engine.storageConfig.videoDirURL == nil)

            Divider().frame(height: 18)

            // SAM3
            Button(action: {
                if engine.sam3Cfg.apiToken.isEmpty { showSAM3Sheet = true } else { engine.runSAM3() }
            }) { Label("SAM3 自动标注", systemImage: "wand.and.stars") }
            .disabled(!engine.isLoaded)

            Button(action: { showSAM3Sheet = true }) { Image(systemName: "gearshape") }

            sam3Badge

            if engine.isLoaded {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Image(systemName: engine.hasUnsaved ? "circle.fill" : "checkmark.circle.fill")
                            .foregroundColor(engine.hasUnsaved ? .orange : .green)
                            .font(.system(size: 8))
                        Text(engine.projectFileURL?.lastPathComponent ?? "未命名修订版")
                            .font(.caption.monospaced())
                            .foregroundColor(.secondary)
                    }
                    Text("修订版保存路径：\(engine.standardProjectPathText)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .help(engine.standardProjectPathText)
                    if let base = engine.baseAnnotationPathText {
                        Text("基线标注来源：\(base)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .help(base)
                    }
                }
            }

            Spacer()

            Button(action: { doSaveProject() }) {
                Label(engine.hasUnsaved ? "保存项目" : "另存项目", systemImage: "square.and.arrow.down")
            }
            .disabled(!engine.isLoaded)

            Button(action: { engine.revealProjectInFinder() }) {
                Label("打开保存位置", systemImage: "folder")
            }
            .disabled(!engine.isLoaded)

            // 导出菜单
            Menu {
                Button("保存/另存标注 JSON...") { doSaveProject(forcePanel: true) }
                Button("导出 YOLO 数据集...") { doExportYOLO() }
                Divider()
                Button("导入标注 JSON...") { doImportJSON() }
            } label: { Label("导出", systemImage: "square.and.arrow.up") }
            .disabled(!engine.isLoaded)

            // 统计信息
            if engine.isLoaded {
                let s = engine.stats
                VStack(alignment: .trailing, spacing: 1) {
                    Text("已标注 \(s.annotated) 帧 | 共 \(s.boxes) 框")
                        .font(.caption.monospaced()).foregroundColor(.secondary)
                    Text("覆盖率 \(String(format: "%.1f%%", s.coverage*100))")
                        .font(.system(size: 9, design: .monospaced)).foregroundColor(.secondary)
                }
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: SAM3 状态徽章
    // ═══════════════════════════════════════════════════════════

    @ViewBuilder
    private var sam3Badge: some View {
        switch engine.sam3.status {
        case .idle: EmptyView()
        case .uploading:
            HStack(spacing: 4) { ProgressView().controlSize(.small); Text("上传中...").font(.caption) }
        case .processing(let m):
            HStack(spacing: 4) { ProgressView().controlSize(.small); Text(m).font(.caption).lineLimit(1) }
        case .extracting(let f, let t):
            HStack(spacing: 4) { ProgressView(value: Double(f), total: Double(t)).frame(width: 80); Text("提取 \(f)/\(t)").font(.caption) }
        case .done(let n):
            HStack(spacing: 4) { Image(systemName: "checkmark.circle.fill").foregroundColor(.green); Text("完成 \(n) 框").font(.caption) }
        case .failed(let e):
            HStack(spacing: 4) { Image(systemName: "exclamationmark.triangle.fill").foregroundColor(.red); Text(e).font(.caption).lineLimit(1) }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 帧导航条
    // ═══════════════════════════════════════════════════════════

    private var frameNav: some View {
        VStack(spacing: 4) {
            HStack {
                Text(String(format: "帧 %d / %d", engine.currentFrame, engine.totalFrames))
                    .font(.caption.monospaced())
                Spacer()
                if engine.timelineAlignedWithReader {
                    Text("PTS 时间轴")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.trailing, 6)
                }
                Text(String(format: "%.2f fps | %dx%d", engine.fps, Int(engine.videoSize.width), Int(engine.videoSize.height)))
                    .font(.caption.monospaced()).foregroundColor(.secondary)
            }
            .padding(.horizontal, 12)

            HStack(spacing: 10) {
                Text("SAM3 帧偏移").font(.caption)
                Stepper(value: Binding(
                    get: { engine.project?.sam3MaskOffset ?? 0 },
                    set: { engine.setSam3MaskOffset($0) }
                ), in: -120...120, step: 1) {
                    Text("\(engine.project?.sam3MaskOffset ?? 0)")
                        .font(.caption.monospaced())
                        .frame(width: 40, alignment: .trailing)
                }
                .disabled(!engine.isLoaded || engine.project == nil)
                Button("重映射并保存") {
                    engine.applySam3RemapToLocalFrames()
                }
                .font(.caption)
                .disabled(!engine.isLoaded || (engine.project?.sam3MaskOffset ?? 0) == 0)
            }
            .padding(.horizontal, 12)
            Text("框相对车滞后则增大偏移，超前则减小；对齐后点「重映射」把键位写回 JSON 并归零偏移。")
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)

            HStack(spacing: 8) {
                Button(action: { engine.prev() }) { Image(systemName: "backward.frame") }
                    .keyboardShortcut("a", modifiers: [])
                Button(action: { engine.togglePlay() }) {
                    Image(systemName: engine.isPlaying ? "pause.fill" : "play.fill")
                }
                Button(action: { engine.next() }) { Image(systemName: "forward.frame") }
                    .keyboardShortcut("d", modifiers: [])
            }

            Slider(value: Binding(
                get: { Double(engine.currentFrame) },
                set: { engine.seekTo(Int($0)) }
            ), in: 0...max(1, Double(engine.totalFrames - 1)), step: 1)
            .padding(.horizontal, 12)

            HStack(spacing: 8) {
                Text("跳帧步长").font(.caption)
                Stepper(value: $jumpStep, in: 1...300) {
                    Text("\(jumpStep)").font(.caption.monospaced()).frame(width: 36)
                }
                Button("← \(jumpStep)") { engine.seekTo(max(0, engine.currentFrame - jumpStep)) }.font(.caption)
                Button("\(jumpStep) →") { engine.seekTo(min(engine.totalFrames - 1, engine.currentFrame + jumpStep)) }.font(.caption)
            }
            .padding(.horizontal, 12).padding(.bottom, 4)
        }
        .padding(.vertical, 4)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 空状态
    // ═══════════════════════════════════════════════════════════

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "film").font(.system(size: 64)).foregroundColor(.secondary)
            Text("打开视频或拖放到此处开始标注").font(.title2).foregroundColor(.secondary)
            Text("支持 MP4 / MOV / AVI 格式").font(.caption).foregroundColor(.secondary)
            VStack(spacing: 4) {
                Text("打开视频后会自动加载对应的已有标注")
                    .font(.caption).foregroundColor(.secondary)
                Text("播放时标注框会叠加显示在视频上")
                    .font(.caption).foregroundColor(.secondary)
            }
            Button("选择视频文件...") { showOpen = true }.buttonStyle(.borderedProminent)
            if !engine.configuredVideoNames.isEmpty {
                Text("或从上方【快捷打开】菜单选择已配置的视频")
                    .font(.system(size: 10)).foregroundColor(.secondary.opacity(0.7))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 右侧面板
    // ═══════════════════════════════════════════════════════════

    private var rightPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // ── 当前帧标注列表 ──
                GroupBox("当前帧标注 (\(engine.currentBoxes.count))") {
                    if engine.currentBoxes.isEmpty {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("当前帧无标注")
                                .font(.caption).foregroundColor(.secondary)
                            Text("在画面上拖拽即可新建标注框，或点击【复制上帧】沿用上一帧标注")
                                .font(.system(size: 9)).foregroundColor(.secondary.opacity(0.7))
                        }.padding(4)
                    } else {
                        VStack(spacing: 4) {
                            ForEach(engine.currentBoxes) { box in
                                HStack(spacing: 6) {
                                    Circle().fill(box.source == .sam3 ? Color.cyan : Color.green).frame(width: 8, height: 8)
                                    VStack(alignment: .leading, spacing: 1) {
                                        Text("RC Car").font(.caption.monospaced())
                                        Text(String(format: "(%.3f, %.3f, %.3f, %.3f)", box.cx, box.cy, box.w, box.h))
                                            .font(.system(size: 9, design: .monospaced)).foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    Text(box.source == .sam3 ? "SAM3" : "手动")
                                        .font(.system(size: 9)).padding(.horizontal, 5).padding(.vertical, 2)
                                        .background(box.source == .sam3 ? Color.cyan.opacity(0.15) : Color.green.opacity(0.15))
                                        .cornerRadius(3)
                                    Button(action: { engine.removeBox(box.id) }) {
                                        Image(systemName: "trash").font(.caption).foregroundColor(.red)
                                    }.buttonStyle(.plain)
                                }
                                .padding(.vertical, 2).padding(.horizontal, 4)
                                .background(engine.selectedBoxId == box.id ? Color.accentColor.opacity(0.1) : Color.clear)
                                .cornerRadius(4)
                                .onTapGesture { engine.selectedBoxId = box.id }
                            }
                        }
                    }
                }

                // ── 选中框编辑 ──
                GroupBox("选中框编辑") {
                    if let box = engine.selectedBox {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("ID").font(.caption).foregroundColor(.secondary)
                                Text(box.id.uuidString.prefix(8))
                                    .font(.system(size: 10, design: .monospaced))
                                Spacer()
                                Button(role: .destructive, action: { engine.removeBox(box.id) }) {
                                    Label("删除", systemImage: "trash")
                                }
                                .font(.caption)
                            }

                            HStack(spacing: 8) {
                                Text("来源").font(.caption).frame(width: 36, alignment: .leading)
                                Picker("来源", selection: Binding(
                                    get: { box.source },
                                    set: { engine.updateBox(box.id, source: $0) }
                                )) {
                                    ForEach(AnnBox.BoxSource.allCases, id: \.self) { src in
                                        Text(src == .sam3 ? "SAM3" : "手动").tag(src)
                                    }
                                }
                                .pickerStyle(.segmented)
                            }

                            HStack(spacing: 8) {
                                Text("类别").font(.caption).frame(width: 36, alignment: .leading)
                                Stepper(value: Binding(
                                    get: { box.classId },
                                    set: { engine.updateBox(box.id, classId: $0) }
                                ), in: 0...99) {
                                    Text("class \(box.classId)")
                                        .font(.caption.monospaced())
                                }
                            }

                            scalarEditor("中心 X", value: Double(box.cx), range: 0...1, step: 0.001) {
                                engine.updateBox(box.id, cx: CGFloat($0))
                            }
                            scalarEditor("中心 Y", value: Double(box.cy), range: 0...1, step: 0.001) {
                                engine.updateBox(box.id, cy: CGFloat($0))
                            }
                            scalarEditor("宽度 W", value: Double(box.w), range: 0.001...1, step: 0.001) {
                                engine.updateBox(box.id, w: CGFloat($0))
                            }
                            scalarEditor("高度 H", value: Double(box.h), range: 0.001...1, step: 0.001) {
                                engine.updateBox(box.id, h: CGFloat($0))
                            }
                            scalarEditor("置信度", value: Double(box.confidence), range: 0...1, step: 0.01, digits: 2) {
                                engine.updateBox(box.id, confidence: Float($0))
                            }

                            Divider()

                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("微调步长").font(.caption)
                                    Slider(value: $geomStep, in: 0.001...0.02)
                                    Text(String(format: "%.3f", geomStep)).font(.caption.monospaced()).frame(width: 42)
                                }
                                HStack(spacing: 6) {
                                    nudgeButton("←", systemImage: "arrow.left", action: { engine.nudgeSelected(dx: -CGFloat(geomStep)) })
                                    nudgeButton("→", systemImage: "arrow.right", action: { engine.nudgeSelected(dx: CGFloat(geomStep)) })
                                    nudgeButton("↑", systemImage: "arrow.up", action: { engine.nudgeSelected(dy: -CGFloat(geomStep)) })
                                    nudgeButton("↓", systemImage: "arrow.down", action: { engine.nudgeSelected(dy: CGFloat(geomStep)) })
                                }
                                HStack(spacing: 6) {
                                    nudgeButton("变窄", systemImage: "arrow.left.and.right.righttriangle.left.righttriangle.right", action: { engine.nudgeSelected(dw: -CGFloat(geomStep)) })
                                    nudgeButton("变宽", systemImage: "arrow.left.and.right.righttriangle.left.righttriangle.right.fill", action: { engine.nudgeSelected(dw: CGFloat(geomStep)) })
                                }
                                HStack(spacing: 6) {
                                    nudgeButton("变矮", systemImage: "arrow.up.and.down.righttriangle.up.righttriangle.down", action: { engine.nudgeSelected(dh: -CGFloat(geomStep)) })
                                    nudgeButton("变高", systemImage: "arrow.up.and.down.righttriangle.up.righttriangle.down.fill", action: { engine.nudgeSelected(dh: CGFloat(geomStep)) })
                                }
                            }
                        }
                        .padding(4)
                    } else {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("先在画面中点击一个框，随后即可编辑。")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("提示：修改标注后会自动保存，并在顶部显示【已保存标注】确认。")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary.opacity(0.8))
                        }
                        .padding(4)
                    }
                }

                // ── 快捷操作 ──
                GroupBox("快捷操作") {
                    VStack(spacing: 6) {
                        HStack(spacing: 8) {
                            Button(action: { engine.copyPrev() }) {
                                Label("复制上帧", systemImage: "doc.on.doc")
                            }.font(.caption).disabled(engine.currentFrame == 0)

                            Button(action: { engine.copyCurrentToNext() }) {
                                Label("复制到下一帧", systemImage: "arrow.right.doc.on.clipboard")
                            }.font(.caption).disabled(engine.currentBoxes.isEmpty || engine.currentFrame >= engine.totalFrames - 1)
                        }

                        HStack(spacing: 8) {
                            Button(action: { engine.clearFrame() }) {
                                Label("清空当前帧", systemImage: "trash")
                            }.font(.caption).disabled(engine.currentBoxes.isEmpty)

                            Button(action: { _ = engine.saveProject() }) {
                                Label(engine.hasUnsaved ? "保存当前项目" : "已保存", systemImage: "square.and.arrow.down")
                            }
                            .font(.caption)
                            .disabled(!engine.hasUnsaved)
                        }

                        Divider()

                        // 插值
                        VStack(alignment: .leading, spacing: 4) {
                            Text("线性插值").font(.caption.bold())
                            HStack(spacing: 4) {
                                Text("从帧").font(.caption)
                                TextField("起始", text: $interpFrom).frame(width: 50).textFieldStyle(.roundedBorder).font(.caption)
                                Text("到帧").font(.caption)
                                TextField("结束", text: $interpTo).frame(width: 50).textFieldStyle(.roundedBorder).font(.caption)
                                Button("插值") {
                                    guard let a = Int(interpFrom), let b = Int(interpTo) else { return }
                                    engine.interpolate(from: a, to: b)
                                }.font(.caption)
                            }
                        }
                    }.padding(4)
                }

                // ── 快捷键与操作说明 ──
                GroupBox("快捷键与操作说明") {
                    VStack(alignment: .leading, spacing: 3) {
                        kh("A / D", "上一帧 / 下一帧")
                        kh("← / →", "上一帧 / 下一帧")
                        kh("Space", "播放 / 暂停")
                        kh("Delete", "删除选中框")
                        kh("拖拽画面", "新建标注框")
                        kh("点击框", "选中后可编辑")
                        kh("拖动框", "移动位置")
                        kh("拖动角点", "调整大小")
                        Divider()
                        Text("标注流程：打开视频 → 自动加载已有标注 → 逐帧查看并修正 → 自动保存")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                        Text("每次修改标注后会自动保存，顶部会显示【已保存标注】提示。")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    }.padding(4)
                }

                Spacer()
            }
            .padding(8)
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 辅助视图
    // ═══════════════════════════════════════════════════════════

    private func kh(_ key: String, _ desc: String) -> some View {
        HStack {
            Text(key).font(.system(size: 10, design: .monospaced)).frame(width: 80, alignment: .leading)
            Text(desc).font(.system(size: 10)).foregroundColor(.secondary)
        }
    }

    private func scalarEditor(_ title: String,
                              value: Double,
                              range: ClosedRange<Double>,
                              step: Double,
                              digits: Int = 4,
                              onChange: @escaping (Double) -> Void) -> some View {
        let binding = Binding<Double>(get: { value }, set: onChange)
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.caption)
                Spacer()
                TextField("", value: binding, format: .number.precision(.fractionLength(digits)))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 78)
                    .font(.system(.caption, design: .monospaced))
            }
            HStack(spacing: 8) {
                Slider(value: binding, in: range)
                Stepper("", value: binding, in: range, step: step)
                    .labelsHidden()
            }
        }
    }

    private func nudgeButton(_ title: String, systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .labelStyle(.titleAndIcon)
        }
        .buttonStyle(.bordered)
        .font(.caption)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 导出操作
    // ═══════════════════════════════════════════════════════════

    private func doSaveProject(forcePanel: Bool = false) {
        if !forcePanel, let saved = engine.saveProject() {
            print("[Annotator] 项目已保存到: \(saved.path)")
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType.json]
        panel.nameFieldStringValue = engine.projectFileURL?.lastPathComponent ?? "annotations.json"
        if panel.runModal() == .OK, let url = panel.url {
            _ = engine.saveProject(to: url)
        }
    }

    private func doExportYOLO() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false; panel.canChooseDirectories = true
        panel.canCreateDirectories = true; panel.message = "选择 YOLO 数据集输出目录"
        if panel.runModal() == .OK, let url = panel.url { engine.exportYOLO(to: url) }
    }

    private func doExportJSON() {
        doSaveProject(forcePanel: true)
    }

    private func doImportJSON() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType.json]; panel.canChooseFiles = true
        if panel.runModal() == .OK, let url = panel.url { engine.importJSON(from: url) }
    }
}

// ═══════════════════════════════════════════════════════════════

struct SAM3Sheet: View {
    @Binding var cfg: SAM3Cfg
    @Binding var isPresented: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("SAM3 自动标注配置").font(.headline)
            Text("使用 Replicate API 进行自动标注。需要提供 API Token。")
                .font(.caption).foregroundColor(.secondary)

            Divider()

            HStack {
                Text("API Token").font(.caption).frame(width: 80, alignment: .leading)
                SecureField("Replicate API Token", text: $cfg.apiToken)
                    .textFieldStyle(.roundedBorder)
            }

            HStack {
                Text("Prompt").font(.caption).frame(width: 80, alignment: .leading)
                TextField("检测目标描述", text: $cfg.prompt)
                    .textFieldStyle(.roundedBorder)
            }

            HStack {
                Text("负面提示").font(.caption).frame(width: 80, alignment: .leading)
                TextField("排除目标描述", text: $cfg.negativePrompt)
                    .textFieldStyle(.roundedBorder)
            }

            Divider()

            HStack {
                Spacer()
                Button("取消") { isPresented = false }
                Button("确定") { isPresented = false }.buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 480)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 标注 CLI 入口视图
// ═══════════════════════════════════════════════════════════════

struct AnnotationCLIView: View {
    @StateObject private var engine = VideoAnnotationEngine()
    @State private var started = false

    var body: some View {
        VStack(spacing: 20) {
            if engine.isLoaded {
                VideoAnnotationView()
            } else {
                Image(systemName: "pencil.and.ruler")
                    .font(.system(size: 48))
                    .foregroundColor(.orange)
                Text(started ? "正在加载视频..." : "准备中...")
                    .font(.title2.bold())
                Text("命令行标注模式")
                    .font(.body.monospaced())
                    .foregroundColor(.secondary)
                ProgressView()
            }
        }
        .frame(minWidth: 700, minHeight: 500)
        .onAppear {
            guard !started else { return }
            started = true
            // 尝试自动加载第一个配置视频
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                if let firstName = engine.configuredVideoNames.first {
                    _ = engine.quickLoadConfiguredVideo(named: firstName)
                }
            }
        }
    }
}
