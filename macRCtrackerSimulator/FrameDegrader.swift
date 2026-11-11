//
//  FrameDegrader.swift
//  macRCtrackerSimulator
//
//  帧降质模块：在虚拟赛道渲染帧上叠加真实世界噪声
//  让仿真环境逼近真实视频的脏数据特性
//
//  用法：在 VirtualRaceSimulator.renderFrame() 返回后、送入 YOLO 前调用
//
//  Created for sim-to-real gap bridging.
//

import Foundation
import CoreImage
import CoreGraphics
import CoreVideo
import AppKit

// MARK: - 降质配置

struct DegradeConfig {
    // 运动模糊
    var motionBlurProb: Double = 0.30        // 每帧触发概率
    var motionBlurRadiusRange: ClosedRange<Double> = 4...18  // CIMotionBlur radius

    // JPEG 压缩伪影
    var jpegProb: Double = 0.50
    var jpegQualityRange: ClosedRange<Double> = 0.25...0.65  // 越低越糊

    // 曝光/白平衡跳变
    var exposureProb: Double = 0.40
    var exposureEVRange: ClosedRange<Double> = -1.2...1.2    // EV 偏移

    // 高斯噪声 (模拟高 ISO)
    var noiseProb: Double = 0.35
    var noiseAmountRange: ClosedRange<Double> = 0.01...0.04

    // 色温偏移 (模拟 AWB 失误)
    var colorTempProb: Double = 0.25
    var tempShiftRange: ClosedRange<Double> = -1500...1500   // 开尔文偏移

    // 总开关 & 强度
    var enabled: Bool = true
    /// 整体强度 0~1，乘到各 prob 上，方便 UI 调节
    var intensity: Double = 1.0

    /// 预设：轻度 (手持三脚架、光线好)
    static let mild = DegradeConfig(
        motionBlurProb: 0.15, motionBlurRadiusRange: 3...10,
        jpegProb: 0.30, jpegQualityRange: 0.45...0.75,
        exposureProb: 0.20, exposureEVRange: -0.6...0.6,
        noiseProb: 0.15, noiseAmountRange: 0.005...0.02,
        colorTempProb: 0.10, tempShiftRange: -800...800
    )

    /// 预设：重度 (手持跟拍、逆光、低码率直播)
    static let harsh = DegradeConfig(
        motionBlurProb: 0.50, motionBlurRadiusRange: 8...25,
        jpegProb: 0.70, jpegQualityRange: 0.15...0.45,
        exposureProb: 0.55, exposureEVRange: -2.0...2.0,
        noiseProb: 0.50, noiseAmountRange: 0.02...0.06,
        colorTempProb: 0.35, tempShiftRange: -2000...2000
    )
}

// MARK: - 帧降质器

final class FrameDegrader {

    private let ci = CIContext(options: [.useSoftwareRenderer: false])
    var config: DegradeConfig

    init(config: DegradeConfig = DegradeConfig()) {
        self.config = config
    }

    /// 对 CGImage 施加随机降质，返回降质后的 CGImage
    /// 每帧独立随机，各效果独立触发
    func degrade(_ input: CGImage) -> CGImage {
        guard config.enabled else { return input }

        var image = CIImage(cgImage: input)
        let intensity = config.intensity

        // 1. 运动模糊
        if shouldApply(config.motionBlurProb * intensity) {
            image = applyMotionBlur(image)
        }

        // 2. 曝光跳变
        if shouldApply(config.exposureProb * intensity) {
            image = applyExposureShift(image)
        }

        // 3. 色温偏移
        if shouldApply(config.colorTempProb * intensity) {
            image = applyColorTempShift(image)
        }

        // 4. 高斯噪声
        if shouldApply(config.noiseProb * intensity) {
            image = applyNoise(image)
        }

        // 5. JPEG 压缩伪影 (放最后，模拟视频编码)
        if shouldApply(config.jpegProb * intensity) {
            if let compressed = applyJPEGArtifacts(image, originalSize: input) {
                return compressed
            }
        }

        // 回转为 CGImage
        let rect = CGRect(x: 0, y: 0, width: input.width, height: input.height)
        return ci.createCGImage(image, from: rect) ?? input
    }

    /// 对 CVPixelBuffer 降质（就地替换）
    /// 返回降质后的新 pixelBuffer + CGImage
    func degrade(cgImage: CGImage, pixelBuffer: CVPixelBuffer) -> (CGImage, CVPixelBuffer?) {
        let degraded = degrade(cgImage)
        let pb = makePixelBuffer(from: degraded,
                                  width: degraded.width,
                                  height: degraded.height)
        return (degraded, pb)
    }

    // MARK: - 各效果实现

    private func applyMotionBlur(_ image: CIImage) -> CIImage {
        guard let filter = CIFilter(name: "CIMotionBlur") else { return image }
        let radius = Double.random(in: config.motionBlurRadiusRange)
        let angle = Double.random(in: 0...(.pi))
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(radius, forKey: kCIInputRadiusKey)
        filter.setValue(angle, forKey: kCIInputAngleKey)
        return filter.outputImage ?? image
    }

    private func applyExposureShift(_ image: CIImage) -> CIImage {
        guard let filter = CIFilter(name: "CIExposureAdjust") else { return image }
        let ev = Double.random(in: config.exposureEVRange)
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(ev, forKey: kCIInputEVKey)
        return filter.outputImage ?? image
    }

    private func applyColorTempShift(_ image: CIImage) -> CIImage {
        guard let filter = CIFilter(name: "CITemperatureAndTint") else { return image }
        let shift = Double.random(in: config.tempShiftRange)
        // 基准中性白约 6500K
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(CIVector(x: CGFloat(6500 + shift), y: 0), forKey: "inputNeutral")
        filter.setValue(CIVector(x: 6500, y: 0), forKey: "inputTargetNeutral")
        return filter.outputImage ?? image
    }

    private func applyNoise(_ image: CIImage) -> CIImage {
        let amount = Double.random(in: config.noiseAmountRange)

        // 生成噪声层
        guard let noiseGen = CIFilter(name: "CIRandomGenerator") else { return image }
        guard let noiseImage = noiseGen.outputImage else { return image }

        // 缩放噪声强度
        let whitening = CIFilter(name: "CIColorMatrix")!
        whitening.setValue(noiseImage, forKey: kCIInputImageKey)
        let s = CGFloat(amount)
        whitening.setValue(CIVector(x: s, y: 0, z: 0, w: 0), forKey: "inputRVector")
        whitening.setValue(CIVector(x: 0, y: s, z: 0, w: 0), forKey: "inputGVector")
        whitening.setValue(CIVector(x: 0, y: 0, z: s, w: 0), forKey: "inputBVector")
        whitening.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")
        whitening.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputBiasVector")
        guard let scaledNoise = whitening.outputImage else { return image }

        // 叠加
        let blend = CIFilter(name: "CIAdditionCompositing")!
        blend.setValue(image, forKey: kCIInputImageKey)
        blend.setValue(scaledNoise.cropped(to: image.extent), forKey: kCIInputBackgroundImageKey)
        return blend.outputImage ?? image
    }

    private func applyJPEGArtifacts(_ image: CIImage, originalSize: CGImage) -> CGImage? {
        let rect = CGRect(x: 0, y: 0, width: originalSize.width, height: originalSize.height)
        guard let cg = ci.createCGImage(image, from: rect) else { return nil }
        let bitmap = NSBitmapImageRep(cgImage: cg)
        let quality = Double.random(in: config.jpegQualityRange)
        guard let jpegData = bitmap.representation(using: .jpeg,
                                                     properties: [.compressionFactor: quality]),
              let compressed = NSBitmapImageRep(data: jpegData),
              let result = compressed.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else { return nil }
        return result
    }

    // MARK: - 工具

    private func shouldApply(_ prob: Double) -> Bool {
        Double.random(in: 0...1) < prob
    }

    private func makePixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &pb)
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
}
