//
//  Calibrator.swift
//  macRCtrackerSimulator
//
//  标定模块：从图像或现有 FOVMap 参数得到相机内参/外参，供 CalibratedFOVMap 使用。
//
//  Phase1 集成（不改动现有裁切流程）：
//  1. 在 SimEngine 中用 CameraCalibration.fromFOVMap(resolution:srcH:srcV:panC:tiltC:camH:camV:)
//     根据当前 fov 与视频分辨率生成等效标定，再建 CalibratedFOVMap。
//  2. PID 误差改为角度差：用 fovMap.inverseProjectNormalized(normalizedX:normalizedY:) 得到
//     (idealPan, idealTilt)，误差 = (idealPan - gimbal.panPos, idealTilt - gimbal.tiltPos)，
//     再传入 PID（或新开一个“角度空间 PID”接口）。
//  3. 裁切仍用现有 FOVMap.crop，待 Phase2 再切到 CalibratedFOVMap.project()。
//

import Foundation
import CoreGraphics
import simd

// MARK: - 标定结果

/// 标定结果，包含相机内参和外参
struct CameraCalibration {
    /// 图像分辨率 (宽高)
    let resolution: CGSize
    /// 焦距 (像素)
    let focalLength: Double
    /// 主点 (像素，通常为图像中心)
    let principalPoint: CGPoint
    /// 相机位置 (米)，仿真中可忽略
    let position: simd_double3
    /// 旋转矩阵 (世界 → 相机)，标定姿态下的取景方向
    let rotation: simd_double3x3
    /// 舵机水平中心角（度），与云台约定一致；由 fromFOVMap 填入，供 CalibratedFOVMap 对齐坐标系
    let servoPanCenter: Double?
    /// 舵机俯仰中心角（度）
    let servoTiltCenter: Double?

    /// Brown-Conrady 径向畸变系数。典型运动相机 k1≈-0.28, k2≈0.08；源视频已 rectified 则全 0
    let k1: Double
    let k2: Double
    let k3: Double

    /// 水平视场角 (度)
    var hFov: Double {
        2 * atan((resolution.width / 2.0) / focalLength) * 180.0 / .pi
    }
    /// 垂直视场角 (度)
    var vFov: Double {
        2 * atan((resolution.height / 2.0) / focalLength) * 180.0 / .pi
    }
    /// 内参矩阵 3x3 (像素坐标用)
    var intrinsicMatrix: simd_double3x3 {
        simd_double3x3(
            SIMD3<Double>(focalLength, 0, 0),
            SIMD3<Double>(0, focalLength, 0),
            SIMD3<Double>(principalPoint.x, principalPoint.y, 1)
        )
    }

    var hasDistortion: Bool { k1 != 0 || k2 != 0 || k3 != 0 }

    init(resolution: CGSize, focalLength: Double, principalPoint: CGPoint, position: simd_double3, rotation: simd_double3x3, servoPanCenter: Double? = nil, servoTiltCenter: Double? = nil, k1: Double = 0, k2: Double = 0, k3: Double = 0) {
        self.resolution = resolution
        self.focalLength = focalLength
        self.principalPoint = principalPoint
        self.position = position
        self.rotation = rotation
        self.servoPanCenter = servoPanCenter
        self.servoTiltCenter = servoTiltCenter
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
    }

    /// 去畸变: 畸变图像像素 → 理想针孔像素
    func undistort(pixel: CGPoint) -> CGPoint {
        guard hasDistortion else { return pixel }
        let cx = principalPoint.x
        let cy = principalPoint.y
        let fx = focalLength
        let fy = focalLength
        let xd = (pixel.x - cx) / fx
        let yd = (pixel.y - cy) / fy
        var xu = xd
        var yu = yd
        for _ in 0..<5 {
            let r2u = xu * xu + yu * yu
            let radial = 1.0 + k1 * r2u + k2 * r2u * r2u + k3 * r2u * r2u * r2u
            xu = xd / radial
            yu = yd / radial
        }
        return CGPoint(x: xu * fx + cx, y: yu * fy + cy)
    }

    /// 加畸变: 理想针孔像素 → 畸变图像像素
    func distort(pixel: CGPoint) -> CGPoint {
        guard hasDistortion else { return pixel }
        let cx = principalPoint.x
        let cy = principalPoint.y
        let fx = focalLength
        let fy = focalLength
        let xu = (pixel.x - cx) / fx
        let yu = (pixel.y - cy) / fy
        let r2 = xu * xu + yu * yu
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return CGPoint(x: xu * radial * fx + cx, y: yu * radial * fy + cy)
    }
}

// MARK: - 标定器

final class Calibrator {

    enum CalibrationError: Error {
        case generic(String)
    }

    /// 从单张图片进行标定（简化版）
    /// 真实实现需消失点/单应性等 CV 算法，此处用 PDF 典型值作为后备
    func calibrate(from image: CGImage) -> Result<CameraCalibration, CalibrationError> {
        let imageSize = CGSize(width: image.width, height: image.height)
        let pdfFocalLength = 554.3
        let pdfResolution = CGSize(width: 1920, height: 1080)
        let pdfTiltAngle = -26.0 * .pi / 180.0

        let focalLength = pdfFocalLength * (imageSize.width / pdfResolution.width)
        let principalPoint = CGPoint(x: imageSize.width / 2.0, y: imageSize.height / 2.0)
        let position = simd_double3(0.0, 2.5, -5.1)

        let cosT = cos(pdfTiltAngle)
        let sinT = sin(pdfTiltAngle)
        let initialRotation = simd_double3x3(
            SIMD3<Double>(1, 0, 0),
            SIMD3<Double>(0, cosT, -sinT),
            SIMD3<Double>(0, sinT, cosT)
        )

        let result = CameraCalibration(
            resolution: imageSize,
            focalLength: focalLength,
            principalPoint: principalPoint,
            position: position,
            rotation: initialRotation
        )
        return .success(result)
    }
}

// MARK: - 从现有 FOVMap 反推标定 (Phase1 用)

extension CameraCalibration {

    /// 由水平 FOV + 用户仰俯角（舵机角度，与 FOVMap 一致：120° 为中性）快速构建标定
    static func quickCalibrate(
        videoResolution: CGSize,
        sourceFovH: Double,
        cameraTiltDeg: Double,
        cameraHeightM: Double,
        k1: Double = 0,
        k2: Double = 0,
        k3: Double = 0
    ) -> CameraCalibration {
        let w = videoResolution.width
        let h = videoResolution.height
        let principalPoint = CGPoint(x: w / 2.0, y: h / 2.0)
        let halfH = (sourceFovH * .pi / 180.0) / 2.0
        let focalLengthH = (w / 2.0) / tan(halfH)
        let halfV = atan((h / w) * tan(halfH))
        let focalLengthV = (h / 2.0) / tan(halfV)
        let focalLength = (focalLengthH + focalLengthV) / 2.0

        let panC = 120.0
        let tiltRad = (cameraTiltDeg - 120.0) * .pi / 180.0
        let panRad = (panC - 120.0) * .pi / 180.0
        let cosT = cos(tiltRad)
        let sinT = sin(tiltRad)
        let cosP = cos(panRad)
        let sinP = sin(panRad)
        let rTilt = simd_double3x3(
            SIMD3(1, 0, 0),
            SIMD3(0, cosT, -sinT),
            SIMD3(0, sinT, cosT)
        )
        let rPan = simd_double3x3(
            SIMD3(cosP, 0, -sinP),
            SIMD3(0, 1, 0),
            SIMD3(sinP, 0, cosP)
        )
        let rotation = rTilt * rPan

        return CameraCalibration(
            resolution: videoResolution,
            focalLength: focalLength,
            principalPoint: principalPoint,
            position: simd_double3(0, cameraHeightM, 0),
            rotation: rotation,
            servoPanCenter: panC,
            servoTiltCenter: cameraTiltDeg,
            k1: k1, k2: k2, k3: k3
        )
    }

    /// 用当前 FOVMap 的线性模型反推一套等效标定，使 CalibratedFOVMap 与现有取景/控制一致
    /// - Parameters:
    ///   - resolution: 源视频分辨率（若未知可用 1920x1080）
    ///   - srcH: 源水平 FOV 度
    ///   - srcV: 源垂直 FOV 度
    ///   - panC: 舵机水平中心角
    ///   - tiltC: 舵机俯仰中心角
    ///   - camH: 取景水平 FOV 度
    ///   - camV: 取景垂直 FOV 度
    static func fromFOVMap(resolution: CGSize, srcH: Double, srcV: Double, panC: Double, tiltC: Double, camH: Double, camV: Double, k1: Double = 0, k2: Double = 0, k3: Double = 0) -> CameraCalibration {
        let w = resolution.width
        let h = resolution.height
        let principalPoint = CGPoint(x: w / 2.0, y: h / 2.0)
        // 使 hFov = camH, vFov = camV
        let halfH = (camH * .pi / 180.0) / 2.0
        let halfV = (camV * .pi / 180.0) / 2.0
        let focalLengthH = (w / 2.0) / tan(halfH)
        let focalLengthV = (h / 2.0) / tan(halfV)
        let focalLength = (focalLengthH + focalLengthV) / 2.0

        // 中心姿态：仅俯仰对应 tiltC（向下为正），水平对应 panC
        let tiltRad = (tiltC - 120.0) * .pi / 180.0
        let panRad = (panC - 120.0) * .pi / 180.0
        let cosT = cos(tiltRad)
        let sinT = sin(tiltRad)
        let cosP = cos(panRad)
        let sinP = sin(panRad)
        let rTilt = simd_double3x3(
            SIMD3(1, 0, 0),
            SIMD3(0, cosT, -sinT),
            SIMD3(0, sinT, cosT)
        )
        let rPan = simd_double3x3(
            SIMD3(cosP, 0, -sinP),
            SIMD3(0, 1, 0),
            SIMD3(sinP, 0, cosP)
        )
        let rotation = rTilt * rPan

        return CameraCalibration(
            resolution: resolution,
            focalLength: focalLength,
            principalPoint: principalPoint,
            position: simd_double3(0, 0, 0),
            rotation: rotation,
            servoPanCenter: panC,
            servoTiltCenter: tiltC,
            k1: k1, k2: k2, k3: k3
        )
    }
}
