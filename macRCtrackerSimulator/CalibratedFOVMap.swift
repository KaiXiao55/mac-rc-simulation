//
//  CalibratedFOVMap.swift
//  macRCtrackerSimulator
//
//  基于标定结果的精确 FOV 几何：正向投影(pan/tilt/zoom→裁切框)、反向投影(像素→舵机角度)。
//

import Foundation
import CoreGraphics
import simd

// MARK: - 标定 FOV 映射

final class CalibratedFOVMap {

    let calibration: CameraCalibration

    /// 舵机中心角（标定姿态），与 gimbal 角度一致；由标定填入则与云台 120° 中心对齐
    let panCenterDegrees: Double
    let tiltCenterDegrees: Double

    init(calibration: CameraCalibration) {
        self.calibration = calibration
        let forward = calibration.rotation.columns.2
        let derivedTilt = asin(-forward.y) * 180.0 / .pi
        let derivedPan = atan2(forward.x, forward.z) * 180.0 / .pi
        // 标定若提供舵机中心角则用其对齐坐标系，避免“世界 0°”与“舵机 120°”错配
        self.tiltCenterDegrees = calibration.servoTiltCenter ?? derivedTilt
        self.panCenterDegrees = calibration.servoPanCenter ?? derivedPan
    }

    /// 正向投影：由当前 pan/tilt/zoom 计算源视频上的归一化裁切区域
    /// - Returns: 归一化 CGRect (0–1)，可直接用于裁切
    func project(panDegrees: Double, tiltDegrees: Double, zoom: Double) -> CGRect {
        let effectiveFocalLength = calibration.focalLength * max(zoom, 1.0)

        let panRad = (panDegrees - panCenterDegrees) * .pi / 180.0
        let tiltRad = (tiltDegrees - tiltCenterDegrees) * .pi / 180.0
        let cosP = cos(panRad)
        let sinP = sin(panRad)
        let cosT = cos(tiltRad)
        let sinT = sin(tiltRad)

        let rPan = simd_double3x3(
            SIMD3(cosP, 0, -sinP),
            SIMD3(0, 1, 0),
            SIMD3(sinP, 0, cosP)
        )
        let rTilt = simd_double3x3(
            SIMD3(1, 0, 0),
            SIMD3(0, cosT, -sinT),
            SIMD3(0, sinT, cosT)
        )
        let totalRotation = calibration.rotation * rTilt * rPan
        let invTotalRotation = totalRotation.inverse

        let w = calibration.resolution.width
        let h = calibration.resolution.height
        let cx = calibration.principalPoint.x
        let cy = calibration.principalPoint.y

        let corners: [simd_double3] = [
            simd_double3(-cx, -cy, effectiveFocalLength),
            simd_double3(w - cx, -cy, effectiveFocalLength),
            simd_double3(w - cx, h - cy, effectiveFocalLength),
            simd_double3(-cx, h - cy, effectiveFocalLength)
        ]
        let cornersNorm = corners.map { simd_normalize($0) }

        let K = calibration.intrinsicMatrix
        // rotation = 世界→源相机，故 cornerWorld 转至源图系应为 R * cornerWorld（不是 R^{-1}）
        let R_worldToSrc = calibration.rotation

        let projectedCorners: [CGPoint] = cornersNorm.map { cornerCam in
            let cornerWorld = invTotalRotation * cornerCam
            let cornerInSrcCam = R_worldToSrc * cornerWorld
            let p = K * cornerInSrcCam
            let idealPixel = CGPoint(x: p.x / p.z, y: p.y / p.z)
            return calibration.distort(pixel: idealPixel)
        }

        let minX = projectedCorners.map(\.x).min() ?? 0
        let minY = projectedCorners.map(\.y).min() ?? 0
        let maxX = projectedCorners.map(\.x).max() ?? 0
        let maxY = projectedCorners.map(\.y).max() ?? 0

        return CGRect(
            x: minX / w,
            y: minY / h,
            width: (maxX - minX) / w,
            height: (maxY - minY) / h
        )
    }

    /// 反向投影：由源视频上的像素点计算舵机应指向的 pan/tilt（度）
    /// - Parameter targetPixel: 源视频像素坐标（目标中心等）
    /// - Returns: (panDegrees, tiltDegrees)，与 gimbal 角度一致
    func inverseProject(targetPixel: CGPoint) -> (pan: Double, tilt: Double) {
        let correctedPixel = calibration.undistort(pixel: targetPixel)
        let K_inv = calibration.intrinsicMatrix.inverse
        let p = simd_double3(correctedPixel.x, correctedPixel.y, 1)
        let rayCam = simd_normalize(K_inv * p)
        // rotation 为世界→相机，故 ray_world = rotation.inverse * ray_cam
        let rayWorld = calibration.rotation.inverse * rayCam

        let panRad = atan2(rayWorld.x, rayWorld.z)
        let tiltRad = asin(-rayWorld.y)

        let panDeg = panRad * 180.0 / .pi + panCenterDegrees
        let tiltDeg = tiltRad * 180.0 / .pi + tiltCenterDegrees
        return (panDeg, tiltDeg)
    }

    /// 由归一化坐标 (0–1) 的反向投影，便于与当前裁切坐标系一致
    func inverseProjectNormalized(normalizedX: Double, normalizedY: Double) -> (pan: Double, tilt: Double) {
        let px = normalizedX * calibration.resolution.width
        let py = normalizedY * calibration.resolution.height
        return inverseProject(targetPixel: CGPoint(x: px, y: py))
    }

    /// 等效水平/垂直 FOV（度），与当前 FOVMap.effectiveFov 对齐，供 PID/UI 使用
    func effectiveFov(opticalZoom: CGFloat) -> (h: Double, v: Double) {
        let z = max(Double(opticalZoom), 1.0)
        let f = calibration.focalLength * z
        let w = calibration.resolution.width
        let h = calibration.resolution.height
        let hFov = 2 * atan((w / 2.0) / f) * 180.0 / .pi
        let vFov = 2 * atan((h / 2.0) / f) * 180.0 / .pi
        return (hFov, vFov)
    }
}
