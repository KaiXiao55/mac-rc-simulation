//
//  VirtualRaceTrackGenerator.swift
//  macRCtrackerSimulator
//
//  程序化 3D 赛车赛道生成器 — 10组用户手绘RC赛道
//
//  功能:
//    ① 10组用户手绘的紧凑RC赛道控制点 (已闭合+缩放到真实尺寸)
//    ② Catmull-Rom 样条曲线生成闭合赛道中心线
//    ③ 沿中心线挤出路面几何体 (带纹理 UV)
//    ④ 支持高度起伏 (飞坡/坡道)
//    ⑤ 生成护栏、路缘石
//    ⑥ 输出 SceneKit 节点 + 路径数据供车辆物理使用
//

import SceneKit
import Foundation

// ═══════════════════════════════════════════════════════════════
// MARK: - 赛道形状枚举 (10组用户手绘RC赛道)
// ═══════════════════════════════════════════════════════════════

enum TrackShape: Int, CaseIterable {
    case track1               = 0   // Track 1 (12x10m)
    case track2               = 1   // Track 2 (12x10m)
    case track3               = 2   // Track 3 (12x10m)
    case track4               = 3   // Track 4 (12x10m)
    case track5               = 4   // Track 5 (12x10m)
    case track6               = 5   // Track 6 (12x10m)
    case track7               = 6   // Track 7 (12x10m)
    case track8               = 7   // Track 8 (12x10m)
    case track9               = 8   // Track 9 (12x10m)
    case track10              = 9   // Track 10 (12x10m)

    var displayName: String {
        switch self {
        case .track1              : return "Track 1"
        case .track2              : return "Track 2"
        case .track3              : return "Track 3"
        case .track4              : return "Track 4"
        case .track5              : return "Track 5"
        case .track6              : return "Track 6"
        case .track7              : return "Track 7"
        case .track8              : return "Track 8"
        case .track9              : return "Track 9"
        case .track10             : return "Track 10"
        }
    }

    /// 场地尺寸 (宽, 高) 单位: 米 — 所有赛道均为 24x20m 场地 (原始12x10×2)
    var fieldSize: (CGFloat, CGFloat) {
        switch self {
        case .track1              : return (24, 20)
        case .track2              : return (24, 20)
        case .track3              : return (24, 20)
        case .track4              : return (24, 20)
        case .track5              : return (24, 20)
        case .track6              : return (24, 20)
        case .track7              : return (24, 20)
        case .track8              : return (24, 20)
        case .track9              : return (24, 20)
        case .track10             : return (24, 20)
        }
    }

    /// 赛道宽度 (米) — 原始0.6m×2 = 1.2m
    var recommendedWidth: CGFloat {
        switch self {
        case .track1              : return 1.2
        case .track2              : return 1.2
        case .track3              : return 1.2
        case .track4              : return 1.2
        case .track5              : return 1.2
        case .track6              : return 1.2
        case .track7              : return 1.2
        case .track8              : return 1.2
        case .track9              : return 1.2
        case .track10             : return 1.2
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 赛道数据
// ═══════════════════════════════════════════════════════════════

struct TrackData {
    let centerPoints: [SCNVector3]
    let tangents: [SCNVector3]
    let totalLength: CGFloat
    let trackNode: SCNNode
    let controlPoints: [SCNVector3]
    let rampSegments: [RampSegment]
}

struct RampSegment {
    let startProgress: CGFloat
    let endProgress: CGFloat
    let peakHeight: CGFloat
    let launchAngle: CGFloat
}

// ═══════════════════════════════════════════════════════════════
// MARK: - 赛道生成器
// ═══════════════════════════════════════════════════════════════

final class VirtualRaceTrackGenerator {

    let pointCount: Int
    let radius: CGFloat
    let width: CGFloat
    let seed: UInt64
    let trackShape: TrackShape
    let enableRamps: Bool
    let rampCount: Int

    private let subdivisions: Int = 70

    init(pointCount: Int = 18,
         radius: CGFloat = 15,
         width: CGFloat = 2.5,
         seed: UInt64 = 42,
         trackShape: TrackShape = .track1,
         enableRamps: Bool = false,
         rampCount: Int = 2) {
        self.pointCount = max(4, pointCount)
        self.radius = radius
        self.width = width
        self.seed = seed
        self.trackShape = trackShape
        self.enableRamps = enableRamps
        self.rampCount = rampCount
    }

    func generate() -> TrackData {
        srand48(Int(seed))

        let controlPts = generateControlPoints()
        var (centerPts, tangents) = catmullRomInterpolate(controlPts)

        var ramps: [RampSegment] = []
        if enableRamps {
            ramps = generateRampSegments(count: rampCount, totalPoints: centerPts.count)
            applyRampHeights(to: &centerPts, ramps: ramps)
            tangents = recalculateTangents(centerPts)
        }

        var totalLen: CGFloat = 0
        for i in 1..<centerPts.count {
            totalLen += scnDist(centerPts[i-1], centerPts[i])
        }
        totalLen += scnDist(centerPts.last!, centerPts.first!)

        let actualWidth = trackShape.recommendedWidth
        let trackNode = buildTrackGeometry(center: centerPts, tangents: tangents, ramps: ramps, trackWidth: actualWidth)

        return TrackData(
            centerPoints: centerPts,
            tangents: tangents,
            totalLength: totalLen,
            trackNode: trackNode,
            controlPoints: controlPts,
            rampSegments: ramps
        )
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 控制点 — 10组用户手绘RC赛道
    // ═══════════════════════════════════════════════════════════

    private func generateControlPoints() -> [SCNVector3] {
        switch trackShape {

        case .track1:
            // Track 1: 12x10m field, 40 pts, w=0.6m, bbox=5.2x5.7m
            return [
                SCNVector3(  -8.834, 0,    6.000),
                SCNVector3(  -8.766, 0,   -4.300),
                SCNVector3(  -6.366, 0,   -5.034),
                SCNVector3(  -5.866, 0,   -2.866),
                SCNVector3(  -5.866, 0,   -1.534),
                SCNVector3(  -6.534, 0,   -0.466),
                SCNVector3(  -6.900, 0,    0.200),
                SCNVector3(  -5.566, 0,    0.500),
                SCNVector3(  -4.200, 0,    0.200),
                SCNVector3(  -4.100, 0,   -1.166),
                SCNVector3(  -4.100, 0,   -2.134),
                SCNVector3(  -4.100, 0,   -4.066),
                SCNVector3(  -4.666, 0,   -4.766),
                SCNVector3(  -4.534, 0,   -5.200),
                SCNVector3(  -3.300, 0,   -5.234),
                SCNVector3(  -1.800, 0,   -5.234),
                SCNVector3(  -2.034, 0,   -4.266),
                SCNVector3(  -2.166, 0,   -2.234),
                SCNVector3(  -2.434, 0,   -1.266),
                SCNVector3(  -1.900, 0,   -0.734),
                SCNVector3(  -0.334, 0,   -1.500),
                SCNVector3(  -0.334, 0,   -2.800),
                SCNVector3(  -0.266, 0,   -4.834),
                SCNVector3(   0.400, 0,   -4.834),
                SCNVector3(   1.266, 0,   -4.100),
                SCNVector3(   1.266, 0,   -3.434),
                SCNVector3(   1.266, 0,   -0.734),
                SCNVector3(   1.266, 0,    0.300),
                SCNVector3(   0.134, 0,    0.966),
                SCNVector3(  -1.334, 0,    1.234),
                SCNVector3(  -3.000, 0,    1.400),
                SCNVector3(  -5.700, 0,    2.034),
                SCNVector3(  -7.534, 0,    2.566),
                SCNVector3(  -7.234, 0,    3.734),
                SCNVector3(  -2.800, 0,    3.666),
                SCNVector3(  -0.666, 0,    3.266),
                SCNVector3(   1.666, 0,    3.866),
                SCNVector3(   1.666, 0,    5.300),
                SCNVector3(   0.800, 0,    6.134),
                SCNVector3(  -7.834, 0,    5.900)
            ]

        case .track2:
            // Track 2: 12x10m field, 26 pts, w=0.6m, bbox=6.8x6.6m
            return [
                SCNVector3(  -2.266, 0,    6.900),
                SCNVector3(  -8.734, 0,    6.900),
                SCNVector3(  -9.000, 0,    2.166),
                SCNVector3(  -9.700, 0,   -3.434),
                SCNVector3( -10.200, 0,   -5.834),
                SCNVector3(   1.400, 0,   -6.234),
                SCNVector3(   2.600, 0,   -4.400),
                SCNVector3(   2.700, 0,   -0.066),
                SCNVector3(   0.866, 0,   -0.566),
                SCNVector3(   0.634, 0,   -4.166),
                SCNVector3(  -0.534, 0,   -4.566),
                SCNVector3(  -4.034, 0,   -4.200),
                SCNVector3(  -6.734, 0,   -4.366),
                SCNVector3(  -6.866, 0,   -2.366),
                SCNVector3(  -4.900, 0,   -1.566),
                SCNVector3(  -1.166, 0,   -1.534),
                SCNVector3(  -1.066, 0,    1.300),
                SCNVector3(  -2.634, 0,    1.300),
                SCNVector3(  -4.900, 0,    1.266),
                SCNVector3(  -6.600, 0,    1.066),
                SCNVector3(  -6.700, 0,    2.900),
                SCNVector3(   2.700, 0,    3.534),
                SCNVector3(   3.334, 0,    4.800),
                SCNVector3(   2.766, 0,    6.100),
                SCNVector3(   1.166, 0,    6.100),
                SCNVector3(  -1.300, 0,    6.634)
            ]

        case .track3:
            // Track 3: 12x10m field, 37 pts, w=0.6m, bbox=7.3x8.3m
            return [
                SCNVector3(  -0.100, 0,    6.734),
                SCNVector3(  -7.300, 0,    7.000),
                SCNVector3(  -9.234, 0,    6.134),
                SCNVector3(  -9.566, 0,    2.034),
                SCNVector3(  -9.566, 0,   -1.466),
                SCNVector3(  -9.966, 0,   -5.500),
                SCNVector3( -10.534, 0,   -9.700),
                SCNVector3(  -3.434, 0,   -8.766),
                SCNVector3(  -4.766, 0,   -6.834),
                SCNVector3(  -8.734, 0,   -7.400),
                SCNVector3(  -7.466, 0,   -2.234),
                SCNVector3(  -7.466, 0,    2.500),
                SCNVector3(  -7.366, 0,    3.666),
                SCNVector3(  -5.400, 0,    4.266),
                SCNVector3(  -5.600, 0,   -1.700),
                SCNVector3(  -5.600, 0,   -4.166),
                SCNVector3(  -1.966, 0,   -6.066),
                SCNVector3(  -1.900, 0,   -6.766),
                SCNVector3(  -1.834, 0,   -9.134),
                SCNVector3(  -0.300, 0,   -9.034),
                SCNVector3(   0.266, 0,   -6.434),
                SCNVector3(   0.234, 0,   -4.766),
                SCNVector3(  -2.100, 0,   -2.900),
                SCNVector3(  -3.400, 0,   -1.334),
                SCNVector3(   0.034, 0,   -1.466),
                SCNVector3(   2.800, 0,   -1.034),
                SCNVector3(   2.800, 0,    0.200),
                SCNVector3(  -0.034, 0,    0.734),
                SCNVector3(  -3.466, 0,    2.234),
                SCNVector3(  -3.366, 0,    4.600),
                SCNVector3(   1.900, 0,    4.334),
                SCNVector3(  -0.166, 0,    2.934),
                SCNVector3(   1.166, 0,    2.000),
                SCNVector3(   3.500, 0,    2.234),
                SCNVector3(   4.100, 0,    4.766),
                SCNVector3(   3.966, 0,    6.166),
                SCNVector3(   0.200, 0,    6.700)
            ]

        case .track4:
            // Track 4: 12x10m field, 34 pts, w=0.6m, bbox=6.0x9.9m
            return [
                SCNVector3(  -1.066, 0,    9.466),
                SCNVector3(  -9.434, 0,    9.466),
                SCNVector3(  -9.866, 0,    7.300),
                SCNVector3(  -9.866, 0,    2.066),
                SCNVector3(  -6.566, 0,    6.334),
                SCNVector3(  -6.566, 0,    7.366),
                SCNVector3(  -7.466, 0,    7.366),
                SCNVector3(  -8.100, 0,    7.834),
                SCNVector3(  -8.100, 0,    8.334),
                SCNVector3(  -5.066, 0,    8.334),
                SCNVector3(  -2.666, 0,    7.600),
                SCNVector3(  -2.666, 0,    6.634),
                SCNVector3(  -4.566, 0,    4.900),
                SCNVector3(  -5.666, 0,    4.666),
                SCNVector3(  -6.300, 0,    3.234),
                SCNVector3(  -7.966, 0,    1.266),
                SCNVector3( -11.600, 0,   -1.566),
                SCNVector3( -11.600, 0,   -6.466),
                SCNVector3( -11.034, 0,   -7.066),
                SCNVector3(  -8.100, 0,   -5.234),
                SCNVector3(  -9.400, 0,   -3.734),
                SCNVector3(  -9.466, 0,   -2.534),
                SCNVector3(  -5.566, 0,   -0.634),
                SCNVector3(  -2.500, 0,   -0.634),
                SCNVector3(  -4.466, 0,   -7.866),
                SCNVector3(  -8.366, 0,   -8.434),
                SCNVector3(  -8.266, 0,  -10.434),
                SCNVector3(  -2.566, 0,  -10.434),
                SCNVector3(  -0.434, 0,   -5.200),
                SCNVector3(  -0.434, 0,    0.666),
                SCNVector3(  -2.666, 0,    2.334),
                SCNVector3(  -4.100, 0,    3.166),
                SCNVector3(  -0.434, 0,    4.966),
                SCNVector3(   0.434, 0,    9.000)
            ]

        case .track5:
            // Track 5: 12x10m field, 28 pts, w=0.6m, bbox=7.8x7.7m
            return [
                SCNVector3(  -1.100, 0,    9.400),
                SCNVector3(  -6.966, 0,    9.600),
                SCNVector3(  -9.666, 0,    9.366),
                SCNVector3( -10.500, 0,    5.766),
                SCNVector3(  -6.866, 0,   -4.266),
                SCNVector3(  -7.400, 0,    3.166),
                SCNVector3(  -7.434, 0,    6.234),
                SCNVector3(  -8.000, 0,    7.266),
                SCNVector3(  -3.766, 0,    7.266),
                SCNVector3(  -3.900, 0,    4.566),
                SCNVector3(  -0.466, 0,    4.066),
                SCNVector3(   2.000, 0,    5.934),
                SCNVector3(  -1.000, 0,    6.800),
                SCNVector3(   1.300, 0,    8.200),
                SCNVector3(   3.534, 0,    7.634),
                SCNVector3(   3.534, 0,    3.400),
                SCNVector3(   0.734, 0,    0.600),
                SCNVector3(  -5.200, 0,    1.734),
                SCNVector3(  -5.200, 0,   -4.666),
                SCNVector3(  -1.866, 0,   -5.066),
                SCNVector3(  -2.700, 0,   -1.266),
                SCNVector3(  -3.334, 0,   -0.300),
                SCNVector3(  -0.034, 0,   -1.734),
                SCNVector3(   1.834, 0,   -3.766),
                SCNVector3(   3.534, 0,   -5.066),
                SCNVector3(   3.900, 0,   -0.800),
                SCNVector3(   5.066, 0,    2.834),
                SCNVector3(   5.000, 0,   10.334)
            ]

        case .track6:
            // Track 6: 12x10m field, 36 pts, w=0.6m, bbox=6.2x8.8m
            return [
                SCNVector3(  -1.266, 0,    8.700),
                SCNVector3(  -7.166, 0,    8.066),
                SCNVector3(  -8.234, 0,    5.100),
                SCNVector3(  -8.234, 0,   -0.400),
                SCNVector3(  -9.666, 0,   -7.466),
                SCNVector3( -10.800, 0,   -1.834),
                SCNVector3( -10.866, 0,    7.934),
                SCNVector3(  -9.600, 0,    9.466),
                SCNVector3(  -3.834, 0,    9.434),
                SCNVector3(  -1.300, 0,   10.166),
                SCNVector3(   0.100, 0,    9.266),
                SCNVector3(  -0.066, 0,    6.000),
                SCNVector3(  -2.066, 0,    6.266),
                SCNVector3(  -2.300, 0,    3.966),
                SCNVector3(  -2.300, 0,   -1.634),
                SCNVector3(   0.134, 0,   -1.600),
                SCNVector3(   0.234, 0,    2.500),
                SCNVector3(   0.634, 0,    4.400),
                SCNVector3(   1.600, 0,    4.400),
                SCNVector3(   1.566, 0,    0.266),
                SCNVector3(   1.566, 0,   -2.600),
                SCNVector3(   1.100, 0,   -4.300),
                SCNVector3(  -0.800, 0,   -7.300),
                SCNVector3(  -3.500, 0,   -6.234),
                SCNVector3(  -6.300, 0,   -3.334),
                SCNVector3(  -6.900, 0,   -6.866),
                SCNVector3(  -7.966, 0,   -5.266),
                SCNVector3(  -7.134, 0,   -0.800),
                SCNVector3(  -5.800, 0,   -1.066),
                SCNVector3(  -4.934, 0,    0.334),
                SCNVector3(  -5.134, 0,    4.300),
                SCNVector3(  -6.634, 0,    4.400),
                SCNVector3(  -5.966, 0,    5.966),
                SCNVector3(  -3.334, 0,    6.300),
                SCNVector3(  -2.600, 0,    7.466),
                SCNVector3(  -1.266, 0,    8.700)
            ]

        case .track7:
            // Track 7: 12x10m field, 41 pts, w=0.6m, bbox=8.2x7.9m
            return [
                SCNVector3(  -1.600, 0,    8.600),
                SCNVector3( -10.366, 0,    9.434),
                SCNVector3(  -7.300, 0,    5.766),
                SCNVector3(  -4.234, 0,    6.666),
                SCNVector3(  -6.800, 0,    7.166),
                SCNVector3(  -3.534, 0,    7.100),
                SCNVector3(  -2.334, 0,    6.900),
                SCNVector3(  -2.666, 0,    1.466),
                SCNVector3(  -7.800, 0,    2.034),
                SCNVector3(  -6.300, 0,    3.300),
                SCNVector3(  -6.300, 0,    4.166),
                SCNVector3(  -8.600, 0,    4.166),
                SCNVector3( -10.366, 0,    4.034),
                SCNVector3( -11.000, 0,   -2.934),
                SCNVector3(  -8.434, 0,   -0.834),
                SCNVector3(  -9.200, 0,    1.066),
                SCNVector3(  -4.100, 0,   -2.000),
                SCNVector3(  -6.966, 0,   -3.100),
                SCNVector3(  -8.000, 0,   -3.700),
                SCNVector3(  -7.366, 0,   -6.466),
                SCNVector3(  -0.266, 0,   -6.466),
                SCNVector3(  -0.566, 0,   -4.800),
                SCNVector3(  -3.900, 0,   -4.666),
                SCNVector3(  -1.800, 0,   -2.234),
                SCNVector3(  -3.034, 0,   -1.200),
                SCNVector3(   1.900, 0,   -1.066),
                SCNVector3(   1.700, 0,    0.266),
                SCNVector3(  -0.466, 0,    0.434),
                SCNVector3(  -0.466, 0,    2.234),
                SCNVector3(   1.734, 0,    4.234),
                SCNVector3(   3.000, 0,    3.900),
                SCNVector3(   1.300, 0,    1.600),
                SCNVector3(   3.234, 0,    1.600),
                SCNVector3(   5.434, 0,    2.734),
                SCNVector3(   5.400, 0,    5.366),
                SCNVector3(   3.966, 0,    5.366),
                SCNVector3(   1.166, 0,    5.566),
                SCNVector3(   0.400, 0,    5.566),
                SCNVector3(   0.234, 0,    6.700),
                SCNVector3(  -0.100, 0,    7.966),
                SCNVector3(  -0.666, 0,    8.266)
            ]

        case .track8:
            // Track 8: 12x10m field, 37 pts, w=0.6m, bbox=9.0x6.8m
            return [
                SCNVector3(   0.400, 0,    9.634),
                SCNVector3(  -7.600, 0,    9.634),
                SCNVector3(  -8.400, 0,    8.566),
                SCNVector3(  -9.800, 0,    1.800),
                SCNVector3( -10.800, 0,    6.700),
                SCNVector3( -12.600, 0,    5.866),
                SCNVector3( -12.534, 0,    3.266),
                SCNVector3( -12.400, 0,    2.666),
                SCNVector3( -10.934, 0,    0.400),
                SCNVector3(  -6.700, 0,    0.034),
                SCNVector3(  -7.266, 0,    3.700),
                SCNVector3(  -6.934, 0,    5.966),
                SCNVector3(  -5.034, 0,    7.766),
                SCNVector3(  -2.800, 0,    7.766),
                SCNVector3(  -2.366, 0,    7.366),
                SCNVector3(  -2.600, 0,    6.866),
                SCNVector3(  -4.434, 0,    5.500),
                SCNVector3(  -5.200, 0,    2.934),
                SCNVector3(  -4.966, 0,   -1.734),
                SCNVector3(  -2.900, 0,   -0.500),
                SCNVector3(  -2.900, 0,    2.266),
                SCNVector3(  -2.900, 0,    3.334),
                SCNVector3(  -2.266, 0,    3.600),
                SCNVector3(   0.500, 0,    7.166),
                SCNVector3(  -0.200, 0,    7.666),
                SCNVector3(   3.434, 0,    7.666),
                SCNVector3(   3.434, 0,    8.066),
                SCNVector3(   3.266, 0,    4.766),
                SCNVector3(  -0.334, 0,    1.834),
                SCNVector3(  -0.334, 0,    0.066),
                SCNVector3(   1.400, 0,   -3.166),
                SCNVector3(   2.600, 0,   -1.900),
                SCNVector3(   3.134, 0,    0.866),
                SCNVector3(   4.834, 0,    4.500),
                SCNVector3(   4.834, 0,    6.200),
                SCNVector3(   4.866, 0,    7.066),
                SCNVector3(   5.334, 0,   10.366)
            ]

        case .track9:
            // Track 9: 12x10m field, 31 pts, w=0.6m, bbox=7.8x6.8m
            return [
                SCNVector3(   3.800, 0,    8.134),
                SCNVector3(  -4.500, 0,    8.134),
                SCNVector3(  -6.666, 0,    8.934),
                SCNVector3(  -8.466, 0,    8.466),
                SCNVector3( -10.734, 0,    4.566),
                SCNVector3( -10.466, 0,   -0.966),
                SCNVector3(  -9.966, 0,   -4.566),
                SCNVector3(  -4.334, 0,   -4.000),
                SCNVector3(  -7.600, 0,   -1.934),
                SCNVector3(  -7.834, 0,    2.300),
                SCNVector3(  -8.234, 0,    4.700),
                SCNVector3(  -8.134, 0,    6.066),
                SCNVector3(  -6.966, 0,    6.066),
                SCNVector3(  -5.534, 0,    5.834),
                SCNVector3(  -3.600, 0,    6.066),
                SCNVector3(  -3.634, 0,    4.200),
                SCNVector3(  -5.400, 0,    0.934),
                SCNVector3(  -5.134, 0,    2.034),
                SCNVector3(  -5.634, 0,    1.966),
                SCNVector3(  -5.666, 0,   -0.366),
                SCNVector3(  -1.466, 0,   -3.366),
                SCNVector3(  -1.300, 0,    0.634),
                SCNVector3(  -2.666, 0,    1.134),
                SCNVector3(  -1.466, 0,    4.866),
                SCNVector3(  -1.000, 0,    6.334),
                SCNVector3(   1.134, 0,    6.266),
                SCNVector3(   1.534, 0,    3.734),
                SCNVector3(   0.934, 0,   -1.100),
                SCNVector3(   0.934, 0,   -3.466),
                SCNVector3(   3.066, 0,   -3.900),
                SCNVector3(   4.934, 0,    7.466)
            ]

        case .track10:
            // Track 10: 12x10m field, 26 pts, w=0.6m, bbox=7.1x8.0m
            return [
                SCNVector3(   0.400, 0,    7.500),
                SCNVector3(  -9.266, 0,    7.934),
                SCNVector3(  -9.666, 0,    5.434),
                SCNVector3(  -0.600, 0,    5.366),
                SCNVector3(   1.000, 0,    5.034),
                SCNVector3(   1.000, 0,    1.366),
                SCNVector3(  -7.600, 0,    2.600),
                SCNVector3(  -9.634, 0,    2.600),
                SCNVector3( -10.034, 0,    1.866),
                SCNVector3( -10.134, 0,   -1.500),
                SCNVector3(  -4.034, 0,   -1.500),
                SCNVector3(  -0.666, 0,   -1.500),
                SCNVector3(  -0.066, 0,   -1.500),
                SCNVector3(   1.766, 0,   -1.966),
                SCNVector3(   1.600, 0,   -4.000),
                SCNVector3(  -0.800, 0,   -4.234),
                SCNVector3(  -6.534, 0,   -3.500),
                SCNVector3( -10.834, 0,   -3.800),
                SCNVector3( -10.834, 0,   -6.600),
                SCNVector3(  -4.334, 0,   -7.066),
                SCNVector3(   2.866, 0,   -6.566),
                SCNVector3(   3.334, 0,   -4.900),
                SCNVector3(   3.366, 0,    7.566),
                SCNVector3(   0.400, 0,    7.500),
                SCNVector3(  -2.066, 0,    8.900),
                SCNVector3(   0.400, 0,    7.500)
            ]

        }
    }

    // ═══════════════════════════════════════════════════════════

    private func generateRampSegments(count: Int, totalPoints: Int) -> [RampSegment] {
        var ramps: [RampSegment] = []
        let segLen: CGFloat = 0.04
        let minGap: CGFloat = 0.15

        for ri in 0..<count {
            let baseStart = CGFloat(ri) / CGFloat(count) + CGFloat(drand48()) * minGap
            let start = min(baseStart.truncatingRemainder(dividingBy: 1.0), 1.0 - segLen * 2)
            let end = start + segLen
            let peakH = CGFloat(0.3 + drand48() * 0.5)
            let launchAngle = CGFloat(0.2 + drand48() * 0.3)

            ramps.append(RampSegment(
                startProgress: start,
                endProgress: end,
                peakHeight: peakH,
                launchAngle: launchAngle
            ))
        }
        return ramps
    }

    private func applyRampHeights(to points: inout [SCNVector3], ramps: [RampSegment]) {
        let n = points.count
        for ramp in ramps {
            let startIdx = Int(ramp.startProgress * CGFloat(n))
            let endIdx = Int(ramp.endProgress * CGFloat(n))
            guard startIdx < endIdx && endIdx < n else { continue }

            let rampLen = endIdx - startIdx

            for i in startIdx...endIdx {
                let localT = CGFloat(i - startIdx) / CGFloat(rampLen)
                let h = ramp.peakHeight * sin(localT * .pi)
                points[i].y += h
            }

            let dropStart = endIdx + 1
            let dropEnd = min(dropStart + rampLen / 3, n - 1)
            for i in dropStart...dropEnd {
                let dropT = CGFloat(i - dropStart) / CGFloat(dropEnd - dropStart + 1)
                points[i].y += ramp.peakHeight * 0.3 * (1 - dropT)
            }
        }
    }

    private func recalculateTangents(_ points: [SCNVector3]) -> [SCNVector3] {
        let n = points.count
        var tangents: [SCNVector3] = []
        for i in 0..<n {
            let prev = points[(i - 1 + n) % n]
            let next = points[(i + 1) % n]
            let tx = next.x - prev.x
            let ty = next.y - prev.y
            let tz = next.z - prev.z
            let tLen = sqrt(tx * tx + ty * ty + tz * tz)
            if tLen > 0.001 {
                tangents.append(SCNVector3(tx / tLen, ty / tLen, tz / tLen))
            } else {
                tangents.append(SCNVector3(0, 0, 1))
            }
        }
        return tangents
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: Catmull-Rom 插值
    // ═══════════════════════════════════════════════════════════

    private func catmullRomInterpolate(_ pts: [SCNVector3]) -> ([SCNVector3], [SCNVector3]) {
        let n = pts.count
        var result: [SCNVector3] = []
        var tangentResult: [SCNVector3] = []

        for i in 0..<n {
            let p0 = pts[(i - 1 + n) % n]
            let p1 = pts[i]
            let p2 = pts[(i + 1) % n]
            let p3 = pts[(i + 2) % n]

            for j in 0..<subdivisions {
                let t = CGFloat(j) / CGFloat(subdivisions)
                let t2 = t * t
                let t3 = t2 * t

                let x = 0.5 * ((2 * p1.x) +
                    (-p0.x + p2.x) * t +
                    (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
                    (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3)

                let y = 0.5 * ((2 * p1.y) +
                    (-p0.y + p2.y) * t +
                    (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
                    (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)

                let z = 0.5 * ((2 * p1.z) +
                    (-p0.z + p2.z) * t +
                    (2 * p0.z - 5 * p1.z + 4 * p2.z - p3.z) * t2 +
                    (-p0.z + 3 * p1.z - 3 * p2.z + p3.z) * t3)

                let tx = 0.5 * ((-p0.x + p2.x) +
                    2 * (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t +
                    3 * (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t2)

                let ty = 0.5 * ((-p0.y + p2.y) +
                    2 * (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t +
                    3 * (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t2)

                let tz = 0.5 * ((-p0.z + p2.z) +
                    2 * (2 * p0.z - 5 * p1.z + 4 * p2.z - p3.z) * t +
                    3 * (-p0.z + 3 * p1.z - 3 * p2.z + p3.z) * t2)

                result.append(SCNVector3(x, y, z))

                let tLen = sqrt(tx * tx + ty * ty + tz * tz)
                if tLen > 0.001 {
                    tangentResult.append(SCNVector3(tx / tLen, ty / tLen, tz / tLen))
                } else {
                    tangentResult.append(SCNVector3(0, 0, 1))
                }
            }
        }

        return (result, tangentResult)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 几何体构建
    // ═══════════════════════════════════════════════════════════

    private func buildTrackGeometry(center: [SCNVector3], tangents: [SCNVector3], ramps: [RampSegment], trackWidth: CGFloat) -> SCNNode {
        let rootNode = SCNNode()
        rootNode.name = "track_root"

        let roadNode = buildRoadSurface(center: center, tangents: tangents, trackWidth: trackWidth)
        rootNode.addChildNode(roadNode)

        // 护栏改为沿赛道外包络(AABB)边界放置
        let boundaryFence = buildBoundaryFence(center: center, tangents: tangents, trackWidth: trackWidth)
        rootNode.addChildNode(boundaryFence)

        let startLine = buildStartLine(center: center, tangents: tangents, trackWidth: trackWidth)
        rootNode.addChildNode(startLine)

        if enableRamps && !ramps.isEmpty {
            let rampNodes = buildRampGeometry(center: center, tangents: tangents, ramps: ramps, trackWidth: trackWidth)
            for rn in rampNodes {
                rootNode.addChildNode(rn)
            }
        }

        return rootNode
    }

    private func buildRoadSurface(center: [SCNVector3], tangents: [SCNVector3], trackWidth: CGFloat) -> SCNNode {
        let n = center.count
        let halfW = trackWidth / 2

        var vertices: [SCNVector3] = []
        var normals: [SCNVector3] = []
        var texCoords: [CGPoint] = []
        var indices: [Int32] = []

        var accLen: CGFloat = 0

        for i in 0..<n {
            let p = center[i]
            let t = tangents[i]
            let normal = SCNVector3(-t.z, 0, t.x)

            let left = SCNVector3(p.x - normal.x * halfW, p.y + 0.05, p.z - normal.z * halfW)
            let right = SCNVector3(p.x + normal.x * halfW, p.y + 0.05, p.z + normal.z * halfW)

            vertices.append(left)
            vertices.append(right)
            normals.append(SCNVector3(0, 1, 0))
            normals.append(SCNVector3(0, 1, 0))

            if i > 0 {
                accLen += scnDist(center[i-1], center[i])
            }
            let u = accLen / trackWidth
            texCoords.append(CGPoint(x: 0, y: u))
            texCoords.append(CGPoint(x: 1, y: u))

            if i < n - 1 {
                let base = Int32(i * 2)
                indices.append(contentsOf: [base, base + 1, base + 2])
                indices.append(contentsOf: [base + 1, base + 3, base + 2])
            }
        }

        let last = Int32((n - 1) * 2)
        indices.append(contentsOf: [last, last + 1, 0])
        indices.append(contentsOf: [last + 1, 1, 0])

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let normalSource = SCNGeometrySource(normals: normals)
        let texSource = SCNGeometrySource(textureCoordinates: texCoords)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)

        let geometry = SCNGeometry(sources: [vertexSource, normalSource, texSource], elements: [element])

        let material = SCNMaterial()
        material.diffuse.contents = NSColor(red: 0.25, green: 0.25, blue: 0.28, alpha: 1)
        material.roughness.contents = NSColor(white: 0.8, alpha: 1)
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "road_surface"

        let lineNode = buildCenterLine(center: center, tangents: tangents, trackWidth: trackWidth)
        node.addChildNode(lineNode)

        return node
    }

    private func buildCenterLine(center: [SCNVector3], tangents: [SCNVector3], trackWidth: CGFloat) -> SCNNode {
        let n = center.count
        let lineW: CGFloat = 0.06
        let dashLen: CGFloat = 1.5
        let gapLen: CGFloat = 1.0

        var vertices: [SCNVector3] = []
        var indices: [Int32] = []
        var accLen: CGFloat = 0

        for i in 0..<n {
            if i > 0 { accLen += scnDist(center[i-1], center[i]) }
            let phase = accLen.truncatingRemainder(dividingBy: dashLen + gapLen)
            if phase > dashLen { continue }

            let p = center[i]
            let t = tangents[i]
            let normal = SCNVector3(-t.z, 0, t.x)

            let left = SCNVector3(p.x - normal.x * lineW, p.y + 0.06, p.z - normal.z * lineW)
            let right = SCNVector3(p.x + normal.x * lineW, p.y + 0.06, p.z + normal.z * lineW)

            let base = Int32(vertices.count)
            vertices.append(left)
            vertices.append(right)

            if vertices.count >= 4 {
                indices.append(contentsOf: [base - 2, base - 1, base])
                indices.append(contentsOf: [base - 1, base + 1, base])
            }
        }

        guard !vertices.isEmpty && !indices.isEmpty else { return SCNNode() }

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: [vertexSource], elements: [element])

        let material = SCNMaterial()
        material.diffuse.contents = NSColor.white
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "center_line"
        return node
    }

    /// 沿赛道外包络(AABB)边界放置护栏——真实RC赛道风格
    private func buildBoundaryFence(center: [SCNVector3], tangents: [SCNVector3], trackWidth: CGFloat) -> SCNNode {
        let fenceRoot = SCNNode()
        fenceRoot.name = "boundary_fence"

        let halfW = trackWidth / 2
        // 计算赛道路面外边缘的AABB包围盒
        var minX: CGFloat = .greatestFiniteMagnitude
        var maxX: CGFloat = -.greatestFiniteMagnitude
        var minZ: CGFloat = .greatestFiniteMagnitude
        var maxZ: CGFloat = -.greatestFiniteMagnitude

        for i in 0..<center.count {
            let p = center[i]
            let t = tangents[i]
            let normal = SCNVector3(-t.z, 0, t.x)
            let lx = p.x - normal.x * halfW
            let lz = p.z - normal.z * halfW
            let rx = p.x + normal.x * halfW
            let rz = p.z + normal.z * halfW
            minX = min(minX, min(lx, rx))
            maxX = max(maxX, max(lx, rx))
            minZ = min(minZ, min(lz, rz))
            maxZ = max(maxZ, max(lz, rz))
        }

        // 紧贴赛道路面边缘 10mm
        let margin: CGFloat = 0.01
        minX -= margin
        maxX += margin
        minZ -= margin
        maxZ += margin

        let fenceH: CGFloat = 0.017    // 护栏高度 (再降2/3: 0.05→0.017)
        let postR: CGFloat = 0.003    // 立柱半径 (直径缩小一半: 0.006→0.003)
        let railR: CGFloat = 0.002    // 横杆半径 (直径缩小一半: 0.004→0.002)
        let postSpacing: CGFloat = 3.2 // 立柱间距 (稀疏)

        // 护栏材质
        let postMat = SCNMaterial()
        postMat.diffuse.contents = NSColor(white: 0.92, alpha: 1)
        postMat.roughness.contents = NSColor(white: 0.6, alpha: 1)

        let railMat = SCNMaterial()
        railMat.diffuse.contents = NSColor(red: 0.85, green: 0.1, blue: 0.1, alpha: 1)
        railMat.roughness.contents = NSColor(white: 0.5, alpha: 1)

        // 四条边的护栏
        let edges: [(start: (CGFloat, CGFloat), end: (CGFloat, CGFloat))] = [
            (start: (minX, minZ), end: (maxX, minZ)),  // 底边
            (start: (maxX, minZ), end: (maxX, maxZ)),  // 右边
            (start: (maxX, maxZ), end: (minX, maxZ)),  // 顶边
            (start: (minX, maxZ), end: (minX, minZ)),  // 左边
        ]

        for edge in edges {
            let dx = edge.end.0 - edge.start.0
            let dz = edge.end.1 - edge.start.1
            let edgeLen = sqrt(dx * dx + dz * dz)
            let numPosts = max(2, Int(edgeLen / postSpacing) + 1)

            var postPositions: [(CGFloat, CGFloat)] = []
            for j in 0..<numPosts {
                let frac = CGFloat(j) / CGFloat(numPosts - 1)
                let px = edge.start.0 + dx * frac
                let pz = edge.start.1 + dz * frac
                postPositions.append((px, pz))

                // 立柱
                let post = SCNCylinder(radius: postR, height: fenceH)
                post.firstMaterial = postMat
                let postNode = SCNNode(geometry: post)
                postNode.position = SCNVector3(px, fenceH / 2, pz)
                fenceRoot.addChildNode(postNode)
            }

            // 横杆（上下两根）
            for railIdx in 0..<2 {
                let railY = railIdx == 0 ? fenceH * 0.4 : fenceH * 0.85
                for j in 0..<(postPositions.count - 1) {
                    let (x1, z1) = postPositions[j]
                    let (x2, z2) = postPositions[j + 1]
                    let segDx = x2 - x1
                    let segDz = z2 - z1
                    let segLen = sqrt(segDx * segDx + segDz * segDz)
                    guard segLen > 0.01 else { continue }

                    let rail = SCNCylinder(radius: railR, height: segLen)
                    rail.firstMaterial = railMat
                    let railNode = SCNNode(geometry: rail)
                    railNode.position = SCNVector3((x1 + x2) / 2, railY, (z1 + z2) / 2)
                    // 旋转圆柱使其水平躺倒并朝向正确方向
                    let heading = atan2(segDx, segDz)
                    railNode.eulerAngles = SCNVector3(0, 0, CGFloat.pi / 2)
                    railNode.eulerAngles.y = heading + CGFloat.pi / 2
                    // 用平放的box替代旋转圆柱，更简单可靠
                    railNode.removeFromParentNode()

                    let railBox = SCNBox(width: segLen, height: railR * 2, length: railR * 2, chamferRadius: railR * 0.5)
                    railBox.firstMaterial = railMat
                    let boxNode = SCNNode(geometry: railBox)
                    boxNode.position = SCNVector3((x1 + x2) / 2, railY, (z1 + z2) / 2)
                    boxNode.eulerAngles.y = atan2(segDx, segDz) + CGFloat.pi / 2
                    fenceRoot.addChildNode(boxNode)
                }
            }
        }

        return fenceRoot
    }

    private func buildStartLine(center: [SCNVector3], tangents: [SCNVector3], trackWidth: CGFloat) -> SCNNode {
        guard center.count > 2 else { return SCNNode() }

        let p = center[0]
        let t = tangents[0]
        let normal = SCNVector3(-t.z, 0, t.x)
        let halfW = trackWidth / 2

        let lineW: CGFloat = 0.3
        let left = SCNVector3(p.x - normal.x * halfW, p.y + 0.07, p.z - normal.z * halfW)
        let right = SCNVector3(p.x + normal.x * halfW, p.y + 0.07, p.z + normal.z * halfW)
        let leftFwd = SCNVector3(left.x + t.x * lineW, left.y, left.z + t.z * lineW)
        let rightFwd = SCNVector3(right.x + t.x * lineW, right.y, right.z + t.z * lineW)

        let vertices = [left, right, leftFwd, rightFwd]
        let indices: [Int32] = [0, 1, 2, 1, 3, 2]

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: [vertexSource], elements: [element])

        let material = SCNMaterial()
        material.diffuse.contents = NSColor.white
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "start_line"
        return node
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 飞坡几何体
    // ═══════════════════════════════════════════════════════════

    private func buildRampGeometry(center: [SCNVector3], tangents: [SCNVector3], ramps: [RampSegment], trackWidth: CGFloat) -> [SCNNode] {
        var nodes: [SCNNode] = []
        let n = center.count

        for ramp in ramps {
            let startIdx = Int(ramp.startProgress * CGFloat(n))
            let endIdx = Int(ramp.endProgress * CGFloat(n))
            guard startIdx < endIdx && endIdx < n else { continue }

            for side: CGFloat in [-1, 1] {
                let halfW = trackWidth / 2
                for i in stride(from: startIdx, to: endIdx, by: 3) {
                    let p = center[i]
                    let t = tangents[i]
                    let normal = SCNVector3(-t.z, 0, t.x)
                    let wx = p.x + normal.x * (halfW + 0.2) * side
                    let wz = p.z + normal.z * (halfW + 0.2) * side

                    let wallH: CGFloat = 0.25
                    let wall = SCNBox(width: 0.08, height: wallH, length: 0.3, chamferRadius: 0.01)
                    wall.firstMaterial?.diffuse.contents = NSColor(red: 1.0, green: 0.6, blue: 0.0, alpha: 1)
                    wall.firstMaterial?.roughness.contents = NSColor(white: 0.7, alpha: 1)
                    let wallNode = SCNNode(geometry: wall)
                    wallNode.position = SCNVector3(wx, p.y + wallH / 2, wz)
                    let heading = atan2(t.x, t.z)
                    wallNode.eulerAngles.y = heading
                    nodes.append(wallNode)
                }
            }

            let peakIdx = (startIdx + endIdx) / 2
            if peakIdx < n {
                let pp = center[peakIdx]
                let marker = SCNCylinder(radius: 0.15, height: 0.05)
                marker.firstMaterial?.diffuse.contents = NSColor.yellow
                let markerNode = SCNNode(geometry: marker)
                markerNode.position = SCNVector3(pp.x, pp.y + 0.03, pp.z)
                nodes.append(markerNode)
            }
        }

        return nodes
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: 工具
    // ═══════════════════════════════════════════════════════════

    private func scnDist(_ a: SCNVector3, _ b: SCNVector3) -> CGFloat {
        let dx = a.x - b.x
        let dy = a.y - b.y
        let dz = a.z - b.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private func makeCurbTexture(side: CGFloat) -> NSImage {
        let w = 256, h = 64
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return NSImage(size: NSSize(width: w, height: h))
        }
        let stripeW = 32
        for i in 0..<(w / stripeW + 1) {
            let isRed = (i + (side > 0 ? 0 : 1)) % 2 == 0
            if isRed {
                ctx.setFillColor(NSColor(red: 0.85, green: 0.1, blue: 0.1, alpha: 1).cgColor)
            } else {
                ctx.setFillColor(NSColor(white: 0.92, alpha: 1).cgColor)
            }
            ctx.fill(CGRect(x: i * stripeW, y: 0, width: stripeW, height: h))
        }
        for _ in 0..<100 {
            let rx = Int.random(in: 0..<w)
            let ry = Int.random(in: 0..<h)
            let g = CGFloat.random(in: 0.3...0.7)
            ctx.setFillColor(NSColor(white: g, alpha: 0.15).cgColor)
            ctx.fill(CGRect(x: rx, y: ry, width: Int.random(in: 1...3), height: Int.random(in: 1...2)))
        }
        if let cgImage = ctx.makeImage() {
            return NSImage(cgImage: cgImage, size: NSSize(width: w, height: h))
        }
        return NSImage(size: NSSize(width: w, height: h))
    }
}
