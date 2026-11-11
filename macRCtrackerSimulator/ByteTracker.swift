// ByteTracker.swift
// ByteTrack 多目标追踪算法的 Swift 原生实现
// 基于论文: ByteTrack: Multi-Object Tracking by Associating Every Detection Box (Zhang et al., 2022)
//
// 核心特性:
//   1. 卡尔曼滤波器预测目标运动轨迹
//   2. 匈牙利算法进行最优匹配
//   3. 级联关联策略: 高分框优先匹配 → 低分框二次匹配
//   4. 丢失轨迹保持与重新关联
//
// 适配 RC 赛车追踪场景:
//   - 使用归一化坐标 [0,1]
//   - 针对小目标和高速移动优化参数
//   - 提供主车选择接口
//
// ★ 修复记录 (2026/3/27):
//   - trackBuffer: 30 → 5 (快速清理误检轨迹)
//   - 新增 minTrackletAge=3 (过滤短命误检轨迹)
//   - 新增 useCenterDistance=true (中心点距离对归一化坐标下小目标更鲁棒)
//   - 新增 maxCenterDist=0.08 (归一化坐标下最大匹配距离)
//   - update() 返回值添加 trackletLen >= minTrackletAge 过滤
//   - 所有匹配阶段统一使用 configuredDistance() 配置感知距离函数

import Foundation
import CoreGraphics  // CGRect, CGPoint 等
import Accelerate   // 用于高效矩阵运算

// MARK: - 轨迹状态枚举

enum BTTrackState: Int {
    case new      = 0
    case tracked  = 1
    case lost     = 2
    case removed  = 3
}

// MARK: - 8维卡尔曼滤波器 (用于边界框追踪)

/// 状态向量: [cx, cy, a, h, vx, vy, va, vh]
///   cx, cy = 边界框中心坐标
///   a      = 宽高比 (width / height)
///   h      = 高度
///   vx, vy, va, vh = 对应的速度
///
/// 观测向量: [cx, cy, a, h]
/// 运动模型: 常速模型 (constant velocity)
final class BTKalmanFilter {

    // 状态转移矩阵 F (8x8)
    private var motionMat: [Double]   // row-major 8x8
    // 观测矩阵 H (4x8)
    private var updateMat: [Double]   // row-major 4x8

    private let stdWeightPosition: Double = 1.0 / 20.0
    private let stdWeightVelocity: Double = 1.0 / 160.0

    init() {
        // F = I(8) + dt*shift
        motionMat = [Double](repeating: 0, count: 64)
        for i in 0..<8 { motionMat[i * 8 + i] = 1.0 }
        for i in 0..<4 { motionMat[i * 8 + (i + 4)] = 1.0 }  // dt = 1

        // H = [I(4) | 0(4x4)]
        updateMat = [Double](repeating: 0, count: 32)
        for i in 0..<4 { updateMat[i * 8 + i] = 1.0 }
    }

    /// 从首次观测创建新轨迹
    func initiate(measurement: [Double]) -> (mean: [Double], covariance: [Double]) {
        // measurement = [cx, cy, a, h]
        let h = measurement[3]
        var mean = [Double](repeating: 0, count: 8)
        for i in 0..<4 { mean[i] = measurement[i] }
        // 速度初始化为 0

        let std: [Double] = [
            2 * stdWeightPosition * h,
            2 * stdWeightPosition * h,
            1e-2,
            2 * stdWeightPosition * h,
            10 * stdWeightVelocity * h,
            10 * stdWeightVelocity * h,
            1e-5,
            10 * stdWeightVelocity * h
        ]
        // 协方差 = diag(std^2)
        var cov = [Double](repeating: 0, count: 64)
        for i in 0..<8 { cov[i * 8 + i] = std[i] * std[i] }

        return (mean, cov)
    }

    /// 卡尔曼预测步骤
    func predict(mean: inout [Double], covariance: inout [Double]) {
        let h = mean[3]
        let stdPos: [Double] = [
            stdWeightPosition * h,
            stdWeightPosition * h,
            1e-2,
            stdWeightPosition * h
        ]
        let stdVel: [Double] = [
            stdWeightVelocity * h,
            stdWeightVelocity * h,
            1e-5,
            stdWeightVelocity * h
        ]

        // 过程噪声 Q = diag([stdPos, stdVel]^2)
        var motionCov = [Double](repeating: 0, count: 64)
        for i in 0..<4 { motionCov[i * 8 + i] = stdPos[i] * stdPos[i] }
        for i in 0..<4 { motionCov[(i+4) * 8 + (i+4)] = stdVel[i] * stdVel[i] }

        // mean = F * mean
        let newMean = matVecMul8x8(motionMat, mean)
        mean = newMean

        // cov = F * cov * F^T + Q
        let temp = matMul8x8(motionMat, covariance)
        let FcovFT = matMulTranspose8x8(temp, motionMat)
        for i in 0..<64 { covariance[i] = FcovFT[i] + motionCov[i] }
    }

    /// 将状态投影到观测空间
    func project(mean: [Double], covariance: [Double]) -> (projMean: [Double], projCov: [Double]) {
        let h = mean[3]
        let std: [Double] = [
            stdWeightPosition * h,
            stdWeightPosition * h,
            1e-1,
            stdWeightPosition * h
        ]
        // 创新噪声 R = diag(std^2)
        var innovCov = [Double](repeating: 0, count: 16)
        for i in 0..<4 { innovCov[i * 4 + i] = std[i] * std[i] }

        // projMean = H * mean
        var projMean = [Double](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                projMean[i] += updateMat[i * 8 + j] * mean[j]
            }
        }

        // projCov = H * cov * H^T + R
        // H is 4x8, cov is 8x8 → H*cov is 4x8
        var HCov = [Double](repeating: 0, count: 32) // 4x8
        for i in 0..<4 {
            for j in 0..<8 {
                var sum = 0.0
                for k in 0..<8 {
                    sum += updateMat[i * 8 + k] * covariance[k * 8 + j]
                }
                HCov[i * 8 + j] = sum
            }
        }
        // HCov * H^T → 4x4
        var projCov = [Double](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                var sum = 0.0
                for k in 0..<8 {
                    sum += HCov[i * 8 + k] * updateMat[j * 8 + k]  // H^T[k,j] = H[j,k]
                }
                projCov[i * 4 + j] = sum
            }
        }
        // + R
        for i in 0..<16 { projCov[i] += innovCov[i] }

        return (projMean, projCov)
    }

    /// 卡尔曼更新步骤
    func update(mean: inout [Double], covariance: inout [Double], measurement: [Double]) {
        let (projMean, projCov) = project(mean: mean, covariance: covariance)

        // Kalman gain K = cov * H^T * (projCov)^{-1}
        // 先求 cov * H^T (8x4)
        var covHT = [Double](repeating: 0, count: 32) // 8x4
        for i in 0..<8 {
            for j in 0..<4 {
                var sum = 0.0
                for k in 0..<8 {
                    sum += covariance[i * 8 + k] * updateMat[j * 8 + k]
                }
                covHT[i * 4 + j] = sum
            }
        }

        // 求 projCov 的逆 (4x4)
        let projCovInv = invert4x4(projCov)

        // K = covHT * projCovInv (8x4)
        var K = [Double](repeating: 0, count: 32)
        for i in 0..<8 {
            for j in 0..<4 {
                var sum = 0.0
                for k in 0..<4 {
                    sum += covHT[i * 4 + k] * projCovInv[k * 4 + j]
                }
                K[i * 4 + j] = sum
            }
        }

        // innovation = measurement - projMean
        var innovation = [Double](repeating: 0, count: 4)
        for i in 0..<4 { innovation[i] = measurement[i] - projMean[i] }

        // new_mean = mean + K * innovation
        for i in 0..<8 {
            var sum = 0.0
            for j in 0..<4 {
                sum += K[i * 4 + j] * innovation[j]
            }
            mean[i] += sum
        }

        // new_cov = cov - K * projCov * K^T
        // K * projCov → 8x4
        var KS = [Double](repeating: 0, count: 32)
        for i in 0..<8 {
            for j in 0..<4 {
                var sum = 0.0
                for k in 0..<4 {
                    sum += K[i * 4 + k] * projCov[k * 4 + j]
                }
                KS[i * 4 + j] = sum
            }
        }
        // KS * K^T → 8x8
        var KSKT = [Double](repeating: 0, count: 64)
        for i in 0..<8 {
            for j in 0..<8 {
                var sum = 0.0
                for k in 0..<4 {
                    sum += KS[i * 4 + k] * K[j * 4 + k]
                }
                KSKT[i * 8 + j] = sum
            }
        }
        for i in 0..<64 { covariance[i] -= KSKT[i] }
    }

    // MARK: - 矩阵工具函数

    /// 8x8矩阵 * 8维向量
    private func matVecMul8x8(_ mat: [Double], _ vec: [Double]) -> [Double] {
        var result = [Double](repeating: 0, count: 8)
        for i in 0..<8 {
            var sum = 0.0
            for j in 0..<8 {
                sum += mat[i * 8 + j] * vec[j]
            }
            result[i] = sum
        }
        return result
    }

    /// 8x8矩阵乘法 A * B
    private func matMul8x8(_ A: [Double], _ B: [Double]) -> [Double] {
        var C = [Double](repeating: 0, count: 64)
        for i in 0..<8 {
            for j in 0..<8 {
                var sum = 0.0
                for k in 0..<8 {
                    sum += A[i * 8 + k] * B[k * 8 + j]
                }
                C[i * 8 + j] = sum
            }
        }
        return C
    }

    /// A * B^T (两个8x8矩阵)
    private func matMulTranspose8x8(_ A: [Double], _ B: [Double]) -> [Double] {
        var C = [Double](repeating: 0, count: 64)
        for i in 0..<8 {
            for j in 0..<8 {
                var sum = 0.0
                for k in 0..<8 {
                    sum += A[i * 8 + k] * B[j * 8 + k]  // B^T[k,j] = B[j,k]
                }
                C[i * 8 + j] = sum
            }
        }
        return C
    }

    /// 4x4 矩阵求逆 (使用 Cramer's rule，对小矩阵足够高效)
    private func invert4x4(_ m: [Double]) -> [Double] {
        // 使用 Accelerate 框架的 LAPACK 求逆
        var matrix = m
        var rowN = __CLPK_integer(4)
        var colN = __CLPK_integer(4)
        var ldaN = __CLPK_integer(4)
        var pivots = [__CLPK_integer](repeating: 0, count: 4)
        var workspace = [Double](repeating: 0, count: 4)
        var error: __CLPK_integer = 0

        // LU 分解
        dgetrf_(&rowN, &colN, &matrix, &ldaN, &pivots, &error)
        if error != 0 {
            // 奇异矩阵，返回单位矩阵作为fallback
            var identity = [Double](repeating: 0, count: 16)
            for i in 0..<4 { identity[i * 4 + i] = 1.0 }
            return identity
        }

        // 求逆
        var invN = __CLPK_integer(4)
        var ldaInv = __CLPK_integer(4)
        var lwork = __CLPK_integer(4)
        dgetri_(&invN, &matrix, &ldaInv, &pivots, &workspace, &lwork, &error)
        if error != 0 {
            var identity = [Double](repeating: 0, count: 16)
            for i in 0..<4 { identity[i * 4 + i] = 1.0 }
            return identity
        }

        return matrix
    }
}

// MARK: - 单个轨迹 (STrack)

final class BTSTrack {
    // 全局 ID 计数器
    private static var idCounter: Int = 0
    static func resetIDCounter() { idCounter = 0 }
    private static func nextID() -> Int {
        idCounter += 1
        return idCounter
    }

    // 轨迹属性
    var trackId: Int = 0
    var state: BTTrackState = .new
    var isActivated: Bool = false
    var score: Double = 0
    var frameId: Int = 0
    var startFrame: Int = 0
    var trackletLen: Int = 0

    // 卡尔曼滤波状态
    var mean: [Double] = [Double](repeating: 0, count: 8)
    var covariance: [Double] = [Double](repeating: 0, count: 64)
    private var _tlwh: [Double]  // 初始 tlwh (在 activate 前使用)
    private var kalmanFilter: BTKalmanFilter?

    var endFrame: Int { return frameId }

    init(tlwh: [Double], score: Double) {
        self._tlwh = tlwh
        self.score = score
    }

    /// 当前位置 (top-left x, top-left y, width, height)
    var tlwh: [Double] {
        if kalmanFilter == nil {
            return _tlwh
        }
        // 从 xyah 恢复 tlwh
        let cx = mean[0], cy = mean[1], a = mean[2], h = mean[3]
        let w = a * h
        return [cx - w / 2, cy - h / 2, w, h]
    }

    /// 当前位置 (x1, y1, x2, y2)
    var tlbr: [Double] {
        let t = tlwh
        return [t[0], t[1], t[0] + t[2], t[1] + t[3]]
    }

    /// 中心点坐标
    var center: (x: Double, y: Double) {
        let t = tlwh
        return (t[0] + t[2] / 2, t[1] + t[3] / 2)
    }

    /// tlwh → xyah 格式
    static func tlwhToXyah(_ tlwh: [Double]) -> [Double] {
        let cx = tlwh[0] + tlwh[2] / 2
        let cy = tlwh[1] + tlwh[3] / 2
        let a = tlwh[2] / max(tlwh[3], 1e-6)
        let h = tlwh[3]
        return [cx, cy, a, h]
    }

    /// tlbr → tlwh 格式
    static func tlbrToTlwh(_ tlbr: [Double]) -> [Double] {
        return [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]
    }

    /// 激活新轨迹
    func activate(kf: BTKalmanFilter, frameId: Int) {
        self.kalmanFilter = kf
        self.trackId = BTSTrack.nextID()
        let xyah = BTSTrack.tlwhToXyah(_tlwh)
        let (m, c) = kf.initiate(measurement: xyah)
        self.mean = m
        self.covariance = c
        self.trackletLen = 0
        self.state = .tracked
        // ★ RC赛车场景: 新轨迹立即激活 (YOLO置信度已经很高，且归一化坐标下小目标帧间IoU低，延迟激活会导致永远无法匹配)
        self.isActivated = true
        self.frameId = frameId
        self.startFrame = frameId
    }

    /// 重新激活丢失的轨迹
    func reActivate(newTrack: BTSTrack, frameId: Int, newId: Bool = false) {
        let xyah = BTSTrack.tlwhToXyah(newTrack.tlwh)
        kalmanFilter?.update(mean: &mean, covariance: &covariance, measurement: xyah)
        self.trackletLen = 0
        self.state = .tracked
        self.isActivated = true
        self.frameId = frameId
        if newId {
            self.trackId = BTSTrack.nextID()
        }
        self.score = newTrack.score
    }

    /// 用新检测更新已匹配的轨迹
    func update(newTrack: BTSTrack, frameId: Int) {
        self.frameId = frameId
        self.trackletLen += 1
        let xyah = BTSTrack.tlwhToXyah(newTrack.tlwh)
        kalmanFilter?.update(mean: &mean, covariance: &covariance, measurement: xyah)
        self.state = .tracked
        self.isActivated = true
        self.score = newTrack.score
    }

    /// 卡尔曼预测
    func predict() {
        if state != .tracked {
            mean[7] = 0  // 丢失状态下不预测高度变化速度
        }
        kalmanFilter?.predict(mean: &mean, covariance: &covariance)
    }

    func markLost() { state = .lost }
    func markRemoved() { state = .removed }
}

// MARK: - 匈牙利算法 (线性分配)

/// 经典匈牙利算法实现
/// 输入: cost矩阵 (n x m), 阈值
/// 输出: 匹配对, 未匹配行, 未匹配列
struct HungarianResult {
    var matches: [(Int, Int)]
    var unmatchedRows: [Int]
    var unmatchedCols: [Int]
}

func linearAssignment(costMatrix: [[Double]], thresh: Double, numRows: Int? = nil, numCols: Int? = nil) -> HungarianResult {
    let nRows = numRows ?? costMatrix.count
    let nCols = numCols ?? (costMatrix.first?.count ?? 0)

    if nRows == 0 || nCols == 0 {
        return HungarianResult(
            matches: [],
            unmatchedRows: Array(0..<nRows),
            unmatchedCols: Array(0..<nCols)
        )
    }

    // 使用 Jonker-Volgenant / 匈牙利算法
    let assignment = hungarianSolve(costMatrix: costMatrix)

    var matches: [(Int, Int)] = []
    var unmatchedRows: [Int] = []
    var unmatchedCols = Set(0..<nCols)

    for row in 0..<nRows {
        let col = assignment[row]
        if col < 0 || col >= nCols || costMatrix[row][col] > thresh {
            unmatchedRows.append(row)
        } else {
            matches.append((row, col))
            unmatchedCols.remove(col)
        }
    }

    return HungarianResult(
        matches: matches,
        unmatchedRows: unmatchedRows,
        unmatchedCols: Array(unmatchedCols).sorted()
    )
}

/// 匈牙利算法核心 (Kuhn-Munkres)
/// 返回每行分配的列索引 (-1 表示未分配)
private func hungarianSolve(costMatrix: [[Double]]) -> [Int] {
    let n = costMatrix.count
    let m = costMatrix.first?.count ?? 0
    let dim = max(n, m)

    // 扩展为方阵
    var cost = [[Double]](repeating: [Double](repeating: 0, count: dim), count: dim)
    let bigVal = 1e9
    for i in 0..<dim {
        for j in 0..<dim {
            if i < n && j < m {
                cost[i][j] = costMatrix[i][j]
            } else {
                cost[i][j] = bigVal
            }
        }
    }

    // 行归约
    for i in 0..<dim {
        let minVal = cost[i].min() ?? 0
        for j in 0..<dim { cost[i][j] -= minVal }
    }
    // 列归约
    for j in 0..<dim {
        var minVal = Double.infinity
        for i in 0..<dim { minVal = min(minVal, cost[i][j]) }
        for i in 0..<dim { cost[i][j] -= minVal }
    }

    // 匈牙利算法主循环
    var rowMatch = [Int](repeating: -1, count: dim)
    var colMatch = [Int](repeating: -1, count: dim)

    for _ in 0..<dim {
        // 尝试增广路径
        var visited = [Bool](repeating: false, count: dim)
        var slack = [Double](repeating: Double.infinity, count: dim)
        var slackRow = [Int](repeating: -1, count: dim)
        var prev = [Int](repeating: -1, count: dim)
        var queue = [Int]()

        // 找一个未匹配的行
        var startRow = -1
        for i in 0..<dim {
            if rowMatch[i] == -1 {
                startRow = i
                break
            }
        }
        if startRow == -1 { break }

        // BFS 初始化
        for j in 0..<dim {
            slack[j] = cost[startRow][j]
            slackRow[j] = startRow
        }

        var matched = false
        var matchedCol = -1
        var curRow = startRow

        while !matched {
            // 找最小 slack 的未访问列
            var minSlack = Double.infinity
            var minCol = -1
            for j in 0..<dim {
                if !visited[j] && slack[j] < minSlack {
                    minSlack = slack[j]
                    minCol = j
                }
            }

            if minCol == -1 { break }

            // 更新 slack
            for j in 0..<dim {
                if !visited[j] {
                    slack[j] -= minSlack
                }
            }
            // 也需要更新已访问列对应行的 cost
            for j in 0..<dim {
                if visited[j] {
                    let r = colMatch[j]
                    if r >= 0 {
                        for jj in 0..<dim {
                            if !visited[jj] {
                                // 不需要额外操作，slack已经在上面减了
                            }
                        }
                    }
                }
            }

            visited[minCol] = true
            prev[minCol] = slackRow[minCol]

            if colMatch[minCol] == -1 {
                // 找到增广路径
                matched = true
                matchedCol = minCol
            } else {
                // 继续扩展
                let nextRow = colMatch[minCol]
                for j in 0..<dim {
                    if !visited[j] {
                        let newSlack = cost[nextRow][j] - minSlack
                        if newSlack < slack[j] {
                            slack[j] = newSlack
                            slackRow[j] = nextRow
                        }
                    }
                }
            }
        }

        if matched {
            // 沿增广路径更新匹配
            var col = matchedCol
            while col != -1 {
                let row = prev[col]
                let prevCol = rowMatch[row]
                rowMatch[row] = col
                colMatch[col] = row
                col = prevCol
            }
        }
    }

    // 返回原始维度的结果
    var result = [Int](repeating: -1, count: n)
    for i in 0..<n {
        if rowMatch[i] >= 0 && rowMatch[i] < m {
            result[i] = rowMatch[i]
        }
    }
    return result
}

// MARK: - 距离计算

/// 中心点距离匹配 (归一化坐标下小目标专用)
/// 返回 cost matrix，值域 [0, 1]，距离超过 maxDist 的设为 1.0
func centerDistance(aTracks: [BTSTrack], bTracks: [BTSTrack], maxDist: Double = 0.08) -> [[Double]] {
    let n = aTracks.count
    let m = bTracks.count
    var costMatrix = [[Double]](repeating: [Double](repeating: 1.0, count: m), count: n)
    for i in 0..<n {
        let ac = aTracks[i].center
        for j in 0..<m {
            let bc = bTracks[j].center
            let dx = ac.x - bc.x
            let dy = ac.y - bc.y
            let dist = sqrt(dx * dx + dy * dy)
            if dist < maxDist {
                costMatrix[i][j] = dist / maxDist  // 归一化到 [0, 1)
            }
        }
    }
    return costMatrix
}

/// 混合距离: 优先用 IoU，当 IoU 为 0 时 fallback 到中心点距离
func hybridDistance(aTracks: [BTSTrack], bTracks: [BTSTrack], maxCenterDist: Double = 0.08) -> [[Double]] {
    let iouCost = iouDistanceFromBoxes(aBoxes: aTracks.map { $0.tlbr }, bBoxes: bTracks.map { $0.tlbr })
    let centerCost = centerDistance(aTracks: aTracks, bTracks: bTracks, maxDist: maxCenterDist)
    let n = aTracks.count
    let m = bTracks.count
    var costMatrix = [[Double]](repeating: [Double](repeating: 1.0, count: m), count: n)
    for i in 0..<n {
        for j in 0..<m {
            let iouDist = iouCost[i][j]
            if iouDist < 1.0 {
                costMatrix[i][j] = iouDist  // IoU 有效时用 IoU
            } else {
                costMatrix[i][j] = centerCost[i][j]  // IoU=0 时 fallback 中心点距离
            }
        }
    }
    return costMatrix
}

/// ★ 配置感知的距离函数 (ByteTracker 内部使用)
/// useCenterDistance=true 时使用纯中心点距离，否则使用混合距离 (IoU + 中心点 fallback)
func configuredDistance(aTracks: [BTSTrack], bTracks: [BTSTrack], config: BTConfig) -> [[Double]] {
    if config.useCenterDistance {
        return centerDistance(aTracks: aTracks, bTracks: bTracks, maxDist: config.maxCenterDist)
    } else {
        return hybridDistance(aTracks: aTracks, bTracks: bTracks, maxCenterDist: config.maxCenterDist)
    }
}

/// 默认 IoU 距离 (保留向后兼容，外部调用者可继续使用)
func iouDistance(aTracks: [BTSTrack], bTracks: [BTSTrack]) -> [[Double]] {
    return hybridDistance(aTracks: aTracks, bTracks: bTracks, maxCenterDist: 0.08)
}

func iouDistanceFromBoxes(aBoxes: [[Double]], bBoxes: [[Double]]) -> [[Double]] {
    let n = aBoxes.count
    let m = bBoxes.count
    var costMatrix = [[Double]](repeating: [Double](repeating: 1.0, count: m), count: n)

    for i in 0..<n {
        for j in 0..<m {
            let iou = computeIoU(a: aBoxes[i], b: bBoxes[j])
            costMatrix[i][j] = 1.0 - iou
        }
    }
    return costMatrix
}

/// 计算两个 tlbr 框的 IoU
private func computeIoU(a: [Double], b: [Double]) -> Double {
    let x1 = max(a[0], b[0])
    let y1 = max(a[1], b[1])
    let x2 = min(a[2], b[2])
    let y2 = min(a[3], b[3])

    let interW = max(0, x2 - x1)
    let interH = max(0, y2 - y1)
    let interArea = interW * interH

    let areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    let areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    let unionArea = areaA + areaB - interArea

    if unionArea <= 0 { return 0 }
    return interArea / unionArea
}

/// 融合 IoU 距离和检测分数 (与原版 fuse_score 一致)
func fuseScore(costMatrix: [[Double]], detections: [BTSTrack]) -> [[Double]] {
    if costMatrix.isEmpty { return costMatrix }
    var fused = costMatrix
    for i in 0..<fused.count {
        for j in 0..<fused[i].count {
            let iouSim = 1.0 - costMatrix[i][j]
            let score = detections[j].score
            let fuseSim = iouSim * score
            fused[i][j] = 1.0 - fuseSim
        }
    }
    return fused
}

// MARK: - ByteTracker 主追踪器

/// ByteTrack 配置
struct BTConfig {
    var trackThresh: Double = 0.3     // 高分检测阈值 (RC赛车小目标置信度偏低)
    var matchThresh: Double = 0.8     // 第一级匹配的 IoU 距离阈值
    var secondMatchThresh: Double = 0.5  // 第二级匹配阈值
    var newTrackThresh: Double = 0.4  // 新建轨迹的最低分数 (= trackThresh + 0.1)
    var lowThresh: Double = 0.1       // 低分检测的下限
    var trackBuffer: Int = 5          // ★ 丢失轨迹保持帧数 (从30降到5，快速清理误检轨迹)
    var fuseScoreEnabled: Bool = true // 是否融合检测分数
    var minTrackletAge: Int = 3       // ★ 最小轨迹存活帧数 (只返回存活≥3帧的轨迹，过滤短命误检)
    var useCenterDistance: Bool = true // ★ 使用中心点距离匹配 (归一化坐标下小目标更鲁棒)
    var maxCenterDist: Double = 0.08  // ★ 中心点距离阈值 (归一化坐标)

    /// RC 赛车默认配置
    static var rcDefault: BTConfig {
        return BTConfig(
            trackThresh: 0.3,
            matchThresh: 0.8,
            secondMatchThresh: 0.5,
            newTrackThresh: 0.4,
            lowThresh: 0.1,
            trackBuffer: 5,
            fuseScoreEnabled: true,
            minTrackletAge: 3,
            useCenterDistance: true,
            maxCenterDist: 0.08
        )
    }
}

/// 检测框输入格式
struct BTDetection {
    var x1: Double  // tlbr 格式
    var y1: Double
    var x2: Double
    var y2: Double
    var score: Double
    var classId: Int

    var tlwh: [Double] {
        return [x1, y1, x2 - x1, y2 - y1]
    }
    var tlbr: [Double] {
        return [x1, y1, x2, y2]
    }
    var center: (x: Double, y: Double) {
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    }
}

/// ByteTrack 主追踪器
final class ByteTracker {

    private let config: BTConfig
    private let kalmanFilter = BTKalmanFilter()

    // 轨迹池
    private(set) var trackedStracks: [BTSTrack] = []
    private(set) var lostStracks: [BTSTrack] = []
    private(set) var removedStracks: [BTSTrack] = []

    private(set) var frameId: Int = 0
    private let maxTimeLost: Int

    // 主车追踪 ID (用于 RC 赛车场景)
    private(set) var primaryTrackId: Int = -1

    init(config: BTConfig = .rcDefault) {
        self.config = config
        self.maxTimeLost = config.trackBuffer
    }

    /// 重置追踪器状态
    func reset() {
        trackedStracks.removeAll()
        lostStracks.removeAll()
        removedStracks.removeAll()
        frameId = 0
        primaryTrackId = -1
        BTSTrack.resetIDCounter()
    }

    /// 设置主车追踪 ID
    func setPrimaryTrack(id: Int) {
        primaryTrackId = id
    }

    /// 获取当前主车轨迹 (如果存在)
    func getPrimaryTrack() -> BTSTrack? {
        // 先在 tracked 中找
        if let t = trackedStracks.first(where: { $0.trackId == primaryTrackId && $0.isActivated }) {
            return t
        }
        // 再在 lost 中找 (短暂丢失)
        if let t = lostStracks.first(where: { $0.trackId == primaryTrackId }) {
            return t
        }
        return nil
    }

    // MARK: - 核心更新方法

    /// 输入当前帧的所有检测结果，返回所有活跃轨迹
    ///
    /// - Parameter detections: YOLO 输出的检测框列表
    /// - Returns: 当前帧所有已激活且存活超过 minTrackletAge 的轨迹
    @discardableResult
    func update(detections: [BTDetection]) -> [BTSTrack] {
        frameId += 1

        var activatedStracks: [BTSTrack] = []
        var refindStracks: [BTSTrack] = []
        var lostThisFrame: [BTSTrack] = []
        var removedThisFrame: [BTSTrack] = []

        // ========== Step 1: 分离高分和低分检测 ==========
        let scores = detections.map { $0.score }

        var highDets: [BTDetection] = []
        var lowDets: [BTDetection] = []
        for det in detections {
            if det.score >= config.trackThresh {
                highDets.append(det)
            } else if det.score >= config.lowThresh {
                lowDets.append(det)
            }
        }

        // 创建高分 STrack
        let highSTracks = highDets.map { BTSTrack(tlwh: $0.tlwh, score: $0.score) }
        // 创建低分 STrack
        let lowSTracks = lowDets.map { BTSTrack(tlwh: $0.tlwh, score: $0.score) }

        // ========== Step 2: 分离已确认和未确认轨迹 ==========
        var unconfirmed: [BTSTrack] = []
        var confirmedTracked: [BTSTrack] = []
        for track in trackedStracks {
            if !track.isActivated {
                unconfirmed.append(track)
            } else {
                confirmedTracked.append(track)
            }
        }

        // ========== Step 3: 合并 tracked + lost 作为候选池 ==========
        var strackPool = jointStracks(confirmedTracked, lostStracks)

        // 卡尔曼预测所有候选轨迹
        for track in strackPool {
            track.predict()
        }
        // 也预测未确认轨迹
        for track in unconfirmed {
            track.predict()
        }

        // ========== Step 4: 第一级匹配 — 高分检测 vs 候选池 ==========
        // ★ 使用配置感知的距离函数 (中心点距离对小目标更鲁棒)
        var dists = configuredDistance(aTracks: strackPool, bTracks: highSTracks, config: config)
        if config.fuseScoreEnabled {
            dists = fuseScore(costMatrix: dists, detections: highSTracks)
        }

        let firstResult = linearAssignment(costMatrix: dists, thresh: config.matchThresh, numRows: strackPool.count, numCols: highSTracks.count)

        for (iTracked, iDet) in firstResult.matches {
            let track = strackPool[iTracked]
            let det = highSTracks[iDet]
            if track.state == .tracked {
                track.update(newTrack: det, frameId: frameId)
                activatedStracks.append(track)
            } else {
                track.reActivate(newTrack: det, frameId: frameId, newId: false)
                refindStracks.append(track)
            }
        }

        // ========== Step 5: 第二级匹配 — 低分检测 vs 未匹配的 tracked ==========
        let unmatchedTrackedIndices = firstResult.unmatchedRows
        let rTrackedStracks = unmatchedTrackedIndices
            .filter { strackPool[$0].state == .tracked }
            .map { strackPool[$0] }

        // ★ 第二级也使用配置感知的距离函数
        let dists2 = configuredDistance(aTracks: rTrackedStracks, bTracks: lowSTracks, config: config)
        let secondResult = linearAssignment(costMatrix: dists2, thresh: config.secondMatchThresh, numRows: rTrackedStracks.count, numCols: lowSTracks.count)

        for (iTracked, iDet) in secondResult.matches {
            let track = rTrackedStracks[iTracked]
            let det = lowSTracks[iDet]
            if track.state == .tracked {
                track.update(newTrack: det, frameId: frameId)
                activatedStracks.append(track)
            } else {
                track.reActivate(newTrack: det, frameId: frameId, newId: false)
                refindStracks.append(track)
            }
        }

        // 未匹配的 tracked → 标记为 lost
        for it in secondResult.unmatchedRows {
            let track = rTrackedStracks[it]
            if track.state != .lost {
                track.markLost()
                lostThisFrame.append(track)
            }
        }
        // 第一级中未匹配且不在 rTrackedStracks 中的 lost 轨迹也保持 lost
        for idx in unmatchedTrackedIndices {
            let track = strackPool[idx]
            if track.state == .lost {
                // 已经是 lost，保持
            }
        }

        // ========== Step 6: 处理未确认轨迹 ==========
        let remainingHighDets = firstResult.unmatchedCols.map { highSTracks[$0] }
        let dists3 = configuredDistance(aTracks: unconfirmed, bTracks: remainingHighDets, config: config)
        let thirdResult = linearAssignment(costMatrix: dists3, thresh: 0.7, numRows: unconfirmed.count, numCols: remainingHighDets.count)

        for (iTracked, iDet) in thirdResult.matches {
            unconfirmed[iTracked].update(newTrack: remainingHighDets[iDet], frameId: frameId)
            activatedStracks.append(unconfirmed[iTracked])
        }
        for it in thirdResult.unmatchedRows {
            let track = unconfirmed[it]
            track.markRemoved()
            removedThisFrame.append(track)
        }

        // ========== Step 7: 初始化新轨迹 ==========
        let finalUnmatchedDets = thirdResult.unmatchedCols.map { remainingHighDets[$0] }
        for det in finalUnmatchedDets {
            if det.score < config.newTrackThresh { continue }
            det.activate(kf: kalmanFilter, frameId: frameId)
            activatedStracks.append(det)
        }

        // ========== Step 8: 清理超时丢失轨迹 ==========
        for track in lostStracks {
            if frameId - track.endFrame > maxTimeLost {
                track.markRemoved()
                removedThisFrame.append(track)
            }
        }

        // ========== Step 9: 更新轨迹池 ==========
        trackedStracks = trackedStracks.filter { $0.state == .tracked }
        trackedStracks = jointStracks(trackedStracks, activatedStracks)
        trackedStracks = jointStracks(trackedStracks, refindStracks)
        lostStracks = subStracks(lostStracks, trackedStracks)
        lostStracks.append(contentsOf: lostThisFrame)
        lostStracks = subStracks(lostStracks, removedStracks)
        removedStracks.append(contentsOf: removedThisFrame)
        let (deduped1, deduped2) = removeDuplicateStracks(trackedStracks, lostStracks)
        trackedStracks = deduped1
        lostStracks = deduped2

        // ★ DEBUG: 输出内部状态
        if frameId <= 10 || frameId % 100 == 0 {
            let dbg = NSTemporaryDirectory() + "bt_internal_debug.log"
            var lines = "[Frame \(frameId)] highDets=\(highDets.count) lowDets=\(lowDets.count) confirmedTracked=\(confirmedTracked.count) unconfirmed=\(unconfirmed.count) strackPool=\(strackPool.count)\n"
            lines += "  1stMatch: matches=\(firstResult.matches.count) unmatchedRows=\(firstResult.unmatchedRows.count) unmatchedCols=\(firstResult.unmatchedCols.count)\n"
            lines += "  2ndMatch: matches=\(secondResult.matches.count)\n"
            lines += "  3rdMatch: matches=\(thirdResult.matches.count)\n"
            lines += "  newTracks=\(finalUnmatchedDets.filter { $0.score >= config.newTrackThresh }.count)\n"
            lines += "  activated=\(activatedStracks.count) refind=\(refindStracks.count) lost=\(lostThisFrame.count) removed=\(removedThisFrame.count)\n"
            lines += "  trackedStracks=\(trackedStracks.count) lostStracks=\(lostStracks.count)\n"
            for t in trackedStracks {
                lines += "    track id=\(t.trackId) len=\(t.trackletLen) activated=\(t.isActivated) state=\(t.state.rawValue) score=\(String(format: "%.3f", t.score)) center=(\(String(format: "%.4f", t.center.x)),\(String(format: "%.4f", t.center.y)))\n"
            }
            if let d = lines.data(using: .utf8) {
                if let h = try? FileHandle(forWritingTo: URL(fileURLWithPath: dbg)) {
                    h.seekToEndOfFile(); h.write(d); h.closeFile()
                } else { try? d.write(to: URL(fileURLWithPath: dbg)) }
            }
        }

        // ★ 返回已激活且存活超过 minTrackletAge 的轨迹 (过滤短命误检轨迹)
        return trackedStracks.filter { $0.isActivated && $0.trackletLen >= config.minTrackletAge }
    }

    // MARK: - 主车选择逻辑 (RC 赛车专用)

    /// 从当前帧的活跃轨迹中选择主车
    /// 策略: 优先保持已有主车ID → 如果丢失则选择最近/最大/最高分的目标
    ///
    /// - Parameters:
    ///   - activeTracks: update() 返回的活跃轨迹
    ///   - lastRect: 上一帧主车的归一化矩形 (可选，用于距离匹配)
    /// - Returns: 选中的主车轨迹，nil 表示本帧无法追踪
    func selectPrimaryCar(activeTracks: [BTSTrack], lastRect: CGRect? = nil) -> BTSTrack? {
        if activeTracks.isEmpty { return nil }

        // 1. 如果已有主车 ID，优先保持
        if primaryTrackId > 0 {
            if let track = activeTracks.first(where: { $0.trackId == primaryTrackId }) {
                return track
            }
        }

        // 2. 主车丢失，需要重新选择
        // 策略: 如果有 lastRect，选择与之 IoU 最大的; 否则选最高分的
        if let lastRect = lastRect {
            let lastTlbr = [
                Double(lastRect.minX), Double(lastRect.minY),
                Double(lastRect.maxX), Double(lastRect.maxY)
            ]
            var bestTrack: BTSTrack? = nil
            var bestIoU: Double = 0
            for track in activeTracks {
                let iou = computeIoU(a: track.tlbr, b: lastTlbr)
                if iou > bestIoU {
                    bestIoU = iou
                    bestTrack = track
                }
            }
            if let t = bestTrack, bestIoU > 0.1 {
                primaryTrackId = t.trackId
                return t
            }
        }

        // 3. 没有 lastRect 或 IoU 都太低 → 选最高分
        if let best = activeTracks.max(by: { $0.score < $1.score }) {
            primaryTrackId = best.trackId
            return best
        }

        return nil
    }

    // MARK: - 辅助函数

    /// 合并两个轨迹列表 (去重)
    private func jointStracks(_ listA: [BTSTrack], _ listB: [BTSTrack]) -> [BTSTrack] {
        var exists = Set<Int>()
        var result: [BTSTrack] = []
        for t in listA {
            exists.insert(t.trackId)
            result.append(t)
        }
        for t in listB {
            if !exists.contains(t.trackId) {
                exists.insert(t.trackId)
                result.append(t)
            }
        }
        return result
    }

    /// 从 listA 中移除 listB 中存在的轨迹
    private func subStracks(_ listA: [BTSTrack], _ listB: [BTSTrack]) -> [BTSTrack] {
        let bIds = Set(listB.map { $0.trackId })
        return listA.filter { !bIds.contains($0.trackId) }
    }

    /// 移除重复轨迹 (距离 < 0.15 的保留存活时间更长的)
    private func removeDuplicateStracks(_ strackA: [BTSTrack], _ strackB: [BTSTrack]) -> ([BTSTrack], [BTSTrack]) {
        // ★ 使用配置感知的距离函数
        let dists = configuredDistance(aTracks: strackA, bTracks: strackB, config: config)
        var dupaSet = Set<Int>()
        var dupbSet = Set<Int>()

        for i in 0..<dists.count {
            for j in 0..<(dists[i].count) {
                if dists[i][j] < 0.15 {  // 距离 < 0.15 视为重复
                    let timeA = strackA[i].frameId - strackA[i].startFrame
                    let timeB = strackB[j].frameId - strackB[j].startFrame
                    if timeA > timeB {
                        dupbSet.insert(j)
                    } else {
                        dupaSet.insert(i)
                    }
                }
            }
        }

        let resA = strackA.enumerated().filter { !dupaSet.contains($0.offset) }.map { $0.element }
        let resB = strackB.enumerated().filter { !dupbSet.contains($0.offset) }.map { $0.element }
        return (resA, resB)
    }
}

// MARK: - CGRect 扩展 (方便与现有系统交互)

extension BTSTrack {
    /// 返回归一化的 CGRect (与现有系统的 CGRect 格式一致)
    var normalizedRect: CGRect {
        let t = tlwh
        return CGRect(x: t[0], y: t[1], width: t[2], height: t[3])
    }
}

extension BTDetection {
    /// 从归一化 CGRect 和分数创建检测
    init(rect: CGRect, score: Double, classId: Int = 0) {
        self.x1 = Double(rect.minX)
        self.y1 = Double(rect.minY)
        self.x2 = Double(rect.maxX)
        self.y2 = Double(rect.maxY)
        self.score = score
        self.classId = classId
    }
}
