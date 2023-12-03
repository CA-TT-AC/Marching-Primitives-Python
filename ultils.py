import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from skimage import measure
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def parse_input_args(voxel_grid, **kwargs):
    # 设置默认参数
    defaults = {
        'verbose': True,
        'padding_size': int(np.ceil(12 * voxel_grid['truncation'] / voxel_grid['interval'])),
        'min_area': int(np.ceil(voxel_grid['size'][0] / 20)),
        'max_division': 50,
        'scaleInitRatio': 0.1,
        'nanRange': 0.5 * voxel_grid['interval'],
        'w': 0.99,
        'tolerance': 1e-6,
        'relative_tolerance': 1e-4,
        'switch_tolerance': 1e-1,
        'maxSwitch': 2,
        'iter_min': 2,
        'maxOptiIter': 2,
        'maxIter': 15,
        'activeMultiplier': 3
    }

    # 使用从kwargs传入的参数来更新默认参数
    for key, value in kwargs.items():
        if key in defaults:
            defaults[key] = value

    return defaults

def idx3d_flatten(indices, grid):
    # 将3D索引平铺为1D索引]
    return indices[:, 0] + grid['size'][0] * (indices[:, 1] - 1) + grid['size'][0] * grid['size'][1] * (indices[:, 2] - 1)

def idx2Coordinate(indices, grid):
    # 从网格索引转换为3D坐标
    x = grid['x'][indices[:, 0] - 1]
    y = grid['y'][indices[:, 1] - 1]
    z = grid['z'][indices[:, 2] - 1]
    return np.vstack([x, y, z]).T

import numpy as np

def rotz(x):
    x_rad = np.radians(x)
    rotm = np.array([[np.cos(x_rad), -np.sin(x_rad), 0],
                     [np.sin(x_rad), np.cos(x_rad), 0],
                     [0, 0, 1]])
    return rotm

def rotm2eul(rotm):
    # 创建旋转对象
    rotation = R.from_matrix(rotm)
    
    # 将旋转转换为欧拉角
    # 'XYZ' 是旋转顺序，可以根据需要更改（例如，'ZYX', 'YZX', 'XZY' 等）
    eul = rotation.as_euler('XYZ')
    
    return eul

def eul2rotm(eul):
    ct = np.cos(eul)
    st = np.sin(eul)

    R = np.zeros((3, 3, eul.shape[0]))
    R[0, 0, :] = ct[:, 1] * ct[:, 0]
    R[0, 1, :] = st[:, 2] * st[:, 1] * ct[:, 0] - ct[:, 2] * st[:, 0]
    R[0, 2, :] = ct[:, 2] * st[:, 1] * ct[:, 0] + st[:, 2] * st[:, 0]
    R[1, 0, :] = ct[:, 1] * st[:, 0]
    R[1, 1, :] = st[:, 2] * st[:, 1] * st[:, 0] + ct[:, 2] * ct[:, 0]
    R[1, 2, :] = ct[:, 2] * st[:, 1] * st[:, 0] - st[:, 2] * ct[:, 0]
    R[2, 0, :] = -st[:, 1]
    R[2, 1, :] = st[:, 2] * ct[:, 1]
    R[2, 2, :] = ct[:, 2] * ct[:, 1]

    return R

def difference_sqsdf(params, sdf, points, truncation, weight):
    R = eul2rotm(params[5:8])
    t = params[8:11]
    X = R.T @ points - R.T @ t[:, np.newaxis]

    r0 = np.linalg.norm(X, axis=0)

    scale = ((((X[0, :] / params[2]) ** 2) ** (1 / params[1]) + 
              ((X[1, :] / params[3]) ** 2) ** (1 / params[1])) ** (params[1] / params[0]) + 
             ((X[2, :] / params[4]) ** 2) ** (1 / params[0])) ** (-params[0] / 2)

    sdf_para = r0 * (1 - scale)

    if truncation != 0:
        sdf_para = np.clip(sdf_para, -truncation, truncation)

    dist = (sdf_para - sdf) * np.sqrt(weight)
    
    return dist

def cost_switched(params, sdf, points, truncation, weight):
    value = np.zeros(params.shape[0])

    for i in range(params.shape[0]):
        diff = difference_sqsdf(params[i, :], sdf, points, truncation, weight)
        value[i] = np.sum(diff ** 2)

    return value


def sdf_superquadric(params, points, truncation):
    R = eul2rotm(params[5:8])
    t = params[8:11]
    X = R.T @ points - R.T @ t[:, np.newaxis]

    r0 = np.linalg.norm(X, axis=0)
    scale = ((((X[0, :] / params[2]) ** 2) ** (1 / params[1]) + 
              ((X[1, :] / params[3]) ** 2) ** (1 / params[1])) ** (params[1] / params[0]) + 
             ((X[2, :] / params[4]) ** 2) ** (1 / params[0])) ** (-params[0] / 2)

    sdf = r0 * (1 - scale)

    if truncation != 0:
        sdf = np.clip(sdf, -truncation, truncation)

    return sdf

def inlier_weight(sdf_active, active_idx, sdf_current, sigma2, w, truncation):
    in_idx = sdf_active < 0.0 * truncation
    sdf_current = sdf_current[active_idx]

    const = w / ((1 - w) * np.sqrt(2 * np.pi * sigma2) * truncation)
    dist_current = np.clip(sdf_current[in_idx], -truncation, truncation) - sdf_active[in_idx]

    weight = np.ones(sdf_active.shape)
    p = np.exp(-0.5 / sigma2 * dist_current ** 2)
    p = p / (const + p)
    weight[in_idx] = p

    return weight

def fit_superquadric_tsdf(sdf, x_init, truncation, points, roi_idx, bounding_points, para):
    # 初始化有效性向量
    valid = np.zeros(6)
    print('bounding_points:', bounding_points.shape)
    # 定位上下界
    t_lb = bounding_points[:, 0]
    t_ub = bounding_points[:, 7]

    # 设置上下界
    lb = np.array([0.0, 0.0, truncation, truncation, truncation, -2 * np.pi, -2 * np.pi, -2 * np.pi] + t_lb.tolist())
    ub = np.array([2, 2, 1, 1, 1, 2 * np.pi, 2 * np.pi, 2 * np.pi] + t_ub.tolist())

    # 初始化
    x = x_init.copy()
    cost = 0
    switched = 0
    nan_idx = ~np.isnan(sdf)
    sigma2 = np.exp(truncation) ** 2

    for iter in range(para['max_iter']):
        Rot = eul2rotm(x[5:8])
        check_points = np.array([
            x[8:11] - Rot[:, 0] * x[2],
            x[8:11] + Rot[:, 0] * x[2],
            x[8:11] - Rot[:, 1] * x[3],
            x[8:11] + Rot[:, 1] * x[3],
            x[8:11] - Rot[:, 2] * x[4],
            x[8:11] + Rot[:, 2] * x[4]
        ])
        valid[:3] = np.min(check_points, axis=0) >= t_lb - truncation
        valid[3:6] = np.max(check_points, axis=0) <= t_ub + truncation

        if not np.all(valid):
            break

        sdf_current = sdf_superquadric(x, points, 0)
        active_idx = (sdf_current < para['active_multiplier'] * truncation) & \
                     (sdf_current > -para['active_multiplier'] * truncation) & \
                     nan_idx

        points_active = points[:, active_idx]
        sdf_active = sdf[active_idx]

        weight = inlier_weight(sdf_active, active_idx, sdf_current, sigma2, para['w'], truncation)

        Rot = eul2rotm(x[5:8])
        bP_body = Rot.T @ (bounding_points - x[8:11][:, np.newaxis])
        scale_limit = np.mean(np.abs(bP_body), axis=1)
        ub[2:5] = scale_limit

        def cost_func(params):
            return difference_sqsdf(params, sdf_active, points_active, truncation, weight)

        result = least_squares(cost_func, x, bounds=(lb, ub))
        x_n = result.x
        cost_n = result.cost

        sigma2_n = cost_n / np.sum(weight)
        cost_n /= len(sdf_active)

        relative_cost = abs(cost - cost_n) / cost_n

        if (cost_n < para['tolerance'] and iter > 1) or \
           (relative_cost < para['relative_tolerance'] and switched >= para['max_switch'] and iter > para['iter_min']):
            x = x_n
            break
        # 案例1 - 轴不匹配相似性
        axis_0 = eul2rotm(x[5:8])
        axis_1 = np.roll(axis_0, 2, axis=1)
        axis_2 = np.roll(axis_0, 1, axis=1)
        eul_1 = rotm2eul(axis_1)
        eul_2 = rotm2eul(axis_2)
        x_axis = [[x[1], x[0], x[3], x[4], x[2]] + eul_1.tolist() + x[8:11].tolist(),
                    [x[1], x[0], x[4], x[2], x[3]] + eul_2.tolist() + x[8:11].tolist()]

        # 案例2 - 对偶相似性和组合
        scale_ratio = np.roll(x[2:5], 2) / x[2:5]
        scale_idx = np.where((scale_ratio > 0.8) & (scale_ratio < 1.2))[0]
        x_rot = []
        
        rot_idx = 0

        if 1 in scale_idx:
            eul_rot = rotm2eul(np.dot(axis_0, rotz(45)))
            if x[1] <= 1:
                new_x = [x[0], 2 - x[1]] + [((1 - np.sqrt(2)) * x[1] + np.sqrt(2)) * min(x[2], x[3])] * 2 + [x[4]] + eul_rot.tolist() + x[8:11].tolist()
            else:
                new_x = [x[0], 2 - x[1]] + [((np.sqrt(2) / 2 - 1) * x[1] + 2 - np.sqrt(2) / 2) * min(x[2], x[3])] * 2 + [x[4]] + eul_rot.tolist() + x[8:11].tolist()
            x_rot.append(new_x)
            rot_idx += 1

        if 2 in scale_idx:
            eul_rot = rotm2eul(np.dot(axis_1, rotz(45)))
            if x[0] <= 1:
                new_x = [x[1], 2 - x[0]] + [((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[3], x[4])] * 2 + [x[2]] + eul_rot.tolist() + x[8:11].tolist()
            else:
                new_x = [x[1], 2 - x[0]] + [((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[3], x[4])] * 2 + [x[2]] + eul_rot.tolist() + x[8:11].tolist()
            x_rot.append(new_x)
            rot_idx += 1

        if 3 in scale_idx:
            eul_rot = rotm2eul(np.dot(axis_2, rotz(45)))
            if x[0] <= 1:
                new_x = [x[1], 2 - x[0]] + [((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[4], x[2])] * 2 + [x[3]] + eul_rot.tolist() + x[8:11].tolist()
            else:
                new_x = [x[1], 2 - x[0]] + [((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[4], x[2])] * 2 + [x[3]] + eul_rot.tolist() + x[8:11].tolist()
            x_rot.append(new_x)
            # 生成候选配置列表及其成本
        # 生成候选配置列表及其成本
        x_candidate = np.vstack([x_axis, x_rot])
        cost_candidate = cost_switched(x_candidate, sdf_active, points_active, truncation, weight)

        # 筛选有效成本
        valid_indices = np.logical_and(~np.isnan(cost_candidate), ~np.isinf(cost_candidate))
        cost_candidate = cost_candidate[valid_indices]
        x_candidate = x_candidate[valid_indices, :]

        # 对成本进行排序
        sorted_indices = np.argsort(cost_candidate)
        switch_success = False

        for i_candidate in sorted_indices:
            # 更新上界
            Rot = eul2rotm(x_candidate[i_candidate, 5:8])
            bP_body = Rot.T @ (bounding_points - x_candidate[i_candidate, 8:11][:, np.newaxis])
            scale_limit = np.mean(np.abs(bP_body), axis=1)
            ub[2:5] = scale_limit

            # 最小二乘优化
            result = least_squares(cost_func, x_candidate[i_candidate, :], bounds=(lb, ub))
            x_switch = result.x
            cost_switch = result.cost

            if cost_switch / len(sdf_active) < min(cost_n, cost):
                x = x_switch
                cost = cost_switch / len(sdf_active)
                sigma2 = cost_switch / np.sum(weight)
                switch_success = True
                break

        if not switch_success:
            cost = cost_n
            x = x_n
            sigma2 = sigma2_n

        switched += 1
    # 根据最终超四面体参数计算 SDF
    sdf_occ = sdf_superquadric(x, points, 0)

    # 确定占据的体素
    occ = sdf_occ < para['nan_range']
    occ_idx = roi_idx[occ]

    # 确定内部体素
    occ_in = sdf_occ <= 0

    # 统计相关数量指标
    num_idx = np.zeros(3)
    num_idx[0] = np.sum(np.logical_or(sdf[occ_in] <= 0, np.isnan(sdf[occ_in])))
    num_idx[1] = np.sum(sdf[occ_in] > 0)
    num_idx[2] = np.sum(sdf[occ_in] <= 0)

    # 最终检查超四面体大小有效性
    Rot = eul2rotm(x[5:8])
    check_points = np.array([
        x[8:11] - Rot[:, 0] * x[2],
        x[8:11] + Rot[:, 0] * x[2],
        x[8:11] - Rot[:, 1] * x[3],
        x[8:11] + Rot[:, 1] * x[3],
        x[8:11] - Rot[:, 2] * x[4],
        x[8:11] + Rot[:, 2] * x[4]
    ])

    valid[:3] = np.min(check_points, axis=0) >= t_lb - truncation
    valid[3:6] = np.max(check_points, axis=0) <= t_ub + truncation

    return x, occ_idx, valid, num_idx
        


def mps(sdf, voxel_grid, **kargs):
    # 这里假设parameters是一个包含算法超参数的字典
    parameters = parse_input_args(voxel_grid, **kargs)
    # 初始化一些变量
    num_division = 1
    x = []
    dratio = 3/5
    conn_ratio = [1, dratio, dratio**2, dratio**3, dratio**4,
                  dratio**5, dratio**6, dratio**7, dratio**8]

    # 初始化连接指针和区域数量
    conn_pointer = 1
    num_region = 1

    while num_division < parameters['max_division']:
        # 1-连接性行进
        if conn_pointer != 1 and num_region != 0:
            conn_pointer = 1

        # 设置连接阈值
        conn_threshold = conn_ratio[conn_pointer - 1] * np.min(sdf)
        if conn_threshold > -voxel_grid['truncation'] * 0.3:
            break

        # 将sdf重排成三维数组，用于连接性检查
        sdf3d_region = sdf.reshape((voxel_grid['size'][0], voxel_grid['size'][1], voxel_grid['size'][2]))

        # 连接性检查和初步划分
        labeled_region = label(sdf3d_region <= conn_threshold)
        regions = regionprops(labeled_region)

        # 计算感兴趣区域的大小
        regions = [region for region in regions if region.area >= parameters['min_area']]
        num_region = len(regions)

        if parameters['verbose']:
            print(f"Number of regions: {num_region}")

        if num_region == 0:
            if conn_pointer < len(conn_ratio):
                conn_pointer += 1
            else:
                break
        # 2-概率基元行进
        # 针对每个区域寻找最佳的超四面体表示
        num_region = len(regions)
        x_temp = np.zeros((num_region, 11))
        del_idx = np.zeros(num_region, dtype=int)
        occ_idx_in = [None] * num_region
        num_idx = np.zeros((num_region, 3))
        # 创建一个字典来存储每个区域的像素索引
        region_pixel_idx_lists = {region.label: np.where(labeled_region == region.label) for region in regions}
        for i in range(num_region):
            # 获取并调整边界框
            bbox = regions[i].bbox
            print('bbox:', len(bbox))
            bbox = np.ceil(bbox).astype(int)
            bbox[3:] = np.minimum(bbox[:3] + bbox[3:] + parameters['padding_size'], 
                                [voxel_grid['size'][1], voxel_grid['size'][0], voxel_grid['size'][2]])
            bbox[:3] = np.maximum(bbox[:3] - parameters['padding_size'], 1)
            # 计算激活的体素索引
            idx_x, idx_y, idx_z = np.mgrid[bbox[1]:bbox[4], bbox[0]:bbox[3], bbox[2]:bbox[5]]
            indices = np.vstack([idx_x.ravel(), idx_y.ravel(), idx_z.ravel()]).T
            roi_idx = idx3d_flatten(indices, voxel_grid)
            regions[i].idx = roi_idx

            # 计算边界点坐标
            bounding_points = idx2Coordinate(np.array([
                [bbox[1], bbox[1], bbox[4], bbox[4], bbox[1], bbox[1], bbox[4], bbox[4]],
                [bbox[0], bbox[0], bbox[0], bbox[0], bbox[3], bbox[3], bbox[3], bbox[3]],
                [bbox[2], bbox[5], bbox[2], bbox[5], bbox[2], bbox[5], bbox[2], bbox[5]]
            ]), voxel_grid)
            regions[i].bounding_points = bounding_points
            # 确定中心点并向下取整
            centroid = np.maximum(np.floor(regions[i].centroid), 1).astype(int)
            # 将中心点坐标转换为一维索引
            centroid_flatten = idx3d_flatten(np.array([[centroid[1], centroid[0], centroid[2]]]), voxel_grid)
            # 获取区域的三维坐标
            coords = regions[i].coords
            # 将三维坐标转换为一维线性索引
            # pixel_idx_list = np.ravel_multi_index(coords, labeled_region.shape)
            pixel_idx_list = idx3d_flatten(coords, voxel_grid)
            print('pixel_idx_list:', pixel_idx_list.shape, pixel_idx_list[0])
            # 检查中心点一维索引是否在像素索引列表中
            if centroid_flatten[0] in pixel_idx_list:
                centroid = centroid
            else:
                pixel_coords = coords
                # 使用 KDTree 进行最近邻搜索
                kdtree = cKDTree(pixel_coords)
                print("centroid:", centroid.shape)
                # query_point = voxel_grid['points'][centroid]
                query_point = centroid
                # print("query_point:", query_point.shape)
                _, nearest_point_idx = kdtree.query(query_point)
                # print("nearest_point_idx:", nearest_point_idx)
                # print("new centroid:", pixel_coords[nearest_point_idx])

                # 更新区域中心点坐标
                centroid = pixel_coords[nearest_point_idx]

            x_temp = np.zeros((num_region, 11))  # 超四面体参数的临时存储
            num_idx_temp = np.zeros((num_region, 3))  # 数量指标的临时存储
            valid_temp = np.zeros((num_region, 6))  # 有效性的临时存储

            valid = np.zeros(6, dtype=bool)
            while not np.all(valid):
                # 获取并调整边界框
                bbox = regions[i].bbox
                bbox = np.ceil(bbox).astype(int)
                bbox[3:] = np.minimum(bbox[:3] + bbox[3:] + parameters['padding_size'], 
                                    [voxel_grid['size'][1], voxel_grid['size'][0], voxel_grid['size'][2]])
                bbox[:3] = np.maximum(bbox[:3] - parameters['padding_size'], 1)

                # 初始化超四面体尺度
                bbox = np.array(bbox)  # 确保 bbox 是 numpy 数组
                scale_init = parameters['scaleInitRatio'] * (bbox[3:] - bbox[:3]) * voxel_grid['interval']


                # 初始化超四面体参数
                centroid = np.maximum(np.floor(regions[i]['centroid']), 1).astype(int)
                x_init = [1, 1] + scale_init[[1, 0, 2]].tolist() + [0, 0, 0] + centroid.tolist()

                # 为每个区域找到最佳的超四面体表示
                x_temp[i, :], occ_idx, valid, num_idx_temp[i, :] = fit_superquadric_tsdf(
                    sdf[regions[i]['idx']],
                    x_init,
                    voxel_grid['truncation'],
                    voxel_grid['points'][regions[i]['idx'], :],
                    regions[i]['idx'],
                    bounding_points,
                    parameters
                )
        # 更新划分深度
        num_division += 1

    return x

def connectivity_check(sdf, grid, threshold):
    """
    检查和标记连接区域。
    返回标记的区域和每个区域的属性（如中心点、面积等）。
    """
    # 这里可以使用skimage的label和regionprops函数
    # ...

def probabilistic_primitive_marching(sdf, grid, parameters):
    """
    在每个划分的区域中寻找最优的超四面体表示。
    这可能涉及复杂的几何计算和优化算法。
    """
    # ...

def optimize_superquadric(sdf, initial_guess, grid, parameters):
    """
    使用非线性最小二乘法或其他优化方法来调整超四面体参数，以便最佳地拟合给定的SDF。
    """
    # ...