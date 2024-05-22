import sys
import os
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch as tc
import cv2

import utils as u
import utils_tc as utc

import sift_ransac as sr
import superpoint_superglue as sg
import superpoint_ransac as spr

### 注意 ###
# 假设initial_registration.py中的所有算法都以PyTorch格式返回2x3变换矩阵
############

### 算法 ###

def identity_initial_registration(source, target, params):
    return u.create_identity_displacement_field(source)

def rotated_based_combination(source, target, params):
    step = params['angle_step']
    device = params['device']
    resolution = params['registration_size']
    echo = params['echo']
    num_features = params['num_features']
    keypoint_size = params['keypoint_size']
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)

    # SIFT函数
    sift = cv2.SIFT_create(num_features)  # 256
    keypoints, target_descriptors = sift.detectAndCompute(
        (resampled_target[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), None)
    if echo:
        print(f"Number of evaluation keypoints: {len(keypoints)}")

    best_transform = tc.eye(3, device=source.device)[0:2, :].unsqueeze(0)
    _, source_descriptors = sift.compute((resampled_source[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), keypoints)
    costs = np.mean((source_descriptors - target_descriptors) ** 2, axis=1)
    lowest_costs = np.sort(costs)[0:keypoint_size]
    best_cost = np.mean(lowest_costs)

    ## 旋转变换
    for angle in range(-180, 180, step):
        _, _, y_size, x_size = source.shape
        x_origin = x_size // 2
        y_origin = y_size // 2
        # 生成刚性矩阵
        r_transform = utc.generate_rigid_matrix(angle, x_origin, y_origin, 0, 0)
        r_transform = utc.affine2theta(r_transform, (source.size(2), source.size(3))).to(device).unsqueeze(0)
        # 将变换张量转换为位移场张量
        current_displacement_field = utc.tc_transform_to_tc_df(r_transform, (1, 1, source.size(2), source.size(3)))
        # 用给定的位移场变换张量
        transformed_source = utc.warp_tensor(source, current_displacement_field)

        transforms = []
        registration_sizes = params['registration_sizes']
        for registration_size in registration_sizes:
            ex_params = {**params, **{'registration_size': registration_size}}

            current_transform = sr.sift_ransac(transformed_source, target, ex_params)
            transforms.append(current_transform)

            current_transform = sg.superpoint_superglue(transformed_source, target, ex_params)
            transforms.append(current_transform)

            current_transform = spr.superpoint_ransac(transformed_source, target, ex_params)
            transforms.append(current_transform)

        if echo:
            print(f"Initial cost: {best_cost}")
        for transform in transforms:
            # 组合变换矩阵
            transform = utc.compose_transforms(r_transform[0], transform).unsqueeze(0).to(device)

            displacement_field = utc.tc_transform_to_tc_df(transform, resampled_source.size())
            warped_source = u.warp_image(resampled_source, displacement_field)
            _, source_descriptors = sift.compute(
                (warped_source[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), keypoints)
            costs = np.mean((source_descriptors - target_descriptors) ** 2, axis=1)
            lowest_costs = np.sort(costs)[0:keypoint_size]
            current_cost = np.mean(lowest_costs)
            if echo:
                print(f"Current cost: {current_cost}")
            if current_cost < best_cost:
                best_cost = current_cost
                best_transform = transform
                if echo:
                    print(f"Current best: {best_cost}")
    if echo:
        print(f"Current best: {best_cost}")
        print(f"Final transform: {best_transform}")
    return best_transform

### 公用函数：返回属性名函数体 ###
def get_function(function_name):
    return getattr(current_file, function_name)