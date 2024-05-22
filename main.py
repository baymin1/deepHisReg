# 本代码进行CZ数据测试

### Python 导入 ###
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import natsort
import time

### 额外的导入 ###
import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import pandas as pd
import cv2

### 内部函数 ###
import preprocessing as pre
import postprocessing as pst
import regularizers as rg
import utils as u
import utils_tc as utc

import instance_optimization as io

import initial_registration as ir

### 导入损失函数库 ###
import cost_functions as cf

### 导入配准文件包 ###
import configs


### 方法 ###

def affine_iterative_nonrigid(source, target, **config):
    """
        仿射变换+非刚性配准
        :param source: 固定图像
        :param target: 配准图像
    """
    tc.cuda.empty_cache()

    ### 初始对齐 + 仿射变换 ###
    # 计算仿射配准位移场
    displacement_field_ini = affine_iterative(source, target, **config)
    tc.cuda.empty_cache()

    ### 非刚性配准 ###
    # 获得非刚性配准参数
    nonrigid_params = config['nonrigid_params']
    # 计算非刚性配准位移场值
    displacement_field_nr = instance_optimization_nonrigid_registration(source, target, displacement_field_ini,
                                                                        nonrigid_params)
    tc.cuda.empty_cache()

    return displacement_field_nr


def affine_iterative(source, target, **config):
    """
        仿射变换迭代优化函数
        :param source: 固定图像
        :param target: 配准图像
    """
    tc.cuda.empty_cache()
    ### 初始校准 ###
    affine_params = config['affine_params']
    b_t = time.time()
    # BEWARE - REVERSED
    transform = ir.rotated_based_combination(target, source, affine_params)
    e_t = time.time()
    print(f"Elapsed time: {e_t - b_t}")
    final_transform = tc.eye(3)
    final_transform[0:2, 0:3] = transform
    final_transform = tc.linalg.inv(final_transform)
    transform = final_transform[0:2, 0:3].unsqueeze(0).to("cuda:0")

    iterative_affine_params = config['iterative_affine_params']
    # 实例化仿射配准
    iterative_transform = instance_optimization_affine_registration(source, target, transform, iterative_affine_params)
    transforms = [transform, iterative_transform]

    best_cost = np.inf
    best_transform = transform
    resampled_source, resampled_target = u.initial_resampling(source, target, 256)

    # SIFT函数
    sift = cv2.SIFT_create(256)  # 256
    for tr in transforms:
        displacement_field = utc.tc_transform_to_tc_df(tr, resampled_source.size())
        warped_source = u.warp_image(resampled_source, displacement_field)
        keypoints, source_descriptors = sift.detectAndCompute(
            (warped_source[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), None)
        _, target_descriptors = sift.compute(
            (resampled_target[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), keypoints)
        try:
            costs = np.mean((source_descriptors - target_descriptors) ** 2, axis=1)
            lowest_costs = np.sort(costs)[0:8]
            current_cost = np.mean(lowest_costs)
        except:
            current_cost = np.inf
        if current_cost < best_cost:
            best_cost = current_cost
            best_transform = tr
    displacement_field_ini = utc.tc_transform_to_tc_df(best_transform, source.size())
    return displacement_field_ini


def instance_optimization_affine_registration(source, target, initial_transform, params):
    """
        实例优化仿射变换配准
        :param source: 固定图像
        :param target: 配准图像
        :param params: 配准参数
    """
    device = params['device']
    echo = params['echo']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    resolution = params['registration_size']
    num_levels = params['num_levels']
    used_levels = params['used_levels']
    iterations = params['iterations']
    learning_rate = params['learning_rate']

    if type(cost_function) == str:
        cost_function = cf.get_function(cost_function)

    ### 初始重采样 ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")
    initial_cost_function = cost_function(resampled_source, resampled_target, device=device, **cost_function_params)
    if echo:
        print(f"Initial objective function: {initial_cost_function.item()}")

    ### 仿射配准 ###
    transform = io.affine_registration(resampled_source, resampled_target, num_levels, used_levels, iterations,
                                       learning_rate, cost_function, cost_function_params, device=device,
                                       initial_transform=initial_transform, echo=echo, return_best=True)
    if echo:
        print(f"Final transform: {transform}")
    return transform


def instance_optimization_nonrigid_registration(source, target, initial_displacement_field, params):
    """
        实例优化非刚性配准
        :param source: 固定图像
        :param target: 配准图像
        :param initial_displacement_field: 初始位移常
        :param params: 配准参数
        :return: displacement_field: 实例计算位移值
    """
    device = params['device']
    echo = params['echo']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    regularization_function = params['regularization_function']
    regularization_function_params = params['regularization_function_params']
    resolution = params['registration_size']

    num_levels = params['num_levels']
    used_levels = params['used_levels']
    iterations = params['iterations']
    learning_rates = params['learning_rates']
    alphas = params['alphas']

    if type(cost_function) == str:
        cost_function = cf.get_function(cost_function)
    if type(regularization_function) == str:
        regularization_function = rg.get_function(regularization_function)

    ### 初始重采样 ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    initial_cost_function = cost_function(resampled_source, resampled_target, device=device, **cost_function_params)
    if echo:
        print(f"Initial objective function: {initial_cost_function.item()}")

    ### 非刚性配准 ###
    if initial_displacement_field is None:
        initial_df = None
        displacement_field = io.nonrigid_registration(resampled_source, resampled_target, num_levels, used_levels,
                                                      iterations, learning_rates, alphas, cost_function,
                                                      regularization_function, cost_function_params,
                                                      regularization_function_params,
                                                      initial_displacement_field=initial_df, device=device, echo=echo)
    else:
        initial_df = utc.resample_displacement_field_to_size(initial_displacement_field,
                                                             (resampled_source.size(2), resampled_source.size(3)))
        with tc.set_grad_enabled(False):
            warped_source = utc.warp_tensor(resampled_source, initial_df, mode='bicubic')
        displacement_field = io.nonrigid_registration(warped_source, resampled_target, num_levels, used_levels,
                                                      iterations, learning_rates, alphas, cost_function,
                                                      regularization_function, cost_function_params,
                                                      regularization_function_params,
                                                      initial_displacement_field=None, device=device, echo=echo)

        displacement_field = utc.compose_displacement_fields(initial_df, displacement_field)

    if echo:
        print(f"Registered displacement field size: {displacement_field.size()}")
    displacement_field = utc.resample_displacement_field_to_size(displacement_field, (source.size(2), source.size(3)),
                                                                 mode='bicubic')
    if echo:
        print(f"Output displacement field size: {displacement_field.size()}")
    return displacement_field


### 主函数 ###
if __name__ == "__main__":
    ## 加载非刚性配准参数配准
    config = configs.affine_nonrigid_config()

    ## 参数配准传递
    # 固定文件地址——HE
    input_HE_datapath = config['input_HE_datapath']
    # 配准文件地址——IHC
    input_IHC_datapath = config['input_IHC_datapath']
    # 输出地址——结果
    output_path = config['output_path']
    # 读取图像级别（分辨率金字塔）
    level = config['level']
    # 配准方法传递
    registration_method = affine_iterative_nonrigid
    # 配准参数
    registration_params = config['registration_params']
    # 预处理参数
    preprocessing_params = config['preprocessing_params']
    # 结果输出表
    # output_csv_path = output_path + 'result.csv'

    # HE是目标图像（target），IHC是需要配准的图像（source）
    source_paths = natsort.natsorted(os.listdir(input_IHC_datapath))
    target_paths = natsort.natsorted(os.listdir(input_HE_datapath))

    # 显卡地址
    device = "cuda:0"

    # 配准过程（循环处理，一次一图像）
    for i in range(0, len(source_paths)):
        tc.cuda.empty_cache()
        ### 加载图像
        source_path = source_paths[i]
        target_path = target_paths[i]
        print(f"Source path: {source_path}")
        print(f"Target path: {target_path}")
        source, source_slide = u.load_slide(os.path.join(input_IHC_datapath, source_path), level, load_slide=True)
        target, target_slide = u.load_slide(os.path.join(input_HE_datapath, target_path), level, load_slide=True)

        print(f"Source dimensions: {source_slide.level_dimensions}")
        print(f"Target dimensions: {target_slide.level_dimensions}")

        source = u.image_to_tensor(source, device)
        target = u.image_to_tensor(target, device)

        ### 预处理
        print(f"Original source shape: {source.shape}")
        print(f"Original target shape: {target.shape}")
        preprocessing_function = pre.get_function(preprocessing_params['preprocessing_function'])

        # 无图像坐标输入
        ori_source, ori_target, pre_source, pre_target, postprocessing_params = preprocessing_function(source, target,
                                                                                                       preprocessing_params)
        print(f"Preprocessed source shape: {pre_source.shape}")
        print(f"Preprocessed target shape: {pre_target.shape}")

        ### 进行配准
        # 得到配准的位移数据（预处理数据）
        displacement_field = registration_method(pre_source, pre_target, **registration_params)

        ### 保存处理数据
        # 根据位移值进行扭曲校对(原始图像)
        warped_source = u.warp_image(ori_source, displacement_field)
        # 得到数据图像的名字编号
        case_id = target_path.split("BC2_1_HE_1_")[1].split(".")[0]
        output_case_path = output_path + str(case_id)
        if not os.path.isdir(output_case_path):
            os.makedirs(output_case_path)
        u.save_image(ori_source[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy(), output_case_path + '/source.jpg',
                     renormalize=True)
        del ori_source
        u.save_image(ori_target[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy(), output_case_path + '/target.jpg',
                     renormalize=True)
        del ori_target
        u.save_image(warped_source[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy(),
                     output_case_path + '/warped_source.jpg', renormalize=True)
        del warped_source
