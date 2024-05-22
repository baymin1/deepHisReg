import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch as tc
import torch.nn.functional as F
import torch.optim as optim

import utils_tc as utc

from typing import Callable

def affine_registration(source: tc.Tensor, target: tc.Tensor, num_levels: int, used_levels: int, num_iters: list,
    learning_rate: float, cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float], cost_function_params: dict={}, 
    device: str="cpu", initial_transform=None, echo: bool=False, return_best: bool=False):
    """
    使用实例优化技术(一个原型)执行仿射变换配准。
    对于实时DirectX实现，必须用解析梯度计算代替autograd，并使用无矩阵运算和拟牛顿算法来实现优化。

    Parameters
    ----------
    source : tc.Tensor
        The source tensor (1x1 x size)
    target : tc.Tensor
        The target tensor (1x1 x size)
    num_levels : int
        The number of resolution levels
    used_levels : int
        The number of actually used resolution levels (must be lower (or equal) than the num_levels)
    num_iters : int
        The nubmer of iterations per resolution
    learning_rate : float
        The learning rate for the optimizer
    cost_function : Callable[tc.Tensor, tc.Tensor, dict] -> float
        The cost function being optimized
    cost_function_params : dict (default: {})
        The optional cost function parameters
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")

    Returns
    ----------
    transformation : tc.Tensor
        The affine transformation matrix (1 x transformation_size (2x3 or 3x4))
    """
    ndim = len(source.size()) - 2
    if initial_transform is None:
        if ndim == 2:
            transformation = tc.zeros((1, 2, 3), dtype=source.dtype, device=device)
            transformation[0, 0, 0] = 1.0
            transformation[0, 1, 1] = 1.0
            transformation = transformation.detach().clone()
            transformation.requires_grad = True
        elif ndim == 3:
            transformation = tc.zeros((1, 3, 4), dtype=source.dtype, device=device)
            transformation[0, 0, 0] = 1.0
            transformation[0, 1, 1] = 1.0
            transformation[0, 2, 2] = 1.0
            transformation = transformation.detach().clone()
            transformation.requires_grad = True
        else:
            raise ValueError("Unsupported number of dimensions.")
    else:
        transformation = initial_transform.detach().clone()
        transformation.requires_grad = True

    optimizer = optim.Adam([transformation], learning_rate)
    source_pyramid = utc.create_pyramid(source, num_levels=num_levels)
    target_pyramid = utc.create_pyramid(target, num_levels=num_levels)
    if return_best:
        best_transformation = transformation.clone()
        best_cost = 1000.0
    ### 金字塔层次 仿射优化 ###
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        ### 迭代优化 ###
        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                sampling_grid = F.affine_grid(transformation, size=current_source.size(), align_corners=False)
                warped_source = utc.transform_tensor(current_source, sampling_grid, device=device)
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)    
                cost.backward()
                optimizer.step()
                current_cost = cost.item()
            optimizer.zero_grad()
            if echo:
                print(f"Iter: {i}, Current cost: {current_cost}")
            if return_best:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_transformation = transformation.clone()
    if return_best:
        return best_transformation
    else:
        return transformation

def nonrigid_registration(source: tc.Tensor, target: tc.Tensor, num_levels: int, used_levels: int, num_iters: list, learning_rates: list, alphas: list,
                          cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float], regularization_function: Callable[[tc.Tensor, dict], float],
                          cost_function_params: dict={}, regularization_function_params: dict={}, penalty_function: Callable=None, penalty_function_params: dict={},
                          initial_displacement_field: tc.Tensor=None, device: str="cpu", echo: bool=False):
    """
    使用实例优化技术(一个原型)执行非刚性配准。
    对于实时DirectX实现，必须用解析梯度计算代替autograd，并使用无矩阵运算和拟牛顿算法来实现优化。

    Parameters
    ----------
    source : tc.Tensor
        The source 张量 (1x1 x size)
    target : tc.Tensor
        The target 张量 (1x1 x size)
    num_levels : int
        分辨率级别的数量
    used_levels : int
        实际使用的分辨率级别的数量(必须低于(或等于)num_levels)
    num_iters : int0
        每个分辨率的迭代次数
    learning_rate : float
        优化器的学习率
    alpha : float
        正则化权值
    cost_function : Callable[tc.Tensor, tc.Tensor, dict] -> float
        被优化的损失函数
    regularization_function : Callable[tc.Tensor,  dict] -> float
        正则化函数
    cost_function_params : dict (default: {})
        可选的损失函数参数
    regularization_function_params : dict (default: {})
        可选的正则化函数参数
    penalty_function : Callable
        可选惩罚函数(必须是可微的)
    penalty_function_params : dict(default: {})
        可选的惩罚函数参数
    initial_displacement_field : tc.Tensor (default None)
        可选的惩罚函数参数
    device : str
        用于CUDA的设备(如:cpu或cuda:0)

    Returns
    ----------
    displacement_field : tc.Tensor
        计算的位移场(将使用来自utils_tc的warp_tensor)
    """
    source_pyramid = utc.create_pyramid(source, num_levels=num_levels)
    target_pyramid = utc.create_pyramid(target, num_levels=num_levels)
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        if j == 0:
            if initial_displacement_field is None:
                displacement_field = utc.create_identity_displacement_field(current_source).detach().clone()
                displacement_field.requires_grad = True
            else:
                displacement_field = utc.resample_displacement_field_to_size(initial_displacement_field, (current_source.size(2), current_source.size(3))).detach().clone()
                displacement_field.requires_grad = True
            optimizer = optim.Adam([displacement_field], learning_rates[j])
        else:
            displacement_field = utc.resample_displacement_field_to_size(displacement_field, (current_source.size(2), current_source.size(3))).detach().clone()
            displacement_field.requires_grad = True
            optimizer = optim.Adam([displacement_field], learning_rates[j])

        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                warped_source = utc.warp_tensor(current_source, displacement_field, device=device)
                if i == 0:
                    if echo:
                        print(f"Initial cost: {cost_function(current_source, current_target, device=device, **cost_function_params)}")
                        print(f"First warp cost: {cost_function(warped_source, current_target, device=device, **cost_function_params)}")
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)   
                reg = regularization_function(displacement_field, device=device, **regularization_function_params)
                loss = cost + alphas[j]*reg
                if penalty_function is not None:
                    loss = loss + penalty_function(penalty_function_params) 
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            if echo:
                print("Iter: ", i, "Current cost: ", cost.item(), "Current reg: ", reg.item(), "Current loss: ", loss.item())
    if used_levels != num_levels:
        displacement_field = utc.resample_displacement_field_to_size(displacement_field, (source.size(2), source.size(3)))
    return displacement_field
