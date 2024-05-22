import sys
current_file = sys.modules[__name__]

import numpy as np
import torch as tc
import cv2

import utils as u

### 预处理 ###

def basic_preprocessing(source, target, params):
    """
    TODO - documentation
    """
    postprocessing_params = dict()
    postprocessing_params['original_size'] = source.shape[0:2] if isinstance(source, np.ndarray) else (source.size(2), source.size(3))

    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resolution = params['initial_resolution']
        source_y_size, source_x_size, target_y_size, target_x_size = u.get_combined_size(source, target)
        initial_resample_ratio = u.calculate_resampling_ratio((source_x_size, target_x_size), (source_y_size, target_y_size), initial_resolution)
        initial_smoothing = max(initial_resample_ratio - 1, 0.1)
        source = u.resample(u.gaussian_smoothing(source, initial_smoothing), initial_resample_ratio, cval=params['pad_value'])
        target = u.resample(u.gaussian_smoothing(target, initial_smoothing), initial_resample_ratio, cval=params['pad_value'])
        postprocessing_params['initial_resample_ratio'] = initial_resample_ratio

    postprocessing_params['initial_resampling'] = initial_resampling 

    normalization = params['normalization']
    if normalization:
        source, target = u.normalize(source), u.normalize(target)

    pad_to_same_size = params['pad_to_same_size']
    if pad_to_same_size:
        source, target, padding_params = u.pad_to_same_size(source, target, params['pad_value'])

        postprocessing_params['padding_params'] = padding_params
    postprocessing_params['pad_to_same_size'] = pad_to_same_size

    late_resample = params['late_resample']
    if late_resample:
        late_resample_ratio = params['late_resample_ratio']
        late_smoothing = max(late_resample_ratio - 1, 0.1)
        source = u.resample(u.gaussian_smoothing(source, late_smoothing), late_resample_ratio, cval=params['pad_value'])
        target = u.resample(u.gaussian_smoothing(target, late_smoothing), late_resample_ratio, cval=params['pad_value'])

        postprocessing_params['late_resample_ratio'] = late_resample_ratio
    postprocessing_params['late_resample'] = late_resample

    # 灰度转换
    convert_to_gray = params['convert_to_gray']
    if convert_to_gray:
        pre_source = 1 - u.convert_to_gray(source)
        pre_target = 1 - u.convert_to_gray(target)

        # 图像色彩均值化
        clahe = params['clahe']
        if clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            src = clahe.apply((pre_source[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
            trg = clahe.apply((pre_target[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
            pre_source = tc.from_numpy((src.astype(np.float32) / 255)).to(pre_source.device).unsqueeze(0).unsqueeze(0)
            pre_target = tc.from_numpy((trg.astype(np.float32) / 255)).to(pre_target.device).unsqueeze(0).unsqueeze(0)

    return source, target, pre_source, pre_target, postprocessing_params


def target_landmarks_preprocessing(target_landmarks, params):
    """
    TODO - documentation
    """
    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resample_ratio = params['initial_resample_ratio']
        target_landmarks = target_landmarks / initial_resample_ratio

    pad_to_same_size = params['pad_to_same_size']
    if pad_to_same_size:
        padding_params = params['padding_params']
        target_landmarks = u.pad_landmarks(target_landmarks, padding_params['pad_2'])

    late_resample = params['late_resample']
    if late_resample:
        late_resample_ratio = params['late_resample_ratio']
        target_landmarks = target_landmarks / late_resample_ratio
    return target_landmarks


### Utilities ###

def get_function(function_name):
    return getattr(current_file, function_name)