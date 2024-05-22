## 配准所需的参数文件
import cost_functions as cf

# 基础文件地址
openslide_path = "C:/Program Files/openslide/bin"  # TODO
code_path = "E:/work/HE_ICH_imageRE/python/whd_DPReg/"  # TODO
models_path = code_path + "/models"
superpoint_model_path = models_path + "/superpoint_v1.pth"
superglue_model_path = models_path + "/superglue_outdoor.pth"

def affine_nonrigid_config():
    config = dict()
    ### 初始配置 ###
    config['input_HE_datapath'] = "G:/datasets/CQCH/CQCH_HE_HER2_svs"  # HE图像地址
    config['input_IHC_datapath'] = "G:/datasets/CQCH/CQCH_IHC_HER2_svs"  # IHC图像地址
    config['output_path'] = "G:/datasets/CQCH/Affine_Results/"
    config['level'] = 1
    config['registration_method'] = "affine_iterative_nonrigid"
    config['registration_params'] = dict()

    ### 预处理 ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = False
    preprocessing_params['normalization'] = True
    preprocessing_params['pad_to_same_size'] = True
    preprocessing_params['late_resample'] = False
    preprocessing_params['late_resample_ratio'] = 1.0
    preprocessing_params['pad_value'] = 1.0
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True

    ### 仿射参数 ###
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['registration_sizes'] = [100, 150, 200, 250, 300, 400, 500, 600]
    affine_params['transform_type'] = 'rigid'
    affine_params['keypoint_threshold'] = 0.005
    affine_params['match_threshold'] = 0.3
    affine_params['sinkhorn_iterations'] = 50
    affine_params['show'] = False
    affine_params['angle_step'] = 30
    affine_params['num_features'] = 256
    #affine_params['sparse_size'] = 45
    affine_params['keypoint_size'] = 32
    affine_params['device'] = "cuda:0"  # 显卡运行

    ### 配准——迭代仿射参数 ###
    config['registration_params']['iterative_affine_params'] = dict()
    config['registration_params']['iterative_affine_params']['device'] = "cuda:0"
    config['registration_params']['iterative_affine_params']['echo'] = True
    config['registration_params']['iterative_affine_params']['cost_function'] = cf.get_function("ncc_local_tc")
    config['registration_params']['iterative_affine_params']['cost_function_params'] = {'win_size': 7}
    config['registration_params']['iterative_affine_params']['registration_size'] = 1024
    config['registration_params']['iterative_affine_params']['num_levels'] = 4
    config['registration_params']['iterative_affine_params']['used_levels'] = 4
    config['registration_params']['iterative_affine_params']['iterations'] = [200, 200, 200, 200]
    config['registration_params']['iterative_affine_params']['learning_rate'] = 0.02

    ### 非刚性配准参数 ###
    nonrigid_params = dict()
    nonrigid_params['device'] = "cuda:0"
    nonrigid_params['echo'] = True
    nonrigid_params['cost_function'] = cf.get_function("ncc_local_tc")
    nonrigid_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_params['regularization_function'] = "diffusion_relative_tc"
    nonrigid_params['regularization_function_params'] = dict()
    nonrigid_params['registration_size'] = 2048
    nonrigid_params['num_levels'] = 7
    nonrigid_params['used_levels'] = 7
    nonrigid_params['iterations'] = 7*[400]
    nonrigid_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015]
    nonrigid_params['alphas'] = [1.2, 1.2, 1.2, 1.2, 1.2, 1.0, 0.6]

    config['preprocessing_params'] = preprocessing_params
    config['registration_params']['affine_params'] = affine_params
    config['registration_params']['nonrigid_params'] = nonrigid_params
    return config