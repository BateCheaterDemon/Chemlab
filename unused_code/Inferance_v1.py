import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from isaacsim import SimulationApp
import cv2
from datetime import datetime
import json
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import omni.usd
from omegaconf import OmegaConf
import hydra
from collections import deque
# 导入自定义组件
from robots.franka import Franka
from utils.object_utils import ObjectUtils
from controllers.pick_controller import PickController
from controllers.grapper_manager import Gripper
from controllers.trajectory_controller import FrankaTrajectoryController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
# 导入Diffusion Policy相关组件
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
# 导入动作转换函数
from utils.action_utils import joint_positions_to_action

# 注册OmegaConf解析器
OmegaConf.register_new_resolver("eval", lambda x: eval(x))

# 加载配置文件
# cfg = load_camera_config("/home/ubuntu/Documents/IsaacLabSim/collected_data/20250228_113514")
# config_path = "/home/ubuntu/Documents/IsaacLabSim/data/outputs/2025.02.28/11.44.44_train_diffusion_unet_image_real_image/.hydra/config.yaml"
# model_path = "/home/ubuntu/Documents/IsaacLabSim/data/outputs/2025.02.28/11.44.44_train_diffusion_unet_image_real_image/checkpoints/latest.ckpt"

# 不采用视觉，纯使用关节数据训练的：
# cfg_camera = load_camera_config("/home/ubuntu/AIlab project/chemistry-lab-simulator/collected_data/20250306_173208rgb")
# config_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.06/20.46.14_train_diffusion_unet_image_real_image/.hydra/config.yaml"
# model_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.06/20.46.14_train_diffusion_unet_image_real_image/checkpoints/epoch=0120-train_loss=0.002.ckpt"

# 加了视觉出现问题的
# cfg_camera = load_camera_config("/home/ubuntu/AIlab project/chemistry-lab-simulator/collected_data/20250306_173208rgb")
# config_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.07/16.19.03_train_diffusion_unet_image_real_image/.hydra/config.yaml"
# model_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.07/16.19.03_train_diffusion_unet_image_real_image/checkpoints/epoch=0040-train_loss=0.004.ckpt"

# 周日加了视觉会徘徊在烧杯处就不提起
config_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.09/12.51.12_train_diffusion_unet_image_real_image/.hydra/config.yaml"
model_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.09/12.51.12_train_diffusion_unet_image_real_image/checkpoints/epoch=0080-train_loss=0.0019.ckpt"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 加载配置文件
cfg = OmegaConf.load(config_path)

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# 从配置文件初始化策略模型
policy = hydra.utils.instantiate(cfg.policy)
policy.load_state_dict(checkpoint['state_dicts']['model'])
policy.eval()
policy.to(device)

dataset: BaseImageDataset
dataset = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(dataset, BaseImageDataset)
normalizer = dataset.get_normalizer()
normalizer.to(device)
        
policy.set_normalizer(normalizer)

val_dataset = dataset.get_validation_dataset()  # 获取验证集，用于对比
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # 批量大小设为1，便于逐个对比

# 在主循环前初始化观察历史缓存
n_obs_steps = cfg.n_obs_steps  # 从配置文件获取所需的观察帧数
obs_history_camera1 = []  # 存储camera1的历史帧
obs_history_camera2 = []  # 存储camera2的历史帧
obs_history_pose = []    # 存储机器人姿态的历史

frame_idx = 0
reset_needed = False
video_frames = []
episode_count = 0
is_generate_trajectory = True
# 在主循环前添加帧数上限常量
MAX_FRAMES = 1000  # 设置最大帧数

while True:    
    if reset_needed:    
        obs_history_camera1.clear()
        obs_history_camera2.clear()
        obs_history_pose.clear()
        continue

    frame_idx += 1
    if frame_idx < 5:
        continue
    elif frame_idx >= MAX_FRAMES:
        reset_needed = True
        continue
    
    # 获取配置指定类型的图像用于模型输入
    camera1_data = val_dataloader
    camera2_data, _ = get_camera_data(camera2, cfg_camera.camera_2.image_type)
    
    robot_pose = my_franka.get_joint_positions()[:-1]
    
    if camera1_data is not None and camera2_data is not None and robot_pose is not None:
        obs_history_camera1.append(np.array(camera1_data, dtype=np.int8))
        obs_history_camera2.append(np.array(camera2_data, dtype=np.int8))
        obs_history_pose.append(robot_pose)
        while len(obs_history_camera1) > n_obs_steps:
            obs_history_camera1.pop(0)
            obs_history_camera2.pop(0)
            obs_history_pose.pop(0)
    
    # 如果当前轨迹已完成且有足够的观察历史,则进行新的推理
    if trajectory_controller.is_trajectory_complete() and len(obs_history_camera1) == n_obs_steps:
        # 将历史观察数据堆叠并转换为张量
        camera1_tensor = torch.from_numpy(np.stack(obs_history_camera1)).float() / 255
        camera2_tensor = torch.from_numpy(np.stack(obs_history_camera2)).float() / 255
        pose_tensor = torch.from_numpy(np.stack(obs_history_pose)).float()
        
        obs_dict = {
            'camera_1': camera1_tensor.unsqueeze(0).to(device),  # 添加batch维度 [1,T,C,H,W]
            'camera_2': camera2_tensor.unsqueeze(0).to(device),
            'agent_pose': pose_tensor.unsqueeze(0).to(device)    # [1,T,D]
        }
        with torch.no_grad():
            prediction = policy.predict_action(obs_dict)
            predicted_action = prediction['action']

            # print("Inferance action:",prediction)
            # # 指定保存路径并确保目录存在
            # save_dir = '/home/ubuntu/AIlab project/chemistry-lab-simulator'  # 保存目录
            # save_path = os.path.join(save_dir, 'Inferance_results.txt')  # 完整文件路径
            # os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
            
            # # 将 prediction 追加写入 TXT 文件
            # with open(save_path, 'a') as f:  # 'a' 表示追加模式
            #     f.write(f"{str(prediction)}\n\n")  # 写入 prediction，添加换行分隔
