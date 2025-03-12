import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from isaacsim import SimulationApp
import cv2
from datetime import datetime
import json
simulation_app = SimulationApp({"headless": False})
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

# 初始化仿真环境
world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")

# 设置机器人和场景
my_franka = Franka(position=np.array((-0.4, -0, 0.71)))
usd_path = "asserts/chemistry_lab/table.usd"
add_reference_to_stage(usd_path=os.path.abspath(usd_path), prim_path="/World")

def load_camera_config(data_dir):
    """加载相机配置"""
    config_path = os.path.join(data_dir, "camera_config.yaml")
    return OmegaConf.load(config_path)

# 添加视频保存工具函数
def save_video(frames, filename, fps=30):
    """
    将图像帧保存为视频
    Args:
        frames: 包含(camera1_frame, camera2_frame)元组的列表
        filename: 输出文件名
        fps: 视频帧率
    """
    if not frames:
        return

    h, w = frames[0][0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w*2, h))  # 双倍宽度用于并排显示
    
    for camera1_frame, camera2_frame in frames:
        camera1_frame = np.array(camera1_frame)
        camera2_frame = np.array(camera2_frame)
        combined_frame = np.hstack((camera1_frame, camera2_frame))
        out.write(combined_frame)
    
    out.release()
    print(f"Video saved to {filename}")

# 修改相机数据获取部分
def get_camera_data(camera1, image_type):
    if image_type == "rgb":
        img = camera1.get_rgb()
        img_for_record = np.transpose(img, (2, 1, 0))
        img_for_display = img[..., ::-1]
        return img_for_record, img_for_display
    elif image_type == "depth":
        depth = camera1.get_depth()
        if depth is not None:
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_for_record = depth_normalized.astype(np.uint8)
            depth_for_record = depth_for_record[np.newaxis, :, :]
            depth_for_record = np.repeat(depth_for_record, 3, axis=0)
            depth_for_display = cv2.applyColorMap(depth_for_record[0], cv2.COLORMAP_JET)
            return depth_for_record, depth_for_display
        else:
            print("Warning: Depth data not available.")
            return None, None

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
cfg_camera = load_camera_config("/home/ubuntu/AIlab project/chemistry-lab-simulator/collected_data/20250306_173208rgb")
config_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.09/12.51.12_train_diffusion_unet_image_real_image/.hydra/config.yaml"
model_path = "/home/ubuntu/AIlab project/chemistry-lab-simulator/data/outputs/2025.03.09/12.51.12_train_diffusion_unet_image_real_image/checkpoints/latest.ckpt"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置相机
camera1 = Camera(
    prim_path=cfg_camera.camera_1.prim_path,
    translation=np.array(cfg_camera.camera_1.translation),
    frequency=20,
    name=cfg_camera.camera_1.name,
    resolution=tuple(cfg_camera.camera_1.resolution)
)
camera1.set_focal_length(cfg_camera.camera_1.focal_length)
camera1.set_local_pose(orientation=np.array(cfg_camera.camera_1.orientation), camera_axes="usd")
world.scene.add(camera1)

camera2 = Camera(
    prim_path=cfg_camera.camera_2.prim_path,
    translation=np.array(cfg_camera.camera_2.translation),
    frequency=20,
    name=cfg_camera.camera_2.name,
    resolution=tuple(cfg_camera.camera_2.resolution)
)
camera2.set_focal_length(cfg_camera.camera_2.focal_length)
camera2.set_local_pose(orientation=np.array(cfg_camera.camera_2.orientation), camera_axes="usd")
world.scene.add(camera2)

stage = omni.usd.get_context().get_stage()

# 初始化控制器
object_utils = ObjectUtils(stage)
articulation_controller = my_franka.get_articulation_controller()
pick_controller = PickController(
    name="pick_controller",
    cspace_controller=RMPFlowController(name="target_follower_controller", robot_articulation=my_franka),
    gripper=my_franka.gripper,
)
gripper_control = Gripper()

# 修改控制器初始化部分
trajectory_controller = FrankaTrajectoryController(
    name="trajectory_controller",
    robot_articulation=my_franka
)

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

# 在主循环前初始化观察历史缓存
n_obs_steps = cfg.n_obs_steps  # 从配置文件获取所需的观察帧数
obs_history_camera1 = []  # 存储camera1的历史帧
obs_history_camera2 = []  # 存储camera2的历史帧
obs_history_pose = []    # 存储机器人姿态的历史

world.reset()
my_franka.initialize()
camera1.initialize()
camera1.add_distance_to_image_plane_to_frame()
camera2.initialize()
camera2.add_distance_to_image_plane_to_frame()
    
articulation_controller = my_franka.get_articulation_controller()
frame_idx = 0
reset_needed = False
video_frames = []
episode_count = 0
beaker_position = np.array([np.random.uniform(0.2, 0.25), np.random.uniform(-0.05, 0.05), 0.77])
object_utils.set_object_position(obj_path="/World/beaker", position=beaker_position)
is_generate_trajectory = True
# 在主循环前添加帧数上限常量
is_start_action = True
MAX_FRAMES = 10000  # 设置最大帧数

while simulation_app.is_running():
    world.step(render=True)
    if world.is_stopped() and not reset_needed:
        reset_needed = True
    
    if world.is_playing():
        if reset_needed:
            # 保存上一轮的视频
            if video_frames:
                video_path = f"videos/{timestamp}/episode_{episode_count}.mp4"
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                save_video(video_frames, video_path)
                video_frames = []  # 清空帧列表
                episode_count += 1
            is_start_action = True
            frame_idx = 0
            reset_needed = False
            
            world.reset()
            pick_controller.reset()  
            trajectory_controller.reset()
                                  
            my_franka.initialize()
    
            obs_history_camera1.clear()
            obs_history_camera2.clear()
            obs_history_pose.clear()

            beaker_position = np.array([np.random.uniform(0.2, 0.25), np.random.uniform(-0.05, 0.05), 0.77])
            object_utils.set_object_position(obj_path="/World/beaker", position=beaker_position)
            continue

        frame_idx += 1
        if frame_idx < 20:
            continue
        elif frame_idx >= MAX_FRAMES:
            reset_needed = True
            continue
        
        # if is_start_action:
        #     is_start_action = False
        #     predicted_action = np.array([[ 6.7467e-03, -5.8784e-01, -2.9321e-03, -2.8271e+00, -1.7062e-03,
        #    2.9650e+00,  7.3938e-01,  3.5510e-02],
        #  [ 2.0667e-03, -5.8572e-01, -9.7320e-04, -2.8278e+00, -1.3311e-03,
        #    2.9508e+00,  7.3987e-01,  3.6195e-02],
        #  [ 3.0778e-04, -5.8786e-01, -3.3113e-03, -2.8312e+00,  4.9892e-03,
        #    2.9353e+00,  7.4076e-01,  3.6775e-02],
        #  [ 5.3338e-04, -5.8961e-01, -1.9967e-03, -2.8340e+00, -3.4814e-04,
        #    2.9212e+00,  7.4077e-01,  3.7263e-02],
        #  [ 1.7995e-03, -5.8887e-01, -4.7999e-03, -2.8373e+00,  4.1765e-03,
        #    2.9028e+00,  7.4213e-01,  3.7679e-02],
        #  [ 1.6486e-03, -5.8513e-01, -6.4282e-03, -2.8387e+00, -8.7792e-05,
        #    2.8873e+00,  7.4083e-01,  3.8003e-02]])
            
        #     trajectory_controller.add_trajectory(predicted_action)


        # 获取相机数据并保存帧
        _, camera1_data_rgb = get_camera_data(camera1, "rgb")  # 始终获取RGB用于保存视频
        _, camera2_data_rgb = get_camera_data(camera2, "rgb")
        if camera1_data_rgb is not None and camera2_data_rgb is not None:
            video_frames.append((camera1_data_rgb, camera2_data_rgb))
        
        # 获取配置指定类型的图像用于模型输入
        camera1_data, _ = get_camera_data(camera1, cfg_camera.camera_1.image_type)
        camera2_data, _ = get_camera_data(camera2, cfg_camera.camera_2.image_type)
        
        robot_pose = my_franka.get_joint_positions()[:-1]
        
        if camera1_data is not None and camera2_data is not None and robot_pose is not None:
            obs_history_camera1.append(np.array(camera1_data, dtype=np.uint8))
            obs_history_camera2.append(np.array(camera2_data, dtype=np.uint8))
            obs_history_pose.append(robot_pose)
            while len(obs_history_camera1) > n_obs_steps:
                obs_history_camera1.pop(0)
                obs_history_camera2.pop(0)
                obs_history_pose.pop(0)
        
        # 如果当前轨迹已完成且有足够的观察历史,则进行新的推理is_start_action
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
            print(obs_dict['agent_pose'])
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

            # print("Predicted action:", predicted_action)
            joint_positions = predicted_action[0].cpu().numpy()
            # print("Predicted joint positions:", joint_positions)
            if is_generate_trajectory:
                trajectory_controller.generate_trajectory(joint_positions)
            else:
                trajectory_controller.add_trajectory(joint_positions)
            
        action = trajectory_controller.get_next_action()
        if action is not None:
            # print(action)
            articulation_controller.apply_action(action)

simulation_app.close()
# 保存最后一轮的视频
if video_frames:
    video_path = f"/home/ubuntu/Documents/IsaacLabSim/videos/{timestamp}/episode_{episode_count}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    save_video(video_frames, video_path)