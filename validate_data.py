import os
import numpy as np
import h5py
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
import omni.usd
from collections import deque
import json
import hydra
from omegaconf import OmegaConf

# 导入自定义组件
from robots.franka import Franka
from utils.object_utils import ObjectUtils
from utils.action_utils import joint_positions_to_action

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

stage = omni.usd.get_context().get_stage()
object_utils = ObjectUtils(stage)
articulation_controller = my_franka.get_articulation_controller()

def load_episode_data(data_file, episode_idx):
    """从单个HDF5文件中加载指定场景的数据"""
    with h5py.File(data_file, 'r') as f:
        # 构造数据集名称
        episode_group = f[f'episode_{episode_idx:04d}']
        joint_angles_dataset = episode_group['agent_pose']
        camera1_dataset = episode_group['camera_1']
        camera2_dataset = episode_group['camera_2']
        
        # 读取数据
        joint_angles = joint_angles_dataset[:]
        camera1_images = camera1_dataset[:]
        camera2_images = camera2_dataset[:]
        
    return joint_angles, camera1_images, camera2_images

def validate_episode(episode_number, data_file):
    """验证单个场景的数据"""
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return False
    
    try:
        # 加载数据
        joint_angles, camera1_images, camera2_images = load_episode_data(data_file, episode_number)
    except KeyError:
        print(f"Episode {episode_number} data not found in the file!")
        return False
    except Exception as e:
        print(f"Error loading episode {episode_number}: {str(e)}")
        return False
    
    # 重置环境
    world.reset()
    my_franka.initialize()
    
    # 设置烧杯位置
    beaker_position = np.array([0.32, 0.0, 0.77])
    object_utils.set_object_position(obj_path="/World/beaker", position=beaker_position)
    
    # 执行动作序列
    for step, target_joints in enumerate(joint_angles):
        # 将关节角度转换为动作
        action = joint_positions_to_action(target_joints)
        
        # 应用动作
        articulation_controller.apply_action(action)
        
        # 步进仿真
        world.step(render=True)
        
        # 获取当前相机图像
        # current_camera1 = camera1.get_rgb()
        # current_camera2 = camera2.get_rgb()
        
        # 打印进度
        print(f"Step {step}/{len(joint_angles)}", end='\r')
    
    print(f"\nFinished validating episode {episode_number}")
    return True

def main():
    """主函数"""
    data_dir = "/home/ubuntu/Documents/IsaacLabSim/collected_data/20250228_113514"
    data_file = os.path.join(data_dir, "episode_data.hdf5")
    
    # 加载相机配置
    cfg = load_camera_config(data_dir)
    
    # 设置相机
    camera = Camera(
        prim_path=cfg.camera_1.prim_path,
        translation=np.array(cfg.camera_1.translation),
        frequency=20,
        name=cfg.camera_1.name,
        resolution=tuple(cfg.camera_1.resolution)
    )
    camera.set_focal_length(cfg.camera_1.focal_length)
    camera.set_local_pose(orientation=np.array(cfg.camera_1.orientation), camera_axes="usd")
    world.scene.add(camera)
    
    camera2 = Camera(
        prim_path=cfg.camera_2.prim_path,
        translation=np.array(cfg.camera_2.translation),
        frequency=20,
        name=cfg.camera_2.name,
        resolution=tuple(cfg.camera_2.resolution)
    )
    camera2.set_focal_length(cfg.camera_2.focal_length)
    camera2.set_local_pose(orientation=np.array(cfg.camera_2.orientation), camera_axes="usd")
    world.scene.add(camera2)
    
    # 首先检查文件是否存在
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    
    # 获取可用的场景数量
    with h5py.File(data_file, 'r') as f:
        # 获取所有数据集名称
        episode_keys = [k for k in f.keys() if k.startswith('episode_')]
        num_episodes = len(episode_keys)
    
    print(f"Found {num_episodes} episodes in the data file")
    
    # 验证前几个场景
    for episode in range(min(3, num_episodes)):  # 这里选择验证前3个场景或更少
        print(f"\nValidating episode {episode}")
        validate_episode(episode, data_file)
        
        # 等待用户确认继续
        input("Press Enter to continue to next episode...")

if __name__ == "__main__":
    main()
