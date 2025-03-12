import argparse
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World, PhysicsContext
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from pxr import Usd, UsdGeom, Gf
import omni.usd

# User import
# from controllers.controller_manager import ControllerManager
# from isaacsim.robot.manipulators.grippers.gripper import ParallelGripperpper
import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from robots.franka import Franka
from utils.object_utils import ObjectUtils
from data_collector import DataCollector
from controllers.pick_controller import PickController
from controllers.grapper_manager import Gripper

import hydra
from omegaconf import OmegaConf, DictConfig
import os

# create the world
world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
# world = World(stage_units_in_meters = 1, device = "cpu")
# physx_interface = omni.physx.get_physx_interface()
# physx_interface.overwrite_gpu_setting(1)

my_franka = Franka(position=np.array((-0.4, -0, 0.71)))
# usd_path = "asserts/chemistry_lab/table.usd"
usd_path = "/home/ubuntu/IL_my/assert/chemistry_lab/table4.usd"
add_reference_to_stage(usd_path=usd_path, prim_path="/World/lab")

stage = omni.usd.get_context().get_stage()

# 修改 hydra 配置部分
@hydra.main(version_base=None, config_path="config", config_name="camera_config")
def main(cfg: DictConfig):
    # 设置最大录制集数
    MAX_EPISODES = 30
    
    # 创建数据采集器
    data_collector = DataCollector(save_dir="collected_data", max_episodes=MAX_EPISODES)
    
    # 保存配置到数据集目录
    config_save_path = os.path.join(data_collector.session_dir, "camera_config.yaml")
    OmegaConf.save(cfg, config_save_path)
    
    # 设置相机
    camera = Camera(
        prim_path=cfg.camera_1.prim_path,
        translation=np.array(cfg.camera_1.translation),
        name=cfg.camera_1.name,
        frequency=20,
        resolution=tuple(cfg.camera_1.resolution)
    )
    camera.set_focal_length(cfg.camera_1.focal_length)
    camera.set_local_pose(orientation=np.array(cfg.camera_1.orientation), camera_axes="usd")
    world.scene.add(camera)
    
    camera2 = Camera(
        prim_path=cfg.camera_2.prim_path,
        translation=np.array(cfg.camera_2.translation),
        name=cfg.camera_2.name,
        frequency=20,
        resolution=tuple(cfg.camera_2.resolution)
    )
    camera2.set_focal_length(cfg.camera_2.focal_length)
    camera2.set_local_pose(orientation=np.array(cfg.camera_2.orientation), camera_axes="usd")
    world.scene.add(camera2)
    
    object_utils = ObjectUtils(stage)

    articulation_controller = my_franka.get_articulation_controller()

    # 创建pick controller
    pick_controller = PickController(
        name="pick_controller",
        cspace_controller=RMPFlowController(name="target_follower_controller", robot_articulation=my_franka),
        gripper=my_franka.gripper,
    )
    gripper_control = Gripper()

    world.reset()
    my_franka.initialize()
    articulation_controller = my_franka.get_articulation_controller()

    reset_needed = False
    frame_idx = 0
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_stopped() and not reset_needed:
            reset_needed = True
        if world.is_playing():
            if reset_needed:
                # 检查是否达到最大录制集数
                if data_collector.episode_count >= MAX_EPISODES:
                    data_collector.close()
                    simulation_app.close()
                    break
                    
                world.reset()
                my_franka.initialize()
                reset_needed = False
                frame_idx = 0
                pick_controller.reset()
                # 随机化烧杯位置
                # beaker_position = np.array([np.random.uniform(0, 0.45), np.random.uniform(-0.4, 0.4), 0.77])
                # beaker_position = np.array([np.random.uniform(0.2, 0.25), np.random.uniform(-0.05, 0.05), 0.77])
                # object_utils.set_object_position(obj_path="/World/beaker", position=beaker_position)
            frame_idx += 1
            if frame_idx < 5:
                continue
            
            object_position = object_utils.get_object_position(obj_path="/World/lab/beaker")
            object_size = object_utils.get_object_size(obj_path="/World/lab/beaker")
            
            if object_position is None:
                print("Warning: Object position not found.")
                continue
            joint_positions = my_franka.get_joint_positions()
            
            joint_state = my_franka.get_joints_state()
            
            # 执行抓取动作
            if not pick_controller.is_done():
                action = pick_controller.forward(
                    picking_position=object_position,
                    current_joint_positions=joint_positions,
                    object_size=object_size,
                    object_name="beaker",
                    gripper_control=gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
            else:
                # 获取最后一个状态的joint positions作为最后一个动作
                final_joint_positions = joint_positions[:-1]
                # 将所有缓存的数据写入文件
                data_collector.write_cached_data(final_joint_positions)
                reset_needed = True
            
            # 获取相机图像
            camera_data = camera.get_rgb()
            camera2_data = camera2.get_rgb()
            camera_data = np.transpose(camera_data, (2, 1, 0))
            camera2_data = np.transpose(camera2_data, (2, 1, 0))
            # 获取当前关节角度
            joint_angles = my_franka.get_joint_positions()[:-1]
            
            # 缓存当前状态
            data_collector.cache_step(
                camera_data,
                camera2_data,
                joint_angles
            )
                
            if action is not None:
                articulation_controller.apply_action(action)

if __name__ == "__main__":
    main()