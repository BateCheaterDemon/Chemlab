import argparse
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

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
import cv2

# create the world
world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")

stage = omni.usd.get_context().get_stage()

my_franka = Franka(position=np.array((-1.25, -0.72, -1.11)))

usd_path = "/home/ubuntu/IL_my/assert/chemistry_lab/table6.usd"
add_reference_to_stage(usd_path=usd_path, prim_path="/World/lab")

def process_camera_image(camera,image_type):
    if image_type == "rgb":
        img = camera.get_rgb()
        img_for_record = np.transpose(img, (2, 1, 0))
        img_for_display = img[..., ::-1]
        return img_for_record, img_for_display
    elif image_type == "depth":
        depth = camera.get_depth()
        # 数据记录用的深度图需要归一化到0-255
        if depth is not None:
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_for_record = depth_normalized.astype(np.uint8)
            depth_for_record = depth_for_record[np.newaxis, :, :]
            # 显示用的深度图需要添加颜色映射
            depth_for_display = cv2.applyColorMap(depth_for_record[0], cv2.COLORMAP_JET)
            return depth_for_record, depth_for_display
        else:
            print("Warning: Depth data not available.")
            return None, None

@hydra.main(version_base=None, config_path="config", config_name="camera_config")
def main(cfg:DictConfig):
    # 设置最大录制集数
    MAX_EPISODES = 80
    
    # 创建数据采集器
    data_collector = DataCollector(save_dir="policy_data",max_episodes=MAX_EPISODES)

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

    # controller
    pick_controller = PickController(
        name="pick_controller",
        cspace_controller=RMPFlowController(name="target_follower_controller", robot_articulation=my_franka),
        gripper=my_franka.gripper,
    )
    gripper_control = Gripper()

    world.reset()
    my_franka.initialize()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    camera2.initialize()
    camera2.add_distance_to_image_plane_to_frame()

    articulation_controller = my_franka.get_articulation_controller()

    reset_needed = False
    frame_idx = 0

    while simulation_app.is_running():
        world.step(render=True)
        if world.is_stopped() and not reset_needed:
            reset_needed = True
        if world.is_playing():
            if reset_needed:
                if data_collector.episode_count >= MAX_EPISODES:
                    data_collector.close()
                    simulation_app.close()
                    break

                world.reset()
                my_franka.initialize()
                reset_needed = False
                frame_idx = 0
                pick_controller.reset()
                
                # 暂时先不管这个
                # beaker_position = np.array([np.random.uniform(-0.55, -0.45), np.random.uniform(-0.6, -0.7), -0.7])
                # beaker_position = np.array([-0.5, -0.7, -0.7])
                # beaker_position = np.array([-0.3, -0.9, -0.7])
                # object_utils.set_object_position(obj_path="/World/lab/beaker", position=beaker_position)
            frame_idx += 1

            if frame_idx < 5: 
                continue
            
            object_position = object_utils.get_object_position(obj_path="/World/lab/beaker")
            # print("object_position", object_position)
            object_size = object_utils.get_object_size(obj_path="/World/lab/beaker")

            if object_position is None:
                print("Warning: Object position not found.")
                continue
            joint_positions = my_franka.get_joint_positions()
            # joint_state = my_franka.get_joints_state()
            
            if not pick_controller.is_done():
                action = pick_controller.forward(
                    picking_position=object_position,
                    current_joint_positions=joint_positions,
                    object_size=object_size,
                    object_name="beaker",
                    gripper_control=gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
            # print("action controller",action)
            # print("action type",type(action))
            else:
                # 判断最终烧杯位置
                final_beaker_position = object_utils.get_object_xform_position(obj_path="/World/lab/beaker/mesh")
                # print("final_baker_position:",final_beaker_position)
                if final_beaker_position is not None and final_beaker_position[2] > -1.0:
                    print("Task successed!!!")
                    # 成功完成任务，获取最后一个状态的joint positions作为最后一个动作
                    final_joint_positions = joint_positions[:-1]
                    # 将所有缓存的数据写入文件
                    data_collector.write_cached_data(final_joint_positions)
                else:
                    print("Task failed: beaker not in target position")
                    # 任务失败，清空当前缓存的数据
                    data_collector.temp_camera_1.clear()
                    data_collector.temp_camera_2.clear()
                    data_collector.temp_agent_pose.clear()
                    print("Task failed: beaker not in target position, clearing cached data")
                reset_needed = True

            # 获取并处理相机图像
            camera1_record, camera1_display = process_camera_image(camera, cfg.camera_1.image_type)
            camera2_record, camera2_display = process_camera_image(camera2, cfg.camera_2.image_type)

            joint_angles = my_franka.get_joint_positions()[:-1]

            if camera1_display is not None and camera2_display is not None:
                # 合并两个相机图像用于显示
                combined_img = np.hstack((camera1_display, camera2_display))
                
                # 添加标签
                label1 = f"Camera 1 ({cfg.camera_1.image_type})"
                label2 = f"Camera 2 ({cfg.camera_2.image_type})"
                cv2.putText(combined_img, label1, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined_img, label2, (camera1_display.shape[1] + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 显示合并后的图像
                cv2.imshow('Camera Views', combined_img)
                cv2.waitKey(1)

                # print(f"camera1_record shape: {camera1_record.shape}, min: {camera1_record.min()}, max: {camera1_record.max()}")
                # print(f"camera2_record shape: {camera2_record.shape}, min: {camera2_record.min()}, max: {camera2_record.max()}")

                # 使用记录用的图像保存到数据集
                data_collector.cache_step(
                    camera1_record,
                    camera2_record,
                    joint_angles
                )

            # actions_recordform = action.joint_positions        
            # def is_numpy_array(data):
            #     return isinstance(data,np.ndarray)       
            # if is_numpy_array(actions_recordform):
            #     actions_recordform = torch.from_numpy(actions_recordform)
            # print("actions_recordform",actions_recordform)
            # print("type",type(actions_recordform))

            if action is not None:
                articulation_controller.apply_action(action)
        
    cv2.destroyAllWindows()

# simulation_app.close()

if __name__ == "__main__":
    main()
    # simulation_app.close()
