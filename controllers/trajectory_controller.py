import os
import numpy as np
from typing import List, Optional

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.prims.impl.single_articulation import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController

class FrankaTrajectoryController(RMPFlowController):
    """Franka机械臂轨迹控制器，支持连续轨迹生成和执行"""

    def __init__(
        self, 
        name: str, 
        robot_articulation: SingleArticulation, 
        physics_dt: float = 1.0/60.0
    ) -> None:
        super().__init__(name=name, robot_articulation=robot_articulation, physics_dt=physics_dt)
        
        # 获取motion_generation配置文件路径
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        
        # 初始化轨迹生成器
        self._c_space_trajectory_generator = mg.LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        # 初始化运动学求解器
        self._kinematics_solver = mg.LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        self._action_sequence = []
        self._action_sequence_index = 0
        self._end_effector_name = "panda_hand"
        self._physics_dt = physics_dt

    def add_trajectory(
        self, 
        waypoints: np.ndarray,
    ) -> None:
        """生成关节空间轨迹

        Args:
            waypoints (np.ndarray): Shape (N, 8) 的数组，包含N个路点的关节角度和夹爪位置
        """
        # 分离关节角度和夹爪位置
        joint_waypoints = waypoints[:, :7]  # 前7个关节

        self.gripper_positions = waypoints[:, 7]  # 最后一个是夹爪位置
        self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, waypoints.shape[0], dtype=int)

        # 初始化 action_sequence 为空列表
        self._action_sequence = []
        self._action_sequence_index = 0
        # 为每个 waypoint 创建 ArticulationAction 并添加到序列中
        for joint_pos in joint_waypoints:
            # 创建 ArticulationAction 对象，只设置 joint_positions
            action = ArticulationAction(joint_positions=joint_pos)
            self._action_sequence.append(action)

    

    def generate_trajectory(
        self, 
        waypoints: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> None:
        """生成关节空间轨迹

        Args:
            waypoints (np.ndarray): Shape (N, 8) 的数组，包含N个路点的关节角度和夹爪位置
        """
        # 分离关节角度和夹爪位置
        joint_waypoints = waypoints[:, :7]  # 前7个关节
        self.gripper_positions = waypoints[:, 7]  # 最后一个是夹爪位置

        if timestamps is not None:
            trajectory = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(
                joint_waypoints, timestamps
            )
        else:
            trajectory = self._c_space_trajectory_generator.compute_c_space_trajectory(joint_waypoints)

        if trajectory is not None:
            # 将轨迹转换为动作序列
            articulation_trajectory = mg.ArticulationTrajectory(
                self._articulation_motion_policy._robot_articulation,
                trajectory,
                self._physics_dt
            )
            self._action_sequence = articulation_trajectory.get_action_sequence()
            # 计算每个动作对应的夹爪位置索引
            total_actions = len(self._action_sequence)
            self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
            self._action_sequence_index = 0
        else:
            print("Warning: Failed to generate trajectory")
            self._action_sequence = []
            self.gripper_positions = []
            self.gripper_indices = []

    def get_next_action(self) -> Optional[ArticulationAction]:
        """获取序列中的下一个动作

        Returns:
            Optional[ArticulationAction]: 下一个要执行的动作，如果序列已完成则返回None
        """
        if not self._action_sequence or self._action_sequence_index >= len(self._action_sequence):
            return None
            
        action = self._action_sequence[self._action_sequence_index]
        
        # 将夹爪位置添加到动作中
        if hasattr(self, 'gripper_positions') and len(self.gripper_positions) > 0:
            gripper_idx = self.gripper_indices[self._action_sequence_index]
            gripper_position = self.gripper_positions[gripper_idx]
            
            # 创建包含夹爪位置的新动作
            # Franka机器人的夹爪需要两个关节位置，所以添加相同的值
            joint_positions = np.concatenate([
                action.joint_positions,
                np.array([gripper_position, gripper_position], dtype=np.float32)
            ])
            # joint_velocities = np.concatenate([
            #     action.joint_velocities,
            #     np.array([0.0, 0.0], dtype=np.float32)
            # ])
            
            new_action = ArticulationAction(
                joint_positions=joint_positions,
                # joint_velocities=joint_velocities,
                joint_efforts=action.joint_efforts
            )
            action = new_action
        
        self._action_sequence_index += 1
        return action

    def is_trajectory_complete(self) -> bool:
        """检查轨迹是否执行完成

        Returns:
            bool: 如果轨迹执行完成返回True，否则返回False
        """
        return len(self._action_sequence) == 0 or self._action_sequence_index >= len(self._action_sequence)

    def reset(self) -> None:
        """重置控制器状态"""
        super().reset()
        self._action_sequence = []
        self._action_sequence_index = 0
        self.gripper_positions = []
        self.gripper_indices = []
