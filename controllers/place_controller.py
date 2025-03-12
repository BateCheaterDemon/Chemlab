from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat

import numpy as np
import typing
from isaacsim.robot.manipulators.grippers.gripper import Gripper

class PlaceController(BaseController):
    """
    一个放置状态机控制器。

    该控制器处理将物体放置到指定位置的过程。

    放置阶段包括：
    - 阶段 0：将末端执行器移动到放置位置上方的指定高度。
    - 阶段 1：将末端执行器降低到放置位置。
    - 阶段 2：等待机械臂惯性稳定。
    - 阶段 3：打开夹具以释放物体。
    - 阶段 4：抬高末端执行器。

    参数：
        name (str): 控制器的标识符。
        cspace_controller (BaseController): 返回 ArticulationAction 类型的笛卡尔空间控制器。
        gripper (Gripper): 用于开闭夹具的夹具控制器。
        end_effector_initial_height (float, optional): 末端执行器的初始高度。默认为 0.3 米。
        events_dt (list of float, optional): 每个阶段的时间持续。默认为 [0.005, 0.02, 0.2, 0.2, 0.01] 除以速度。
        speed (float, optional): 阶段持续时间的速度倍率。默认为 1.0。

    异常：
        Exception: 如果 'events_dt' 不是列表或 numpy 数组。
        Exception: 如果 'events_dt' 的长度超过 7。
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1.0
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.32 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if events_dt is None:
            self._events_dt = [dt / speed for dt in [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt 需要是列表或 numpy 数组")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt 长度必须小于 7")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self._start = True
        self.target_position = None  # 用于存储目标放置位置
        return

    def is_paused(self) -> bool:
        """
        检查状态机是否暂停。

        Returns:
            bool: 如果暂停，返回 True，否则返回 False。
        """
        return self._pause

    def get_current_event(self) -> int:
        """
        获取当前状态机的阶段。

        Returns:
            int: 当前阶段编号。
        """
        return self._event

    def forward(
        self,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        执行控制器的一步操作。

        Args:
            place_position (np.ndarray): 放置目标位置。
            current_joint_positions (np.ndarray): 机械臂当前的关节位置。
            end_effector_offset (np.ndarray, optional): 末端执行器的偏移。默认为 None。
            end_effector_orientation (np.ndarray, optional): 末端执行器的朝向。默认为 None。

        Returns:
            ArticulationAction: 由 ArticulationController 执行的动作。
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._start:
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        if self._event == 0:
            # 阶段 0：移动到放置位置上方
            target_position = place_position.copy()
            target_position[2] += 0.25 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 1:
            # 阶段 1：降低到放置位置
            target_position = place_position.copy()
            target_position[2] += 0.05 / get_stage_units()
            target_position = place_position.copy()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 2:
            # 阶段 2：等待惯性稳定
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            # 阶段 3：打开夹具以释放物体
            target_joint_positions = self._gripper.forward(action="open")
            self.target_position = place_position.copy()
            self.target_position[2] += 0.05
            self.target_position[0] -= 0.1
            gripper_control.release_object()
        elif self._event == 4:
            # 阶段 4：抬高末端执行器
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        else:
            # 所有阶段完成，保持当前位置
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joint_positions

    def _get_alpha(self):
        """
        根据当前阶段计算插值因子。

        Returns:
            float: 插值因子。
        """
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError("无效的事件阶段")

    def _get_target_hs(self, target_height):
        """
        根据当前阶段计算末端执行器的目标高度。

        Args:
            target_height (float): 末端执行器的目标高度。

        Returns:
            float: 计算后的目标高度。
        """
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError("无效的事件阶段")
        return h

    def _mix_sin(self, t):
        """
        使用正弦函数计算插值因子，以实现平滑过渡。

        Args:
            t (float): 时间参数。

        Returns:
            float: 插值因子。
        """
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        """
        对两个值进行凸组合。

        Args:
            a (float): 第一个值。
            b (float): 第二个值。
            alpha (float): 插值因子。

        Returns:
            float: 凸组合后的值。
        """
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """
        重置状态机，从第一个阶段开始。

        Args:
            end_effector_initial_height (float, optional): 末端执行器的初始高度。默认为 None。
            events_dt (list of float, optional): 每个阶段的时间持续。默认为 None。

        异常：
            Exception: 如果 'events_dt' 不是列表或 numpy 数组。
            Exception: 如果 'events_dt' 的长度超过 10。
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities 需要是列表或 numpy 数组")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt 长度必须小于 10")
        return

    def is_done(self) -> bool:
        """
        检查状态机是否已经完成所有阶段。

        Returns:
            bool: 如果所有阶段完成，返回 True，否则返回 False。
        """
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        """
        暂停状态机的时间和阶段。
        """
        self._pause = True
        return

    def resume(self) -> None:
        """
        恢复状态机的时间和阶段。
        """
        self._pause = False
        return