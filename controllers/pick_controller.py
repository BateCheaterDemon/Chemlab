from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units, get_current_stage
from isaacsim.core.utils.prims import move_prim, get_prim_at_path
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import Gf, UsdGeom, Usd, Sdf
import numpy as np
import typing
from isaacsim.robot.manipulators.grippers.gripper import Gripper

class PickController(BaseController):
    """
    A pick state machine controller.

    This controller handles the process of picking up an object.

    The pick phases are:
    - Phase 0: Move end_effector above the object at 'end_effector_initial_height'.
    - Phase 1: Lower end_effector to grip the object.
    - Phase 2: Wait for the robot's inertia to settle.
    - Phase 3: Close the gripper to pick up the object.
    - Phase 4: Lift the object by raising the end_effector.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction type.
        gripper (Gripper): A gripper controller for open/close actions.
        end_effector_initial_height (float, optional): Initial height for the end effector. Defaults to 0.3 meters if not specified.
        events_dt (list of float, optional): Time duration for each phase. Defaults to [0.005, 0.005, 0.02, 0.02, 0.005, 0.005, 0.005] divided by speed if not specified.
        speed (float, optional): Speed multiplier for phase durations. Defaults to 1.0.

    Raises:
        Exception: If 'events_dt' is not a list or numpy array.
        Exception: If 'events_dt' length is greater than 7.
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
            self._events_dt = [dt / speed for dt in [0.005, 0.01, 0.01, 0.2, 0.05, 0.01]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 7")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self._start = True
        self.object_size = None
        return

    def is_paused(self) -> bool:
        """
        Check if the state machine is paused.

        Returns:
            bool: True if paused, False otherwise.
        """
        return self._pause

    def get_current_event(self) -> int:
        """
        Get the current phase/event of the state machine.

        Returns:
            int: Current phase/event number.
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_size: np.ndarray,
        object_name: str,
        gripper_control,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute one step of the controller.

        Args:
            picking_position (np.ndarray): Position of the object to be picked.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (np.ndarray, optional): Offset of the end effector target. Defaults to None.
            end_effector_orientation (np.ndarray, optional): Orientation of the end effector. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        if self.object_size is None:
            self.object_size = object_size
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._start:
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = 0.04 / get_stage_units()
            target_joint_positions[8] = 0.04 / get_stage_units() 
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
                
        if self._event == 0:
            picking_position[0] -= 0.4 / get_stage_units()
            picking_position[2] += self.object_size[2]
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=picking_position
                )
        elif self._event == 1:
            picking_position[0] -= 0.1 / get_stage_units()
            picking_position[2] -= self.object_size[2] / 3  # +0.06
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=picking_position, target_end_effector_orientation=end_effector_orientation
                )
        elif self._event == 2:
            # qwx add for baker pick(use controller)
            # picking_position[2] -= 0.08
            ############################
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=picking_position, target_end_effector_orientation=end_effector_orientation
                )
        elif self._event == 3:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 4:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = self.get_gripper_distance(object_name) / get_stage_units()
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.target_position = picking_position
            self.target_position[2] += self.object_size[2] * 1.5 # origen
            if "glass" in object_name:
                gripper_control.add_object_to_gripper("/World/lab/Desk1/glass_rod", "/World/Franka/panda_hand/tool_center")
        else:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position, 
                target_end_effector_orientation=end_effector_orientation
            )

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joint_positions

    def _get_alpha(self):
        """
        Calculate the interpolation factor based on the current phase.

        Returns:
            float: Interpolation factor.
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
            raise ValueError()

    def _get_target_hs(self, target_height):
        """
        Calculate the target height for the end effector based on the current phase.

        Args:
            target_height (float): Target height for the end effector.

        Returns:
            float: Calculated target height.
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
            raise ValueError()
        return h

    def _mix_sin(self, t):
        """
        Calculate the interpolation factor using a sine function for smooth transitions.

        Args:
            t (float): Time parameter.

        Returns:
            float: Interpolation factor.
        """
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        """
        Perform convex combination of two values.

        Args:
            a (float): First value.
            b (float): Second value.
            alpha (float): Interpolation factor.

        Returns:
            float: Convex combination of the two values.
        """
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            end_effector_initial_height (float, optional): Initial height for the end effector. Defaults to None.
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 10.
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
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        self._start = True
        self.object_size = None
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        """
        Pause the state machine's time and phase.
        """
        self._pause = True
        return

    def resume(self) -> None:
        """
        Resume the state machine's time and phase.
        """
        self._pause = False
        return
    
    def get_gripper_distance(self, item_name):
        gripper_distances = {
            "rod": 0.003,
            "tube": 0.01,
            "beaker": 0.017,
            "Erlenmeyer flask": 0.018,
            "cylinder": 0.016,
            "Petri dish": 0.005,
            "pipette": 0.008,
            "microscope slide": 0.002
        }
        
        for key in gripper_distances:
            if key in item_name.lower():
                return gripper_distances[key]
        
        return 0.02
    
