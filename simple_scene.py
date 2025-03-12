import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World, PhysicsContext
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.sensors.camera import Camera
from pxr import Usd, UsdGeom, Gf
import omni.usd
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController

# User import
# from controllers.controller_manager import ControllerManager
from controllers.pick_controller import PickController
# from isaacsim.robot.manipulators.grippers.gripper import ParallelGripperpper
from robots.franka import Franka
from utils.object_utils import ObjectUtils
from scipy.spatial.transform import Rotation as R
from controllers.grapper_manager import Gripper

import copy
import numpy as np
import torch

# create the world
world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")

stage = omni.usd.get_context().get_stage()

my_franka = Franka(position=np.array((-1.25, -1.0, -1.11)))
    
usd_path = "/home/ubuntu/IL_my/assert/chemistry_lab/table4.usd"
add_reference_to_stage(usd_path=usd_path, prim_path="/World/lab")

# Add a camera to the scene
camera = Camera(
    prim_path="/World/Camera",
    translation=np.array([1.9, -0, 2]),
    frequency=20,
    resolution=(256, 256),
    orientation=np.array([0.61237, -0.45266, 0.54279, 0.42862])
)

camera.set_focal_length(2)
camera.set_local_pose(orientation=np.array([0.61237, 0.35355, 0.34281, 0.61845]), camera_axes="usd")
world.scene.add(camera)

object_utils = ObjectUtils(stage)
reset_needed = False
frame_idx = 0

# controller
pick_controller = PickController(
    name="pick_controller",
    cspace_controller=RMPFlowController(name="target_follower_controller", robot_articulation=my_franka),
    gripper=my_franka.gripper,
)
gripper_control = Gripper()

world.reset()
my_franka.initialize()
articulation_controller = my_franka.get_articulation_controller()

# left_contact_sensor, right_contact_sensor = my_franka.get_contact_sensor()

while simulation_app.is_running():
    world.step(render=True)
    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if world.is_playing():
        if reset_needed:
            world.reset(soft=True)
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
        
        if not pick_controller.is_done():
            action = pick_controller.forward(
                picking_position=object_position,
                current_joint_positions=joint_positions,
                object_size=object_size,
                object_name="beaker",
                gripper_control=gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )

        print(action)
        if action is not None:
            print("action", action)
            articulation_controller.apply_action(action)

        # last_action = action
        # if pick_controller.is_done():
        #     action = last_action

simulation_app.close()
