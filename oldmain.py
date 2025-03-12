import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
# import omni.isaac.core.utils.numpy.rotations as rot_utils
# from omni.isaac.core.articulations import ArticulationView
# from omni.isaac.core.objects import DynamicSphere
# from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
# from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
# from omni.isaac.nucleus import get_assets_root_path
# from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
# from omni.isaac.core.utils.string import find_unique_string_name
# from omni.isaac.core.utils.prims import is_prim_path_valid
# from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
# from omni.isaac.core.utils.stage import get_stage_units
from isaacsim.sensors.camera import Camera
from pxr import Usd, UsdGeom, Gf
import omni.usd
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api import PhysicsContext

from isaacsim.core.api.controllers.articulation_controller import ArticulationController

# User import
from controllers.controller_manager import ControllerManager
from robots.franka import Franka


import copy
import numpy as np
import torch

# create the world

# world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="torch", device="cuda")
world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
# world = World(stage_units_in_meters = 1, device = "cpu")
# physx_interface = omni.physx.get_physx_interface()
# physx_interface.overwrite_gpu_setting(1)

stage = omni.usd.get_context().get_stage()
# print("stage", stage)
# print("type(stage)", type(stage))

my_franka = Franka(position=np.array((-1.25, -1.0, -1.11)))

# my_franka = Franka(position=np.array((-12.5, -10, -11.1)))
# prim = stage.GetPrimAtPath("/World/Franka")
# if prim.IsValid():
#     scale_attr = prim.GetAttribute("xformOp:scale")
#     current_scale = scale_attr.Get()
#     new_scale = (current_scale[0] * 10, current_scale[1] * 10, current_scale[2] * 10)
#     scale_attr.Set(new_scale)
    
usd_path = "/home/ubuntu/IL_my/assert/chemistry_lab/chemistry_lab_addScale.usd"
# relative_path = 'asserts/chemistry_lab/chemistry_lab_addScale.usd' # 相对路径好像在这个版本有问题
# relative_path = "asserts/chemistry_lab/chemistry_lab.usd"
add_reference_to_stage(usd_path=usd_path, prim_path="/World/lab")

# Add a camera to the scene
camera = Camera(
    prim_path="/World/Camera",
    translation=np.array([0.97, -0.93, -0.06]),
    frequency=20,
    resolution=(256, 256),
    orientation=np.array([0.56282, -0.45266, 0.54279, 0.42862])
)
camera.set_focal_length(2)
camera.set_local_pose(orientation=np.array([-0.60563, -0.39358, -0.34952, -0.56677]), camera_axes="usd")
world.scene.add(camera)

world.reset()
my_franka.initialize()
articulation_controller = my_franka.get_articulation_controller()

reset_needed = False
frame_idx = 0

# controller
control_manager = ControllerManager(stage, my_franka, articulation_controller)
# action_list = ['pick beaker(id:1)', 'place beaker(id:1) laboratory_bench(id:7)', 'pick HCl_beaker(id:5)', 'pour HCl_beaker(id:5) beaker(id:1)', 'place HCl_beaker(id:5) laboratory_bench(id:7)', 'pick glass_rod(id:4)', 'place glass_rod(id:4) laboratory_bench(id:7)', 'pick pH_reagent_beaker(id:3)', 'place pH_reagent_beaker(id:3) laboratory_bench(id:7)']
# action_list = ['pick HCl_beaker(id:5)', 'pour HCl_beaker(id:5) beaker(id:1)', 'place HCl_beaker(id:5) laboratory_bench(id:7)']
action_list = ['pick glass_rod(id:4)', 'stir beaker(id:1)', 'place glass_rod(id:4) laboratory_bench(id:7)']

control_manager.load_actions(action_list)

left_contact_sensor, right_contact_sensor = my_franka.get_contact_sensor()

while simulation_app.is_running():
    world.step(render=True)
    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if world.is_playing():
        if reset_needed:
            world.reset(soft=True)
            my_franka.initialize()
            control_manager.reset()
            control_manager.load_actions(action_list)
            reset_needed = False
            frame_idx = 0
        frame_idx += 1

        if frame_idx < 3: 
            continue
        
        if not control_manager.all_tasks_done():
            actions = control_manager.step()
            if actions != None:
                # print(actions)
                articulation_controller.apply_action(actions) 

simulation_app.close()
