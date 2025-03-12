import os
from isaacsim import SimulationApp

# 启动 SimulationApp
simulation_app = SimulationApp({"headless": False})  # headless=False 表示可视化模式

# 基础库导入
from pxr import Usd, UsdGeom
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World
import omni.usd

from robots.franka import Franka
import numpy as np 

# 粒子系统
physx_interface = omni.physx.get_physx_interface()
physx_interface.overwrite_gpu_setting(1)

# 创建模拟世界
world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")

# 指定场景 USD 文件路径
usd_path1 = "/home/ubuntu/IL_my/assert/chemistry_lab/table_test.usd"
# usd_path2 = "/home/ubuntu/IL_my/assert/chemistry_lab/table.usd"

print(usd_path1)
# 将场景引用到 /World 下
add_reference_to_stage(usd_path=usd_path1, prim_path="/World/lab")
add_reference_to_stage(usd_path=usd_path1, prim_path="/World")

# add_reference_to_stage(usd_path=usd_path2, prim_path="/World")

# add franka
my_franka = Franka(position=np.array((-0.004, -0, 0.0071)))

# 打印确认加载的信息
# print(f"Loaded scene: {usd_path} into /World")

# 转到主循环，允许用户查看加载的场景
while simulation_app.is_running():
    world.step(render=True)
