# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 用于操作系统交互
from isaaclab.app import AppLauncher  # 从isaaclab包中导入AppLauncher类

# 添加命令行参数
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")  # 任务名称
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")  # 远程操作设备
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")  # 导出路径
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")  # 环境步骤频率
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")  # 演示次数
parser.add_argument("--num_success_steps", type=int, default=10, help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.")  # 记录成功的步骤数量

# 这是一个关于手持追踪设备可选参数的代码
AppLauncher.add_app_launcher_args(parser)  # 向解析器添加AppLauncher的特定参数
args_cli = parser.parse_args()  # 解析命令行参数

# 处理手持追踪设备的路径
if args_cli.teleop_device.lower() == "handtracking":
    vars(args_cli)["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'

# 启动模拟器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 剩下的代码如下
import contextlib  # 一个上下文本地化工具库
import gymnasium as gym  # 一个广泛使用的RL环境库
import time  # 时间处理库
import torch  # 一个广泛使用的深度学习库
import omni.log  # 用于Isaac Lab Log记录

# 导入远程操作和环境相关的模块
from isaaclab.devices import Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import ViewportCameraController

import isaaclab_tasks  # 导入isaaclab任务模块
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# 定义RateLimiter类帮助控制循环频率
class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """初始化RateLimiter类"""
        self.hz = hz  # 频率值
        self.last_time = time.time()  # 记录上一次时间
        self.sleep_duration = 1.0 / hz  # 计算休眠持续时间
        self.render_period = min(0.033, self.sleep_duration)  # 渲染时间周期

    def sleep(self, env):
        """在规定的频率下尝试休眠"""
        next_wakeup_time = self.last_time + self.sleep_duration  # 预计下一次唤醒时间
        while time.time() < next_wakeup_time:  # 在实际时间小于预计唤醒时间前，循环
            time.sleep(self.render_period)  # 休眠渲染周期时长
            env.sim.render()  # 渲染环境的模拟

        self.last_time = self.last_time + self.sleep_duration  # 更新上次时间

        # 检测时间跳跃（如循环过慢）
        if self.last_time < time.time():  
            while self.last_time < time.time():
                self.last_time += self.sleep_duration  # 时间步进

# 一个用于处理动作和夹紧器控制的函数
def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # 根据环境计算动作
    if "Reach" in args_cli.task:  # 如果任务是Reach
        return delta_pose
    else:  # 否则为多自由度机器人控制
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        return torch.concat([delta_pose, gripper_vel], dim=1)

def main():
    """从环境中收集演示数据，使用远程操作接口。"""

    # 如果选择了手持追踪设备，通过OpenXR实现速率限制
    if args_cli.teleop_device.lower() == "handtracking":
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # 从CLI参数中获得目录路径和文件名
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # 如果目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分析环境配置
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # 从环境中提取成功检验函数，以便在主循环中调用
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # 修改配置以便于环境无限运行直到达成目标或到达其他终止条件
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    # 配置记录器
    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 添加重置当前记录实例的遥控按键
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # 创建控制器
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.2, rot_sensitivity=0.5)
    elif args_cli.teleop_device.lower() == "handtracking":
        from isaacsim.xr.openxr import OpenXRSpec

        teleop_interface = Se3HandTracking(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True)
        teleop_interface.add_callback("RESET", reset_recording_instance)
        viewer = ViewerCfg(eye=(-0.25, -0.3, 0.5), lookat=(0.6, 0, 0), asset_name="viewer")
        ViewportCameraController(env, viewer)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'handtracking'."
        )

    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # 开始前重置
    env.reset()
    teleop_interface.reset()

    # 模拟环境 -- 在推理模式下运行所有东西
    current_recorded_demo_count = 0
    success_step_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # 获取键盘命令
            delta_pose, gripper_command = teleop_interface.advance()
            # 转换为torch张量
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # 计算动作
            actions = pre_process_actions(delta_pose, gripper_command)

            # 在环境中执行动作
            env.step(actions)

            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0

            if should_reset_recording_instance:
                env.recorder_manager.reset()
                env.reset()
                should_reset_recording_instance = False
                success_step_count = 0

            # 如果当前记录的demo计数变了则打印出来
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # 检查模拟是否停止
            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()

if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
