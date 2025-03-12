import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

def joint_positions_to_action(joint_positions: np.ndarray) -> ArticulationAction:
    """
    将关节位置转换为机器人可接受的动作格式
    
    Args:
        joint_positions (np.ndarray): 预测的关节位置数组
        
    Returns:
        ArticulationAction: 格式化的机器人动作
    """
    # 确保输入是numpy数组
    if not isinstance(joint_positions, np.ndarray):
        joint_positions = np.array(joint_positions)
        
    # 创建与关节数量相同的空动作数组
    action_size = joint_positions.shape[0] + 1
    target_joint_positions = [None] * action_size
    
    # 设置关节位置
    for i in range(action_size-1):
        target_joint_positions[i] = joint_positions[i]
    target_joint_positions[-1] = joint_positions[-1]
    
    return ArticulationAction(joint_positions=target_joint_positions)
