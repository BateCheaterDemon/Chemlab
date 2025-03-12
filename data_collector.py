import os
import numpy as np
import cv2
from datetime import datetime
import json
import h5py

class DataCollector:
    def __init__(self, save_dir="policy_data", max_episodes=10):
        """初始化数据采集器
        
        Args:
            save_dir (str): 数据保存的根目录
            max_episodes (int): 最大录制集数
        """
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(save_dir, self.timestamp)
        self.episode_count = 0
        
        # 创建保存目录
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 创建单个HDF5文件
        self.h5_path = os.path.join(self.session_dir, "episode_data.hdf5")
        self.h5_file = h5py.File(self.h5_path, 'w', libver='latest', swmr=True)

        
        # 创建数据组
        self.current_episode = None
        self.new_episode()
        
        # 添加临时存储列表
        self.temp_camera_1 = []
        self.temp_camera_2 = []
        self.temp_agent_pose = []
        self.temp_actions = []
    
    def new_episode(self):
        """开始新的数据采集episode"""
        if self.episode_count >= self.max_episodes:
            self.close()
            return
            
        episode_name = f"episode_{self.episode_count:04d}"
        self.current_episode = self.h5_file.create_group(episode_name)
        
        # 清空临时存储
        self.temp_camera_1 = []
        self.temp_camera_2 = []
        self.temp_agent_pose = []
        self.temp_actions = []
        
    def cache_step(self, rgb_image1, rgb_image2, joint_angles):
        """缓存每一步的数据
        
        Args:
            rgb_image1 (np.ndarray): 第一个相机的RGB图像
            rgb_image2 (np.ndarray): 第二个相机的RGB图像
            joint_angles (np.ndarray): 机器人关节角度
        """
        self.temp_camera_1.append(rgb_image1)
        self.temp_camera_2.append(rgb_image2)
        self.temp_agent_pose.append(joint_angles)
        
    def write_cached_data(self, final_joint_positions):
        """将缓存的数据写入文件"""
        # 将最后一步的joint_positions作为倒数第二步的action
        self.temp_actions = self.temp_agent_pose[1:] + [final_joint_positions]
        
        # 将列表转换为numpy数组
        camera_1_data = np.array(self.temp_camera_1)
        camera_2_data = np.array(self.temp_camera_2)
        agent_pose_data = np.array(self.temp_agent_pose)
        actions_data = np.array(self.temp_actions)
        
        # 直接创建数据集并写入数据
        self.current_episode.create_dataset("camera_1", data=camera_1_data, dtype='uint8')
        self.current_episode.create_dataset("camera_2", data=camera_2_data, dtype='uint8')
        self.current_episode.create_dataset("agent_pose", data=agent_pose_data, dtype='float32')
        self.current_episode.create_dataset("actions", data=actions_data, dtype='float32')
        
        # 清空缓存
        self.temp_camera_1 = []
        self.temp_camera_2 = []
        self.temp_agent_pose = []
        self.temp_actions = []
        
        # 增加episode计数并开始新的episode
        self.episode_count += 1
        self.new_episode()

    def close(self):
        """关闭数据文件"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None