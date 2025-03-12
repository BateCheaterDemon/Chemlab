import os
import numpy as np
import h5py
import torch
import copy
from typing import Dict, Optional, List, Tuple
import glob
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class RobotImageDataset(BaseImageDataset):
    def __init__(self, 
                 shape_meta,
                 dataset_path: str,
                 horizon: int = None,
                 pad_before: int = None,
                 pad_after: int = None,
                 n_obs_steps: int = None,
                 n_latency_steps: int = None,
                 use_cache: bool = True,
                 seed: int = 42,
                 val_ratio: float = 0.00,
                 max_train_episodes: Optional[int] = None,
                 delta_action: bool = False):
        """
        机器人图像数据集加载器
        """
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.shape_meta = shape_meta
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.use_cache = use_cache
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.delta_action = delta_action
        
        # 加载所有数据文件
        self.episode_map = []
        
        # 扫描数据集中的h5文件
        h5_path = os.path.join(dataset_path, "episode_data.hdf5")
        self.h5_file = h5py.File(h5_path, 'r')
        
        # 记录每个episode的信息
        for episode_name in self.h5_file.keys():
            n_frames = self.h5_file[episode_name]['actions'].shape[0]
            self.episode_map.append((
                episode_name,
                n_frames
            ))
        
        # 计算可用的序列数量
        self.sequences = []
        for episode_name, n_frames in self.episode_map:
            total_steps = n_frames
            if self.horizon is not None and self.n_obs_steps is not None:
                # 确保有足够的帧来包含观测和动作
                total_steps = n_frames - (self.horizon + self.n_obs_steps) + 1
            for start_idx in range(total_steps):
                self.sequences.append((episode_name, start_idx))
                
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        
        # 使用 SingleFieldLinearNormalizer 处理动作
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.get_all_actions().numpy())
        
        # 为机器人端点位姿添加归一化器
        all_poses = []
        for episode_name, _ in self.episode_map:
            episode = self.h5_file[episode_name]
            poses = episode['agent_pose'][:].astype(np.float32)
            all_poses.append(poses)
        all_poses = np.concatenate(all_poses, axis=0)
        normalizer['agent_pose'] = SingleFieldLinearNormalizer.create_fit(all_poses)
        
        # 为图像添加范围归一化 [0,1]
        normalizer['camera_1'] = get_image_range_normalizer()
        normalizer['camera_2'] = get_image_range_normalizer()
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode_name, _ in self.episode_map:
            episode = self.h5_file[episode_name]
            actions = torch.from_numpy(episode['actions'][:].astype(np.float32))
            all_actions.append(actions)
        return torch.cat(all_actions, dim=0)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_name, start_idx = self.sequences[idx]
        episode = self.h5_file[episode_name]
        
        # 计算obs和action的起止索引
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps
        action_start_idx = obs_end_idx
        action_end_idx = action_start_idx + self.horizon
        
        # 读取观测数据
        cam1_obs = episode['camera_1'][obs_start_idx:obs_end_idx]  # [T,3,H,W]
        cam3_obs = episode['camera_2'][obs_start_idx:obs_end_idx]  # [T,3,H,W]
        robot_eef_obs = episode['agent_pose'][obs_start_idx:obs_end_idx]  # [T,2]

        if cam1_obs.shape[1] == 1:
            cam1_obs = np.repeat(cam1_obs, 3, axis=1)
        if cam3_obs.shape[1] == 1:
            cam3_obs = np.repeat(cam3_obs, 3, axis=1)
            
        # 读取动作数据
        actions = episode['actions'][action_start_idx:action_end_idx]  # [T,2]
        
        # 转换为tensor并归一化
        cam1_obs = torch.from_numpy(cam1_obs).float() / 255.0
        cam3_obs = torch.from_numpy(cam3_obs).float() / 255.0
        robot_eef_obs = torch.from_numpy(robot_eef_obs).float()
        actions = torch.from_numpy(actions).float()
        
        # 返回符合shape_meta格式的数据
        return {
            'obs': {
                'camera_1': cam1_obs,
                'camera_2': cam3_obs,
                'agent_pose': robot_eef_obs,
            },
            'action': actions,
        }

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
    
    @staticmethod
    def collate_fn(batch):
        """自定义数据打包函数"""
        cam1_images = torch.stack([item['obs']['camera_1'] for item in batch])
        cam3_images = torch.stack([item['obs']['camera_2'] for item in batch])
        robot_eef_pose = torch.stack([item['obs']['agent_pose'] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        
        return {
            'obs': {
                'camera_1': cam1_images,        # [B,T,3,H,W]
                'camera_2': cam3_images,        # [B,T,3,H,W]
                'agent_pose': robot_eef_pose, # [B,T,7]
            },
            'action': actions,                  # [B,T,7]
        }
