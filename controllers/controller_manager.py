from typing import List
import numpy as np
from .pick_controller import PickController
from .pour_controller import PourController
from .place_controller import PlaceController
from .stir_controller import StirController
from .grapper_manager import Gripper
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.stage import get_stage_units

from scipy.spatial.transform import Rotation as R

class ControllerManager:
    def __init__(self, stage, robot, articulation_controller):
        self.gripper_control = Gripper()
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=RMPFlowController(name="target_follower_controller", robot_articulation=robot),
            gripper=robot.gripper,
            speed=1.5
        )

        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=RMPFlowController(name="pour_cspace_controller", robot_articulation=robot),
            gripper=robot.gripper,
        )
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=RMPFlowController(name="target_follower_controller", robot_articulation=robot),
            gripper=robot.gripper,
            speed=1.5
        )
        
        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=RMPFlowController(name="target_follower_controller", robot_articulation=robot),
        )
        self._robot = robot
        self._stage = stage
        self._articulation_controller = articulation_controller
        self.last_pick_position = None
        self.actions_queue = []
        self.current_action_index = 0
        self.pick_position = {
            "rod": 0.03,
            "tube": 0.01,
            "beaker": 0.1,
            "Erlenmeyer flask": 0.018,
            "cylinder": 0,
            "Petri dish": 0.005,
            "pipette": 0.008,
            "microscope slide": 0.002
        }

    def reset(self):
        self.pick_controller.reset()
        self.pour_controller.reset()
        self.place_controller.reset()
        self.stir_controller.reset()
        self.actions_queue = []
        self.last_pick_position = None
        self.current_action_index = 0

    def _parse_object_info(self, obj_info: str):
        """
        解析对象信息，例如 'pH_paper(id:3)'，返回 ('pH_paper', 3)
        """
        name_part, id_part = obj_info.split('(id:')
        obj_name = name_part
        obj_id = int(id_part.rstrip(')'))
        return obj_name, obj_id
    
    def load_actions(self, new_action_list: List[str]):
        """
        加载新的动作列表，清除现有的动作队列。
        """
        # self.reset()
        self.parse_actions(new_action_list)
        
    def all_tasks_done(self) -> bool:
        """
        返回是否所有任务都已经执行完成。
        """
        return self.current_action_index >= len(self.actions_queue)


    def parse_actions(self, action_list: List[str]):
        """
        解析动作列表，处理逻辑以移除无效的 pick/place 和 pour 操作。
        """
        parsed_actions = []
        last_pick_index = -1  # 记录上一次 pick 的索引
        i = 0

        while i < len(action_list):
            action_str = action_list[i]
            parts = action_str.split()
            action_type = parts[0]

            if action_type == 'pick':
                # 解析 pick 操作
                obj_info = parts[1]
                obj_name, obj_id = self._parse_object_info(obj_info)

                # 检查下一个动作是否是 place 且针对同一对象
                if i + 1 < len(action_list):
                    next_action = action_list[i + 1]
                    next_parts = next_action.split()
                    if next_parts[0] == 'place':
                        next_obj_info = next_parts[1]
                        next_obj_name, next_obj_id = self._parse_object_info(next_obj_info)

                        # 如果 pick 和 place 针对同一对象，则跳过这两个动作
                        if obj_name == next_obj_name and obj_id == next_obj_id:
                            i += 2  # 跳过 pick 和 place
                            continue

                # 如果没有被跳过，则记录 pick 动作
                parsed_actions.append({
                    'type': 'pick',
                    'object_name': obj_name,
                    'object_id': obj_id
                })
                last_pick_index = len(parsed_actions) - 1  # 更新最后一个 pick 的索引
                i += 1

            elif action_type == 'place':
                # 解析 place 操作
                obj_info = parts[1]
                target_info = parts[2]
                obj_name, obj_id = self._parse_object_info(obj_info)
                target_name, target_id = self._parse_object_info(target_info)

                parsed_actions.append({
                    'type': 'place',
                    'object_name': obj_name,
                    'object_id': obj_id,
                    'target_name': target_name,
                    'target_id': target_id
                })
                i += 1

            elif action_type == 'pour':
                # 解析 pour 操作
                source_info = parts[1]
                target_info = parts[2]
                source_name, source_id = self._parse_object_info(source_info)
                target_name, target_id = self._parse_object_info(target_info)

                # 检查 pour 的 source 是否是最后一个 pick 的对象
                if last_pick_index == -1 or parsed_actions[last_pick_index]['object_name'] != source_name or \
                        parsed_actions[last_pick_index]['object_id'] != source_id:
                    # 如果不匹配，则修改上一个 pick 的对象为 pour 的 source
                    if last_pick_index != -1:
                        parsed_actions[last_pick_index]['object_name'] = source_name
                        parsed_actions[last_pick_index]['object_id'] = source_id

                # 添加 pour 操作
                parsed_actions.append({
                    'type': 'pour',
                    'source_name': source_name,
                    'source_id': source_id,
                    'target_name': target_name,
                    'target_id': target_id
                })
                i += 1

            elif action_type == "stir":
                # 解析 stir 操作
                obj_info = parts[1]
                obj_name, obj_id = self._parse_object_info(obj_info)
                parsed_actions.append({
                    'type': 'stir',
                    'object_name': obj_name,
                    'object_id': obj_id
                })
                i += 1

            else:
                print(f"未知的动作类型: {action_type}")
                i += 1

        self.actions_queue = parsed_actions
        self.current_action_index = 0


    # def parse_actions(self, action_list: List[str]):
    #     """
    #     解析大模型给出的动作字符串列表，并存储为内部队列。
    #     每个动作被解析为一个字典，包含动作类型和相关参数。
    #     """
    #     parsed_actions = []
    #     for action_str in action_list:
    #         parts = action_str.split()
    #         action_type = parts[0]
    #         if action_type == 'pick':
    #             # 示例: 'pick pH_paper(id:3)'
    #             obj_info = parts[1]
    #             obj_name, obj_id = self._parse_object_info(obj_info)
    #             parsed_actions.append({
    #                 'type': 'pick',
    #                 'object_name': obj_name,
    #                 'object_id': obj_id
    #             })
    #         elif action_type == 'place':
    #             # 示例: 'place pH_paper(id:3) laboratory_bench(id:7)'
    #             obj_info = parts[1]
    #             target_info = parts[2]
    #             obj_name, obj_id = self._parse_object_info(obj_info)
    #             target_name, target_id = self._parse_object_info(target_info)
    #             parsed_actions.append({
    #                 'type': 'place',
    #                 'object_name': obj_name,
    #                 'object_id': obj_id,
    #                 'target_name': target_name,
    #                 'target_id': target_id
    #             })
    #         elif action_type == 'pour':
    #             # 示例: 'pour HCl_beaker(id:5) clean_beaker(id:1)'
    #             source_info = parts[1]
    #             target_info = parts[2]
    #             source_name, source_id = self._parse_object_info(source_info)
    #             target_name, target_id = self._parse_object_info(target_info)
    #             parsed_actions.append({
    #                 'type': 'pour',
    #                 'source_name': source_name,
    #                 'source_id': source_id,
    #                 'target_name': target_name,
    #                 'target_id': target_id
    #             })
    #         elif action_type == "stir":
    #             obj_info = parts[1]
    #             obj_name, obj_id = self._parse_object_info(obj_info)
    #             parsed_actions.append({
    #                 'type': 'stir',
    #                 'object_name': obj_name,
    #                 'object_id': obj_id
    #             })
    #         else:
    #             print(f"未知的动作类型: {action_type}")
    #     self.actions_queue = parsed_actions
    #     self.current_action_index = 0
    
    def step(self):
        """
        执行当前任务，并根据控制器的状态决定是否移动到下一个任务。
        """
        if self.current_action_index >= len(self.actions_queue):
            print("All tasks are done!")
            return  # 所有任务已完成

        current_action = self.actions_queue[self.current_action_index]
        action_type = current_action['type']
        actions = None
        # print("action type: ", action_type)
        if action_type == 'pick':
            if self.pick_controller.is_done():
                self.current_action_index += 1
                self.pick_controller.reset()
            else:
                actions = self.pick(
                    object_name=current_action['object_name']
                )
        elif action_type == 'place':
            if self.place_controller.is_done():
                self.current_action_index += 1
                self.last_pick_position = None
                self.place_controller.reset()
            else:
                actions = self.place()
                
        elif action_type == 'pour':
            if self.pour_controller.is_done():
                self.current_action_index += 1
                self.pour_controller.reset()
            else:
                actions = self.pour(current_action['source_name'], current_action['target_name'])
        elif action_type == "stir":

            if self.stir_controller.is_done():
                self.current_action_index += 1
                self.stir_controller.reset()
            else:
                actions = self.stir(object_name=current_action['object_name'])
        elif action_type == "observe":
            #TODO 观测动作
            self.current_action_index += 1
        else:
            print(f"未知的动作类型: {action_type}")
            
        self.gripper_control.update_grasped_object_position()
        return actions

    def pick(self, object_name: str):
        position = self.calculate_pick_position(object_name)
        size = self.calculate_size(object_name)
        if self.last_pick_position is None:
            self.last_pick_position = np.array(position)
        actions = self.pick_controller.forward(
            picking_position=position,
            current_joint_positions=np.array(self._robot.get_joints_state().positions),
            object_size=size,
            object_name=object_name,
            end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            gripper_control=self.gripper_control
        )
        return actions

    def pour(self, object_name, target_name):
        box_size = self.calculate_size(object_name)
        position = self.calculate_position(target_name)
        if position is None:
             return None
        position[2] += 0.3 / get_stage_units()
        position[0] += 0.01 / get_stage_units()
        position[1] -= box_size[2] * 1 / 4
         
        actions = self.pour_controller.forward(
            franka_art_controller=self._articulation_controller,
            target_position = position,
            current_joint_positions=np.array(self._robot.get_joints_state().positions),
            current_joint_velocities=np.array(self._robot.get_joints_state().velocities)
        )
        return actions
    
    def stir(self, object_name):
        position = self.calculate_position(object_name)
        if position is None:
             return None
        actions = self.stir_controller.forward(
            center_position=position,
            current_joint_positions=np.array(self._robot.get_joints_state().positions),
            end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
        )
        return actions
        
    def place(self):
        actions = self.place_controller.forward(
            self.last_pick_position, 
            np.array(self._robot.get_joints_state().positions),
            end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            gripper_control=self.gripper_control
        )
        return actions
    
    def calculate_position(self, object_name: str):
        # print("Calculating position for object: ", object_name)
        obj_path = '/World/lab/Desk1/' + object_name
        obj_prim = self._stage.GetPrimAtPath(obj_path)
        if obj_prim.IsValid():
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(obj_prim)
            min_point = bbox.GetRange().GetMin()
            max_point = bbox.GetRange().GetMax()
            position = (min_point + max_point) / 2.0
            return position
        else:
            return None
    
    def calculate_pick_position(self, object_name: str):
        position = self.calculate_position(object_name)
        for key in self.pick_position:
            if key in object_name.lower():
                position[2] += self.pick_position[key]
                return position
        position[2] += 0.02
        return position

    def calculate_size(self, object_name: str):
        obj_path = '/World/lab/Desk1/' + object_name
        obj_prim = self._stage.GetPrimAtPath(obj_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(obj_prim)
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        return max_point - min_point