from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.stage import get_stage_units
import numpy as np

class ObjectUtils:
    def __init__(self, stage, default_path="/World"):
        self._stage = stage
        self._default_path = default_path
        self.pick_height_offsets = {
            "rod": 0.04,
            "tube": 0.01,
            "beaker": 0.0,
            "erlenmeyer flask": 0.018,
            "cylinder": 0.0,
            "petri dish": 0.005,
            "pipette": 0.008,
            "microscope slide": 0.002
        }

    def _get_object_path(self, object_name: str = None, obj_path: str = None) -> str:
        """获取物体的完整路径"""
        if obj_path is not None:
            return obj_path
        if object_name is not None:
            return f'{self._default_path}/{object_name}'
        raise ValueError("Either object_name or obj_path must be provided")

    def get_object_position(self, object_name: str = None, obj_path: str = None) -> np.ndarray:
        """获取物体的中心位置"""
        path = self._get_object_path(object_name, obj_path)
        obj_prim = self._stage.GetPrimAtPath(path)
        if not obj_prim.IsValid():
            return None
            
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(obj_prim)
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        return (min_point + max_point) / 2.0

    def get_pick_position(self, object_name: str = None, obj_path: str = None) -> np.ndarray:
        """获取物体的抓取位置(考虑高度偏移)"""
        position = self.get_object_position(object_name, obj_path)
        if position is None:
            return None
            
        object_name = object_name or obj_path.split('/')[-1]
        for key, offset in self.pick_height_offsets.items():
            if key in object_name.lower():
                position[2] += offset / get_stage_units()
                return position
                
        position[2] += 0.02 / get_stage_units()
        return position

    def get_object_size(self, object_name: str = None, obj_path: str = None) -> np.ndarray:
        """获取物体的尺寸"""
        path = self._get_object_path(object_name, obj_path)
        obj_prim = self._stage.GetPrimAtPath(path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(obj_prim)
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        return max_point - min_point

    def get_object_xform_position(self, obj_path: str) -> np.ndarray:
        """
        使用USD API直接获取物体的世界坐标位置
        Args:
            obj_path: 物体的路径
        Returns:
            np.ndarray: 物体的世界坐标位置，如果物体不存在则返回None
        """
        obj = self._stage.GetPrimAtPath(obj_path)
        if obj.IsValid():
            xformable = UsdGeom.Xformable(obj)
            # 获取世界变换矩阵
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            # 获取位置（矩阵的第四列前三个元素）
            position = world_transform.ExtractTranslation()
            return np.array([position[0], position[1], position[2]])
        else:
            print(f"Object at path {obj_path} not found.")
            return None

    def set_object_position(self, obj_path, position, local_position=None, position_offset=None):
        """设置对象的位置"""
        obj = self._stage.GetPrimAtPath(obj_path)
        if obj.IsValid():
            xformable = UsdGeom.Xformable(obj)
            xform_ops = xformable.GetOrderedXformOps()
            
            if local_position is not None and position_offset is not None:
                # 使用局部坐标和偏移量设置位置
                new_position = Gf.Vec3d(local_position[0] + position_offset[0], local_position[1] + position_offset[1], local_position[2] + position_offset[2])
                if xform_ops:
                    translate_op = xform_ops[0]
                    translate_op.Set(new_position)
                else:
                    xformable.AddTranslateOp().Set(new_position)
            else:
                # 直接设置世界坐标位置
                new_position = Gf.Vec3d(position[0], position[1], position[2])
                if xform_ops:
                    translate_op = xform_ops[0]
                    translate_op.Set(new_position)
                else:
                    xformable.AddTranslateOp().Set(new_position)
        else:
            print(f"Object at path {obj_path} not found.")