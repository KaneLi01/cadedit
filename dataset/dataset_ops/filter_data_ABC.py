import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import combinations


import utils.cadlib.Brep_utils as Brep_utils
from utils.vis import render_cad
from shape_info import Shapeinfo
from utils.cadlib.curves import *
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape
from utils.cadlib.Brep_utils import Point
from dataclasses import dataclass, asdict
from typing import Optional, List
import argparse


def append_feature_to_jsonl(feature, jsonl_path):
    """将 CADFeature 实例追加写入到 jsonl 文件"""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(asdict(feature)) + "\n")


@dataclass
class CADFeature:
    name: str
    valid: bool = False                  # step文件存在，且能正确读取
    child_num: Optional[int] = None 
    face_num: Optional[List[int]] = None    
    wire_num: Optional[List[int]] = None
    bbox_min_max: Optional[List[List[float]]] = None
    bbox_center: Optional[List[List[float]]] = None



class Shape_Info_From_Step():
    '''从Json文件中读取Brep的基础序列信息'''
    def __init__(self, step_path):
        self.step_path = step_path
        self.cad_name = self.step_path.split('/')[-1].split('_')[0]
        try:
            self.shape = copy.deepcopy(Brep_utils.get_BRep_from_step(self.step_path))
            self.child_num = self.shape.NbChildren()
            if self.child_num == 1:
                wire_num = [len(Brep_utils.get_wires(self.shape))]
                face_num = [len(Brep_utils.get_faces(self.shape))]
                bbox = Brep_utils.get_bbox(self.shape)
                min_max_pt = [[bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z]]
                center = [[bbox.center.x, bbox.center.y, bbox.center.z]]
            elif self.child_num > 1:
                sub_shapes = Brep_utils.get_first_level_shapes(self.shape)
                wire_num = [len(Brep_utils.get_wires(sub_shape)) for sub_shape in sub_shapes]
                face_num = [len(Brep_utils.get_faces(sub_shape)) for sub_shape in sub_shapes]
                bboxs = [Brep_utils.get_bbox(sub_shape) for sub_shape in sub_shapes]
                min_max_pt = [[bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z] for bbox in bboxs]
                center = [[bbox.center.x, bbox.center.y, bbox.center.z] for bbox in bboxs]

            self.shape_info = CADFeature(
                name=self.cad_name,
                valid=True,
                child_num=self.child_num,
                face_num=face_num,
                wire_num=wire_num,
                bbox_min_max=min_max_pt,
                bbox_center=center,
            )
        except Exception as e:
            self.shape_info = CADFeature(name=self.cad_name)
            print(f"Error reading STEP file {self.step_path}: {e}")

        


    

# 根据序号创建json文件。json文件创建新的一列。将数据写入json文件中。
        

class Filter_Name_From_Shape(Shape_Info_From_Step):
    '''从shape出发进行筛选等处理'''
    def __init__(self, step_path):
        super().__init__(step_path)

        self.shapes = [Brep_utils.get_BRep_from_seq(sub_seq) for sub_seq in self.sub_seqs]
        bboxs_temp = [Brep_utils.get_bbox(shape) for shape in self.shapes]
        self.centers = [bbox.center for bbox in bboxs_temp]
        self.min_points = [bbox.min for bbox in bboxs_temp]
        self.max_points = [bbox.max for bbox in bboxs_temp]
        self.bboxs = [Brep_utils.create_box_from_minmax(min_pt, max_pt) 
                      for min_pt, max_pt in zip(self.min_points, self.max_points)]
    
    def get_view(self):
        '''利用包围盒计算观测角度，以让add的sketch不会被mask'''
        base_min, base_max = self.min_points[1], self.max_points[1]
        add_min, add_max = self.min_points[2], self.max_points[2]

        base_min = Point(0,0,0)
        base_max = Point(1,1,1)
        add_min = Point(0,0,1)
        add_max = Point(1,1,2)

        if base_max.z - add_min.z <= 1e-5:
            view = 'up' 
        elif base_min.z - add_max.z >= -1e-5:
            view = 'down'
        elif base_max.x - add_min.x <= 1e-5:
            view = 'front'
        elif base_min.x - add_max.x >= -1e-5:
            view = 'back'
        elif base_max.y - add_min.y <= 1e-5:
            view = 'right'
        elif base_min.y - add_max.y >= -1e-5:
            view = 'left'
        else: raise Exception('please check code')

        return view
        

    def filter_complex_faces(self, thre=30):
        '''筛选shape面数是否小于阈值'''
        num_face = len(Brep_utils.get_faces(self.shapes[0]))
        if num_face < thre:
            return True
        else: return False

    def filter_intersection(self, thre=1e-5):
        '''筛选两个body是否不相交'''
        common = BRepAlgoAPI_Common(self.shapes[1], self.shapes[2])
        common_shape = common.Shape()
        common_volume = Brep_utils.get_volume(common_shape)
        if common_volume < thre:
            return True
        else: return False

    def filter_distance(self, thre=1e-5):
        '''筛选两个body最小距离是否极小'''
        min_dis = Brep_utils.get_min_distance(self.shapes[1], self.shapes[2])
        if min_dis < thre:
            return True
        else: return False
    
    def filter_small_thin(self, shortest=0.04, scale=15):
        '''筛选shape和两个body是否不会过小、是否不是薄面或棍'''
        def check_slice(dx, dy, dz, scale):
            for a, b in combinations([dx, dy, dz], 2):
                longer = max(a, b)
                shorter = min(a, b)
                if longer >= scale * shorter:
                    return True
            return False

        for min_p, max_p in zip(self.min_points, self.max_points):
            dx, dy, dz = max_p.x-min_p.x, max_p.y-min_p.y, max_p.z-min_p.z

            if dx < shortest or dy < shortest or dz < shortest:
                return False
            elif check_slice(dx,dy,dz, scale):
                return False
        return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, required=True, help="Path to the STEP file")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the output json file")

    args = parser.parse_args()
    return args


def cad_info2json(args):
    """
    将CAD文件信息写入json文件
    :param step_path: STEP文件路径
    :param json_path: 输出json文件路径
    """
    shape_info = Shape_Info_From_Step(args.step_path).shape_info

    # 将数据写入json文件
    append_feature_to_jsonl(shape_info, args.json_path)


def test():
    step_path = '/home/lkh/siga/dataset/ABC/temp/step/00/00000002/00000002_1ffb81a71e5b402e966b9341_step_001.step'
    Shape_Info_From_Step(step_path)


def main():
    args = get_args()
    cad_info2json(args)

    # test()


if __name__ == "__main__":
    main()