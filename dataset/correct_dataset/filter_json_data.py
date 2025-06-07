import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import combinations
import utils.cadlib.Brep_utils as Brep_utils
import utils.cadlib.Brep_utils_test as Brep_utils_t
# from utils.vis.show_single import 
from utils.vis import show_single
from shape_info import Shapeinfo
from utils.cadlib.curves import *
from OCC.Core.gp import gp_Pnt, gp_Dir


def check_slice(dx, dy, dz, scale):
    for a, b in combinations([dx, dy, dz], 2):
        longer = max(a, b)
        shorter = min(a, b)
        if longer >= scale * shorter:
            return True
    return False

class Shape_Info_From_Json():
    def __init__(self, cad_json_path):
        self.cad_json_path = cad_json_path
        self.cad_name = self.cad_json_path.split('/')[-1].split('.')[0]
        self.cad_seq = Brep_utils.get_seq_from_json(self.cad_json_path)
        self.cad_body_len = len(self.cad_seq.seq)
        if self.cad_body_len > 1:
            self.sub_seqs = [copy.deepcopy(self.cad_seq.seq), copy.deepcopy(self.cad_seq.seq[:-1]), copy.deepcopy(self.cad_seq.seq[-1:])]            
        else: 
            self.sub_seqs = [self.cad_seq.seq]

        # 要注意这里可能会影响子类的shapes
        # self.last_seq = self.sub_seqs[-1][0]
        # self.profile = copy.deepcopy(self.last_seq.profile)
        # self.loop = self.profile.children[0]
        # self.curve = self.loop.children[0]
        
        # # raise Exception('-')
        # self.shape = Brep_utils_t.get_BRep_from_seq(self.sub_seqs[-1])
        # self.normal = gp_Dir(*self.last_seq.sketch_plane.normal)
        
        # self.ws = Brep_utils_t.get_wireframe_cir(self.shape)


class Filter_Name_From_Seq(Shape_Info_From_Json):
    def __init__(self, cad_json_path):
        super().__init__(cad_json_path)

    def filter_2body(self):
        if self.cad_body_len == 2:
            return self.cad_name
        else: return None

    def filter_union(self):
        if self.last_seq.operation == 0 or self.last_seq.operation == 1:
            return self.cad_name
        else: return None


class Filter_Name_From_Shape(Shape_Info_From_Json):
    def __init__(self, cad_json_path):
        super().__init__(cad_json_path)
        self.shapes = [Brep_utils.get_BRep_from_seq(copy.copy(sub_seq)) for sub_seq in self.sub_seqs]
        show_single.show_BRep(self.shapes[0])
        show_single.show_BRep(self.shapes[1])
        show_single.show_BRep(self.shapes[2])
        


# json -> brep shape 这一步进行数据集的筛选
class CorrectDataFromJSON():
    def __init__(self, cad_json_path):
        self.cad_json_path = cad_json_path
        self.cad_name = self.cad_json_path.split('/')[-1].split('.')[0]
        self.cad_seq = Brep_utils.get_seq_from_json(self.cad_json_path)
        self.cad_body_len = len(self.cad_seq.seq)
        if self.cad_body_len > 1:
            self.sub_seqs = [copy.deepcopy(self.cad_seq.seq), copy.deepcopy(self.cad_seq.seq[:-1]), copy.deepcopy(self.cad_seq.seq[-1:])]
            self.shapes = [Brep_utils.get_BRep_from_seq(sub_seq) for sub_seq in self.sub_seqs]
        else: 
            self.sub_seq = self.cad_seq.seq
            self.shape = Brep_utils.get_BRep_from_seq(self.sub_seq)
        


    def filter_slice(self, shortest=0.04, scale=15):
        if self.cad_body_len > 1:
        # 筛选薄面、很小的对象
            for shape in self.shapes:
                xmin, ymin, zmin, xmax, ymax, zmax = Brep_utils.get_bbox(shape)
                dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
                if dx < shortest or dy < shortest or dz < shortest:
                    # 去掉很短的边
                    return False
                elif check_slice(dx,dy,dz, scale):
                    # 去掉薄边
                    return False
            return True
        else: raise Exception('the shape has only one body')


    def filter_complex_faces(self, thre=30):
    # 筛选复杂的面
        if self.cad_body_len > 1:
            for shape in self.shapes:
                num_face = len(Brep_utils.get_faces_from_BRep(shape))
                if num_face > thre:
                    return False
            return True
        else: raise Exception('the shape has only one body')


def filter_data():

    # 用现有的shape再过滤
    ref_filter_file = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    init_explaination = "init dataset"
    
    with open(ref_filter_file, 'r') as f:
        dirs = json.load(f)
    for dir in dirs:
        if dir.get('explaination') == init_explaination:
            init_name_dir = dir
    ref_names = init_name_dir['file_names']
    c0 = 0  # 统计共有多少参考文件
    c1 = 0  # 统计有多少满足条件的文件

    # 记录过滤后的结果
    result = {'explaination': 'filter edge face', 'file_names': []}
    
    json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json'
    json_classes = os.listdir(json_dir)
    for json_class in json_classes:
        json_class_dir = os.path.join(json_dir, json_class)
        json_names = os.listdir(json_class_dir)
        for name in json_names:
            n = name.split('.')[0]
            if n in ref_names:  
                c0+=1
                # 只找参考文件中的shape
                json_path = os.path.join(json_class_dir, name)
                data = CorrectDataFromJSON(json_path)
                if data.filter_slice() and data.filter_complex_faces():
                    c1+=1
                    result['file_names'].append(n)
    print(c0,c1)

    with open(ref_filter_file, 'w') as f:
        json.dump(result, f, indent=4)  # 加 indent 让结果更可读





def make_sketch0():

    # 读取需要处理的数据
    ref_filter_file = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    explaination = "6views valid shape"
    with open(ref_filter_file, 'r') as f:
        dirs = json.load(f)
    for dir in dirs:
        if dir['explaination'] == explaination:
            d = dir
    json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp'
    flag = 0
    for name in dir['file_names']:
        path_dir = name[:4]
        if int(path_dir)<=33:
            if name == '00019606':
                flag = 1
            if flag == 1:
                print(f'{name}------------------')
                for i in range(0,6):
                    shape = Shapeinfo(name, i)
                    output_path = os.path.join(output_dir, name+"_"+f"{i}.png")
                    show_single.save_BRep_wire_img_temp(shape.wires, campos=shape.campos, seeat=shape.seeat, output_path=output_path)


def filter_cir():
    ref_filter_file = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    explaination = "6views valid shape"
    with open(ref_filter_file, 'r') as f:
        dirs = json.load(f)
    for dir in dirs:
        if dir['explaination'] == explaination:
            d = dir
    json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp'
    flag = 0
    for i, name in enumerate(dir['file_names']):

        path_dir = name[:4]   
        shape_path = os.path.join(json_dir, path_dir, name+'.json')
        shape = Filter_Name_From_Seq(shape_path)

        if isinstance(shape.curve, Circle):
            with open('temp_cir.txt', 'a') as file:
                file.write(f'\n{name}')  # \n 表示换行

def write_cir():
    result = {'explaination': 'sketch single circle', 'file_names': []}
    file_path = 'temp_cir.txt'
    cir_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        result['file_names'].append(line[:8])

    json_path = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    with open(json_path, 'r') as file:
        current_data = json.load(file)  # data 是一个列表，列表的元素是字典
    current_data.append(result)

    with open (json_path, "w") as f:
        json.dump(current_data, f, indent=4)
        f.write("\n")


def make_sketch_circle0():
    ref_filter_file = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    explaination = "sketch single circle"
    with open(ref_filter_file, 'r') as f:
        dirs = json.load(f)
    for dir in dirs:
        if dir['explaination'] == explaination:
            d = dir
    json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp1'
    flag = 0
    for i, name in enumerate(dir['file_names']):
        
        path_dir = name[:4]   
        if int(path_dir)<=50:
            print(f'------------------{name}')
            for i in range(0,6):
                shape = Shapeinfo(name, i)
                output_path = os.path.join(output_dir, name+"_"+f"{i}.png")
                show_single.save_BRep_wire_img_temp(shape.wires, campos=shape.campos, seeat=shape.seeat, output_path=output_path)

def make_sketch_circle1():
    ref_filter_file = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    explaination = "sketch single circle"
    with open(ref_filter_file, 'r') as f:
        dirs = json.load(f)
    for dir in dirs:
        if dir['explaination'] == explaination:
            d = dir
    json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp1'
    flag = 0
    for i, name in enumerate(dir['file_names']):
        
        path_dir = name[:4]   
        if int(path_dir)>50:
            print(f'------------------{name}')
            for i in range(0,6):
                shape = Shapeinfo(name, i)
                output_path = os.path.join(output_dir, name+"_"+f"{i}.png")
                show_single.save_BRep_wire_img_temp(shape.wires, campos=shape.campos, seeat=shape.seeat, output_path=output_path)
        


def test():
    # 薄片：00020002 长方体 ； 00020013 环; 00020244 复杂的形状；00020328两个圆柱； 00020845 有交集的形状
    # name = ['00020002', '00020013', '00020244', '00020328', '00020845', '00021807']
    name = ['00020328']
    b = []
    test_json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json/0002'
    for n in name:
        test_json_path = os.path.join(test_json_dir, n+'.json')
        a = Filter_Name_From_Shape(test_json_path)





def main():
    test()


if __name__ == "__main__":
    main()