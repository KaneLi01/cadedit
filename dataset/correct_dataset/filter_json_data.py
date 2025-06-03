import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import combinations
import utils.cadlib.Brep_utils as Brep_utils
from utils.vis.show_single import show_BRep


def check_slice(dx, dy, dz, scale):
    for a, b in combinations([dx, dy, dz], 2):
        longer = max(a, b)
        shorter = min(a, b)
        if longer >= scale * shorter:
            return True
    return False


# json -> brep shape 这一步进行数据集的筛选
class CorrectDataFromJSON():
    def __init__(self, cad_json_path):
        self.cad_json_path = cad_json_path
        self.cad_name = self.cad_json_path.split('/')[-1].split('.')[0]
        self.cad_seq = Brep_utils.get_seq_from_json(self.cad_json_path)
        self.cad_body_len = len(self.cad_seq.seq)
        if self.cad_body_len > 1:
            self.sub_seqs = [self.cad_seq.seq, copy.deepcopy(self.cad_seq.seq[:-1]), copy.deepcopy(self.cad_seq.seq[-1:])]
        else: raise Exception('the shape has only one body')
        self.shapes = [Brep_utils.get_BRep_from_seq(sub_seq) for sub_seq in self.sub_seqs]


    def filter_slice(self, shortest=0.04, scale=15):
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


    def filter_complex_faces(self, thre=30):
    # 筛选复杂的面
        for shape in self.shapes:
            num_face = len(Brep_utils.get_faces_from_BRep(shape))
            if num_face > thre:
                return False
        return True


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


def test():
    # 薄片：00020085
    name = ['00020085', '00026913']
    b = []
    test_json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json/0002'
    for n in name:
        test_json_path = os.path.join(test_json_dir, n+'.json')
        a = CorrectDataFromJSON(test_json_path)
        if a.filter_slice() and a.filter_complex_faces():
            b.append(a.cad_name)
    print(b)
    # show_BRep(a.shape, save_path=save_path)


def main():
    filter_data()


if __name__ == "__main__":
    main()