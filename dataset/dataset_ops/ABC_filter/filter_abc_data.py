import blenderproc as bproc
import sys
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from multiprocessing import current_process

if current_process().name == 'MainProcess':
    import bpy

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import glob
import json
import numpy as np
import random
import h5py
import random
from utils.util import *
from joblib import Parallel, delayed
from trimesh.sample import sample_surface
import argparse
import os
import glob

from data.extrude import CADSequence
from utils.exporter import  out_color_boundary
from data.CC3DRawData import *
from OCC.Core.BRepCheck import BRepCheck_Analyzer
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    Textures
)
import matplotlib.pyplot as plt
import trimesh
import matplotlib
import transforms3d as t3d
from skimage import io
from PIL import Image
from torch_scatter import scatter_add, scatter_mean

import multiprocessing as mp
from PIL import Image



from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section

# import polyscope as ps
import trimesh as tri
import networkx as nx
import potpourri3d as pp3d
import pymeshlab as ml
from scipy import stats

from OCC.Core.TopoDS import TopoDS_Wire, TopoDS_Edge
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from OCC.Core.Addons import Font_FontAspect_Regular, text_to_brep
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.Graphic3d import Graphic3d_NOM_STONE
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln, gp_Ax2
from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
from OCC.Display.SimpleGui import init_display
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.GeomProjLib import geomprojlib_Curve2d
from OCC.Core.BRep import BRep_Tool_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face, TopoDS_Edge, topods

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.TopoDS import TopoDS_Shape, topods_Shell, topods_Solid, TopoDS_Shell
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SHELL
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepCheck import BRepCheck_Shell

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Compound, topods_Face, topods_Edge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ExtCC
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Circ, gp_Pln, gp_Vec, gp_Ax3, gp_Ax2, gp_Lin
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from copy import copy
from data.extrude import *
from data.sketch import Loop, Profile
from data.curves import *
import os
import trimesh
from trimesh.sample import sample_surface
import random
from copy import deepcopy
from scipy.stats import mode

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import time
sys.path.append("/media/lida/softwares/Wonder3D/neus")
from neus.utils.cadrender import *





def face_to_trimesh(face, linear_deflection=0.001):

    bt = BRep_Tool()
    BRepMesh_IncrementalMesh(face, linear_deflection, True)
    location = TopLoc_Location()
    facing = bt.Triangulation(face, location)
    if facing is None:
        return None
    triangles = facing.Triangles()

    vertices = []
    faces = []
    offset = face.Location().Transformation().Transforms()

    for i in range(1, facing.NbNodes() + 1):
        node = facing.Node(i)
        coord = [node.X() + offset[0], node.Y() + offset[1], node.Z() + offset[2]]
        # coord = [node.X(), node.Y() , node.Z() ]
        vertices.append(coord)

    for i in range(1, facing.NbTriangles() + 1):
        triangle = triangles.Value(i)
        index1, index2, index3 = triangle.Get()
        if face.Orientation()!=TopAbs_REVERSED:
            tface = [index1 - 1, index2 - 1, index3 - 1]
        else:
            tface = [index1 - 1, index3 - 1, index2 - 1]
        faces.append(tface)
    tmesh = tri.Trimesh(vertices=vertices, faces=faces, process=False)


    return tmesh

def getVertex(compound):
    vs = []
    explorer = TopExp_Explorer(compound, TopAbs_VERTEX)
    while explorer.More():
        current_v = topods.Vertex(explorer.Current())
        vs.append(current_v)
        explorer.Next()
    return vs


def getWires(compound):
    wires = []
    wire_explorer = TopExp_Explorer(compound, TopAbs_WIRE)
    while wire_explorer.More():
        wire = topods.Wire(wire_explorer.Current())
        wires.append(wire)
        wire_explorer.Next()

    return wires

def getEdges(compound):
    edges = []
    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        current_edge = topods.Edge(explorer.Current())
        edges.append(current_edge)
        explorer.Next()
    return edges

def getFaces(compound):
    faces = []
    explorer = TopExp_Explorer(compound, TopAbs_FACE)
    while explorer.More():
        current_face = topods.Face(explorer.Current())
        faces.append(current_face)
        explorer.Next()
    return faces

def face2wire(face):
    c_wire = BRepBuilderAPI_MakeWire()


    for edge in getEdges(face):
        e = edge.Oriented(TopAbs_FORWARD)
        c_wire.Add(e)
    wire = c_wire.Wire()
    return wire


def vec2CADsolid(vec, is_numerical=True, n=256):
    cad = CADSequence.from_vector(vec, is_numerical=is_numerical, n=256)
    cad = create_CAD(cad)
    return cad


def create_CAD(cad_seq: CADSequence):
    """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
    body = create_by_extrude(cad_seq.seq[0])
    for extrude_op in cad_seq.seq[1:]:
        new_body = create_by_extrude(extrude_op)
        if extrude_op.operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation") or \
                extrude_op.operation == EXTRUDE_OPERATIONS.index("JoinFeatureOperation"):
            body = BRepAlgoAPI_Fuse(body, new_body).Shape()
        elif extrude_op.operation == EXTRUDE_OPERATIONS.index("CutFeatureOperation"):
            body = BRepAlgoAPI_Cut(body, new_body).Shape()
        elif extrude_op.operation == EXTRUDE_OPERATIONS.index("IntersectFeatureOperation"):
            body = BRepAlgoAPI_Common(body, new_body).Shape()
    return body


def create_by_extrude(extrude_op: Extrude):
    """create a solid body from Extrude instance."""
    profile = copy(extrude_op.profile) # use copy to prevent changing extrude_op internally
    profile.denormalize(extrude_op.sketch_size)

    sketch_plane = copy(extrude_op.sketch_plane)
    sketch_plane.origin = extrude_op.sketch_pos

    face = create_profile_face(profile, sketch_plane)
    normal = gp_Dir(*extrude_op.sketch_plane.normal)
    ext_vec = gp_Vec(normal).Multiplied(extrude_op.extent_one)
    body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
    if extrude_op.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
        body_sym = BRepPrimAPI_MakePrism(face, ext_vec.Reversed()).Shape()
        body = BRepAlgoAPI_Fuse(body, body_sym).Shape()
    if extrude_op.extent_type == EXTENT_TYPE.index("TwoSidesFeatureExtentType"):
        ext_vec = gp_Vec(normal.Reversed()).Multiplied(extrude_op.extent_two)
        body_two = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
        body = BRepAlgoAPI_Fuse(body, body_two).Shape()
    return body


def create_profile_face(profile: Profile, sketch_plane: CoordSystem):
    """create a face from a sketch profile and the sketch plane"""
    origin = gp_Pnt(*sketch_plane.origin)
    normal = gp_Dir(*sketch_plane.normal)
    x_axis = gp_Dir(*sketch_plane.x_axis)
    gp_face = gp_Pln(gp_Ax3(origin, normal, x_axis))

    all_loops = [create_loop_3d(loop, sketch_plane) for loop in profile.children]
    topo_face = BRepBuilderAPI_MakeFace(gp_face, all_loops[0])
    for loop in all_loops[1:]:
        topo_face.Add(loop.Reversed())
    return topo_face.Face()


def create_loop_3d(loop: Loop, sketch_plane: CoordSystem):
    """create a 3D sketch loop"""
    topo_wire = BRepBuilderAPI_MakeWire()
    for curve in loop.children:
        topo_edge = create_edge_3d(curve, sketch_plane)
        if topo_edge == -1: # omitted
            continue
        topo_wire.Add(topo_edge)
    return topo_wire.Wire()


def create_edge_3d(curve: CurveBase, sketch_plane: CoordSystem):
    """create a 3D edge"""
    if isinstance(curve, Line):
        if np.allclose(curve.start_point, curve.end_point):
            return -1
        start_point = point_local2global(curve.start_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        topo_edge = BRepBuilderAPI_MakeEdge(start_point, end_point)
    elif isinstance(curve, Circle):
        center = point_local2global(curve.center, sketch_plane)
        axis = gp_Dir(*sketch_plane.normal)
        gp_circle = gp_Circ(gp_Ax2(center, axis), abs(float(curve.radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)
    elif isinstance(curve, Arc):
        # print(curve.start_point, curve.mid_point, curve.end_point)
        start_point = point_local2global(curve.start_point, sketch_plane)
        mid_point = point_local2global(curve.mid_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        arc = GC_MakeArcOfCircle(start_point, mid_point, end_point).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc)
    else:
        raise NotImplementedError(type(curve))
    return topo_edge.Edge()


def point_local2global(point, sketch_plane: CoordSystem, to_gp_Pnt=True):
    """convert point in sketch plane local coordinates to global coordinates"""
    g_point = point[0] * sketch_plane.x_axis + point[1] * sketch_plane.y_axis + sketch_plane.origin
    if to_gp_Pnt:
        return gp_Pnt(*g_point)
    return g_point


def CADsolid2pc(shape, n_points, name=None):
    """convert opencascade solid to point clouds"""
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    if bbox.IsVoid():
        raise ValueError("box check failed")

    if name is None:
        name = random.randint(100000, 999999)
    write_stl_file(shape, "tmp_out_{}.stl".format(name))
    out_mesh = trimesh.load("tmp_out_{}.stl".format(name))
    os.system("rm tmp_out_{}.stl".format(name))
    out_pc, _ = sample_surface(out_mesh, n_points)
    return out_pc


def check_multi_side_extrude(cad_seq):
    for i in range(len(cad_seq.seq)):
        seq_one = cad_seq.seq[i]
        if seq_one.extent_type == EXTENT_TYPE.index(
                "SymmetricFeatureExtentType") or seq_one.extent_type == EXTENT_TYPE.index("TwoSidesFeatureExtentType"):
            return True
    return False


def check_shape_is_right(learn_sketch_shapes, learn_sketch_meshes, learn_sketch_types):
    sketch_face_areas = []
    for idx_i in range(len(learn_sketch_shapes)):
        face = learn_sketch_shapes[idx_i]
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        sketch_face_areas.append(area)

    distance_graph = np.zeros((len(learn_sketch_shapes), len(learn_sketch_shapes)))

    sketch_inter_size = 0
    sketch_adj_size = 0
    for idx_i in range(len(learn_sketch_shapes)):
        for idx_j in range(len(learn_sketch_shapes)):
            if idx_j > idx_i:
                face1 = learn_sketch_shapes[idx_i]
                face2 = learn_sketch_shapes[idx_j]

                face1_mesh = learn_sketch_meshes[idx_i]
                face2_mesh = learn_sketch_meshes[idx_j]

                face1_normal = face1_mesh.vertex_normals.mean(axis=0)
                face2_normal = face2_mesh.vertex_normals.mean(axis=0)

                face1_type = learn_sketch_types[idx_i]
                face2_type = learn_sketch_types[idx_j]

                intersection = BRepAlgoAPI_Common(face1, face2).Shape()
                # print("intersection: ", not intersection.IsNull())
                props = GProp_GProps()
                brepgprop.SurfaceProperties(intersection, props)
                area = props.Mass()

                dist = BRepExtrema_DistShapeShape(face1, face2)
                dist.Perform()
                min_distance = dist.Value()
                if min_distance < 1e-5 and np.abs(
                        face1_normal.dot(face2_normal) - 1) < 1e-5 and face1_type == face2_type:
                    sketch_adj_size += 1

                if min_distance == 0.0:
                    distance_graph[idx_i, idx_j] = 1
                    distance_graph[idx_j, idx_i] = 1

                area_ratio_1 = area / sketch_face_areas[idx_i]
                area_ratio_2 = area / sketch_face_areas[idx_j]
                if area_ratio_1 > 1e-5 or area_ratio_2 > 1e-5:
                    sketch_inter_size += 1

    print("sketch intersection", sketch_inter_size)
    print("sketch adj size", sketch_adj_size)
    # if sketch_inter_size > 0 or sketch_adj_size > 0:
    #     return False, distance_graph
    return True, distance_graph


def save_mask_image(tensor, fp):
    mask_set = set(tensor.reshape(-1).tolist())
    tensor_out = torch.zeros(tensor.shape)
    count = 0
    for i in range(len(mask_set)):
        mask_index = torch.where(tensor == mask_set.pop())
        tensor_out[mask_index] = count
        count += 1

    max_v = tensor_out.max()
    min_v = tensor_out.min()
    range_v = max_v - min_v
    print(max_v, min_v, range_v)
    if range_v != 0:
        tensor_out = (tensor_out - min_v) / range_v

    ndarr = tensor_out.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_face_type_images(ndarr, fp):
    ndarr = ndarr.to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_direct_pixels(pixels, fp):
    im = Image.fromarray(pixels)
    im.save(fp)
    return pixels


def save_image(tensor, fp):
    max_v = tensor.max()
    min_v = tensor.min()
    range_v = max_v - min_v
    print(max_v, min_v, range_v)
    if range_v != 0:
        tensor = (tensor - min_v) / range_v

    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


def generate_distinct_colors(num_colors):
    base_colors = plt.cm.tab10.colors
    colors = []
    for i in range(num_colors):
        color = to_rgba(base_colors[i % len(base_colors)])
        colors.append(tuple(int(c * 255) for c in color[:3]))
    return colors


def render_normal_directions(mesh):
    ps.init()
    ps.remove_all_structures()
    ps_mesh = ps.register_surface_mesh("my_mesh", mesh.vertices, mesh.faces)
    ps_mesh.add_vector_quantity("face_normals", mesh.face_normals, defined_on="faces",  enabled=True, length=0.1, color=(0.8, 0.2, 0.2))

    # Show the Polyscope GUI
    ps.show()



def filter_no_good_boundingbox_faces(faces):
    for face in faces:
        mesh = face_to_trimesh(face)
        if max(mesh.bounding_box.extents) / min(mesh.bounding_box.extents) > 10:
            return True
    return False


def filter_no_good_boundingbox_shape(mesh):
    if max(mesh.bounding_box.extents) / min(mesh.bounding_box.extents) > 10:
        return True
    return False


def get_bounding_box_ratio(mesh):
    bounding_box = np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0)
    ratio = max(bounding_box) / min(bounding_box)
    return ratio, bounding_box

def get_bounding_box_faces_ratio(mesh, labels):
    ratios = []
    bounding_boxes = []

    label_set = set(labels.tolist())
    for label in label_set:
        c_mesh = mesh.submesh(np.where(labels == label))[0]
        bounding_box = np.max(c_mesh.vertices, axis=0) - np.min(c_mesh.vertices, axis=0)
        max_value = max(bounding_box)
        min_value = min(bounding_box)
        center_value = bounding_box.tolist()
        center_value.remove(max_value)
        center_value.remove(min_value)
        center_value = center_value[0]

        ratio1 = max(bounding_box) / min(bounding_box)
        ratio2 = max(bounding_box) / center_value
        ratios.append(min(ratio1, ratio2))
        bounding_boxes.append(bounding_box)
    return np.max(ratios), bounding_boxes





import multiprocessing

# def statistics_deepcad(paths, start, end, save_dir,  tolerance=0.0001):
#     save_path = os.path.join(save_dir, str(start) + '_' + str(end))
#     if not os.path.exists(save_path):
#         num_cores = multiprocessing.cpu_count() - 1
#         with multiprocessing.Pool(processes=num_cores) as pool:
#             # Use imap to process the paths in parallel with a progress bar
#             results = list(tqdm(pool.imap(statistics_deepcad_one, paths[start:end]), total=len(paths[start:end])))
#         # for i in range(len(paths[start:end])):
#         #     results = statistics_deepcad_one(paths[start:end][i])
#         #     if results is None:
#         #         continue
#         #     render_normal_directions(tri.util.concatenate(results['out_mesh']) )
#
#         valid_results = [r for r in results if r is not None]
#         used_paths = [r['path'] for r in valid_results]
#         shape_ratio_data = [r['ratio'] for r in valid_results]
#         face_ratio_data = [r['face_ratio'] for r in valid_results]
#         face_number = [r['face_number'] for r in valid_results]
#         shape_invalid_len = [r['shape_invalid_len'] for r in valid_results]
#         face_invalid_len = [r['face_invalid_len'] for r in valid_results]
#         face_meshes = [r['out_mesh'] for r in valid_results]
#         print(f"Processed {len(valid_results)} valid items")
#         save_cache_dill([used_paths, shape_ratio_data, face_ratio_data, face_number, shape_invalid_len, face_invalid_len], save_path)
#

def analyze_deepcad_statistics(save_dir):
    final_used_paths = []
    final_shape_ratio_data = []
    final_face_ratio_data = []
    final_face_number = []
    final_shape_invalid_len = []
    final_face_invalid_len = []

    for path in os.listdir(save_dir):
        cache_data = os.path.join(save_dir, path)
        used_paths, shape_ratio_data, face_ratio_data, face_number, shape_invalid_len, face_invalid_len = load_cache_dill(cache_data)
        final_used_paths += used_paths
        final_shape_ratio_data += shape_ratio_data
        final_face_ratio_data += face_ratio_data
        final_face_number += face_number
        final_shape_invalid_len += shape_invalid_len
        final_face_invalid_len += face_invalid_len

    return final_used_paths


def out_color_mesh_cc3d(shape_faces, face_meshes):
    all_label_array = []
    all_types = [GeomAbs_Plane,
                 GeomAbs_Cylinder,
                 GeomAbs_Cone,
                 GeomAbs_Sphere,
                 GeomAbs_Torus]
    all_type_label = []
    for i in range(len(shape_faces)):
        current_face = shape_faces[i]
        tmesh = face_meshes[i]
        tmesh_label = np.zeros(len(tmesh.faces))
        current_surface = BRepAdaptor_Surface(current_face)
        current_surface_type = current_surface.GetType()
        add_flag = False
        for type_index in range(len(all_types)):
            # if current_surface.DynamicType().Name() == all_types[type_index].__name__:
            if current_surface_type == all_types[type_index]:
                all_type_label.append(np.ones(len(tmesh.faces)) * (type_index + 1))
                add_flag = True
        if add_flag == False:
            all_type_label.append(np.ones(len(tmesh.faces)) * (7 + 1))
        tmesh_label += i + 1
        all_label_array.append(tmesh_label)
    out_mesh = tri.util.concatenate(face_meshes)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)  # 指定降维到的维度数
    new_vertices = pca.fit_transform(out_mesh.vertices)
    out_mesh.vertices = new_vertices
    out_label = np.concatenate(all_label_array)
    out_type_face_label = np.concatenate(all_type_label)
    return out_mesh, out_label, out_type_face_label



def find_step_files_fast(directory, endix):
    """Finds all .step files in the given directory and its subdirectories."""
    return glob.glob(os.path.join(directory, "**", endix), recursive=True)




class File_StemsUsed:
    def __init__(self, directory, istrain=True):
        self.files = {}
        self.directory = directory
        batches = ["abc_0"+str(i) for i in range(0, 10) ] + ["abc_"+str(i) for i in range(10, 100)]
        self.files['step'] = []

        for current_batch in batches:
            save_path = "/media/bizon/extradisk/cad_datasets/abc/" +current_batch +".cache"
            if not os.path.exists(save_path):
                brep_d = os.path.join(directory,  str(current_batch))
                brep_d_files = dict()
                brep_d_files_list = []
                step_files = find_step_files_fast(brep_d, "*.step")
                brep_d_files_list += step_files
                for step_f in step_files:
                    dd1 = os.path.basename(os.path.dirname(step_f))
                    dd2 = os.path.basename(step_f).split(".step")[0]
                    brep_d_files[dd1 + dd2] = step_f
                for key in brep_d_files.keys():
                    self.files['step'].append(brep_d_files[key])
                save_cache_dill(self.files, save_path)
            else:
                self.files = load_cache_dill(save_path)
        self.current_idx = 0
        self.size = len(self.files['step'])

    def select_idx(self, n):
        self.current_idx = n

    def current_step(self):
        return self.files["step"][self.current_idx]

    def skip_n_stems(self, n):
        self.current_idx = (self.current_idx + n) % len(self.files["step"])

    def next_stem(self):
        self.skip_n_stems(1)

    def prev_stem(self):
        self.skip_n_stems(-1)


def calculate_path_length(points):
    """
    Calculate the total length of a path defined by a list of points.

    :param points: List of tuples or arrays representing the coordinates of the points.
    :return: Total length of the path.
    """
    points_array = np.array(points)
    # Calculate the differences between consecutive points
    deltas = np.diff(points_array, axis=0)
    # Calculate the Euclidean distance for each segment
    segment_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    # Sum the segment lengths to get the total path length
    total_length = segment_lengths.sum()
    return total_length


import errno
import os
import signal
import functools

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

@timeout(2)
def read_step_time_out(file):
    content = read_step_file(file)
    faces = getFaces(content)
    return content, faces

class ABCUsed:
    def __init__(self, directory):
        self.directory = directory
        self.file_items = File_StemsUsed(directory)  # 读取step数据路径
        self.size = self.file_items.size



    def load_step_edge_labels(self, inds):
        used_idxs = []
        for index in inds:
            print('start' , index)
            self.file_items.select_idx(index)
            step_f= self.file_items.current_step()

            if True:
                if not os.path.exists(step_f):
                    continue
                file_size = os.path.getsize(step_f)
                # Convert bytes to kilobytes
                file_size_kb = file_size / 1024
                # Check if the file size is greater than 200 KB
                if file_size_kb > 500:
                    continue

                try:
                    c_step, faces = read_step_time_out(step_f)
                except:
                    continue
                if c_step is None:
                    continue


                if len(faces) > 30:
                    continue
                add_flag = False
                all_types =  [
                     GeomAbs_Cone,
                     GeomAbs_Sphere,
                     GeomAbs_Torus]
                remove_types = [
                    GeomAbs_BezierSurface,
                    GeomAbs_BSplineSurface
                ]
                remove_flag = False
                for i in range(len(faces)):
                    current_face = faces[i]
                    current_surface = BRepAdaptor_Surface(current_face)
                    current_surface_type = current_surface.GetType()
                    for type_index in range(len(remove_types)):
                        if current_surface_type == remove_types[type_index]:
                            remove_flag = True
                    for type_index in range(len(all_types)):
                        if current_surface_type == all_types[type_index]:
                            add_flag = True

                if remove_flag == True:
                    continue
                if add_flag == False:
                    continue
                used_idxs.append(index)
        return used_idxs

    def load_step_edge(self, index):
        self.file_items.select_idx(index)
        step_f= self.file_items.current_step()

        if True:
            try:
                c_step = read_step_file(step_f)
                if len(getFaces(c_step)) > 30:
                    return None
                faces, shell = fix_step(getFaces(c_step))
            except:
                return None

            all_type_label = []
            add_flag = False
            all_types =  [
                 GeomAbs_Cone,
                 GeomAbs_Sphere,
                 GeomAbs_Torus]
            remove_types = [
                GeomAbs_BezierSurface,
                GeomAbs_BSplineSurface
            ]
            remove_flag = False
            for i in range(len(faces)):
                current_face = faces[i]
                current_surface = BRepAdaptor_Surface(current_face)
                current_surface_type = current_surface.GetType()
                for type_index in range(len(remove_types)):
                    if current_surface_type == remove_types[type_index]:
                        remove_flag = True
                for type_index in range(len(all_types)):
                    if current_surface_type == all_types[type_index]:
                        add_flag = True

            if remove_flag == True:
                return None 
            if add_flag == False:
                return None
            


            edge_ratio = []
            for ff in faces:
                ees = getEdges(ff)
                ees_lengths = []
                for ee in ees:
                    ees_lengths.append(calculate_path_length(np.array([i.Coord() for i in discretize_edge(ee, 8)])))
                if len(ees_lengths) <= 0:
                    return None
                edge_ratio.append(np.max(ees_lengths) / np.min(ees_lengths))
            edge_max_ratio = np.max(edge_ratio)
            if edge_max_ratio > 30:
                return None

            wire_number = [len(getWires(face))>3 for face in faces]
            if np.sum(wire_number) > 1:
                return None

            if np.sum([len(getWires(face)) > 1 for face in faces]) > 4 :
                return None

            # if np.sum([len(getWires(face)) > 1 for face in faces]) % 2!=0:
            #     return None

            wire_lengths = [[np.sum([calculate_path_length(np.array([i.Coord() for i in discretize_edge(edge, 8)])) for edge in getEdges(wire)]) for wire in getWires(face) ] for face in faces ]
            for wire_length in wire_lengths:
                if len(wire_length) <=1:
                    continue
                max_wire_length = np.max(wire_length)
                min_wire_length = np.min(wire_length)
                wire_length.remove(max_wire_length)
                ratio = min_wire_length / max_wire_length
                if ratio < 0.1:
                    return None
                if ratio > 0.9:
                    return None
            
                if len(wire_length) >=2:
                    second_ratio = np.max(wire_length) / max_wire_length
                    if second_ratio > 0.9:
                        return None


            face_meshes = [face_to_trimesh(face) for face in faces]
            if None in face_meshes:
                return None

            out_mesh, out_label, out_type_face_label  = out_color_mesh_cc3d(faces, face_meshes)
            if None in face_meshes:
                return None
            if 8 in out_type_face_label:
                return None
            if 5 not in out_type_face_label and 3 not in out_type_face_label and 4 not in out_type_face_label:
                return None

            ratio, bounding_box = get_bounding_box_ratio(out_mesh)
            face_ratio, bounding_boxes = get_bounding_box_faces_ratio(out_mesh, out_label)
            print("begin convert")
            # size_len = len(np.where(bounding_box < 0.05)[0])
            # face_size_len = len(np.where(np.sum(np.array(bounding_boxes) < 0.05, axis=1)>=2)[0])

            size_len = len(np.where(bounding_box < 0.05)[0])
            face_size_len = len(np.where(np.sum(np.array(bounding_boxes) < 0.05, axis=1)>=2)[0])
            return {
                'path': step_f,
                'ratio': ratio,
                'edge_max_ratio': edge_max_ratio,
                'face_ratio': face_ratio,
                'face_number': len(faces),
                'shape_invalid_len': size_len,
                'face_invalid_len': face_size_len,
                'out_mesh':face_meshes,
                'out_cad': faces,
                'holes':  np.sum([len(getWires(face)) > 1 for face in faces])
            }
        # except:
        #     return None




def fix_step(faces):
    # sewing = BRepBuilderAPI_Sewing()
    # sewing.SetTolerance(1e-3)
    # for ff in faces:
    #     sewing.Add(ff)
    # sewing.Perform()
    # sewed_shape = sewing.SewedShape()
    # unifier = ShapeUpgrade_UnifySameDomain(sewed_shape, True, True, True)
    # unifier.Build()
    # sewn_shape = unifier.Shape()
    #
    # # Step 2: Check if it's already a shell
    # if sewn_shape.ShapeType() == TopAbs_SHELL:
    #     shell = topods_Shell(sewn_shape)
    # else:
        # If not, we need to explicitly create a shell
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    for face in faces:
        builder.Add(shell, face)
    # solid_maker = BRepBuilderAPI_MakeSolid()
    # solid_maker.Add(shell)
    # solid = solid_maker.Solid()

    # from OCC.Core.ShapeFix import ShapeFix_Shape
    # fixer = ShapeFix_Shape(shell)
    # fixer.Perform()
    # fixed_shape = fixer.Shape()
    return getFaces(shell), shell

def process_cad_data_labels(data_dir, start, end, save_dir):
    save_path = os.path.join(save_dir, str(start) + '_' + str(end)+'_labelvalid_ids')
    print("begin ", start, "   ", end)
    if not os.path.exists(save_path):
        num_cores = 10
        results = []
        skip = 20
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Use imap to process the paths in parallel with a progress barr
            results = list(tqdm(pool.imap(abc_dataset.load_step_edge_labels, [list(range(i, i+200)) for i in range(start, end, 200)]), total=len(list(range(start, end, 200)))))
            # futures_res = pool.imap(abc_dataset.load_step_edge_labels, list(range(start, end)))
            # inputs = deepcopy(list(range(start, end)))
            # while inputs:
            #     s = inputs.pop(0)
            #     print("remaining ", len(inputs))
            #     try:
            #         res = futures_res.next()
            #         # If successful (no time out), append the result
            #         results.append(res)
            #     except mp.context.TimeoutError:
            #         print(s, "err")
            #         continue

        valid_results = [rr for r in results if r is not None for rr in r if rr is not None]
        save_cache_dill(valid_results, save_path)

def process_cad_data(data_dir, start, end,  ids, save_dir):
    result = {}
    global abc_dataset
    if True:
        save_path = os.path.join(save_dir, str(start) + '_' + str(end))
        if not os.path.exists(save_path):
            num_cores = multiprocessing.cpu_count() - 1
            results = []
            with multiprocessing.Pool(processes=num_cores) as pool:
                # Use imap to process the paths in parallel with a progress barr
                futures_res = pool.imap(abc_dataset.load_step_edge, ids)
                inputs = deepcopy(list(ids))
                while inputs:
                    s = inputs.pop(0)
                    print("remaining ", len(inputs))
                    try:
                        res = futures_res.next(1)
                        # If successful (no time out), append the result
                        if res is not None:
                            print("add one ")
                        results.append(res)
                    except mp.context.TimeoutError:
                        print(s, "err")
                        continue


            # for i in range(start, end):
            #     results = abc_dataset.load_step_edge(i)
            #     if results is None:
            #         continue
            #     # render_normal_directions(tri.util.concatenate(results['out_mesh']) )
            #     render_all_occ(results['out_cad'])

            valid_results = [r for r in results if r is not None]
            used_paths = [r['path'] for r in valid_results]
            holes = [r['holes'] for r in valid_results]
            shape_ratio_data = [r['ratio'] for r in valid_results]
            face_ratio_data = [r['face_ratio'] for r in valid_results]
            face_number = [r['face_number'] for r in valid_results]
            shape_invalid_len = [r['shape_invalid_len'] for r in valid_results]
            face_invalid_len = [r['face_invalid_len'] for r in valid_results]
            max_edge_ratio = [r['edge_max_ratio'] for r in valid_results]
            face_meshes = [r['out_mesh'] for r in valid_results]
            cads = [r['out_cad'] for r in valid_results]
            print(f"Processed {len(valid_results)} valid items")
            save_cache_dill(
                [used_paths, shape_ratio_data, face_ratio_data, face_number, shape_invalid_len, face_invalid_len,
                 max_edge_ratio, holes], save_path)
            save_cache_dill([used_paths, face_meshes, cads], save_path + '_meshcad')

start_i = time.time()
data_dir = "/media/bizon/extradisk/cad_datasets/abc"
save_dir = "/media/bizon/extradisk/cad_datasets/abc/cache_dir"
abc_dataset = ABCUsed(data_dir)

if __name__ == '__main__':
    for i in range(0, abc_dataset.size, 2000):
        process_cad_data_labels(data_dir, i, i+2000, save_dir)
        # save_path = os.path.join(save_dir, str(i) + '_' + str(i+2000)+'_labelvalid_ids')
        # ids = load_cache_dill(save_path)
        # process_cad_data(data_dir, i, i+2000, ids, save_dir)
