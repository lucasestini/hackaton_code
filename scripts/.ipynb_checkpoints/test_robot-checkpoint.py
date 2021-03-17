import os
from utilities import get_dir, combine_left_right
root_dir = get_dir()
import sys
print(os.path.join(root_dir,"vtk_robot/project_module/"))
sys.path.insert(0,os.path.join(root_dir,"vtk_robot/project_module/"))
from project_main import projector as projector_class
import numpy as np
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt
import cProfile
import pstats
from pstats import SortKey
from PIL import Image

data_path = os.path.join(root_dir, "data")
save_path = os.path.join(root_dir, "results")
if not os.path.isdir(save_path): os.makedirs(save_path)

kine_file = os.path.join(data_path,"kine.pkl")
with open(kine_file,"rb") as f:
    kine = pkl.load(f)

kine_i = kine[0]

projector =  projector_class(root_dir)

points_l = projector.get_3Dshape(kine_i,"L")
points_r = projector.get_3Dshape(kine_i,"R")


rend_l, mask_l = projector.project_3Dshape(points_l,"L")
rend_r, mask_r = projector.project_3Dshape(points_r,"R")



rend, mask = combine_left_right(rend_l, mask_l, rend_r, mask_r)

Image.fromarray((rend).astype("uint8")).save(os.path.join(save_path,"rend.jpg"))

print("done")


