from camera_model import camera_model
from robot_model import robot_model
from vtk_render import render_shape
from transformations import build_handeyeMatrix
import os
import numpy as np
import pickle as pkl

def load_mm_list(path):
    with open(path, "rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        mm = u.load()
    return mm

def load_mm_list_exvivo(path):
    with open(path, "rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        mm = u.load()
    mm_out = []
    for i in range(2):
        for j in range(3):
            mm_out.append(mm[i][j])
    return mm_out


def unnormalize(data,mm):
    max_ = mm[0]
    min_ = mm[1]
    return data*(max_ - min_) + min_

def unnormalize_kine(kine_sample,mm):
    kine_out = []
    for i in range(6):
        kine_out.append(unnormalize(kine_sample[i], mm[i]))
    return kine_out


# Main functions --------------------------------------------------------------------------------------------------------


class projector():
    def __init__(self, root_dir, opacity = 0.3, exvivo = True):
        vtkDirectory = os.path.join(root_dir, "vtk_robot")
        data_folder = os.path.join(vtkDirectory,"project_utilities/")
        self.mm = load_mm_list(os.path.join(data_folder,"max_min_joints_smaller.pkl")) if not exvivo else load_mm_list_exvivo(os.path.join(data_folder,"max_min_joints_exvivo.pkl"))

        configfile = os.path.join(data_folder,"robot_config.txt") if not exvivo else os.path.join(data_folder,"robot_config_exvivo.txt")
        self.robot = robot_model(configfile=configfile)

        camera_calibration = "cam_calib/Stras_Calib_Results_5param.mat"
        cam = camera_model(os.path.join(data_folder,camera_calibration))
        w, h = cam.getImSize()
        self.render = render_shape(w, h, self.robot.diameters, self.robot.n_sec, opacity)
        self.render.setCamProperties(cam.getF(), cam.getC(), cam.K)

        self.T_left, self.T_right = build_handeyeMatrix()

    def get_3Dshape(self,kine,arm):
        if not arm in ["R","L"]:
            print("Arm not in [\"L\",\"R\"]")
            exit()
        self.robot.arm = arm
        kine_un = unnormalize_kine(kine,self.mm)
        kine_arm = kine_un[:3] if arm=="L" else kine_un[3:]
        points = self.robot.computeShape(kine_arm)
        return points


    def project_3Dshape(self, points, arm):
        T_handeye = self.T_left if arm == "L" else self.T_right
        rendering, mask = self.render.project_3Dshape(points, T_handeye)
        return rendering, mask





