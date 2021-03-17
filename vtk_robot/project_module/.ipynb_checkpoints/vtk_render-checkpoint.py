import numpy as np
import cv2
import vtk
from vtk.util import numpy_support
from PIL import Image

import matplotlib.pyplot as plt


class render_shape():
    """
    VTK rendering for continuum robots
    """

    def __init__(self, w, h, diameters, n_sections, opacity=0.3):

        self.w = w
        self.h = h
        self.initCam = False
        self.n_sections = n_sections
        self.diameters = diameters
        self.points = []
        self.lines = []
        self.linesource = []
        self.filter = []
        self.tubeMapper = []
        self.tubeActor = []
        self.opacity = opacity

        self.init_vtk()

    def setCamProperties(self, f, c, K):
        self.f = f
        self.c = c
        self.K = K

    def init_vtk(self):
        """
        Initialization of VTK for rendering
        :return:
        """

        self.points = []
        self.lines = []
        self.linesource = []
        self.filter = []
        self.tubeMapper = []
        self.tubeActor = []

        # For each section of the robot, create a tube model of the right dimensions and a corresponding VTK rendering pipeline
        opacity = self.opacity
        for i in range(self.n_sections):
            points =  vtk.vtkPoints()
            points.SetNumberOfPoints(2)
            points.SetPoint(0, 0, 0, 0)
            points.SetPoint(1, 0, 0, 1)
            self.points.append(points)

            lines = vtk.vtkCellArray()
            lines.InsertNextCell(2)
            lines.InsertCellPoint(0)
            lines.InsertCellPoint(1)
            self.lines.append(lines)

            linesource = vtk.vtkPolyData()
            linesource.SetPoints(self.points[i])
            linesource.SetLines(self.lines[i])
            self.linesource.append(linesource)

            filter = vtk.vtkTubeFilter()
            filter.SetNumberOfSides(8)
            filter.SetRadius(self.diameters[i]/2.0)
            filter.SetInputData(self.linesource[i])
            self.filter.append(filter)

            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.filter[i].GetOutputPort())
            tubeMapper.Update()
            self.tubeMapper.append(tubeMapper)

            tubeActor = vtk.vtkActor()
            tubeActor.SetMapper(self.tubeMapper[i])
            tubeActor.GetProperty().SetColor(opacity, opacity, opacity)

            self.tubeActor.append(tubeActor)

        # Add a sphere at the end (robot specific - to be changed)
        self.sphereSource = vtk.vtkSphereSource()
        self.sphereSource.SetCenter(0, 0, 0)
        self.sphereSource.SetRadius(0./2.0)
        self.sphereMapper = vtk.vtkPolyDataMapper()
        self.sphereMapper.SetInputConnection(self.sphereSource.GetOutputPort())
        self.sphereactor = vtk.vtkActor()
        self.sphereactor.SetMapper(self.sphereMapper)

        # Initialize the renderer and window
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetUseImageBasedLighting(1)
        self.renderer.SetUseShadows(0)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetOffScreenRendering(1)



        # Add actors AFTER rendering initialization
        for actor in self.tubeActor:
            self.renderer.AddActor(actor)

        self.renderer.AddActor(self.sphereactor)

        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.renderer.ResetCamera()

        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetSize(self.w, self.h)

        self.renWin.Render() #----------------------- error

    def project_3Dshape(self, shapes, T):
        for i in range(len(shapes)):
            # Create a list of homogeneous coordinates and transform it with T
            shape = shapes[i]
            N = shape.shape[0]
            s = np.concatenate((shape, np.ones((shape.shape[0], 1))), axis=1)
            if T is None:
                T = np.eye(4)
            shape_trans = np.dot(s, T.T)

            # For each section, apply coordinates to the corresponding actor
            if N > 2:
                self.points[i].SetNumberOfPoints(N)
                self.lines[i].Reset()
                self.lines[i].InsertNextCell(N)

                for j in range(0, N):
                    self.points[i].SetPoint(j, shape_trans[j, 0], shape_trans[j, 1], shape_trans[j, 2])
                    self.lines[i].InsertCellPoint(j)

                self.linesource[i].SetPoints(self.points[i])
                self.linesource[i].SetLines(self.lines[i])
                self.tubeMapper[i].Update()

            # and finally set the tip sphere at the tip of the robot
            if i == len(shapes)-1:
                tip = shape_trans[N-1]
                self.sphereSource.SetCenter(tip[0],tip[1],tip[2])
                self.sphereMapper.Update()


        if (len(shapes)<self.n_sections):
            for i in range(len(shapes),self.n_sections):
                self.points[i].SetNumberOfPoints(0)
                self.lines[i].Reset()
                self.linesource[i].SetPoints(self.points[i])
                self.linesource[i].SetLines(self.lines[i])
                self.tubeMapper[i].Update()

        # Once all positions are set, render the scene using the active camera
        cam = self.renderer.GetActiveCamera()

        # If the camera is not properly initialized, do it here. Parameters are set in such a way that VTK renders using
        # the pinhole model in the "cam" object.
        if not self.initCam:
            near = 0.1
            far = 300.0
            cam.SetClippingRange(near, far)
            cam.SetPosition(0, 0, 0)
            cam.SetFocalPoint(0, 0, 1)
            cam.SetViewUp(0, -1, 0)

            wcx = -2 * (self.c[0] - self.w / 2.0) / self.w
            wcy = 2 * (self.c[1] - self.h / 2.0) / self.h
            cam.SetWindowCenter(wcx, wcy)
            angle = 180 / np.pi * 2.0 * np.arctan2(self.h / 2.0, self.f[1])
            cam.SetViewAngle(angle)

            m = np.eye(4)
            aspect = self.f[1]/self.f[0]
            m[0,0] = 1.0/aspect
            t = vtk.vtkTransform()
            t.SetMatrix(m.flatten())
            cam.SetUserTransform(t)
            self.initCam = True

        # Render the scene and export it to an openCV / numpy array to return it
        self.renWin.Render()

        winToIm = vtk.vtkWindowToImageFilter()
        winToIm.SetInput(self.renWin)
        winToIm.Update()
        vtk_image = winToIm.GetOutput()

        width, height, _ = vtk_image.GetDimensions()

        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()

        arr_ = cv2.flip(numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components), 0)

        arr = 1 - (arr_ != [26, 51, 102]).any(2).astype(int)

        arr = np.clip(np.concatenate([np.zeros([1,760]),arr[:-1,:]], axis=0) + np.concatenate([arr[1:,:], np.zeros([1,760])], axis=0) +np.concatenate([np.zeros([570,1]), arr[:, :-1]], axis=1) + np.concatenate([arr[:,1:], np.zeros([570,1])], axis=1),0.,1.)
        arr = 1 - arr

        arr3d = np.repeat(np.expand_dims(arr,axis=2),repeats=3,axis=2)



        arr_masked=arr_*arr3d
        arr_masked = np.ones(arr_masked.shape) * 255 * (1 - arr3d) + arr_masked * arr3d


        return arr_masked, arr
