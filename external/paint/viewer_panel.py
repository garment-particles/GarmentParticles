import sys
import numpy as np
import vtk
from typing import List
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QLabel, QToolBar, QAction
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


from definitions import *

# ==============================
# VTK 3D ビュー
# ==============================
class VTKPointViewer(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ren = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.GetRenderWindow().GetInteractor()

        self.pointSource = vtk.vtkPoints()
        self.polyData = vtk.vtkPolyData()
        self.vertices = vtk.vtkCellArray()

        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()

        

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()
        self.iren.Initialize()

        
        # 追加: アークボール＝TrackballCamera を明示
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)
    
    # --- 便利関数（任意）：Vertex リスト → xyz の numpy 配列 ---
    def vertices_to_xyz(self, vertices: List[Vertex]):
        return np.array([[p.x, p.y, p.z] for p in vertices], dtype=float)


    def _set_vertices(self, vertices):
        points = self.vertices_to_xyz(vertices)
        self._update_pointcloud(points)

        self.ren.ResetCamera()
        self.iren.Initialize()

    def _arcball_focus(self, arr):
        """点群 arr(np.ndarray [N,3]) の重心を焦点にし、適度な距離から見る"""
        if arr.size == 0:
            return
        center = arr.mean(axis=0)  # 重心
        # 点群の広がり（半径相当）をざっくり計算して距離決め
        radius = np.linalg.norm(arr - center, axis=1).max() if arr.shape[0] > 1 else 100.0
        dist = max(1.0, radius * 2.5)  # 見やすい距離
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(center[0], center[1], center[2])
        # 視点は +Z 方向から（好みで変更可）
        cam.SetPosition(center[0], center[1], center[2] + dist)
        cam.SetViewUp(0, 1, 0)
        self.ren.ResetCameraClippingRange()

    def _update_pointcloud(self, arr):
        self.pointSource.Reset()
        self.vertices.Reset()

        for i, p in enumerate(arr):
            pid = self.pointSource.InsertNextPoint(p)
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(pid)

        self.polyData.SetPoints(self.pointSource)
        self.polyData.SetVerts(self.vertices)
        self.polyData.Modified()

        self.mapper.SetInputData(self.polyData)
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(5)

        # ★ 追加：点群に合わせてアークボール中心を再設定
        self._arcball_focus(arr)

        self.GetRenderWindow().Render()

    def update_points(self, new_points):
        global points
        points = new_points
        self._update_pointcloud(points)

    def set_vertices(self, vertices):
        """vertices: Vertex オブジェクトのリスト (v.x, v.y, v.z, v.flag)"""
        
        # --- 点群データ ---
        
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        self.pointSource.Reset()
        self.vertices.Reset()

        for v in vertices:
            pid = self.pointSource.InsertNextPoint(v.x, v.y, v.z)
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(pid)
            if v.flag == 0:
                colors.InsertNextTuple3(255, 0, 0)  # 赤
            else:
                colors.InsertNextTuple3(0, 255, 0)  # 緑


        self.polyData.SetPoints(self.pointSource)
        self.polyData.SetVerts(self.vertices)
        self.polyData.GetPointData().SetScalars(colors)
        self.polyData.Modified()

        self.mapper.SetInputData(self.polyData)
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(5)

        self._arcball_focus(self.vertices_to_xyz(vertices))

        self.GetRenderWindow().Render()
        self.ren.ResetCamera()
        self.iren.Initialize()