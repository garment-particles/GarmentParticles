import sys
import numpy as np
import random
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPolygon
from PIL import Image, ImageDraw



from definitions import load_vertices, Vertex, Vertex2D
from point_editor import PointEditor
from paint_editor import PaintEditor









    
# ==============================
# 2D ビュー (QLabel + Pillow)
# ==============================
# class ProjectionPanel(PaintEditor):
class ProjectionPanel(PointEditor):
    def __init__(self):
        self.POINT_RADIUS = 0.5 # world space size
        self.CURSOR_RADIUS = 5 #screen space size # 1
        self.MIN_CURSOR_SIZE = 0.5
        super().__init__()
        # 親クラスの __init__ を呼ぶ
    def get_vertex2D(self, v):
        return Vertex2D(v.x, v.y)
    def get_vertex2Ds(self, vertices):
        xys = []
        for v in vertices:
            xys.append(Vertex2D(v.x, v.y))
        return xys
    
    def get_Vertex(self, px, py, flag):
        return Vertex(0,0,px,py,0, flag)