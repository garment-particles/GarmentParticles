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
#class PatternPanel(PaintEditor):
class PatternPanel(PointEditor):
    def __init__(self):
        # 親クラスの __init__ を呼ぶ
        super().__init__()
        self.POINT_RADIUS = 0.005   # pattern space size
        self.CURSOR_RADIUS = 2  #screen space size #0.05
        self.MIN_CURSOR_SIZE = 0.005


    def get_vertex2D(self, v):
        return Vertex2D(v.u, v.v)
    def get_vertex2Ds(self, vertices):
        xys = []
        for v in vertices:
            xys.append(Vertex2D(v.u, v.v))
        return xys
    
    def get_Vertex(self, px, py, flag):
        return Vertex(px,py,0,0,0, flag)