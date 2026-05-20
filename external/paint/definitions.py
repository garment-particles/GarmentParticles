from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPolygon
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import List
import math
import numpy as np
@dataclass
class Vertex:
    u: float
    v: float
    x: float
    y: float
    z: float
    flag: int
    
    @staticmethod
    def distance(x0, y0, x1, y1):
        return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)


class Vertex2D():
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y


def load_vertices(filepath: str) -> List[Vertex]:
    """
    pred_1.txt 形式 (u v x y z flag) を 1 行ごとに読み込み、Vertex のリストで返す。
    先頭が '#' の行や空行はスキップする。
    """
    vertices: List[Vertex] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"[Line {ln}] 要素数が不足しています: {line}")
            try:
                u = float(parts[0])
                v = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                flag = float(parts[5])
            except ValueError as e:
                raise ValueError(f"[Line {ln}] 変換エラー: {e}; 行内容: {line}") from e
            vertices.append(Vertex(u, v, x, y, z, flag))
    return vertices

def load_npy(filepath: str) -> List[Vertex]:
    """
    [[x y z], ,,, ] 
    """
    data = np.load(filepath) 
    vertices: List[Vertex] = []
    for point in data:
        u = 0 #point[0]
        v = 0 #point[1]
        x = point[0]
        y = point[1]
        z = point[2]
        flag = 0 # point[5]
        vertices.append(Vertex(u,v,x,y,z, flag))
    return vertices

def prediction_to_vertices(data):
    vertices: List[Vertex] = []
    for point in data:
        u = point[0]
        v = point[1]
        x = point[2]
        y = point[3]
        z = point[4]
        flag = point[5]
        vertices.append(Vertex(u,v,x,y,z, flag))
    return vertices



# === 使い方例 ===
if __name__ == "__main__":
    verts = load_vertices("pred_1.txt")
    print(f"loaded: {len(verts)} points")
    # xyz = vertices_to_xyz(verts)  # VTK などに渡したい場合に利用
