import sys
import numpy as np
import random
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPolygon
from PIL import Image, ImageDraw



from definitions import load_vertices, Vertex, Vertex2D


IMG_SIZE = 500

POINT_COLOR = (0, 0, 0)  # 黒






    
# ==============================
# 2D ビュー (QLabel + Pillow)
# ==============================
class PointEditor(QLabel):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.setPixmap(self.pil2pixmap(self.image))
     
        self.points2d = []

 

        # ★ 追加：UVの表示範囲（初回計算用）
        self._uv_bbox = None  # (umin, umax, vmin, vmax)
        self._uv_scale = 1.0
        self._uv_origin = (0.0, 0.0)  # 左上(描画原点)に相当するUVの基準
        self._margin = 12  # 余白

        # --- パン・ズーム用 ---
        self._panning = False
        self._last_mouse_pos = None
        self._zoom_min = 0.1
        self._zoom_max = 20.0
        self._zoom_step = 1.1  # ホイール1ノッチでの倍率

        # ★ イベントを確実に受けるための設定
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(IMG_SIZE, IMG_SIZE)
        self.setAttribute(Qt.WA_Hover, True)

        self.brush_color = QColor(255,0, 0)
        self.cursor_pt = None


    # ==============================
    # Pillow画像をQPixmapに変換
    # ==============================
    def pil2pixmap(self, im):
        im = im.convert("RGBA")
        data = im.tobytes("raw", "RGBA")
        qimg = QImage(data, im.size[0], im.size[1], QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    # ★ 新メソッド：UV bbox / scale / offset の初期計算
    def _init_uv_transform(self, vertices):
        vertices = self.get_vertex2Ds(vertices)

        u_vals = [p.x for p in vertices]
        v_vals = [p.y for p in vertices]
        umin, umax = min(u_vals), max(u_vals)
        vmin, vmax = min(v_vals), max(v_vals)

        # 縮退対策
        if umax - umin < 1e-9: umax = umin + 1e-3
        if vmax - vmin < 1e-9: vmax = vmin + 1e-3

        self._uv_bbox = (umin, umax, vmin, vmax)

        M = self._margin
        avail_w = IMG_SIZE - 2 * M
        avail_h = IMG_SIZE - 2 * M
        w_uv = (umax - umin)
        h_uv = (vmax - vmin)

        # スケーリング（等方）
        S = min(avail_w / w_uv, avail_h / h_uv)
        self._uv_scale = S

        # 中央配置用オフセット
        w_px = w_uv * S
        h_px = h_uv * S
        left  = M + (avail_w - w_px) * 0.5
        top   = M + (avail_h - h_px) * 0.5
        self._uv_offset = (left, top)

        self.CURSOR_RADIUS = self.CURSOR_RADIUS / S
        self.POINT_RADIUS = self.POINT_RADIUS 
    # -------------------------
    # 変換ユーティリティ
    # -------------------------
    def _xy_to_pt(self, u, v):
        (umin, umax, vmin, vmax) = self._uv_bbox
        S = self._uv_scale
        left, top = self._uv_offset
        x = left + (u - umin) * S
        y = top  + (vmax - v) * S  # v を上向き表示にするため反転
        return float(x), float(y)
    def _xy_to_pt_scale(self, scale):   
        return scale*self._uv_scale
    def _pt_to_xy_scale(self, scale):   
        return scale/self._uv_scale

    def _pt_to_xy(self, x, y):
        (umin, umax, vmin, vmax) = self._uv_bbox
        S = self._uv_scale
        left, top = self._uv_offset
        u = (x - left) / S + umin
        v = vmax - (y - top) / S
        return float(u), float(v)
    
    def find_nearby_vertex(self, px,py):
        for vertex in self._vertices:
            v = self.get_vertex2D(vertex)
            d = Vertex.distance(v.x, v.y, px, py)
            if (d < self.CURSOR_RADIUS):
                return vertex
        return None
    def find_nearby_vertices(self, px,py):
        nearby_vertices = []
        for vertex in self._vertices:
            v = self.get_vertex2D(vertex)
            d = Vertex.distance(v.x, v.y, px, py)
            if (d < self.CURSOR_RADIUS):
                nearby_vertices.append(vertex)
        return nearby_vertices
        
    def erase_vertices(self, pt):
        px, py = self._pt_to_xy(pt.x(), pt.y())
        new_vertices = []
        for vertex in self._vertices:
            v = self.get_vertex2D(vertex)
            d = Vertex.distance(v.x, v.y, px, py)
            if (d > self.CURSOR_RADIUS):
                new_vertices.append(vertex)
        self._vertices = new_vertices
        self.repaint()
 


    def spray(self, x, y, flag):
        R = self.CURSOR_RADIUS   # 半径（必要に応じて外部定義してもOK）
        MARGIN = 1          # 近すぎとみなす距離の閾値
        N_TRY = 20         # 試行回数

        for _ in range(N_TRY):
            # --- 半径Rの円内にランダムな点を生成 ---
            r = R * math.sqrt(random.random())       # √をかけて一様分布化
            theta = random.uniform(0, 2 * math.pi)
            px = x + r * math.cos(theta)
            py = y + r * math.sin(theta)

            # --- 近い頂点があるか確認 ---
            too_close = False
            for vertex in self._vertices:
                v = self.get_vertex2D(vertex)
                dist = math.hypot(px - v.x, py - v.y)
                if dist < MARGIN:
                    too_close = True
                    break
       
            # --- 近い頂点がなければ追加 ---
            if not too_close:
                vertex = self.get_Vertex(px, py, flag)
                self._vertices.append(vertex)


    def add_vertex(self, pt, flag):
        
        x,y = self._pt_to_xy(pt.x(), pt.y())

        nearby_vertices = self.find_nearby_vertices(x,y)
        if len(nearby_vertices)>0:
            for v in nearby_vertices:
                v.flag = flag
        else:
                vertex = self.get_Vertex(x, y, flag)
                self._vertices.append(vertex)
        self.spray(x, y, flag)
        self.repaint()

    def paint_cursor(self, painter):
        if (self.cursor_pt is None):
            return
        
        pen = QPen()
        pen.setWidth(3)                 # ← 線の太さ（ピクセル単位）
        if self.brush_color is None:
            pen.setColor(QColor(0,0,0))
            pen.setStyle(Qt.DotLine)  
        else:
            pen.setColor(self.brush_color)
            pen.setStyle(Qt.SolidLine)  
        painter.setPen(pen)

        painter.setBrush(Qt.NoBrush)       
        painter.drawEllipse(self.cursor_pt, self._xy_to_pt_scale(self.CURSOR_RADIUS), self._xy_to_pt_scale(self.CURSOR_RADIUS))

    # ---- 再描画（外部/内部どちらからでも呼べる） ----
    def set_vertices(self, vertices):
        self._vertices = vertices  
        self.repaint()
    def render_vertices(self):
        self.repaint()
    def repaint(self):
        verts = self._vertices

        # 画面クリア
        if self.image is None:
            self.image = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
            self.draw = ImageDraw.Draw(self.image)
        else:
            self.image.paste((255, 255, 255, 255), [0, 0, IMG_SIZE, IMG_SIZE])


        if not verts:
            self.setPixmap(self.pil2pixmap(self.image)); self.update(); return

        if self._uv_bbox is None:
            self._init_uv_transform(verts)

        r = self._xy_to_pt_scale(self.POINT_RADIUS)
        for vertex in verts:
            if (vertex.flag == 1):
                v = self.get_vertex2D(vertex)
                x, y = self._xy_to_pt(v.x, v.y)
                xi, yi = int(round(x)), int(round(y))
                self.draw.ellipse((xi - r, yi - r, xi + r, yi + r), fill=(0, 128, 0))
        for vertex in verts:
            if (vertex.flag == 0):
                v = self.get_vertex2D(vertex)
                x, y = self._xy_to_pt(v.x, v.y)
                xi, yi = int(round(x)), int(round(y))
                self.draw.ellipse((xi - r, yi - r, xi + r, yi + r), fill=(255, 0, 0))


   

        self.setPixmap(self.pil2pixmap(self.image))
        self.update()

    # ---- マウス（PyQt5互換: event.pos() を使用） ----
    def mousePressEvent(self, event):



        x, y = event.pos().x(), event.pos().y()   # ★ event.position() → pos()
        inside = (0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE)

        if event.button() == Qt.LeftButton and self.brush_color is not None:
            flag = 1 if (self.brush_color.red() == 0 and self.brush_color.green() == 128 and self.brush_color.blue() == 0) else 0
            self.add_vertex(event.pos(), flag)
            self.cursor_pt = event.pos()
            self.repaint()

        elif event.button() == Qt.LeftButton and self.brush_color is None:
            self.erase_vertices(event.pos())
            self.update()
 
        elif event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self._panning = True
            self._last_mouse_pos = (x, y)
            event.accept()
            return

        elif self._uv_bbox is None or not inside:
            return



    def mouseMoveEvent(self, event):

        if  (event.buttons() & Qt.LeftButton) and self.brush_color is not None:
            flag = 1 if (self.brush_color.red() == 0 and self.brush_color.green() == 128 and self.brush_color.blue() == 0) else 0
            self.add_vertex(event.pos(), flag)
            self.cursor_pt = event.pos()
            self.repaint()
            #self.update()
        elif (event.buttons() & Qt.LeftButton) and self.brush_color is None:
            self.erase_vertices(event.pos())
            self.cursor_pt = event.pos()
            self.update()

        elif (self._panning and self._last_mouse_pos is not None):
            x, y = event.pos().x(), event.pos().y()
            lx, ly = self._last_mouse_pos
            dx, dy = (x - lx), (y - ly)
            left, top = self._uv_offset
            self._uv_offset = (left + dx, top + dy)
            self._last_mouse_pos = (x, y)
            # ★ ここで即再描画
            self.cursor_pt = None
            self.repaint()

        else:
            self.cursor_pt = event.pos()
            self.repaint()


    def mouseReleaseEvent(self, event):

       

        if event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self._last_mouse_pos = None

    def wheelEvent(self, event):
        if self._uv_bbox is None:
            return

        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.2 if delta > 0 else (1.0 / 1.2)
            if self.CURSOR_RADIUS * factor > self.MIN_CURSOR_SIZE:
                self.CURSOR_RADIUS = self.CURSOR_RADIUS * factor 
                self.repaint()
            return
        
        # ★ PyQt5 互換：カーソル座標は pos() を使う
        mx, my = event.pos().x(), event.pos().y()

        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self._zoom_step if delta > 0 else (1.0 / self._zoom_step)

        # 新スケール（クランプ）
        new_scale = self._uv_scale * factor
        new_scale = max(self._uv_scale * self._zoom_min, min(self._uv_scale * self._zoom_max, new_scale))

        # カーソル位置のUVを不動にする補正
        u_fix, v_fix = self._pt_to_xy(mx, my)
        (umin, umax, vmin, vmax) = self._uv_bbox
        new_left = mx - (u_fix - umin) * new_scale
        new_top  = my - (vmax - v_fix) * new_scale

        self._uv_scale = float(new_scale)
        self._uv_offset = (float(new_left), float(new_top))

        # ★ ここで即再描画
        self.repaint()

    def leaveEvent(self, event):
        self.cursor_pt = None
        super().leaveEvent(event)
        self.update()
    def paintEvent(self, event):
        # QLabel の通常描画（背景など）
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

 
        self.paint_cursor(painter)
