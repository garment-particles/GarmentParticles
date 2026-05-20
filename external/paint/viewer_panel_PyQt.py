import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QSplitter, QLabel
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt, QPointF

from definitions import *

IMG_SIZE = 500
class PointCloudWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ★ イベントを確実に受けるための設定
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(IMG_SIZE, IMG_SIZE)
        self.setAttribute(Qt.WA_Hover, True)

        # ==== 点群（ここを自分の verts に差し替えてOK）====
        N = 3000
        self.points = np.random.uniform(-1, 1, (N, 3)).astype(np.float32)
        # 自分の点群が verts (N,3) なら：
        # self.points = verts.astype(np.float32)

        # カメラパラメータ
        self.yaw = 30.0    # 左右回転（度）  左ドラッグで変更
        self.pitch = 20.0  # 上下回転（度）  左ドラッグで変更
        self.distance = 3.0  # 視点距離（大きいほど遠く）固定でもOK
        self.zoom = 1.0      # ズーム倍率（ホイールで変更）

        # 画面平行移動（パン） 中ボタンドラッグで変更
        self.pan_x = 0.0
        self.pan_y = 0.0

        # マウスドラッグ用
        self.last_pos = None
        self.last_button = None

        self.setWindowTitle("3D Point Cloud (mouse: rotate/pan/zoom)")
        self.resize(800, 600)

    def set_vertices(self, vertices):
        self.points = np.zeros((len(vertices),3))
        for i in range(len(vertices)):
            v = vertices[i]
            self.points[i] = (v.x, v.y, v.z)

        self.adjust_view()

    def adjust_view(self):
       # ---------- 新しい点群をセットして自動フィット（カメラは点群の外側） ----------
        """
        新しい点群をセットして、点群の中心を見下ろすように
        カメラを点群の外側に置きつつ、画面にきれいに収まるように
        zoom / pan / yaw / pitch / distance を調整する。
        points: shape (N,3)
        """
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points は (N,3) の配列である必要があります")

        # --- 1. 点群を保存して中心を原点に移動 ---
        center = pts.mean(axis=0)
        pts = pts - center            # 中心を原点へ
        self.points = pts.copy()

        # --- 2. 点群のスケール（半径）に応じてカメラ距離を決める ---
        radii = np.linalg.norm(pts, axis=1)
        r = float(radii.max())
        if not np.isfinite(r) or r < 1e-6:
            r = 1.0  # ほぼ1点しかない場合など

        # 点群の外側から眺めるように、半径の何倍か離す
        self.distance = max(3.0, r * 3.0)  # 係数3.0は好みで調整可

        # --- 3. ビュー状態を初期化（カメラは少し斜め上） ---
        self.yaw = 30.0
        self.pitch = 20.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0

        # --- 4. この距離・姿勢で一度投影して、zoom を決める ---
        w = self.width() if self.width() > 0 else 800
        h = self.height() if self.height() > 0 else 600

        # yaw/pitch から回転行列（paintEvent と同じ）
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)

        Ry = np.array([
            [cos_y, 0.0, sin_y],
            [0.0,   1.0, 0.0  ],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float32)

        Rx = np.array([
            [1.0,  0.0,    0.0   ],
            [0.0,  cos_p, -sin_p],
            [0.0,  sin_p,  cos_p]
        ], dtype=np.float32)

        R = Ry @ Rx

        pts_rot = pts @ R.T  # (N,3)

        d = self.distance
        base_f = 500.0
        z = pts_rot[:, 2] + d
        z = np.clip(z, 0.1, None)

        xs = pts_rot[:, 0] * (base_f / z)
        ys = -pts_rot[:, 1] * (base_f / z)

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        size_x = max_x - min_x
        size_y = max_y - min_y

        if size_x > 0 and size_y > 0:
            margin = 0.8  # 画面の 80% に収める
            scale_x = margin * w / size_x
            scale_y = margin * h / size_y
            self.zoom = min(scale_x, scale_y)
        else:
            self.zoom = 1.0

        # pan は 0 のままで OK（中心を原点にしたので自然に中央付近に来る）
        self.update()


    # ---------- マウス操作 ----------
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        self.last_button = event.button()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return

        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        # 左ボタン：回転
        if self.last_button == Qt.LeftButton:
            # マウス移動量に応じて yaw/pitch を更新
            self.yaw -= dx * 0.5
            self.pitch -= dy * 0.5
            # ピッチが真上・真下を向きすぎないように制限
            self.pitch = max(-89.0, min(89.0, self.pitch))

        # 中ボタン：平行移動（パン）
        elif self.last_button == Qt.MiddleButton:
            # 単純にスクリーン座標上でオフセット
            self.pan_x += dx
            self.pan_y += dy

        self.last_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.last_pos = None
        self.last_button = None

    def wheelEvent(self, event):
        # ホイールでズーム（1ステップ 120）
        delta = event.angleDelta().y() / 120.0
        # 1.1倍ごとの指数的スケーリング
        self.zoom *= 1.1 ** delta
        # ズームの範囲を適当に制限
        self.zoom = max(0.1, min(10.0, self.zoom))
        self.update()

    # ---------- 描画 ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 背景
        painter.fillRect(self.rect(), Qt.black)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0

        # ==== 3D 回転行列を作る (yaw, pitch) ====
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)

        # yaw: y軸回り、pitch: x軸回り
        Ry = np.array([
            [cos_y, 0.0, sin_y],
            [0.0,   1.0, 0.0  ],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float32)

        Rx = np.array([
            [1.0,  0.0,    0.0   ],
            [0.0,  cos_p, -sin_p],
            [0.0,  sin_p,  cos_p]
        ], dtype=np.float32)

        R = Ry @ Rx

        # ==== 3D 点群に回転を適用 ====
        pts = self.points @ R.T  # (N, 3)

        # ==== 簡易透視投影 ====
        d = self.distance
        base_f = 500.0  # 基本の焦点距離
        z = pts[:, 2] + d
        z = np.clip(z, 0.1, None)  # 0 で割らないように

        factor = (self.zoom * base_f) / z

        xs = cx + pts[:, 0] * factor + self.pan_x
        ys = cy - pts[:, 1] * factor + self.pan_y

        # 深度を色とサイズに反映
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

        for x, y, zn in zip(xs, ys, z_norm):
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            size = 1.5 + (1.0 - zn) * 3.0
            intensity = int(80 + (1.0 - zn) * 175)
            color = QColor(intensity, intensity, 255)

            painter.setBrush(QBrush(color, Qt.SolidPattern))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), size, size)


def main():
    app = QApplication(sys.argv)
    w = PointCloudWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
