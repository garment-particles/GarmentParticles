import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QLabel, QToolBar, QAction, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PIL import Image, ImageDraw

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from definitions import *


from pattern_panel import PatternPanel
from projection_panel import ProjectionPanel

from viewer_panel import VTKPointViewer
from viewer_panel_PyQt import PointCloudWidget

from garment import predict_points


from point_editor import PointEditor
from paint_editor import PaintEditor

from inference import *

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D + 3D Point Editor")


        # メニューバー作成
        menubar = self.menuBar()

        # 「ファイル」メニューを追加
        file_menu = menubar.addMenu("File")
        open_txt_action = QAction("Open(txt)", self)
        open_npy_action = QAction("Open(npy)", self)
        inference_action = QAction("Inference", self)
        exit_action = QAction("Exit", self)
        open_txt_action.triggered.connect(self.open_txt_dialog)
        open_npy_action.triggered.connect(self.open_npy_dialog)
        inference_action.triggered.connect(self.inference)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_txt_action)
        file_menu.addAction(open_npy_action)
        file_menu.addAction(inference_action)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        toolbar = QToolBar("tool")
        toolbar.setOrientation(Qt.Vertical)  # 縦方向
        toolbar.setMovable(False)            # 動かせないように固定
        toolbar.setStyleSheet("""
            QToolButton {
                font-size: 20px;        /* ← フォントサイズを指定 */
                padding: 6px 10px;      /* ← 余白を調整 */
                text-align: left;             
            }
        """)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        labels = ["red", "green", "erase", "predict"]
        for label in labels:
            act = QAction(label, self)
            act.triggered.connect(self.on_action_triggered)
            toolbar.addAction(act)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # vertices = load_vertices("./pred_4.txt")
        vertices = load_npy("./example_guide.npy")

        # 2Dビュー
        self.projectionPanel = ProjectionPanel()
        self.projectionPanel.set_vertices(vertices)
        self.projectionPanel.render_vertices() 
        splitter.addWidget(self.projectionPanel)

        # 2Dビュー
        self.patternPanel = PatternPanel()
        self.patternPanel.set_vertices(vertices) 
        self.patternPanel.render_vertices() 
        splitter.addWidget(self.patternPanel)

        # 3Dビュー
        # self.vtkWidget = VTKPointViewer()
        self.vtkWidget = PointCloudWidget()
        self.vtkWidget.set_vertices(vertices)
        splitter.addWidget(self.vtkWidget)
        

        splitter.setSizes([500, 500, 500])

    def show_help(self):
        message = (
            "Left Drag \t draw / erase\n"
            "Middle drag \t pan \n"
            "Wheel \t\t zoom\n"
            "shift Wheel\t brush size \n"
        )
        QMessageBox.information(self, "How to Use", message)

    def open_txt_dialog(self):
        # QFileDialog.getOpenFileName(parent, caption, initial_dir, filter)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",              # ダイアログタイトル
            "",                            # 初期ディレクトリ（空ならカレント）
            "text (*.txt)"
        )

        if file_path:
            print("選択されたファイル:", file_path)
            vertices = load_vertices(file_path)
            self.projectionPanel.set_vertices(vertices) 
            self.vtkWidget.set_vertices(vertices)
        else:
            print("キャンセルされました。")


    def open_npy_dialog(self):
        # QFileDialog.getOpenFileName(parent, caption, initial_dir, filter)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",              # ダイアログタイトル
            "",                            # 初期ディレクトリ（空ならカレント）
            "text (*.npy)"
        )

        if file_path:
            vertices = load_npy(file_path)
            self.projectionPanel.set_vertices(vertices) 
            self.vtkWidget.set_vertices(vertices)
        else:
            print("キャンセルされました。")

    def inference(self):
        inference = Inference()
        vertices = inference.run_inference(self.projectionPanel._vertices)

        self.projectionPanel.set_vertices(vertices) 
        self.patternPanel.set_vertices(vertices) 
        self.vtkWidget.set_vertices(vertices)

    def on_action_triggered(self):
        action = self.sender()  # ← どの QAction が押されたか取得
        print(f"{action.text()} が押されました")
        if action.text() == "red":
            self.projectionPanel.brush_color = QColor(255,0, 0)
            self.patternPanel.brush_color = QColor(255,0, 0)
        elif action.text() == "green":
            self.projectionPanel.brush_color = QColor(0, 128, 0)
            self.patternPanel.brush_color = QColor(0, 128, 0)
        elif action.text() == "erase":
            self.projectionPanel.brush_color = None
            self.patternPanel.brush_color = None
        elif action.text() == "predict":
            if isinstance(self.projectionPanel, PaintEditor):
                self.projectionPanel._vertices = self.projectionPanel.image_to_vertices()
                self.projectionPanel.render_vertices() 
            predict_points(self.projectionPanel._vertices, self.set_vertices )


    
  

# ==============================
# メイン
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1700, 500)
    window.move(100, 100)   
    window.show()
    sys.exit(app.exec_())
