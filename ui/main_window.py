from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QTabWidget, QFileDialog, QScrollArea, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
# from core.image_utils import load_image (불필요한 import 제거됨)


class DepthmapViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depthmap Viewer with Validation Tools")
        self.resize(1280, 720)  # 기본 창 크기 설정
        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()

        self.view_tab = self.create_view_tab()
        self.fringe_tab = self.create_fringe_tab()
        self.phase_tab = self.create_phase_tab()
        self.depth_tab = self.create_depth_tab()

        self.tabs.addTab(self.view_tab, "View")
        self.tabs.addTab(self.fringe_tab, "Fringe")
        self.tabs.addTab(self.phase_tab, "Phase")
        self.tabs.addTab(self.depth_tab, "Depth")

        self.setCentralWidget(self.tabs)

    def create_view_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # 상단 Graycode 이미지 표시 영역
        self.gray_scroll = QScrollArea()
        self.gray_scroll.setWidgetResizable(True)
        self.gray_widget = QWidget()
        self.gray_layout = QHBoxLayout()
        self.gray_layout.setAlignment(Qt.AlignLeft)
        self.gray_widget.setLayout(self.gray_layout)
        self.gray_scroll.setWidget(self.gray_widget)

        # 하단 Fringe 이미지 표시 영역
        self.fringe_scroll = QScrollArea()
        self.fringe_scroll.setWidgetResizable(True)
        self.fringe_widget = QWidget()
        self.fringe_layout = QHBoxLayout()
        self.fringe_layout.setAlignment(Qt.AlignLeft)
        self.fringe_widget.setLayout(self.fringe_layout)
        self.fringe_scroll.setWidget(self.fringe_widget)

        # 버튼
        btn_load_gray = QPushButton("Load Graycode Images")
        btn_load_gray.clicked.connect(self.load_graycode_images)

        btn_load_fringe = QPushButton("Load Fringe Images")
        btn_load_fringe.clicked.connect(self.load_fringe_images)

        layout.addWidget(btn_load_gray)
        layout.addWidget(self.gray_scroll, stretch=1)
        layout.addWidget(btn_load_fringe)
        layout.addWidget(self.fringe_scroll, stretch=1)

        tab.setLayout(layout)
        return tab

    def load_graycode_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Graycode Images", "", "Images (*.bmp *.png *.jpg)")
        if files:
            # 기존 위젯 초기화
            for i in reversed(range(self.gray_layout.count())):
                widget_to_remove = self.gray_layout.itemAt(i).widget()
                self.gray_layout.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)

            for file in sorted(files, key=lambda x: os.path.getsize(x)):
                pixmap = QPixmap(file)
                label = QLabel()
                label.setPixmap(pixmap.scaledToHeight(360, Qt.SmoothTransformation))
                self.gray_layout.addWidget(label)

    def load_fringe_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Fringe Images", "", "Images (*.bmp *.png *.jpg)")
        if files:
            # 기존 위젯 초기화
            for i in reversed(range(self.fringe_layout.count())):
                widget_to_remove = self.fringe_layout.itemAt(i).widget()
                self.fringe_layout.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)

            for file in sorted(files, key=lambda x: os.path.getsize(x)):
                pixmap = QPixmap(file)
                label = QLabel()
                label.setPixmap(pixmap.scaledToHeight(360, Qt.SmoothTransformation))
                self.fringe_layout.addWidget(label)

    def create_fringe_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        btn_fringe = QPushButton("View Fringe Images")
        btn_fringe.clicked.connect(self.view_fringe_images)
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(btn_fringe)
        layout.addWidget(self.image_label)

        tab.setLayout(layout)
        return tab

    def create_phase_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        btn_wrapped = QPushButton("Generate Wrapped Phase")
        btn_unwrap = QPushButton("Unwrap Phase Map")
        btn_qmap = QPushButton("Show Q-map")

        btn_wrapped.clicked.connect(self.generate_wrapped_phase)
        btn_unwrap.clicked.connect(self.unwrap_phase)
        btn_qmap.clicked.connect(self.show_qmap)

        layout.addWidget(btn_wrapped)
        layout.addWidget(btn_unwrap)
        layout.addWidget(btn_qmap)
        layout.addWidget(self.image_label)

        tab.setLayout(layout)
        return tab

    def create_depth_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        btn_depth = QPushButton("Generate Depth Map")
        btn_full = QPushButton("Run Full Pipeline")

        btn_depth.clicked.connect(self.generate_depth_map)
        btn_full.clicked.connect(self.run_full_pipeline)

        layout.addWidget(btn_depth)
        layout.addWidget(btn_full)
        layout.addWidget(self.image_label)

        tab.setLayout(layout)
        return tab

    def view_fringe_images(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Fringe Image", "", "Images (*.bmp *.png *.jpg)")
        if file:
            pixmap = QPixmap(file)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def generate_wrapped_phase(self):
        print("[TODO] Wrapped phase generation logic here.")

    def unwrap_phase(self):
        print("[TODO] Phase unwrapping logic here.")

    def show_qmap(self):
        print("[TODO] Q-map generation and display.")

    def generate_depth_map(self):
        print("[TODO] Depth map calculation logic here.")

    def run_full_pipeline(self):
        print("[TODO] Run full pipeline: fringe -> phase -> unwrap -> depth.")
