import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect

class DrawableLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.start_point = None
        self.end_point = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)

    def mousePressEvent(self, event):
        if self.pixmap():
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_point:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.end_point = event.pos()
        self.update()
        # 여기서 ROI 사용 가능

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())

class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("확실히 보이는 사각형")

        self.label = DrawableLabel()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.load_btn)
        self.setLayout(layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.label.setPixmap(QPixmap(path))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = Viewer()
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec_())
