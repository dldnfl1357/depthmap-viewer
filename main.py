import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import DepthmapViewer

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DepthmapViewer()
    viewer.show()  # 전체화면 대신 일반 창으로 시작
    sys.exit(app.exec_())