# core/image_utils.py

import cv2
import numpy as np

def read_image_unicode(path):
    with open(path, "rb") as f:
        bytes_array = bytearray(f.read())
        np_array = np.asarray(bytes_array, dtype=np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def convert_to_qimage(cv_img):
    """BGR -> RGB -> QImage"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    from PyQt5.QtGui import QImage
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

def extract_roi(image, start, end, widget_size):
    """PyQt 위젯 좌표계 → 실제 이미지 ROI 변환"""
    from PyQt5.QtCore import QRect

    rect = QRect(start, end).normalized()
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

    img_h, img_w, _ = image.shape
    label_w, label_h = widget_size.width(), widget_size.height()
    scale_w, scale_h = img_w / label_w, img_h / label_h

    roi = image[
        int(y1 * scale_h):int(y2 * scale_h),
        int(x1 * scale_w):int(x2 * scale_w)
    ]
    return roi
