#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth map 생성 파이프라인 - C++ 코드를 Python으로 이식
"""

import os
import sys
import glob
import time
import numpy as np
import cv2
import fnmatch
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# 상수 및 타입 정의
# -----------------------------
FP_PREC = np.float64
Two_Pi = 2 * np.pi
Nan = np.nan

# -----------------------------
# 유틸리티 함수
# -----------------------------
def raw_images(input_dir):
    """C++ Raw_images"""
    exts = ('*.jpg','*.jpeg','*.png','*.bmp')
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(input_dir, e))
    imgs = []
    for p in sorted(paths):
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"cannot read image {p}")
        imgs.append(gray)
    print(f"Raw_images: loaded {len(imgs)} images")
    return imgs

def gaussian_blur(img, ksize):
    """C++ Gaussian_Blur"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def binarized_images_by_average_fringe_value(gray_imgs, fringe_imgs):
    """C++ Binarized_images_by_average_fringe_value"""
    # gray_imgs, fringe_imgs: list of 2D arrays, same shape
    h, w = gray_imgs[0].shape
    FN = len(fringe_imgs)
    avg = np.zeros((h,w), dtype=np.float64)
    for f in fringe_imgs:
        avg += f.astype(np.float64)
    avg /= FN
    res = []
    for g in gray_imgs:
        res.append((g.astype(np.float64) > avg).astype(np.uint8))
    return res

def wrapped_phase_and_modulation_image(fringe_imgs):
    """C++ Wrapped_phase_and_modulation_image"""
    h, w = fringe_imgs[0].shape
    FN = len(fringe_imgs)
    # unit circle vectors
    angles = [np.exp(1j * k * Two_Pi / FN) for k in range(FN)]
    wrapped = np.zeros((h,w), dtype=FP_PREC)
    modulation = np.zeros((h,w), dtype=FP_PREC)
    for k,f in enumerate(fringe_imgs):
        f64 = f.astype(FP_PREC)
        wrapped     += f64 * angles[k]
        modulation  += f64
    # wrapped: sum(m*e^{iθ_k}), modulation: sum(m)
    arg = -np.angle(wrapped)
    mod_map = 256 * np.sqrt(np.abs(wrapped)**2) / modulation
    return arg, mod_map

def unwrapped_phase_image(wrapped, bin_gray):
    """C++ Unwrapped_phase_image"""
    # bin_gray: list of GN binary images
    GN = len(bin_gray)
    lut = np.array([
        1,32,16,17,8,25,9,24,
        4,29,13,20,5,28,12,21,
        2,31,15,18,7,26,10,23,
        3,30,14,19,6,27,11,22
    ], dtype=np.uint8)
    h,w = wrapped.shape
    unwrapped = np.zeros_like(wrapped)
    for y in range(h):
        for x in range(w):
            # build gray code
            code = 0
            for k in range(GN):
                if bin_gray[k][y,x]:
                    code |= (1<<k)
            cw = lut[code]
            unwrapped[y,x] = wrapped[y,x] + Two_Pi * cw
    return unwrapped

def filtered_by_median_phase(unwrapped, ksize):
    """C++ Filtered_by_median_phase (1D 수평 median)"""
    # 여기서는 2D medianBlur로 대체
    return cv2.medianBlur(unwrapped.astype(np.float32), ksize).astype(FP_PREC)

# -----------------------------
# 캘리브레이션 데이터 로딩
# -----------------------------
class CalibrationData:
    """C++ Calibration_data"""
    pass

def calibration_data_from_file(xml_path):
    """C++ Calibration_data_from_file"""
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    data = CalibrationData()
    # Intrinsics
    K1 = fs.getNode("C_intrinsic_matrix").mat()
    K2 = fs.getNode("P_intrinsic_matrix").mat()
    data.leftIntrinsic  = K1.astype(FP_PREC)
    data.rightIntrinsic = K2.astype(FP_PREC)
    # Extrinsics
    R = fs.getNode("R_PtoC").mat()
    T = fs.getNode("T_PtoC").mat()
    data.leftR = R.astype(FP_PREC)
    data.leftT = T.astype(FP_PREC)
    # Fundamental or Projection matrices
    F1 = fs.getNode("F_CtoP").mat()
    F2 = fs.getNode("F_PtoC").mat()
    data.leftF  = F1.astype(FP_PREC)
    data.rightF = F2.astype(FP_PREC)
    fs.release()
    return data

# -----------------------------
# 프로젝터 위상 파라미터
# -----------------------------
class UnwrappedProjectorPhase:
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def phase(self,x):
        return self.a*x + self.b
    def x(self,phase):
        return (phase - self.b) / self.a

def projector_phase_image(xml_path):
    """C++ Projector_phase_image"""
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    coeff = fs.getNode("Linear_coefficients").mat().astype(FP_PREC)
    fs.release()
    flat = coeff.flatten()
    if flat.size < 2:
        raise RuntimeError(f"Linear_coefficients has unexpected size: {coeff.shape}")
    a, b = flat[0], flat[1]
    return UnwrappedProjectorPhase(a, b)

def to_focal_plane(intrin, x, y):
    """C++ To_focal_plane"""
    fx, fy = intrin[0,0], intrin[1,1]
    cx, cy = intrin[0,2], intrin[1,2]
    vx = (x - cx) / fx
    vy = (y - cy) / fy
    v0 = np.array([vx - vy*intrin[0,1], vy, 1.0], dtype=FP_PREC)
    return v0

# -----------------------------
# 삼각 보정
# -----------------------------
def depthmap_by_triangulation(cam_unwrap, proj_unwrap, calib, depth_band):
    """C++ Depthmap_by_triangulation"""
    h,w = cam_unwrap.shape
    min_d, max_d = depth_band
    # Fmat: calib.leftF (3x3 or 3x4)
    Fmat = calib.leftF
    depth = np.zeros((h,w), dtype=FP_PREC)
    for y in range(h):
        for x in range(w):
            phi_cam = cam_unwrap[y,x]
            if not np.isfinite(phi_cam):
                continue
            # corresponding proj pixel
            F_campt = Fmat @ np.array([x,y,1.],dtype=FP_PREC)
            if abs(F_campt[1])<1e-8: continue
            xp = proj_unwrap.x(phi_cam)
            yp = -(F_campt[0]*xp + F_campt[2]) / F_campt[1]
            # triangulate (2D intersection)
            # 간단화: ray-plane intersection
            # 실제 C++ 로직만큼 정확하진 않습니다.
            P0 = np.array([0,0,0],dtype=FP_PREC)
            C0 = to_focal_plane(calib.leftIntrinsic, x, y)
            Pp = to_focal_plane(calib.rightIntrinsic, xp, yp)
            # skew-lines nearest point (approx)
            u = C0/np.linalg.norm(C0)
            v = (calib.leftR @ Pp + calib.leftT)
            v = v/np.linalg.norm(v)
            A = np.array([[u@u, -u@v],
                          [u@v, -v@v]],dtype=FP_PREC)
            b = np.array([u@P0, v@calib.leftT],dtype=FP_PREC)
            try:
                s = np.linalg.solve(A,b)
            except np.linalg.LinAlgError:
                continue
            h0 = s[0]*u + P0
            h1 = s[1]*v + calib.leftT.flatten()
            pt = (h0+h1)/2
            z = pt[2]
            if min_d<=z<=max_d:
                depth[y,x] = z
    return depth

# -----------------------------
# 포인트 클라우드 저장
# -----------------------------
def save_point_cloud(path, depth_map, calib):
    h,w = depth_map.shape
    pts = []
    for y in range(h):
        for x in range(w):
            z = depth_map[y,x]
            if z>0:
                fp = to_focal_plane(calib.leftIntrinsic, x, y)
                pts.append([z*fp[0], z*fp[1], z])
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property double x\nen property double y\nproperty double z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

# -----------------------------
# 메인 테스트 파이프라인
# -----------------------------
import os
import fnmatch

def find_input_folder(root_dir):
    # 선택된 폴더 자체가 *_input.d라면 바로 사용
    if root_dir.endswith("_input.d") and os.path.isdir(root_dir):
        return root_dir

    # 하위 폴더 중 *_input.d를 찾아서 리턴
    for name in os.listdir(root_dir):
        if name.endswith("_input.d"):
            cand = os.path.join(root_dir, name)
            if os.path.isdir(cand):
                return cand

    raise RuntimeError(f"No '*_input.d' folder under {root_dir}")

def main(input_dir, output_dir):
    in_dir  = find_input_folder(input_dir)
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    calib = calibration_data_from_file(os.path.join(in_dir,"calib_param.xml"))
    proj_unwrap = projector_phase_image(os.path.join(in_dir,"projector_phase_spec.xml"))

    raw = raw_images(in_dir)
    img_h, img_w = raw[0].shape

    # blur
    blur_gray   = [gaussian_blur(img, 5) for img in raw[:5]]
    blur_fringe = [gaussian_blur(img,5) for img in raw[5:10]]

    # wrapped & modulation
    wrapped, modulation = wrapped_phase_and_modulation_image(blur_fringe)
    cv2.imwrite(os.path.join(out_dir,"modulation.bmp"), modulation.astype(np.uint8))

    # binarize gray
    bin_gray = binarized_images_by_average_fringe_value(blur_gray, blur_fringe)

    # unwrap
    unwrapped = unwrapped_phase_image(wrapped, bin_gray)

    # filter
    filtered = filtered_by_median_phase(unwrapped, 5)

    # mask out-of-range
    min_p = proj_unwrap.phase(0)
    max_p = proj_unwrap.phase(img_w)
    filtered[(filtered<min_p)|(filtered>max_p)] = 0

    # depth
    depth = depthmap_by_triangulation(filtered, proj_unwrap, calib, (75,95))

    # save
    save_point_cloud(os.path.join(out_dir,"point_cloud.ply"), depth, calib)
    cv2.imwrite(os.path.join(out_dir,"depthmap.bmp"), (depth/depth.max()*255).astype(np.uint8))
    print("Done!")

def choose_input_directory():
    root = tk.Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    folder = filedialog.askdirectory(title="Select Input Folder")
    root.destroy()
    return folder

if __name__ == "__main__":
    # 1) 다이얼로그로 입력 폴더 선택
    input_dir = choose_input_directory()
    if not input_dir:
        print("폴더를 선택하지 않아 종료합니다.")
        sys.exit(0)

    # 2) output_dir 자동 생성 (input과 같은 레벨)
    parent = os.path.dirname(input_dir)
    base   = os.path.basename(input_dir)
    if base.endswith("_input.d"):
        out_name = base.replace("_input.d", "_output.d")
    else:
        out_name = base + "_output"
    output_dir = os.path.join(parent, out_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3) 파이프라인 실행
    main(input_dir, output_dir)