# -*- coding: utf-8 -*-
"""
PyQt5 版 - 视频字幕去除器 GUI（延迟导入 torch，不卡主线程）
Author: You
"""

import sys
import os
import cv2
import traceback
import configparser
from threading import Thread
from typing import List, Optional, Tuple
import importlib

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGroupBox, QSlider, QProgressBar, QPlainTextEdit,
    QFrame, QAction, QToolBar, QStyle, QSizePolicy, QMessageBox
)

# --- 复用你的项目结构（只导入轻量模块） ---
try:
    from backend.tools.common_tools import is_image_file
except Exception:
    # 兜底：简单判断图片扩展名
    def is_image_file(path: str) -> bool:
        ext = os.path.splitext(path.lower())[1]
        return ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def bgr_to_qpixmap(img_bgr, dst_size: Optional[Tuple[int, int]] = None) -> QPixmap:
    """OpenCV BGR -> QPixmap，并按需缩放。"""
    if img_bgr is None:
        return QPixmap()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if dst_size:
        pix = pix.scaled(dst_size[0], dst_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


def letterbox(frame, target_w, target_h):
    """等比例缩放并加边到指定尺寸（保持坐标比例一致）。"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (target_h - nh) // 2
    bottom = target_h - nh - top
    left = (target_w - nw) // 2
    right = target_w - nw - left
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas


class SubtitleRemoverWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 路径 / 配置 ---
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.subtitle_config_file = os.path.join(self.app_dir, 'subtitle.ini')

        # --- 窗口 ---
        self.setWindowTitle("Video Subtitle Remover (PyQt5)")
        icon_path = os.path.join(self.app_dir, 'design', 'vsr.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1280, 780)

        # --- 状态 ---
        self.video_paths: List[str] = []
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.frame_count: int = 0
        self.fps: float = 0.0
        self.frame_w: int = 0
        self.frame_h: int = 0

        # 字幕区域：y 起点 / 高度 / x 起点 / 宽度
        self.y_val = 0
        self.h_val = 0
        self.x_val = 0
        self.w_val = 0

        # 预览尺寸
        self.preview_w = 960
        self.preview_h = 540

        # 后端处理器（运行时才导入 backend.main）
        self.sr = None  # backend.main.SubtitleRemover 实例
        self.worker_thread: Optional[Thread] = None

        # UI
        self._build_toolbar()
        self._build_central()

        # 定时器轮询后端进度/预览
        self.timer = QTimer(self)
        self.timer.setInterval(50)  # 20fps
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

        # 样式
        self._apply_qss()

        # 默认区域
        self._load_area_config()

    # ---------------- UI ----------------
    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(Qt.TopToolBarArea, tb)

        act_open = QAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "打开(多选)", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.on_open_files)
        tb.addAction(act_open)

        act_run = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "运行", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self.on_run)
        tb.addAction(act_run)

        tb.addSeparator()

        act_quit = QAction(self.style().standardIcon(QStyle.SP_DialogCloseButton), "退出", self)
        act_quit.triggered.connect(self.close)
        tb.addAction(act_quit)

    def _build_central(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        # 左侧：预览 + 时间轴
        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(self.preview_w, self.preview_h)
        self.lbl_preview.setFrameShape(QFrame.NoFrame)
        self.lbl_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setEnabled(False)
        self.slider_timeline.valueChanged.connect(self.on_seek_frame)

        left = QVBoxLayout()
        left.addWidget(self.lbl_preview, stretch=1)
        left.addWidget(self.slider_timeline)

        # 右侧：控制区
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("输出日志...")

        # 垂直（Y 起点 / 高度）
        grp_v = QGroupBox("Vertical")
        self.slider_y = self._make_vslider()
        self.slider_h = self._make_vslider()
        self.slider_y.valueChanged.connect(self._on_y_changed)
        self.slider_h.valueChanged.connect(self._on_h_changed)
        vbox_v = QVBoxLayout()
        vbox_v.addWidget(self.slider_y)
        vbox_v.addWidget(self.slider_h)
        grp_v.setLayout(vbox_v)

        # 水平（X 起点 / 宽度）
        grp_h = QGroupBox("Horizontal")
        self.slider_x = self._make_vslider()
        self.slider_w = self._make_vslider()
        self.slider_x.valueChanged.connect(self._on_x_changed)
        self.slider_w.valueChanged.connect(self._on_w_changed)
        vbox_h = QVBoxLayout()
        vbox_h.addWidget(self.slider_x)
        vbox_h.addWidget(self.slider_w)
        grp_h.setLayout(vbox_h)

        # 运行 + 进度条
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_run.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        right = QVBoxLayout()
        right.addWidget(self.txt_log, stretch=1)
        groups = QHBoxLayout()
        groups.addWidget(grp_v)
        groups.addWidget(grp_h)
        right.addLayout(groups)
        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_run)
        bottom.addWidget(self.progress)
        right.addLayout(bottom)

        layout = QHBoxLayout(root)
        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=2)

    def _make_vslider(self) -> QSlider:
        s = QSlider(Qt.Vertical)
        s.setRange(0, 0)        # 初始禁用
        s.setEnabled(False)
        s.setSingleStep(1)
        s.setPageStep(8)
        s.setTickPosition(QSlider.NoTicks)
        return s

    def _apply_qss(self):
        qss_path = os.path.join(self.app_dir, "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # ---------------- 逻辑 ----------------
    def log(self, msg: str):
        self.txt_log.appendPlainText(msg)

    def _enable_controls(self, enabled: bool):
        self.slider_timeline.setEnabled(enabled and self.cap is not None)
        for s in (self.slider_y, self.slider_h, self.slider_x, self.slider_w):
            s.setEnabled(enabled and self.frame_w > 0 and self.frame_h > 0)
        self.btn_run.setEnabled(enabled and (self.video_path is not None))

    def on_open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频/图片（可多选）",
            "", "All Files (*.*);;Video (*.mp4 *.flv *.wmv *.avi *.mov *.mkv);;Image (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not files:
            return
        self.video_paths = files[:]
        self.video_path = self.video_paths[0]
        self.log("Open Video/Image Success:")
        for p in self.video_paths:
            self.log(f"  - {p}")

        # 打开第一项获取属性
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(self.video_path) if not is_image_file(self.video_path) else None

        first_frame = None
        if is_image_file(self.video_path):
            img = cv2.imread(self.video_path)
            if img is None:
                QMessageBox.warning(self, "错误", "无法读取图片。")
                return
            self.frame_h, self.frame_w = img.shape[:2]
            self.frame_count = 1
            self.fps = 0
            first_frame = img
            self.slider_timeline.setRange(1, 1)
            self.slider_timeline.setValue(1)
        else:
            if not self.cap or not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开视频。")
                return
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
            self.slider_timeline.setRange(1, max(1, self.frame_count))
            self.slider_timeline.setValue(1)
            ok, frame = self.cap.read()
            if ok:
                first_frame = frame
            else:
                QMessageBox.warning(self, "错误", "无法读取首帧。")
                return

        # 初始化区域滑块
        self._init_area_sliders()
        if first_frame is not None:
            self._update_preview(first_frame)
        self._enable_controls(True)

    def _init_area_sliders(self):
        y_p, h_p, x_p, w_p = self._load_area_config()
        y = int(self.frame_h * y_p)
        h = int(self.frame_h * h_p)
        x = int(self.frame_w * x_p)
        w = int(self.frame_w * w_p)

        self.slider_y.blockSignals(True)
        self.slider_h.blockSignals(True)
        self.slider_x.blockSignals(True)
        self.slider_w.blockSignals(True)

        self.slider_y.setRange(0, self.frame_h if self.frame_h else 0)
        self.slider_y.setValue(y)

        self.slider_h.setRange(0, max(0, self.frame_h - y))
        self.slider_h.setValue(h)

        self.slider_x.setRange(0, self.frame_w if self.frame_w else 0)
        self.slider_x.setValue(x)

        self.slider_w.setRange(0, max(0, self.frame_w - x))
        self.slider_w.setValue(w)

        self.slider_y.blockSignals(False)
        self.slider_h.blockSignals(False)
        self.slider_x.blockSignals(False)
        self.slider_w.blockSignals(False)

    def _update_preview(self, frame):
        # 根据滑块画矩形
        y = self.slider_y.value()
        h = self.slider_h.value()
        x = self.slider_x.value()
        w = self.slider_w.value()

        y = max(0, min(y, self.frame_h))
        x = max(0, min(x, self.frame_w))
        h = max(0, min(h, self.frame_h - y))
        w = max(0, min(w, self.frame_w - x))

        draw = frame.copy()
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 3)

        boxed = letterbox(draw, self.preview_w, self.preview_h)
        pix = bgr_to_qpixmap(boxed, (self.lbl_preview.width(), self.lbl_preview.height()))
        self.lbl_preview.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.video_path:
            frame = None
            if is_image_file(self.video_path):
                frame = cv2.imread(self.video_path)
            elif self.cap:
                pos = self.slider_timeline.value()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, f = self.cap.read()
                if ok:
                    frame = f
            if frame is not None:
                self._update_preview(frame)

    # ---- 滑块联动 ----
    def _on_y_changed(self, val: int):
        self.slider_h.setRange(0, max(0, self.frame_h - val))
        self._preview_current()

    def _on_h_changed(self, _: int):
        self._preview_current()

    def _on_x_changed(self, val: int):
        self.slider_w.setRange(0, max(0, self.frame_w - val))
        self._preview_current()

    def _on_w_changed(self, _: int):
        self._preview_current()

    def _preview_current(self):
        if not self.video_path:
            return
        if is_image_file(self.video_path):
            img = cv2.imread(self.video_path)
            if img is not None:
                self._update_preview(img)
        elif self.cap and self.cap.isOpened():
            pos = self.slider_timeline.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = self.cap.read()
            if ok:
                self._update_preview(frame)

    def on_seek_frame(self, frame_no: int):
        if not self.cap or not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = self.cap.read()
        if ok:
            self._update_preview(frame)

    # ---- 运行处理（延迟导入 torch） ----
    def on_run(self):
        if not self.video_path:
            QMessageBox.information(self, "提示", "请先打开视频或图片。")
            return

        # 计算绝对区域 & 写入配置
        y = int(self.slider_y.value())
        h = int(self.slider_h.value())
        x = int(self.slider_x.value())
        w = int(self.slider_w.value())

        y = max(0, min(y, self.frame_h))
        x = max(0, min(x, self.frame_w))
        h = max(0, min(h, self.frame_h - y))
        w = max(0, min(w, self.frame_w - x))

        y_p = y / self.frame_h if self.frame_h else 0
        h_p = h / self.frame_h if self.frame_h else 0
        x_p = x / self.frame_w if self.frame_w else 0
        w_p = w / self.frame_w if self.frame_w else 0
        self._save_area_config(y_p, h_p, x_p, w_p)

        # 多文件分辨率一致则共用区域
        subtitle_area = None
        if len(self.video_paths) <= 1:
            subtitle_area = (y, y + h, x, x + w)
        else:
            self.log("Processing multiple videos/images ...")
            global_size = None
            same_size = True
            for p in self.video_paths:
                if is_image_file(p):
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    tsize = (img.shape[1], img.shape[0])
                else:
                    tcap = cv2.VideoCapture(p)
                    tsize = (int(tcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(tcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    tcap.release()
                if global_size is None:
                    global_size = tsize
                else:
                    if tsize != global_size:
                        same_size = False
                        break
            if same_size:
                subtitle_area = (y, y + h, x, x + w)
            else:
                self.log("not all video/images in same size, processing in full screen")
                subtitle_area = None

        # 禁用控件
        self._enable_controls(False)

        # 启动后台线程：逐个处理（此处延迟导入 backend.main -> torch）
        def task():
            try:
                backend_main = importlib.import_module("backend.main")
                queue = list(self.video_paths)
                while queue:
                    path = queue.pop(0)
                    if subtitle_area is not None:
                        self.log(f"SubtitleArea: ({y},{y+h},{x},{x+w})")

                    self.sr = backend_main.SubtitleRemover(path, subtitle_area, True)
                    self.sr.run()  # 阻塞直到该文件处理完成
                    self.sr = None
                # 全部完成后启用控件
                self._enable_controls(True)
                self.log("All done.")
            except Exception as e:
                tb = traceback.format_exc()
                self.log(f"[{type(e)}] {e}\n{tb}")
                self._enable_controls(True)
                self.sr = None

        self.worker_thread = Thread(target=task, daemon=True)
        self.worker_thread.start()

        # 当前视频资源释放
        if self.cap:
            self.cap.release()
            self.cap = None

    def _on_timer(self):
        """定时刷新进度与预览（不改控件启用状态，统一在 worker 完成后处理）"""
        if self.sr is not None:
            try:
                # 进度
                prog = getattr(self.sr, "progress_total", None)
                if isinstance(prog, (int, float)):
                    self.progress.setValue(int(prog))

                # 预览帧
                prev = getattr(self.sr, "preview_frame", None)
                if prev is not None:
                    boxed = letterbox(prev, self.preview_w, self.preview_h)
                    pix = bgr_to_qpixmap(boxed, (self.lbl_preview.width(), self.lbl_preview.height()))
                    self.lbl_preview.setPixmap(pix)
            except Exception:
                pass

    # ---- 配置读写 ----
    def _load_area_config(self):
        # 默认：y=.78, h=.21, x=.05, w=.9
        y_p, h_p, x_p, w_p = .78, .21, .05, .9
        if not os.path.exists(self.subtitle_config_file):
            self._save_area_config(y_p, h_p, x_p, w_p)
            return y_p, h_p, x_p, w_p
        try:
            config = configparser.ConfigParser()
            config.read(self.subtitle_config_file, encoding="utf-8")
            y_p = float(config["AREA"]["Y"])
            h_p = float(config["AREA"]["H"])
            x_p = float(config["AREA"]["X"])
            w_p = float(config["AREA"]["W"])
            return y_p, h_p, x_p, w_p
        except Exception:
            self._save_area_config(y_p, h_p, x_p, w_p)
            return y_p, h_p, x_p, w_p

    def _save_area_config(self, y, h, x, w):
        with open(self.subtitle_config_file, "w", encoding="utf-8") as f:
            f.write("[AREA]\n")
            f.write(f"Y = {y}\n")
            f.write(f"H = {h}\n")
            f.write(f"X = {x}\n")
            f.write(f"W = {w}\n")

    def closeEvent(self, event):
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        super().closeEvent(event)


def main():
    # Windows 上可避免某些多进程问题
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
    except Exception:
        pass

    app = QApplication(sys.argv)
    win = SubtitleRemoverWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
