#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt


class VideoScrubber(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3DAMAME - 3D Actuation Magnitude Approximation of Moving Entity")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.cap = None
        self.total_frames = 0
        self.fps = 25.0
        self.current_frame_idx = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame_auto)
        self.current_frame = None
        self.key_frame = None
        self.roi = None
        self.selecting_roi = False
        self.roi_start = None
        self.lower_thresh = 30
        self.upper_thresh = 255
        self.analysis_running = False
        self.diff_intensity = []

        self._build_ui()
        self.setMinimumSize(800, 700)

    def _build_ui(self):


        # Video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.mousePressEvent = self.start_roi_selection
        self.video_label.mouseMoveEvent = self.update_roi_selection
        self.video_label.mouseReleaseEvent = self.finish_roi_selection
        self.video_label.installEventFilter(self)


        self.overlay_label = QtWidgets.QLabel(self.video_label)
        self.overlay_label.setStyleSheet("background-color: rgba(0, 0, 0, 128);")  # semi-transparent black
        self.overlay_label.setVisible(False)
        self.overlay_label.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
        self.overlay_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay_label.setText("Analysis Running...")
        self.overlay_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 128);
            color: white;
            font-size: 24px;
        """)


        # Controls
        self.load_btn = QtWidgets.QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)

        self.prev_btn = QtWidgets.QPushButton("⟨ Prev")
        self.prev_btn.clicked.connect(self.prev_frame)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)

        self.next_btn = QtWidgets.QPushButton("Next ⟩")
        self.next_btn.clicked.connect(self.next_frame)

        self.key_btn = QtWidgets.QPushButton("Set Key Frame")
        self.key_btn.clicked.connect(self.set_key_frame)

        self.roi_btn = QtWidgets.QPushButton("Clear ROI")
        self.roi_btn.clicked.connect(self.clear_roi)

        self.analysis_btn = QtWidgets.QPushButton("Run Analysis")
        self.analysis_btn.clicked.connect(self.run_analysis)

        self.frame_label = QtWidgets.QLabel("Frame: 0 / 0")
        self.fps_label = QtWidgets.QLabel("FPS: -")


        self.logo_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap("logo.png")  # Replace with your logo path
        #pixmap = pixmap.scaledToWidth(150)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setScaledContents(True)  # scale logo to fit label size
        #self.logo_label.setFixedSize(int(185/2), int(155/2))    # width x height in pixels

        # Frame slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setSingleStep(1)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.valueChanged.connect(self.slider_moved)
        self._slider_is_pressed = False

        # Threshold sliders
        self.lower_label = QtWidgets.QLabel(f"Lower Threshold: {self.lower_thresh}")
        self.lower_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lower_slider.setMinimum(0)
        self.lower_slider.setMaximum(255)
        self.lower_slider.setValue(self.lower_thresh)
        self.lower_slider.valueChanged.connect(self.update_lower_thresh)
        self.lower_slider.setFixedWidth(150)  # pixels

        self.upper_label = QtWidgets.QLabel(f"Upper Threshold: {self.upper_thresh}")
        self.upper_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.upper_slider.setMinimum(0)
        self.upper_slider.setMaximum(255)
        self.upper_slider.setValue(self.upper_thresh)
        self.upper_slider.valueChanged.connect(self.update_upper_thresh)
        self.upper_slider.setFixedWidth(150)


        # Layouts
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.key_btn)
        controls.addWidget(self.roi_btn)
        controls.addWidget(self.analysis_btn)
        

        info_layout = QtWidgets.QHBoxLayout()
        info_layout.addWidget(self.frame_label)
        info_layout.addWidget(self.fps_label)
        
        

        threshold_layout = QtWidgets.QVBoxLayout()
        threshold_layout.addWidget(self.lower_label)
        threshold_layout.addWidget(self.lower_slider)
        threshold_layout.addWidget(self.upper_label)
        threshold_layout.addWidget(self.upper_slider)

        logo_layout = QtWidgets.QHBoxLayout()
        logo_layout.addStretch()
        logo_layout.addWidget(self.logo_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.slider)
        layout.addLayout(controls)
        layout.addLayout(info_layout)
        layout.addLayout(threshold_layout)
        layout.addLayout(logo_layout)


        
    def eventFilter(self, source, event):
        if source == self.video_label and event.type() == QtCore.QEvent.Resize:
            self.overlay_label.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
        return super().eventFilter(source, event)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay_label.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
        if self.cap is not None:
            self.show_frame(self.current_frame_idx)

    def load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", QtCore.QDir.homePath(), "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if not path:
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Could not open the selected video file.")
            return

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 25.0
        self.total_frames = total if total > 0 else 0
        self.current_frame_idx = 0

        self.slider.setMaximum(max(0, self.total_frames - 1))
        self.slider.setValue(0)
        self.frame_label.setText(f"Frame: 0 / {self.total_frames}")
        self.fps_label.setText(f"FPS: {self.fps:.2f}")

        self.show_frame(0)

    def set_controls_enabled(self, enabled: bool):
        """Enable or disable UI controls."""
        self.load_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.play_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.key_btn.setEnabled(enabled)
        self.roi_btn.setEnabled(enabled)
        self.analysis_btn.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.lower_slider.setEnabled(enabled)
        self.upper_slider.setEnabled(enabled)
        # Show overlay if disabled (analysis running)
        self.overlay_label.setVisible(not enabled)


    def show_frame(self, frame_idx):
        if self.cap is None:
            return
        if self.total_frames > 0:
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        elif frame_idx < 0:
            frame_idx = 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame_idx = frame_idx
        self.current_frame = frame.copy()
        self.update_ui_after_frame()

        display_frame = frame.copy()
        


        if self.key_frame is not None:
            diff = self.compute_difference(frame, self.key_frame)
            
            if self.roi is not None:
                x, y, w, h = self.roi
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                display_frame[y:y + h, x:x + w] = diff[y:y + h, x:x + w]

            else:
                display_frame = diff

        

        if self.roi and self.selecting_roi:
            x, y, w, h = self.roi
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def set_key_frame(self):
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "No Frame", "No current frame to set as key.")
            return
        self.key_frame = self.current_frame.copy()
        QtWidgets.QMessageBox.information(self, "Key Frame Set", f"Frame {self.current_frame_idx} set as key frame.")

    def clear_roi(self):
        self.roi = None
        self.show_frame(self.current_frame_idx)

    def compute_difference(self, frame, key_frame):
        if frame.shape != key_frame.shape:
            key_frame = cv2.resize(key_frame, (frame.shape[1], frame.shape[0]))
        if self.roi is not None:
            x, y, w, h = self.roi
            roi_frame = frame[y:y + h, x:x + w]
            roi_key = key_frame[y:y + h, x:x + w]
            diff = cv2.absdiff(roi_frame, roi_key)
        else:
            diff = cv2.absdiff(frame, key_frame)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.lower_thresh, self.upper_thresh, cv2.THRESH_BINARY)
        heatmap = cv2.applyColorMap(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)

        if self.roi is not None:
            full_diff = frame.copy()
            full_diff[y:y + h, x:x + w] = heatmap
            return full_diff
        else:
            return heatmap

    def start_roi_selection(self, event):
        if self.current_frame is None:
            return
        self.selecting_roi = True
        self.roi_start = (event.x(), event.y())

    def update_roi_selection(self, event):
        if self.selecting_roi and self.roi_start is not None:
            x0, y0 = self.roi_start
            x1, y1 = event.x(), event.y()
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            self.roi = self._label_to_frame_coords(x, y, w, h)
            self.show_frame(self.current_frame_idx)

    def finish_roi_selection(self, event):
        if self.selecting_roi:
            self.selecting_roi = False
            if self.roi is not None:
                QtWidgets.QMessageBox.information(self, "ROI Set", f"ROI defined: {self.roi}")

    def _label_to_frame_coords(self, x, y, w, h):
        if self.current_frame is None or self.video_label.pixmap() is None:
            return None
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        frame_h, frame_w = self.current_frame.shape[:2]
        scale = min(label_w / frame_w, label_h / frame_h)
        x_offset = (label_w - frame_w * scale) / 2
        y_offset = (label_h - frame_h * scale) / 2
        x = int((x - x_offset) / scale)
        y = int((y - y_offset) / scale)
        w = int(w / scale)
        h = int(h / scale)
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        return (x, y, w, h)

    def update_ui_after_frame(self):
        self.frame_label.setText(f"Frame: {self.current_frame_idx} / {self.total_frames}")
        if not self._slider_is_pressed:
            self.slider.setValue(self.current_frame_idx)

    def update_lower_thresh(self, value):
        self.lower_thresh = value
        self.lower_label.setText(f"Lower Threshold: {value}")
        self.show_frame(self.current_frame_idx)

    def update_upper_thresh(self, value):
        self.upper_thresh = value
        self.upper_label.setText(f"Upper Threshold: {value}")
        self.show_frame(self.current_frame_idx)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.cap is not None:
            self.show_frame(self.current_frame_idx)

    def prev_frame(self):
        if self.cap is None:
            return
        new_idx = max(0, self.current_frame_idx - 1)
        self.show_frame(new_idx)

    def next_frame(self):
        if self.cap is None:
            return
        if self.total_frames > 0:
            new_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
            self.show_frame(new_idx)
        else:
            next_idx = self.current_frame_idx + 1
            self.show_frame(next_idx)

    def next_frame_auto(self):
        if self.cap is None:
            self.timer.stop()
            self.play_btn.setChecked(False)
            self.play_btn.setText("Play")
            return
        if self.total_frames > 0 and self.current_frame_idx >= (self.total_frames - 1):
            self.toggle_play(False)
            return
        self.next_frame()

    def toggle_play(self, checked=None):
        if checked is None:
            checked = not self.play_btn.isChecked()
            self.play_btn.setChecked(checked)
        if checked:
            interval = int(1000.0 / max(1.0, self.fps))
            self.timer.start(interval)
            self.play_btn.setText("Pause")
        else:
            self.timer.stop()
            self.play_btn.setText("Play")

    def slider_pressed(self):
        self._slider_is_pressed = True

    def slider_released(self):
        self._slider_is_pressed = False
        pos = self.slider.value()
        self.show_frame(pos)

    def slider_moved(self, value):
        if self._slider_is_pressed:
            self.show_frame(value)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self.prev_frame()
        elif event.key() == QtCore.Qt.Key_Right:
            self.next_frame()
        elif event.key() == QtCore.Qt.Key_Space:
            self.toggle_play()
        elif event.key() == QtCore.Qt.Key_K:
            self.set_key_frame()
        elif event.key() == QtCore.Qt.Key_R:
            self.clear_roi()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        event.accept()

    def run_analysis(self):
        if self.cap is None or self.key_frame is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Load video and set a key frame first.")
            return

        self.analysis_running = True
        self.diff_intensity = []
        self.current_frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Disable all controls during analysis
        self.set_controls_enabled(False)

        # Disconnect the normal timer
        self.timer.stop()

        # Set up timer for analysis playback
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.process_next_frame_for_analysis)
        self.timer.start(int(1000 / self.fps))
    
    def process_next_frame_for_analysis(self):
        if self.current_frame_idx >= self.total_frames:
            self.timer.stop()
            self.analysis_running = False
            self.plot_analysis()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.analysis_running = False
            self.set_controls_enabled(True)
            self.plot_analysis()
            return

        self.current_frame = frame
        if self.key_frame is not None:
            if self.roi is not None:
                x, y, w, h = self.roi
                diff = self.compute_difference(frame, self.key_frame)
                intensity = self.calculate_diff_intensity(diff[y:y + h, x:x + w])
                self.diff_intensity.append(intensity)
            else:
                diff = self.compute_difference(frame, self.key_frame)
                intensity = self.calculate_diff_intensity(diff)
                self.diff_intensity.append(intensity)

        self.current_frame_idx += 1
        self.frame_label.setText(f"Frame: {self.current_frame_idx} / {self.total_frames}")


    def calculate_diff_intensity(self, diff_frame):
        # Convert diff frame to grayscale and calculate mean intensity
        gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        total_intensity = np.sum(gray)
        return total_intensity

    def plot_analysis(self):
        times = np.arange(len(self.diff_intensity)) / self.fps
        normalised_diff = np.array(self.diff_intensity, dtype=float)
        res = (normalised_diff - normalised_diff.min()) / (normalised_diff.max() - normalised_diff.min())
        plt.figure(figsize=(10, 2))
        plt.plot(times, res)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Normalised\nMovement Intensity')
        plt.title('Movement Intensity Over Time')
        plt.xlim(0, int(self.total_frames/self.fps))
        plt.xticks(range(0,int(self.total_frames/self.fps), 1))
        plt.tight_layout()
        plt.show()
        plt.savefig("Fig.png",dpi=300)
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.next_frame_auto)

    
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = VideoScrubber()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
