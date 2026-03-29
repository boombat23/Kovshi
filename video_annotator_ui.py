import sys
import os
import cv2
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QCheckBox, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class ImageLabel(QLabel):
    mouseMoved = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(640, 480)
        self.show_center_line = False
        self.show_coordinates = False
        self.current_mouse_pos = None

        # Annotation state
        self.annotation_mode = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.bboxes = []  # List of tuples (x, y, w, h) in original image coordinates

    def get_image_coordinate(self, pos):
        if not self.pixmap():
            return None

        pix_size = self.pixmap().size()
        lbl_size = self.size()
        x_offset = (lbl_size.width() - pix_size.width()) // 2
        y_offset = (lbl_size.height() - pix_size.height()) // 2

        img_x = pos.x() - x_offset
        img_y = pos.y() - y_offset

        if 0 <= img_x < pix_size.width() and 0 <= img_y < pix_size.height():
            orig_w = self.property("orig_w") or pix_size.width()
            orig_h = self.property("orig_h") or pix_size.height()

            scale_x = orig_w / pix_size.width()
            scale_y = orig_h / pix_size.height()

            return int(img_x * scale_x), int(img_y * scale_y)
        return None

    def get_widget_coordinate(self, orig_x, orig_y):
        if not self.pixmap():
            return None

        pix_size = self.pixmap().size()
        lbl_size = self.size()
        x_offset = (lbl_size.width() - pix_size.width()) // 2
        y_offset = (lbl_size.height() - pix_size.height()) // 2

        orig_w = self.property("orig_w") or pix_size.width()
        orig_h = self.property("orig_h") or pix_size.height()

        if orig_w == 0 or orig_h == 0:
            return None

        scale_x = pix_size.width() / orig_w
        scale_y = pix_size.height() / orig_h

        widget_x = int(orig_x * scale_x) + x_offset
        widget_y = int(orig_y * scale_y) + y_offset

        return widget_x, widget_y

    def mousePressEvent(self, event):
        if self.annotation_mode and event.button() == Qt.MouseButton.LeftButton:
            img_coord = self.get_image_coordinate(event.pos())
            if img_coord:
                self.drawing = True
                self.start_point = img_coord
                self.end_point = img_coord
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.current_mouse_pos = event.pos()
        self.mouseMoved.emit(event.pos())

        if self.drawing and self.annotation_mode:
            img_coord = self.get_image_coordinate(event.pos())
            if img_coord:
                self.end_point = img_coord

        if self.show_coordinates or self.drawing:
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and self.annotation_mode and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            img_coord = self.get_image_coordinate(event.pos())
            if img_coord:
                self.end_point = img_coord

                # Create bbox
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                if w > 3 and h > 3:  # minimum size
                    self.bboxes.append((x, y, w, h))
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.pixmap():
            return

        painter = QPainter(self)

        # Calculate the actual image area
        pix_size = self.pixmap().size()
        lbl_size = self.size()
        x_offset = (lbl_size.width() - pix_size.width()) // 2
        y_offset = (lbl_size.height() - pix_size.height()) // 2

        if self.show_center_line:
            pen = QPen(QColor(255, 0, 0, 150))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)

            # Draw horizontal center line over the image area
            center_y = y_offset + pix_size.height() // 2
            painter.drawLine(x_offset, center_y, x_offset + pix_size.width(), center_y)

        if self.show_coordinates and self.current_mouse_pos:
            # Convert widget coordinates to image coordinates
            img_x = self.current_mouse_pos.x() - x_offset
            img_y = self.current_mouse_pos.y() - y_offset

            if 0 <= img_x < pix_size.width() and 0 <= img_y < pix_size.height():
                # Map scaled coordinates to original image coordinates
                orig_w = self.property("orig_w") or pix_size.width()
                orig_h = self.property("orig_h") or pix_size.height()

                scale_x = orig_w / pix_size.width()
                scale_y = orig_h / pix_size.height()

                real_x = int(img_x * scale_x)
                real_y = int(img_y * scale_y)

                text = f"X: {real_x}, Y: {real_y}"
                painter.setPen(QColor(255, 255, 0))
                # Draw text background
                font_metrics = painter.fontMetrics()
                rect = font_metrics.boundingRect(text)
                rect.translate(self.current_mouse_pos.x() + 10, self.current_mouse_pos.y() + 10)
                # padding
                rect.adjust(-2, -2, 2, 2)
                painter.fillRect(rect, QColor(0, 0, 0, 150))
                painter.drawText(self.current_mouse_pos.x() + 10, self.current_mouse_pos.y() + 20, text)

        if self.annotation_mode:
            # Draw existing bboxes
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)

            for (bx, by, bw, bh) in self.bboxes:
                w_tl = self.get_widget_coordinate(bx, by)
                w_br = self.get_widget_coordinate(bx + bw, by + bh)
                if w_tl and w_br:
                    painter.drawRect(w_tl[0], w_tl[1], w_br[0] - w_tl[0], w_br[1] - w_tl[1])

            # Draw bbox currently being drawn
            if self.drawing and self.start_point and self.end_point:
                pen = QPen(QColor(0, 255, 255))
                pen.setWidth(2)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)

                x1, y1 = self.start_point
                x2, y2 = self.end_point
                bx = min(x1, x2)
                by = min(y1, y2)
                bw = abs(x2 - x1)
                bh = abs(y2 - y1)

                w_tl = self.get_widget_coordinate(bx, by)
                w_br = self.get_widget_coordinate(bx + bw, by + bh)
                if w_tl and w_br:
                    painter.drawRect(w_tl[0], w_tl[1], w_br[0] - w_tl[0], w_br[1] - w_tl[1])

class VideoAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Frame Annotator")
        self.resize(1024, 768)

        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.current_frame = None
        self.output_dir = Path("annotated_dataset")

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        # Top bar for controls
        self.top_controls = QHBoxLayout()

        self.btn_load_video = QPushButton("Load Video")
        self.btn_set_output = QPushButton("Set Output Folder")
        self.chk_center_line = QCheckBox("Show Center Line")
        self.chk_coordinates = QCheckBox("Show Coordinates")
        self.chk_annotation = QCheckBox("Annotation Mode")

        self.top_controls.addWidget(self.btn_load_video)
        self.top_controls.addWidget(self.btn_set_output)
        self.top_controls.addWidget(self.chk_center_line)
        self.top_controls.addWidget(self.chk_coordinates)
        self.top_controls.addWidget(self.chk_annotation)
        self.top_controls.addStretch()

        self.main_layout.addLayout(self.top_controls)

        # Image display area
        self.image_label = ImageLabel()
        self.main_layout.addWidget(self.image_label, stretch=1)

        # We need this to get coordinate mappings
        self.image_label.setProperty("orig_w", 0)
        self.image_label.setProperty("orig_h", 0)

        # Bottom bar for navigation
        self.bottom_controls = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.bottom_controls.addWidget(self.slider)
        self.bottom_controls.addWidget(self.lbl_frame_info)

        self.main_layout.addLayout(self.bottom_controls)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Use Left/Right arrows to navigate.")

    def connect_signals(self):
        self.btn_load_video.clicked.connect(self.load_video)
        self.slider.valueChanged.connect(self.set_frame)
        self.chk_center_line.stateChanged.connect(self.toggle_center_line)
        self.chk_coordinates.stateChanged.connect(self.toggle_coordinates)
        self.chk_annotation.stateChanged.connect(self.toggle_annotation)
        self.btn_set_output.clicked.connect(self.set_output_folder)

    def set_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = Path(folder)
            self.status_bar.showMessage(f"Output folder set to: {self.output_dir}")

    def toggle_annotation(self, state):
        self.image_label.annotation_mode = (state == Qt.CheckState.Checked.value)
        if not self.image_label.annotation_mode:
            self.image_label.drawing = False
        self.image_label.update()

    def toggle_center_line(self, state):
        self.image_label.show_center_line = (state == Qt.CheckState.Checked.value)
        self.image_label.update()

    def toggle_coordinates(self, state):
        self.image_label.show_coordinates = (state == Qt.CheckState.Checked.value)
        self.image_label.update()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if not file_path:
            return

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.status_bar.showMessage(f"Failed to open {file_path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setEnabled(True)
        self.slider.setValue(0)

        self.status_bar.showMessage(f"Loaded {os.path.basename(file_path)}")
        self.update_frame(0)

    def update_frame(self, frame_idx):
        if self.cap is None or not self.cap.isOpened():
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            self.lbl_frame_info.setText(f"Frame: {frame_idx + 1} / {self.total_frames}")
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)
            self.image_label.bboxes = []  # Clear bboxes when changing frame
            self.display_image()

    def set_frame(self, value):
        self.update_frame(value)

    def save_frame_and_annotations(self):
        if self.current_frame is None:
            return

        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"frame_{self.current_frame_idx:06d}"
        img_path = images_dir / f"{base_name}.jpg"
        lbl_path = labels_dir / f"{base_name}.txt"

        # Save image
        cv2.imwrite(str(img_path), self.current_frame)

        # Save YOLO annotations if any
        if self.image_label.bboxes:
            h, w = self.current_frame.shape[:2]
            with open(lbl_path, "w") as f:
                for (bx, by, bw, bh) in self.image_label.bboxes:
                    # Convert to YOLO format (class x_center y_center width height)
                    x_center = (bx + bw / 2.0) / w
                    y_center = (by + bh / 2.0) / h
                    norm_w = bw / w
                    norm_h = bh / h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        else:
            # Create an empty file to indicate no objects (optional, but good practice)
            lbl_path.touch()

        self.status_bar.showMessage(f"Saved {base_name} and annotations.")

    def display_image(self):
        if self.current_frame is None:
            return

        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        self.image_label.setProperty("orig_w", w)
        self.image_label.setProperty("orig_h", h)

        # Create QImage from numpy array
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale pixmap to fit label while keeping aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            if self.current_frame_idx < self.total_frames - 1:
                self.update_frame(self.current_frame_idx + 1)
        elif event.key() == Qt.Key.Key_Left:
            if self.current_frame_idx > 0:
                self.update_frame(self.current_frame_idx - 1)
        elif event.key() == Qt.Key.Key_Space:
            self.save_frame_and_annotations()
        elif event.key() == Qt.Key.Key_U:
            # Undo last bbox
            if self.image_label.bboxes:
                self.image_label.bboxes.pop()
                self.image_label.update()
        elif event.key() == Qt.Key.Key_C:
            # Clear bboxes
            self.image_label.bboxes = []
            self.image_label.update()
        else:
            super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    window = VideoAnnotator()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
