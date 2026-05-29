import cv2
import numpy as np
import time
import sys
from threading import Thread, Lock


class CameraBuffer:
    def __init__(self, source, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._lock     = Lock()
        self._frame    = None
        self._frame_id = 0
        self._running  = True
        self._thread   = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self._lock:
                    self._frame    = frame
                    self._frame_id += 1

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None, -1
            return True, self._frame.copy(), self._frame_id

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


class CaptureFrames:
    DETECT_INTERVAL = 10

    def __init__(self, bs, source, show_mask=False):
        self.batch_size          = bs
        self.show_mask           = show_mask
        self._last_face          = None
        self._face_confirm_count = 0
        self._confirmed_face     = None
        self._miss_count         = 0
        self.latest_hr           = 0.0
        self.latest_bvp          = 0.0
        self.target_res          = (64, 64)
        self._roi_coords         = None

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError("Haar cascade not found.")

    def __call__(self, pipe, result_pipe, source):
        self.pipe        = pipe
        self.result_pipe = result_pipe
        self._capture_loop(source)

    def _detect_face(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3,
                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
            )

        if len(faces) > 0:
            best = max(faces, key=lambda b: b[2] * b[3])
            if self._confirmed_face is not None:
                cx, cy, cw, ch = self._confirmed_face
                bx, by, bw, bh = best
                dist_x = abs((bx + bw//2) - (cx + cw//2))
                dist_y = abs((by + bh//2) - (cy + ch//2))
                if dist_x > cw * 0.6 or dist_y > ch * 0.6:
                    self._face_confirm_count = 0
                    self._last_face = None
                    return
            self._last_face = best
            self._face_confirm_count += 1
            if self._face_confirm_count >= 3:
                self._confirmed_face = best
            self._miss_count = 0
        else:
            self._face_confirm_count = 0
            self._last_face = None
            self._miss_count += 1
            if self._miss_count > 30:
                self._confirmed_face = None
                self._miss_count = 0

    def _get_forehead(self, frame_rgb):
        face = self._confirmed_face
        if face is None:
            return None, False

        H, W = frame_rgb.shape[:2]
        x, y, fw, fh = face

        cx = x + fw // 2
        cy = y + fh // 2

        roi_half_w = int(fw * 0.22)
        roi_half_h = int(fh * 0.10)
        offset_up  = int(fh * 0.20)

        top   = max(0, cy - offset_up - roi_half_h)
        bot   = min(H, cy - offset_up + roi_half_h)
        left  = max(0, cx - roi_half_w)
        right = min(W, cx + roi_half_w)

        if bot <= top or right <= left:
            return None, False

        roi     = frame_rgb[top:bot, left:right]
        resized = cv2.resize(roi, self.target_res, interpolation=cv2.INTER_LINEAR)
        self._roi_coords = (top, bot, left, right)

        return resized, True

    def _capture_loop(self, source):
        camera        = CameraBuffer(source)
        time.sleep(1.5)
        frames_seen   = 0
        last_frame_id = -1

        fps_counter    = 0
        fps_start_time = time.time()
        current_fps    = 0.0

        while True:
            if self.result_pipe.poll():
                data = self.result_pipe.recv()
                if isinstance(data, tuple):
                    self.latest_hr, self.latest_bvp = data
                else:
                    self.latest_hr  = data
                    self.latest_bvp = 0.0

            grabbed, orig, frame_id = camera.read()
            if not grabbed or orig is None or frame_id == last_frame_id:
                time.sleep(0.004)
                continue
            last_frame_id = frame_id

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self._shutdown(camera)
                break

            frame_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

            if frames_seen % self.DETECT_INTERVAL == 0:
                self._detect_face(orig)

            forehead, face_found = self._get_forehead(frame_rgb)
            self.pipe.send(forehead if face_found else None)

            if self.show_mask:
                self._draw_display(orig, face_found, current_fps)

            frames_seen += 1

            fps_counter += 1
            if fps_counter >= 15:
                elapsed     = time.time() - fps_start_time
                current_fps = fps_counter / elapsed if elapsed > 0 else 0.0
                fps_counter    = 0
                fps_start_time = time.time()

                face_status = "✅ Face" if face_found else "❌ No face"
                sys.stdout.write(
                    f"\r[rPPG] FPS: {current_fps:>5.1f} | "
                    f"HR: {self.latest_hr:>5.1f} BPM | "
                    f"BVP: {self.latest_bvp:>6.3f} | "
                    f"Face: {face_status}   "
                )
                sys.stdout.flush()

    def _draw_display(self, orig, face_found, current_fps):
        # --- Layout: 50/50 split ---
        # Left:  camera feed at 480x360
        # Right: HR info panel at 480x360
        PANEL_W = 480
        PANEL_H = 360

        # --- Left panel: camera feed ---
        cam_view = cv2.resize(orig, (PANEL_W, PANEL_H),
                              interpolation=cv2.INTER_LINEAR)

        # FPS top-left
        cv2.putText(cam_view, f"FPS: {current_fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        face = self._confirmed_face
        if face_found and face is not None:
            sx = PANEL_W / orig.shape[1]
            sy = PANEL_H / orig.shape[0]
            x, y, fw, fh = face

            # Face box only — NO forehead ROI box
            cv2.rectangle(cam_view,
                          (int(x*sx), int(y*sy)),
                          (int((x+fw)*sx), int((y+fh)*sy)),
                          (0, 255, 80), 2)

            # HR label above face box
            hr_str = f"HR: {self.latest_hr:.1f} BPM" if self.latest_hr > 0 else "Measuring..."
            cv2.putText(cam_view, hr_str,
                        (int(x*sx), max(50, int(y*sy) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2)
        else:
            cv2.putText(cam_view, "Searching for face...",
                        (20, PANEL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        # --- Right panel: HR info display ---
        right_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

        # Dark background with subtle border
        cv2.rectangle(right_panel, (0, 0), (PANEL_W-1, PANEL_H-1),
                      (30, 30, 30), 1)

        # Title
        cv2.putText(right_panel, "Heart Rate Monitor",
                    (60, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (180, 180, 180), 1)

        # Large HR number — centered
        hr_display = f"{self.latest_hr:.0f}" if self.latest_hr > 0 else "--"
        cv2.putText(right_panel, hr_display,
                    (PANEL_W//2 - 90, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5.0, (0, 255, 80), 6)

        # BPM label below number
        cv2.putText(right_panel, "BPM",
                    (PANEL_W//2 - 35, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 200, 60), 2)

        # Status indicator
        if face_found and self.latest_hr > 0:
            status_color = (0, 255, 80)
            status_text  = "● MEASURING"
        elif face_found:
            status_color = (0, 200, 255)
            status_text  = "● BUFFERING..."
        else:
            status_color = (0, 80, 255)
            status_text  = "● NO FACE DETECTED"

        cv2.putText(right_panel, status_text,
                    (PANEL_W//2 - 100, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, status_color, 2)

        # BVP value small text at bottom
        cv2.putText(right_panel,
                    f"BVP: {self.latest_bvp:.3f}",
                    (10, PANEL_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (80, 80, 80), 1)

        # --- Combine 50/50 ---
        combined = np.hstack([cam_view, right_panel])
        cv2.imshow('rPPG', combined)

    def _shutdown(self, camera):
        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()
        print("\nCamera released.")