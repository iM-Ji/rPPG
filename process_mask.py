import os
import sys
import numpy as np
import cv2
from threading import Thread, Lock
import time


class RunPOS:
    def __init__(self, batch_size, framerate, model_path, plot=True):
        self.fps         = int(framerate)
        self.plot        = plot
        self.model_path  = model_path
        self.window_len  = 128
        self.spatial     = 64
        self.latest_hr   = 0.0
        self.latest_bvp  = 0.0
        self.latest_wave = None
        self.inference_running = False

    def _init_model(self):
        self.buffer      = []
        self.hr_history  = []
        self.buffer_lock = Lock()
        self.plotter     = None
        self.new_result  = False
        self._last_plot_time = 0.0

        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == ".pt":
            self._load_torchscript()
        elif ext == ".onnx":
            self._load_onnx()
        elif ext == ".pth":
            self._load_pth()
        else:
            raise ValueError(f"Unsupported model format: {ext}")

        if self.plot:
            from plot_cont import DynamicPlot
            self.plotter = DynamicPlot(window_size=128)

        print(f"✅ Model loaded: {os.path.basename(self.model_path)}")
        print(f"   Backend: {self.backend} | Input: {self.window_len}x{self.spatial}x{self.spatial}")

    def _load_torchscript(self):
        import torch
        torch.set_num_threads(2)
        self.session = torch.jit.load(self.model_path, map_location="cpu")
        self.session.eval()
        self.backend = "TorchScript"
        self._infer  = self._infer_torchscript

    def _load_onnx(self):
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads     = 2
        sess_opts.inter_op_num_threads     = 1
        sess_opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session    = ort.InferenceSession(
            self.model_path, sess_opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.backend    = "ONNX"
        self._infer     = self._infer_onnx

    def _load_pth(self):
        import torch
        torch.set_num_threads(2)
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models import PhysNet
        model = PhysNet(T=self.window_len)
        model.load_state_dict(
            torch.load(self.model_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        self.session = model
        self.backend = "PyTorch (.pth)"
        self._infer  = self._infer_torchscript

    def _preprocess(self, frames_snapshot):
        resized = np.stack(
            [cv2.resize(f, (self.spatial, self.spatial),
                        interpolation=cv2.INTER_LINEAR)
             for f in frames_snapshot], axis=0
        )
        tensor = resized.transpose(3, 0, 1, 2).astype(np.float32) / 255.0
        return tensor[np.newaxis]

    def _infer_torchscript(self, input_np):
        import torch
        with torch.no_grad():
            t = torch.from_numpy(input_np)
            return self.session(t).numpy().flatten()

    def _infer_onnx(self, input_np):
        outputs = self.session.run(None, {self.input_name: input_np})
        return outputs[0].flatten()

    def extract_bpm(self, signal):
        fft   = np.abs(np.fft.rfft(signal - signal.mean()))
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.fps)
        valid = (freqs >= 0.75) & (freqs <= 3.0)
        if not valid.any():
            return 0.0
        return float(freqs[valid][np.argmax(fft[valid])] * 60.0)

    def _run_inference(self, frames_snapshot):
        try:
            input_np = self._preprocess(frames_snapshot)
            pred_bvp = self._infer(input_np)

            std = pred_bvp.std()
            if std > 1e-6:
                wave_norm = (pred_bvp - pred_bvp.mean()) / std
            else:
                wave_norm = np.zeros_like(pred_bvp)

            hr = self.extract_bpm(pred_bvp)
            if 45 < hr < 180:
                self.hr_history.append(hr)
                if len(self.hr_history) > 10:
                    self.hr_history.pop(0)

            self.latest_hr   = float(np.median(self.hr_history)) if self.hr_history else 0.0
            self.latest_bvp  = float(pred_bvp[-1])
            self.latest_wave = wave_norm
            self.new_result  = True

        except Exception as e:
            print(f"\n⚠️  Inference error: {e}")
        finally:
            self.inference_running = False

    def __call__(self, pipe, result_pipe=None):
        self._init_model()

        while True:
            try:
                forehead_pixels = pipe.recv()

                if forehead_pixels is None:
                    if result_pipe:
                        result_pipe.send((self.latest_hr, self.latest_bvp))
                    continue

                with self.buffer_lock:
                    self.buffer.append(forehead_pixels)
                    if len(self.buffer) > self.window_len:
                        self.buffer = self.buffer[-self.window_len:]
                    buf_len = len(self.buffer)

                if result_pipe:
                    result_pipe.send((self.latest_hr, self.latest_bvp))

                if self.new_result:
                    now = time.time()
                    if now - self._last_plot_time >= 0.5:
                        if self.plotter and self.latest_wave is not None:
                            self.plotter.update_wave(self.latest_wave, self.latest_hr)
                        self._last_plot_time = now
                    self.new_result = False

                if buf_len >= self.window_len and not self.inference_running:
                    with self.buffer_lock:
                        snapshot = list(self.buffer)
                    self.inference_running = True
                    Thread(
                        target=self._run_inference,
                        args=(snapshot,),
                        daemon=True
                    ).start()

            except EOFError:
                break
