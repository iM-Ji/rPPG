import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import butter, filtfilt
import time


def bandpass_filter(wave, lowcut=0.7, highcut=3.5, fs=30, order=3):
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    try:
        return filtfilt(b, a, wave)
    except Exception:
        return wave.copy()


class DynamicPlot:
    def __init__(self, window_size=128, fps=30):
        self.window_size     = window_size
        self.fps             = fps
        self.display_len     = window_size * 3
        self.bvp_display     = np.zeros(self.display_len)
        self.hr_display      = np.full(self.display_len, np.nan)
        self._launched       = False
        self._counter        = 0
        self._latest_hr      = 0.0
        self._last_draw_time = 0.0
        self.DRAW_INTERVAL   = 1.5

    def _launch(self):
        plt.ion()
        self.fig = plt.figure(figsize=(6, 4), facecolor='white')
        self.fig.canvas.manager.set_window_title('BVP Monitor')

        gs = gridspec.GridSpec(2, 1,
                               height_ratios=[1.2, 1],
                               hspace=0.55,
                               left=0.12, right=0.97,
                               top=0.93, bottom=0.10)

        x = np.arange(self.display_len)

        # ----------------------------------------------------------------
        # Top — BVP waveform
        # ----------------------------------------------------------------
        self.ax_bvp = self.fig.add_subplot(gs[0])
        self.ax_bvp.set_facecolor('white')
        self.ax_bvp.set_xlim(0, self.display_len)
        self.ax_bvp.set_ylim(-3, 3)

        self.line_bvp, = self.ax_bvp.plot(
            x, self.bvp_display,
            color='#1f77b4', linewidth=0.9, antialiased=True
        )

        self.ax_bvp.set_title('BVP', fontsize=10,
                               fontweight='bold', color='black', pad=3)
        self.ax_bvp.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        self.ax_bvp.tick_params(labelsize=7, colors='black')
        self.ax_bvp.grid(False)
        for spine in self.ax_bvp.spines.values():
            spine.set_edgecolor('#aaaaaa')
            spine.set_linewidth(0.7)

        self.hr_text = self.ax_bvp.text(
            0.03, 0.90, 'HR: --',
            transform=self.ax_bvp.transAxes,
            fontsize=9, fontweight='bold',
            color='black', ha='left', va='top'
        )

        # ----------------------------------------------------------------
        # Bottom — Heart Rate trend
        # ----------------------------------------------------------------
        self.ax_hr = self.fig.add_subplot(gs[1])
        self.ax_hr.set_facecolor('white')
        self.ax_hr.set_xlim(0, self.display_len)
        self.ax_hr.set_ylim(40, 160)

        self.line_hr, = self.ax_hr.plot(
            x, self.hr_display,
            color='#1f77b4', linewidth=1.4,
            drawstyle='steps-post', antialiased=True
        )

        self.ax_hr.set_title('Heart Rate Trend', fontsize=10,
                              fontweight='bold', color='black', pad=3)
        self.ax_hr.set_yticks([40, 60, 80, 100, 120, 140, 160])
        self.ax_hr.tick_params(labelsize=7, colors='black')
        self.ax_hr.grid(False)
        for spine in self.ax_hr.spines.values():
            spine.set_edgecolor('#aaaaaa')
            spine.set_linewidth(0.7)

        plt.show(block=False)
        self._launched = True

    def update_wave(self, waveform: np.ndarray, hr: float):
        if not self._launched:
            self._launch()

        filtered = bandpass_filter(waveform, fs=self.fps)
        n = len(filtered)

        self.bvp_display = np.roll(self.bvp_display, -n)
        self.bvp_display[-n:] = filtered

        self.hr_display = np.roll(self.hr_display, -n)
        if hr > 0:
            self.hr_display[-n:] = hr
            self._latest_hr = hr
        else:
            self.hr_display[-n:] = (self._latest_hr
                                    if self._latest_hr > 0 else np.nan)

        self._counter += 1
        if self._counter % 2 == 0:
            hr_val = hr if hr > 0 else self._latest_hr
            self.hr_text.set_text(
                f'HR: {hr_val:.0f}' if hr_val > 0 else 'HR: --'
            )
            if hr_val > 0:
                self.ax_hr.set_ylim(max(40, hr_val - 40),
                                    min(180, hr_val + 60))

        now = time.time()
        if now - self._last_draw_time >= self.DRAW_INTERVAL:
            self.line_bvp.set_ydata(self.bvp_display)
            self.line_hr.set_ydata(self.hr_display)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._last_draw_time = now

    def update(self, bvp: float, hr: float):
        if not self._launched:
            self._launch()
        self.bvp_display = np.roll(self.bvp_display, -1)
        self.bvp_display[-1] = float(bvp)
        self.hr_display = np.roll(self.hr_display, -1)
        self.hr_display[-1] = float(hr) if hr > 0 else np.nan

        now = time.time()
        if now - self._last_draw_time >= self.DRAW_INTERVAL:
            self.line_bvp.set_ydata(self.bvp_display)
            self.line_hr.set_ydata(self.hr_display)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._last_draw_time = now