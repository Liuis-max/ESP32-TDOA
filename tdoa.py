"""
双麦 TDOA 声源定位 (GCC-PHAT)
用法: python tdoa.py <csv文件路径>
示例: python tdoa.py mic_data_20260705_172414.csv
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ============ 参数 ============
SAMPLE_RATE = 16000         # Hz
MIC_DISTANCE = 0.20         # 麦克风间距 20cm
SOUND_SPEED = 340.0         # 声速 m/s
SAMPLES_PER_CH = 512

def gcc_phat(sig1, sig2, n_fft=None):
    """GCC-PHAT 互相关，返回时域互相关序列"""
    if n_fft is None:
        n_fft = len(sig1) * 2
    s1 = np.asarray(sig1, dtype=np.float32)
    s2 = np.asarray(sig2, dtype=np.float32)
    s1_fft = np.fft.rfft(s1, n=n_fft)
    s2_fft = np.fft.rfft(s2, n=n_fft)
    R = s1_fft * np.conj(s2_fft)
    R_phat = R / (np.abs(R) + 1e-8)
    cc = np.fft.irfft(R_phat, n=n_fft)
    return np.real(cc)

def find_peak_subpixel(cc, center):
    """亚采样峰值检测：抛物线插值"""
    idx = np.argmax(np.abs(cc))
    if idx <= 0 or idx >= len(cc) - 1:
        return float(idx - center), 0.0
    y0, y1, y2 = cc[idx-1], cc[idx], cc[idx+1]
    denom = 2 * (y0 - 2*y1 + y2)
    if abs(denom) < 1e-12:
        return float(idx - center), 0.0
    delta = (y0 - y2) / denom
    return float(idx - center) + delta, abs(y1)

def to_angle(lag_samples):
    """样本偏移 → 角度 (度)，正值为声源偏向 MIC1 侧"""
    dt = lag_samples / SAMPLE_RATE
    sin_theta = dt * SOUND_SPEED / MIC_DISTANCE
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_theta))

def parse_csv(path):
    """读取 CSV，返回帧列表 [(frame_num, ts_us, left[], right[]), ...]"""
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                frame_num = int(parts[0])
                ts_us = int(parts[1])
                vals = [int(x) for x in parts[2:]]
                mid = len(vals) // 2
                left = vals[:mid]
                right = vals[mid:]
                if len(left) == SAMPLES_PER_CH and len(right) == SAMPLES_PER_CH:
                    frames.append((frame_num, ts_us, left, right))
            except (ValueError, IndexError):
                continue
    return frames

def main():
    if len(sys.argv) < 2:
        # 自动找最新 CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_files = sorted([f for f in os.listdir(script_dir) if f.startswith("mic_data_") and f.endswith(".csv")])
        if not csv_files:
            print("用法: python tdoa.py <csv文件路径>")
            return
        path = os.path.join(script_dir, csv_files[-1])
        print(f"[auto] {path}")
    else:
        path = sys.argv[1]

    print(f"Loading: {path}")
    frames = parse_csv(path)
    print(f"Valid frames: {len(frames)}")
    if len(frames) == 0:
        print("No data")
        return

    angles = []
    correlations = []
    lags_samples = []

    print(f"\n{'Frame':>6s} {'Time_ms':>10s} {'Lag_samp':>10s} {'Lag_us':>10s} {'Angle':>10s} {'Conf':>8s}")
    print("-" * 62)

    for i, (fc, ts, left, right) in enumerate(frames):
        cc = gcc_phat(left, right)
        center = len(cc) // 2
        lag, confidence = find_peak_subpixel(cc, center)
        angle = to_angle(lag)
        dt_us = lag / SAMPLE_RATE * 1e6

        angles.append(angle)
        lags_samples.append(lag)
        correlations.append(cc)

        if i % 5 == 0 or abs(angle) > 10:
            print(f"{fc:6d} {ts/1000:10.1f} {lag:+10.3f} {dt_us:+10.1f} {angle:+10.2f} {confidence:8.4f}")

    # ====== 统计 ======
    angles_arr = np.array(angles)
    print(f"\n===== Stats ({len(angles)} frames) =====")
    print(f"Mean:  {np.mean(angles_arr):+.2f} deg")
    print(f"Median:{np.median(angles_arr):+.2f} deg")
    print(f"Std:   {np.std(angles_arr):.2f} deg")
    print(f"Range: [{np.min(angles_arr):+.2f}, {np.max(angles_arr):+.2f}] deg")

    # ====== 可视化 ======
    t_arr = np.array([f[1] / 1000.0 for f in frames])  # ms
    angles_arr = np.array(angles)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"TDOA 声源定位 ({len(frames)} 帧, 麦克风间距={MIC_DISTANCE*100:.0f}cm)", fontsize=14)

    # 1. 角度随时间变化
    ax = axes[0, 0]
    ax.plot(t_arr, angles_arr, 'b-', alpha=0.6, linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("角度 (°)")
    ax.set_title("角度-时间曲线")
    ax.grid(True, alpha=0.3)

    # 2. 角度直方图
    ax = axes[0, 1]
    ax.hist(angles_arr, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.median(angles_arr), color='red', linestyle='--', label=f"中值={np.median(angles_arr):.1f}°")
    ax.set_xlabel("角度 (°)")
    ax.set_ylabel("帧数")
    ax.set_title("角度分布直方图")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 中间帧互相关曲线
    mid_idx = len(frames) // 2
    ax = axes[1, 0]
    cc_mid = correlations[mid_idx]
    center = len(cc_mid) // 2
    x_lag = np.arange(len(cc_mid)) - center
    ax.plot(x_lag, cc_mid, 'g-', linewidth=0.8)
    peak_lag = lags_samples[mid_idx]
    ax.axvline(x=peak_lag, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("时延 (采样点, 1点=62.5µs)")
    ax.set_ylabel("相关系数")
    ax.set_title(f"互相关曲线 (第{mid_idx}帧, 峰值偏移={peak_lag:.2f}点)")
    ax.grid(True, alpha=0.3)

    # 4. 单帧波形示例
    ax = axes[1, 1]
    fc, ts, left, right = frames[mid_idx]
    t_wave = np.arange(SAMPLES_PER_CH) / SAMPLE_RATE * 1000
    ax.plot(t_wave, left, 'g-', alpha=0.7, label="左声道 MIC1", linewidth=0.6)
    ax.plot(t_wave, right, 'orange', alpha=0.7, label="右声道 MIC2", linewidth=0.6)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("采样值")
    ax.set_title(f"波形对比 (第{mid_idx}帧, {FRAME_MS}ms)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 输出简单文本结果
    print(f"\nMain source direction: {np.median(angles_arr):+.1f} deg")
    print(f"  (+angle = toward MIC1 side, 0 = front)")


if __name__ == "__main__":
    FRAME_MS = 32
    main()
