"""
3-麦克风 TDOA 声源 2D 定位 (Fang 算法)
依赖: pip install numpy matplotlib
用法: python tdoa_3mic.py [csv_file]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import sys
import os
import glob
from collections import deque

# ====== 配置 ======
SAMPLE_RATE = 16000       # Hz
SPEED_OF_SOUND = 340.0    # m/s
MIC_DISTANCE = 0.20       # 麦克风间距 (m)
SAMPLES_PER_CH = 512
CONFIDENCE_THRESHOLD = 0.15            # 互相关峰值低于此值丢弃
SMOOTH_WINDOW = 5                       # 滑动窗口中值滤波窗口

# 麦克风位置 (L 形阵列)
# MIC1 = 原点, MIC2 = X 轴, MIC3 = Y 轴
MIC1 = np.array([0.0, 0.0])
MIC2 = np.array([MIC_DISTANCE, 0.0])
MIC3 = np.array([0.0, MIC_DISTANCE])
# ==================


def gcc_phat(sig1, sig2, n_fft=1024):
    """GCC-PHAT 互相关, 返回 lag (样本数), 相关系数峰值"""
    s1 = np.array(sig1, dtype=np.float64)
    s2 = np.array(sig2, dtype=np.float64)
    S1 = np.fft.rfft(s1, n=n_fft)
    S2 = np.fft.rfft(s2, n=n_fft)
    R = S1 * np.conj(S2)
    R_phat = R / (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R_phat)
    # cc[0..n_fft/2] = 正时延 (sig2 滞后), cc[n_fft/2+1..] = 负时延 (sig1 滞后)
    max_idx = np.argmax(np.abs(cc))
    if max_idx > n_fft // 2:
        lag = max_idx - n_fft
    else:
        lag = max_idx
    confidence = float(np.abs(cc[max_idx]))
    # 二次插值
    if 0 < max_idx < n_fft - 1:
        y0 = cc[max_idx - 1]
        y1 = cc[max_idx]
        y2 = cc[max_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            sub = (y0 - y2) / (2 * denom)
            lag = lag + sub
    return lag, confidence


def fang_solve(d12, d13, D):
    """Fang 算法: 从两个 TDOA 距离差求解 2D 位置 (x,y)
    MIC1=(0,0), MIC2=(D,0), MIC3=(0,D)
    d12 = r2 - r1,  d13 = r3 - r1"""
    if abs(d12) > D * 1.05 or abs(d13) > D * 1.05:
        return None

    a = (D * D - d12 * d12) / (2.0 * D)
    b = -d12 / D
    c = (D * D - d13 * d13) / (2.0 * D)
    e = -d13 / D

    A = b * b + e * e - 1.0
    B = 2.0 * (a * b + c * e)
    C = a * a + c * c

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    r1_a = (-B + sqrt_disc) / (2.0 * A)
    r1_b = (-B - sqrt_disc) / (2.0 * A)

    candidates = []
    for r1 in [r1_a, r1_b]:
        if r1 <= 0:
            continue
        x = a + b * r1
        y = c + e * r1
        # 验证: sqrt(x²+y²) 应接近 r1
        r1_est = np.sqrt(x * x + y * y)
        if abs(r1_est - r1) > 0.5:
            continue
        if abs(x) > D * 5 or abs(y) > D * 5:
            continue
        candidates.append((x, y, r1))

    if not candidates:
        return None
    return min(candidates, key=lambda t: t[2])  # 取最近的解


def load_csv(path):
    """加载 3 通道 CSV 文件, 返回 [(L, R, C), ...] 列表"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 解析头部获取列信息
    header_line = ""
    for line in lines:
        if not line.startswith("#") and "," in line:
            header_line = line.strip()
            break

    headers = header_line.split(",")
    col_map = {}
    for i, h in enumerate(headers):
        if h.startswith("L_"):
            col_map.setdefault("L", []).append(i)
        elif h.startswith("R_"):
            col_map.setdefault("R", []).append(i)
        elif h.startswith("C_"):
            col_map.setdefault("C", []).append(i)

    frames = []
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            L = [int(parts[i]) for i in col_map["L"][:SAMPLES_PER_CH]]
            R = [int(parts[i]) for i in col_map["R"][:SAMPLES_PER_CH]]
            C = [int(parts[i]) for i in col_map["C"][:SAMPLES_PER_CH]]
        except (ValueError, IndexError, KeyError):
            continue
        if len(L) >= SAMPLES_PER_CH and len(R) >= SAMPLES_PER_CH and len(C) >= SAMPLES_PER_CH:
            frames.append((L[:SAMPLES_PER_CH], R[:SAMPLES_PER_CH], C[:SAMPLES_PER_CH]))
    return frames


def main():
    # 查找 CSV 文件
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_files = sorted(glob.glob(os.path.join(script_dir, "mic_data_*.csv")))
        if not csv_files:
            print("未找到 mic_data_*.csv 文件")
            return
        csv_path = csv_files[-1]
        print(f"[自动选择] {csv_path}")

    print(f"加载 CSV: {csv_path}")
    frames = load_csv(csv_path)
    print(f"有效帧数: {len(frames)}")
    if len(frames) == 0:
        print("无有效数据")
        return

    # TDOA 逐帧计算
    positions = []
    angles_12 = []
    angles_13 = []
    confidences_12 = []
    confidences_13 = []

    for fi, (L, R, C) in enumerate(frames):
        # MIC1-MIC2 (X 轴基线)
        lag_12, conf_12 = gcc_phat(L, R)
        dt_12 = lag_12 / SAMPLE_RATE
        d12 = dt_12 * SPEED_OF_SOUND

        # MIC1-MIC3 (Y 轴基线)
        lag_13, conf_13 = gcc_phat(L, C)
        dt_13 = lag_13 / SAMPLE_RATE
        d13 = dt_13 * SPEED_OF_SOUND

        theta_12 = np.degrees(np.arcsin(np.clip(d12 / MIC_DISTANCE, -1, 1)))
        theta_13 = np.degrees(np.arcsin(np.clip(d13 / MIC_DISTANCE, -1, 1)))
        angles_12.append(theta_12)
        angles_13.append(theta_13)
        confidences_12.append(conf_12)
        confidences_13.append(conf_13)

        if conf_12 < CONFIDENCE_THRESHOLD or conf_13 < CONFIDENCE_THRESHOLD:
            positions.append(None)
            continue

        result = fang_solve(d12, d13, MIC_DISTANCE)
        if result:
            x, y, r1 = result
            positions.append((x, y))
        else:
            positions.append(None)

        if fi % 5 == 0:
            status = f"({x:.2f}, {y:.2f})m" if result else "(无解)"
            print(f"帧 {fi:4d}  lag12={lag_12:+7.3f}  lag13={lag_13:+7.3f}  "
                  f"d12={d12:+7.3f}m  d13={d13:+7.3f}m  {status}")

    # 统计
    valid_pos = [p for p in positions if p is not None]
    n_valid = len(valid_pos)
    print(f"\n===== 统计 ({n_valid}/{len(frames)} 帧有效, {len(frames) - n_valid} 帧丢弃) =====")
    if n_valid > 0:
        xs = [p[0] for p in valid_pos]
        ys = [p[1] for p in valid_pos]
        dists = [np.sqrt(x * x + y * y) for x, y in valid_pos]
        smoothed_x = deque(maxlen=SMOOTH_WINDOW)
        smoothed_y = deque(maxlen=SMOOTH_WINDOW)
        for x, y in valid_pos:
            smoothed_x.append(x)
            smoothed_y.append(y)
        sx = np.median(smoothed_x) if smoothed_x else 0
        sy = np.median(smoothed_y) if smoothed_y else 0
        print(f"X: 均值={np.mean(xs):.3f}m  中值={np.median(xs):.3f}m  范围=[{min(xs):.3f}, {max(xs):.3f}]m")
        print(f"Y: 均值={np.mean(ys):.3f}m  中值={np.median(ys):.3f}m  范围=[{min(ys):.3f}, {max(ys):.3f}]m")
        print(f"距离: 均值={np.mean(dists):.3f}m  中值={np.median(dists):.3f}m")
        print(f"平滑位置: ({sx:.3f}, {sy:.3f})m")
    else:
        xs = ys = dists = []

    # ====== 图表 ======
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"3-Mic TDOA 2D 定位  (麦克风间距={MIC_DISTANCE*100:.0f}cm, {len(frames)}帧)", fontsize=14)

    # 1. 声源位置散点图
    ax = axes[0, 0]
    ax.set_title("声源位置 (俯视图)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.scatter([0, MIC_DISTANCE, 0], [0, 0, MIC_DISTANCE],
               c=["#00cc66", "#ff6633", "#3399ff"], marker="^", s=150, zorder=5,
               label="MIC1(green) MIC2(orange) MIC3(blue)")
    if n_valid > 0:
        ax.scatter(xs, ys, c=range(n_valid), cmap="viridis", s=20, alpha=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    # 2. X 坐标时间序列
    ax = axes[0, 1]
    ax.set_title("X 坐标随时间变化")
    ax.set_xlabel("帧序号")
    ax.set_ylabel("X (m)")
    if n_valid > 0:
        ax.plot(range(n_valid), xs, lw=1)
        ax.axhline(np.median(xs), color="r", lw=1, linestyle="--", label="中值")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Y 坐标时间序列
    ax = axes[0, 2]
    ax.set_title("Y 坐标随时间变化")
    ax.set_xlabel("帧序号")
    ax.set_ylabel("Y (m)")
    if n_valid > 0:
        ax.plot(range(n_valid), ys, lw=1)
        ax.axhline(np.median(ys), color="r", lw=1, linestyle="--", label="中值")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 角度 θ12 (X轴方向)
    ax = axes[1, 0]
    ax.set_title("角度 θ12 (MIC1-MIC2基线)")
    ax.set_xlabel("帧序号")
    ax.set_ylabel("角度 (°)")
    ax.plot(angles_12, lw=1, color="#ff6633")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(True, alpha=0.3)

    # 5. 角度 θ13 (Y轴方向)
    ax = axes[1, 1]
    ax.set_title("角度 θ13 (MIC1-MIC3基线)")
    ax.set_xlabel("帧序号")
    ax.set_ylabel("角度 (°)")
    ax.plot(angles_13, lw=1, color="#3399ff")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(True, alpha=0.3)

    # 6. 互相关置信度
    ax = axes[1, 2]
    ax.set_title("互相关置信度")
    ax.set_xlabel("帧序号")
    ax.set_ylabel("置信度")
    ax.plot(confidences_12, lw=1, color="#ff6633", label="12")
    ax.plot(confidences_13, lw=1, color="#3399ff", label="13")
    ax.axhline(CONFIDENCE_THRESHOLD, color="r", lw=1, linestyle="--", label="阈值")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
