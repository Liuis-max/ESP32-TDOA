"""
三通道 INMP441 麦克风实时波形显示 + CSV 采集 (二进制帧协议)
依赖: pip install pyserial numpy matplotlib
用法: python view.py
"""
import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import struct
import sys
import threading
import time
import os
import queue

# ============ 配置 ============
PORT = "COM11"
BAUDRATE = 921600
PLOT_WINDOW = 1024
Y_RANGE = 500
# =============================

SAMPLE_RATE = 16000
FRAME_MS = 32
SAMPLES_PER_CH = 512
CHANNEL_COUNT = 3
MIC_DISTANCE_M = 0.20

SYNC1, SYNC2 = 0xAA, 0x55
FRAME_DATA_BYTES = CHANNEL_COUNT * SAMPLES_PER_CH * 2
FRAME_SIZE = 2 + 4 + 8 + 1 + 2 + FRAME_DATA_BYTES + 2

ser = None
frame_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=100)

buf_L = deque(maxlen=PLOT_WINDOW)
buf_R = deque(maxlen=PLOT_WINDOW)
buf_C = deque(maxlen=PLOT_WINDOW)
peak_L = peak_R = peak_C = 0
total_frames = 0
bad_crc = 0
last_fc = 0

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
fig.suptitle("INMP441 3-Mic - Live Waveform", fontsize=14)

line_L, = ax1.plot([], [], lw=1, color='#00cc66')
line_R, = ax2.plot([], [], lw=1, color='#ff6633')
line_C, = ax3.plot([], [], lw=1, color='#3399ff')

for ax, title in [(ax1, "MIC1 (L - I2S0 CH_L GPIO40)"), (ax2, "MIC2 (R - I2S0 CH_R GPIO40)"), (ax3, "MIC3 (C - I2S1 CH_L GPIO39)")]:
    ax.set_title(title)
    ax.set_xlim(0, PLOT_WINDOW)
    ax.set_ylim(-Y_RANGE, Y_RANGE)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
ax3.set_xlabel("Sample Index")

info_text = fig.text(0.02, 0.97, "", fontsize=9, fontfamily="monospace",
                     verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# ====== CSV ======
csv_queue = queue.Queue()
csv_file = None
csv_path = ""
csv_frames = 0
csv_running = True


def crc16_update(crc, data):
    crc ^= (data << 8)
    for _ in range(8):
        crc = (crc << 1) ^ 0x1021 if (crc & 0x8000) else (crc << 1)
    return crc & 0xFFFF


def crc16_buf(data):
    crc = 0
    for b in data:
        crc = crc16_update(crc, b)
    return crc


def _csv_writer_thread():
    global csv_file, csv_frames, csv_running
    while csv_running or not csv_queue.empty():
        try:
            item = csv_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        fc, ts, lvals, rvals, cvals = item
        if csv_file is None:
            csv_queue.task_done()
            continue
        row = [str(fc), str(ts)]
        row.extend(str(v) for v in lvals)
        row.extend(str(v) for v in rvals)
        row.extend(str(v) for v in cvals)
        try:
            csv_file.write(",".join(row) + "\n")
        except Exception:
            pass
        csv_frames += 1
        if csv_frames % 50 == 0:
            try:
                csv_file.flush()
            except Exception:
                pass
        csv_queue.task_done()


csv_thread = threading.Thread(target=_csv_writer_thread, daemon=True)


def csv_init():
    global csv_file, csv_path
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(save_dir, f"mic_data_{ts}.csv")
    csv_file = open(csv_path, "w", encoding="utf-8")
    header = ["frame", "ts_us"]
    header += [f"L_{i}" for i in range(SAMPLES_PER_CH)]
    header += [f"R_{i}" for i in range(SAMPLES_PER_CH)]
    header += [f"C_{i}" for i in range(SAMPLES_PER_CH)]
    csv_file.write(f"# sample_rate_hz={SAMPLE_RATE}\n")
    csv_file.write(f"# frame_duration_ms={FRAME_MS}\n")
    csv_file.write(f"# samples_per_channel={SAMPLES_PER_CH}\n")
    csv_file.write(f"# mic_distance_m={MIC_DISTANCE_M}\n")
    csv_file.write("# mic1=I2S0_L(GPIO40)  mic2=I2S0_R(GPIO40)  mic3=I2S1_L(GPIO39)\n")
    csv_file.write("# channel_order=L,R,C\n")
    csv_file.write(",".join(header) + "\n")
    csv_file.flush()
    print(f"[CSV] {csv_path}")


def csv_enqueue(fc, ts, lvals, rvals, cvals):
    csv_queue.put((fc, ts, list(lvals), list(rvals), list(cvals)))


def csv_close():
    global csv_file, csv_running
    csv_running = False
    csv_queue.put(None)
    if csv_file:
        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass
        duration_s = csv_frames * FRAME_MS / 1000.0
        try:
            size_kb = os.path.getsize(csv_path) / 1024
        except Exception:
            size_kb = 0
        print(f"\n[CSV] 总帧: {csv_frames}  时长: {duration_s:.1f}s  大小: {size_kb:.0f} KB")
        csv_file = None


def open_serial():
    global ser
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        ser.dtr = False
        ser.rts = False
        print(f"[OK] 串口 {PORT} 已打开, 波特率 {BAUDRATE}")
    except Exception as e:
        print(f"[FAIL] 无法打开串口 {PORT}: {e}")
        print("可用端口:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device} - {p.description}")
        sys.exit(1)


def serial_reader():
    global last_fc, bad_crc
    buf = bytearray()
    total_bytes = 0
    debug_first = True
    while True:
        try:
            chunk = ser.read(4096)
        except Exception:
            break
        if not chunk:
            continue
        total_bytes += len(chunk)
        buf.extend(chunk)

        if debug_first:
            print(f"[DBG] 收到首批数据 {len(chunk)} 字节, 累计 {total_bytes}")
            print(f"[DBG] 前40字节 HEX: {bytes(chunk[:40]).hex()}")
            debug_first = False

        while len(buf) >= FRAME_SIZE:
            # 找同步头
            idx = buf.find(bytes([SYNC1, SYNC2]))
            if idx < 0:
                # 没有同步头，保留最后 1 字节防 fragment
                if len(buf) > 100:
                    print(f"[DBG] buf={len(buf)}B 无同步头, 前20B: {bytes(buf[:20]).hex()}")
                buf = buf[-1:]
                break
            if idx > 0:
                print(f"[DBG] 同步头前跳过 {idx} 字节")
                buf = buf[idx:]

            if len(buf) < FRAME_SIZE:
                break

            raw = bytes(buf[:FRAME_SIZE])

            # 验证 CRC
            calc_crc = crc16_buf(raw[2:FRAME_SIZE - 2])
            recv_crc = raw[FRAME_SIZE - 2] | (raw[FRAME_SIZE - 1] << 8)
            if calc_crc != recv_crc:
                bad_crc += 1
                if bad_crc <= 3:
                    print(f"[DBG] CRC不匹配 #{bad_crc}: calc=0x{calc_crc:04X} recv=0x{recv_crc:04X}")
                buf = buf[1:]  # 跳过 1 字节重新同步
                continue

            # 解析
            fc = struct.unpack_from("<I", raw, 2)[0]
            ts = struct.unpack_from("<q", raw, 6)[0]
            nch = raw[14]
            nspc = struct.unpack_from("<H", raw, 15)[0]

            if total_frames == 0:
                print(f"[DBG] 首帧解析成功: fc={fc} ts={ts} nch={nch} nspc={nspc}")

            data_start = 17
            expected_data = nch * nspc * 2
            if expected_data > FRAME_DATA_BYTES:
                expected_data = FRAME_DATA_BYTES
                nspc = SAMPLES_PER_CH

            samples = struct.unpack_from(f"<{nch * nspc}h", raw, data_start)
            L = samples[0:nspc]
            R = samples[nspc:2 * nspc]
            C = samples[2 * nspc:3 * nspc] if nch >= 3 else []

            frame_queue.put((fc, ts, L, R, C))
            last_fc = fc
            buf = buf[FRAME_SIZE:]


def update(frame):
    global peak_L, peak_R, peak_C, total_frames

    # 取出所有可用帧
    frames_processed = 0
    while not frame_queue.empty():
        try:
            fc, ts, L, R, C = frame_queue.get_nowait()
        except queue.Empty:
            break

        for v in L:
            buf_L.append(v)
        for v in R:
            buf_R.append(v)
        for v in C:
            buf_C.append(v)

        peak_L = max(peak_L, max(abs(min(L)), abs(max(L))))
        peak_R = max(peak_R, max(abs(min(R)), abs(max(R))))
        peak_C = max(peak_C, max(abs(min(C)), abs(max(C))))

        csv_enqueue(fc, ts, L, R, C)
        total_frames += 1
        frames_processed += 1

    if buf_L:
        x = np.arange(len(buf_L))
        line_L.set_data(x, list(buf_L))
        yl = max(peak_L, 100)
        ax1.set_ylim(-yl, yl)
        limit = max(PLOT_WINDOW, len(buf_L))
        ax1.set_xlim(limit - PLOT_WINDOW, limit)

    if buf_R:
        x = np.arange(len(buf_R))
        line_R.set_data(x, list(buf_R))
        yr = max(peak_R, 100)
        ax2.set_ylim(-yr, yr)
        ax2.set_xlim(limit - PLOT_WINDOW, limit)

    if buf_C:
        x = np.arange(len(buf_C))
        line_C.set_data(x, list(buf_C))
        yc = max(peak_C, 100)
        ax3.set_ylim(-yc, yc)
        ax3.set_xlim(limit - PLOT_WINDOW, limit)

    info_text.set_text(
        f"PORT: {PORT} | Frames: {total_frames} | LastFC: {last_fc} | "
        f"CRCerr: {bad_crc} | Peak L:{peak_L} R:{peak_R} C:{peak_C}"
    )

    return line_L, line_R, line_C, info_text


def main():
    csv_init()
    csv_thread.start()
    open_serial()
    print(f"帧大小: {FRAME_SIZE} 字节 | 采样率: {SAMPLE_RATE} Hz | 帧时长: {FRAME_MS} ms")
    print("关闭窗口停止采集\n")
    thread = threading.Thread(target=serial_reader, daemon=True)
    thread.start()
    ani = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    csv_close()
    if ser:
        ser.close()
    print("退出")


if __name__ == "__main__":
    main()
