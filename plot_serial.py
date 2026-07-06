import argparse
import re
import sys
import threading
import matplotlib
matplotlib.use('TkAgg')  # 添加这行，指定后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

try:
    import serial
except ImportError:
    print("Please install pyserial: pip install pyserial")
    sys.exit(1)

DATA_RE = re.compile(r'^DATA(,(-?\d+))+\r?$')
STAT_RE = re.compile(r'^STAT,(.*)\r?$')


def serial_reader(port, baudrate, data_queue, status_queue, stop_event):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"Failed to open serial port {port}: {e}")
        stop_event.set()
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
        except Exception as e:
            print(f"Serial read error: {e}")
            continue
        if not line:
            continue
        if line.startswith('DATA'):
            parts = line.split(',')[1:]
            try:
                samples = [int(x) for x in parts]
                data_queue.append(samples)
            except ValueError:
                continue
        elif line.startswith('STAT'):
            status_queue.append(line)

    ser.close()


def main():
    parser = argparse.ArgumentParser(description='Realtime serial plot for INMP411 data')
    parser.add_argument('-p', '--port', required=True, help='Serial port, e.g. COM3')
    parser.add_argument('-b', '--baud', default=115200, type=int, help='Baud rate')
    parser.add_argument('-n', '--points', default=128, type=int, help='Points per plot frame')
    args = parser.parse_args()

    data_queue = deque(maxlen=20)
    status_queue = deque(maxlen=5)
    stop_event = threading.Event()

    thread = threading.Thread(target=serial_reader, args=(args.port, args.baud, data_queue, status_queue, stop_event), daemon=True)
    thread.start()

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1)
    ax.set_title('INMP411 Realtime Audio Samples')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-32768, 32767)
    ax.set_xlim(0, args.points - 1)
    ax.grid(True)
    text = fig.text(0.02, 0.95, '', transform=ax.transAxes)

    def animate(frame):
        if data_queue:
            samples = data_queue[-1]
            # 确保样本数据存在且不为空
            if samples and len(samples) > 0:
                x = list(range(min(len(samples), args.points)))
                y = samples[:args.points]
                line.set_data(x, y)
                ax.set_xlim(0, len(y) - 1 if len(y) > 0 else 1)
                text.set_text('Status: ' + (status_queue[-1] if status_queue else 'waiting...'))
            else:
                # 如果没有数据，保持上次的图形
                pass
        else:
            # 没有数据时更新状态文本
            text.set_text('Status: waiting for data...')
        
        # 确保返回所有需要更新的艺术对象
        return [line, text]

    # 使用 blit=False 避免 NoneType 错误
    ani = animation.FuncAnimation(fig, animate, interval=200, blit=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=1)


if __name__ == '__main__':
    main()