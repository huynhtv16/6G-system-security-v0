import threading
import time
import uuid
import logging
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
import psutil
import json
import os
import random
import math
import signal

# Thêm biến điều khiển
running = True

# Thêm hàm xử lý tín hiệu
def signal_handler(signum, frame):
    global running
    print("\n⚠️ Đã nhận tín hiệu thoát. Đang dừng mô phỏng...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Tạo thư mục result nếu chưa tồn tại
os.makedirs('result', exist_ok=True)

# Xóa file log cũ nếu có
os.system('del result\\attacker.log result\\server.log 2>nul')

# Tạo logger riêng cho attacker
attacker_logger = logging.getLogger('attacker')
attacker_logger.setLevel(logging.INFO)
attacker_handler = logging.FileHandler('result\\attacker.log')
attacker_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
attacker_logger.addHandler(attacker_handler)

# Tạo logger riêng cho server
server_logger = logging.getLogger('server')
server_logger.setLevel(logging.INFO)
server_handler = logging.FileHandler('result\\server.log')
server_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
server_logger.addHandler(server_handler)

console = Console()
packet_queue = multiprocessing.Queue()

# Biến toàn cục
packet_count = 0
received_packets = 0
THREAD_COUNT = 1000
SERVER_THREAD_COUNT = 100
sent_history = []
received_history = []
time_history = []
received_lock = threading.Lock()
packets_per_second = 0
pps_history = []
cpu_history = []
ram_history = []
prev_received = 0

CAR_CENTER = (100.0, 100.0)
EARTH_RADIUS = 6371

class PacketGenerator:
    def __init__(self, center_lat=CAR_CENTER[0], center_lon=CAR_CENTER[1], radius_km=1):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_km = radius_km
        self.packet_id = 0
        self.attack_types = ["SYN_Flood", "UDP_Flood", "HTTP_Flood"]
        self.destination_ips = ["10.0.0.50", "172.16.0.10", "192.168.1.200"]
        self.destination_ports = [80, 443, 53, 22]

    def _generate_location(self):
        radius_rad = self.radius_km / EARTH_RADIUS
        bearing = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, radius_rad)

        lat1 = math.radians(self.center_lat)
        lon1 = math.radians(self.center_lon)

        lat2 = math.asin(
            math.sin(lat1) * math.cos(distance) +
            math.cos(lat1) * math.sin(distance) * math.cos(bearing)
        )

        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(distance) * math.cos(lat1),
            math.cos(distance) - math.sin(lat1) * math.sin(lat2)
        )

        return {
            "latitude": math.degrees(lat2),
            "longitude": math.degrees(lon2)
        }

    def _generate_ip(self):
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

    def _generate_payload(self, attack_type):
        if attack_type == "HTTP_Flood":
            return f"GET / HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
        elif attack_type == "UDP_Flood":
            return ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=100))
        else:
            return ""

    def _create_new_packet(self):
        self.packet_id += 1
        location = self._generate_location()
        attack_type = random.choice(self.attack_types)
        protocol = "TCP" if attack_type in ["SYN_Flood", "HTTP_Flood"] else "UDP"
        packet_type = "SYN" if attack_type == "SYN_Flood" else "GET" if attack_type == "HTTP_Flood" else "DATAGRAM"
        flags = ["SYN"] if attack_type == "SYN_Flood" else []

        packet = {
            "packet_id": str(self.packet_id).zfill(8),
            "timestamp": int(time.time()),
            "location": location,
            "source_ip": self._generate_ip(),
            "destination_ip": random.choice(self.destination_ips),
            "source_port": random.randint(1024, 65535),
            "destination_port": random.choice(self.destination_ports),
            "protocol": protocol,
            "packet_type": packet_type,
            "payload": self._generate_payload(attack_type),
            "ttl": random.randint(32, 255),
            "flags": flags,
            "attack_type": attack_type,
            "volume": random.randint(64, 1500),
            "spoofed": random.choice([True, False])
        }
        return packet

generator = PacketGenerator(radius_km=20)

def ddos_attack():
    global packet_count
    while running:
        packet = generator._create_new_packet()
        packet_queue.put(packet)
        packet_count += 1
        attacker_logger.info(f"Generated packet {packet['packet_id']}, Attack Type: {packet['attack_type']}")
        time.sleep(0.001)

def server_listener():
    global received_packets
    BATCH_SIZE = 100
    
    while running:
        if not packet_queue.empty():
            packets_to_process = []
            for _ in range(BATCH_SIZE):
                if not packet_queue.empty():
                    packets_to_process.append(packet_queue.get())
                else:
                    break
                    
            if packets_to_process:
                with received_lock:
                    received_packets += len(packets_to_process)
                
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                
                log_entries = []
                for packet in packets_to_process:
                    log_entries.append(
                        f"Received Packet {packet['packet_id']} | "
                        f"Source IP: {packet['source_ip']} | "
                        f"Destination IP: {packet['destination_ip']} | "
                        f"Source Port: {packet['source_port']} | "
                        f"Destination Port: {packet['destination_port']} | "
                        f"Protocol: {packet['protocol']} | "
                        f"Flags: {','.join(packet['flags'])} | "
                        f"TTL: {packet['ttl']} | "
                        f"Volume: {packet['volume']} | "
                        f"Spoofed: {1 if packet['spoofed'] else 0} | "
                        f"Attack Type: {packet['attack_type']}"
                    )             
                server_logger.info(
                    f"Batch processed {len(packets_to_process)} packets. "
                    f"CPU: {cpu}%, RAM: {ram}%\n" + 
                    "\n".join(log_entries)
                )
                
        time.sleep(0.0001)

def display_status():
    global packets_per_second, prev_received, running
    start_time = time.time()
    last_pps_update = start_time
    display_status.last_second = -1

    while running and (time.time() - start_time < 61):
        try:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            
            # Tính số packet mỗi giây bằng hiệu
            current_time = time.time()
            elapsed_time = int(current_time - start_time)
            if current_time - last_pps_update >= 1.0:
                packets_per_second = received_packets - prev_received
                server_logger.info(f"Packets received in last second: {packets_per_second}")
                prev_received = received_packets
                last_pps_update = current_time
            
            # Lưu lịch sử
            time_history.append(elapsed_time)
            sent_history.append(packet_count)
            received_history.append(received_packets)
            pps_history.append(packets_per_second)
            cpu_history.append(cpu)
            ram_history.append(ram)
            
            # Hiển thị bảng
            if elapsed_time != display_status.last_second:
                table = Table(title="🌐 DDoS Simulation Status")
                table.add_column("Current Second", justify="center")
                table.add_column("Threads (Attack)", justify="center")
                table.add_column("Threads (Server)", justify="center")
                table.add_column("Sent", justify="center")
                table.add_column("Received", justify="center")
                table.add_column("PPS", justify="center")
                table.add_column("CPU", justify="center")
                table.add_column("RAM", justify="center")
                
                table.add_row(
                    str(elapsed_time),
                    str(THREAD_COUNT),
                    str(SERVER_THREAD_COUNT),
                    str(packet_count),
                    str(received_packets),
                    str(packets_per_second),
                    f"{cpu}%",
                    f"{ram}%"
                )
                
                console.clear()
                console.print(table)
                display_status.last_second = elapsed_time
            
            time.sleep(0.1)  # Giảm độ trễ cập nhật
            
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            running = False
            break
    
    # Vẽ biểu đồ gửi/nhận
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, sent_history, label='Packets Sent', color='blue')
    plt.plot(time_history, received_history, label='Packets Received', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Packets')
    plt.title('Packets Sent and Received Over Time')
    plt.legend()
    plt.grid(True)
    plt.subplots_adjust(top=0.9)  # Mở rộng không gian phía trên để hiển thị tiêu đề
    plt.savefig('packets_over_time.png')

    # Vẽ biểu đồ kết hợp Packets/Second, CPU, RAM
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Packets/Second', color='red')
    ax1.plot(time_history, pps_history, label='Packets/Second', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Percentage (%)', color='blue')
    ax2.plot(time_history, cpu_history, label='CPU Usage (%)', color='blue', linestyle='--')
    ax2.plot(time_history, ram_history, label='RAM Usage (%)', color='green', linestyle='-.')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.title('Packets Per Second, CPU, and RAM Usage Over Time')
    plt.subplots_adjust(top=0.9)  # Mở rộng không gian phía trên để hiển thị tiêu đề
    plt.savefig('pps_cpu_ram_over_time.png')

def main():
    for _ in range(THREAD_COUNT):
        threading.Thread(target=ddos_attack, daemon=True).start()
    
    for _ in range(SERVER_THREAD_COUNT):
        threading.Thread(target=server_listener, daemon=True).start()

    print("🛡️ Fake Server & 💣 Attacker đã khởi động. Chạy trong 1 phút...")
    display_status()

    print("\n⛔ Kết thúc phiên mô phỏng.")
    print("✅ Đã dừng simulation. Biểu đồ lưu tại 'packets_over_time.png' và 'pps_cpu_ram_over_time.png'.")
    exit()

if __name__ == "__main__":
    main()