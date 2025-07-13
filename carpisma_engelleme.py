from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from pymavlink import mavutil

NUM_DRONES = 3  # Drone sayısı
BASE_PORT = 14550  # Başlangıç portu
connections = []
ports = [f'udp:127.0.0.1:{BASE_PORT + i*10}' for i in range(NUM_DRONES)]

# Drone bağlantıları
for i, port in enumerate(ports):
    try:
        vehicle = connect(port, wait_ready=True)
        connections.append(vehicle)
        print(f"✓ Bağlandı: Drone {i+1} - {port}")
    except Exception as e:
        print(f"✗ Drone {i+1} bağlantı hatası: {e}")

print(f"Toplam {len(connections)} drone bağlandı.")

# ================= ÇARPIŞMA ENGELLEME ALGORİTMASI =================

class CollisionAvoidance:
    def __init__(self, a=2.0, beta=1.5, r_s=20.0):
        """
        Çarpışma engelleme parametreleri
        a: Kuvvet büyüklüğü
        beta: Kuvvet azalma hızı
        r_s: Etkileşim yarıçapı (metre)
        """
        self.a = a
        self.beta = beta
        self.r_s = r_s
    
    def calculate_repulsive_force(self, pos_i, pos_j):
        """
        İki drone arasındaki itici kuvveti hesapla
        Formül: u_ij = a(e^(-β*||r_ij||) - e^(-β*r_s))
        """
        # Dronlar arası mesafe vektörü
        diff = np.array(pos_i) - np.array(pos_j)
        distance = np.linalg.norm(diff)
        
        # Çok yakın durumda sıfıra bölme hatası önleme
        if distance < 0.1:
            distance = 0.1
            diff = np.array([0.1, 0.0])
        
        # Mesafe etkileşim yarıçapından büyükse kuvvet yok
        if distance > self.r_s:
            return np.array([0.0, 0.0])
        
        # Potansiyel fonksiyonu: u_ij = a(e^(-β*||r_ij||) - e^(-β*r_s))
        force_magnitude = self.a * (np.exp(-self.beta * distance) - np.exp(-self.beta * self.r_s))
        
        # Kuvvet yönü (i'den j'ye doğru, yani itici)
        force_direction = diff / distance
        
        return force_magnitude * force_direction
    
    def calculate_total_force(self, drone_index, positions):
        """
        Bir drone'un diğer tüm drone'lardan aldığı toplam itici kuvvet
        Formül: u_i^T = Σ(j=1 to N) u_ij^T
        """
        total_force = np.array([0.0, 0.0])
        
        for j, pos_j in enumerate(positions):
            if j != drone_index:
                force = self.calculate_repulsive_force(positions[drone_index], pos_j)
                total_force += force
        
        return total_force

# ======================= YARDIMCI FONKSİYONLAR =======================

def send_velocity(vehicle, vx, vy, vz):
    """Drone'a hız komutu gönder"""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def get_drone_id(vehicle):
    """Drone ID'sini al"""
    try:
        return connections.index(vehicle) + 1
    except ValueError:
        return "Bilinmeyen"

def arm(vehicle):
    """Drone'u arm et"""
    drone_id = get_drone_id(vehicle)
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    vehicle.armed = True
    while not vehicle.armed:
        print(f"⏳ Drone {drone_id} arm edilmeyi bekliyor...")
        time.sleep(1)
    print(f"✅ Drone {drone_id} arm edildi")

def takeoff(vehicle, alt):
    """Drone'u havaya kaldır"""
    drone_id = get_drone_id(vehicle)
    vehicle.simple_takeoff(alt)
    while vehicle.location.global_relative_frame.alt < alt * 0.95:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f"🚁 Drone {drone_id} yükseklik: {current_alt:.2f}m / {alt}m")
        time.sleep(1)
    print(f"✅ Drone {drone_id} hedef yüksekliğe ulaştı: {alt}m")

def get_drone_positions(connections, ref_lat, ref_lon):
    """Tüm drone'ların mevcut pozisyonlarını al (metre cinsinden)"""
    positions = []
    for vehicle in connections:
        lat = vehicle.location.global_relative_frame.lat
        lon = vehicle.location.global_relative_frame.lon
        
        # Referans noktaya göre metre cinsinden konum
        d_north = (lat - ref_lat) * 111000
        d_east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
        
        # GPS hata toleransı ekle
        d_north = round(d_north, 2)  # Hassasiyet artırımı
        d_east = round(d_east, 2)
        
        positions.append([d_north, d_east])
    
    return positions

def collision_avoidance_control(connections):
    """Çarpışma engelleme kontrol döngüsü"""
    # Çarpışma engelleme sistemi
    collision_avoidance = CollisionAvoidance(a=1000.0, beta=0.8, r_s=20.0)
    
    # Referans nokta (ilk drone'un başlangıç konumu)
    ref_lat = connections[0].location.global_relative_frame.lat
    ref_lon = connections[0].location.global_relative_frame.lon
    
    print(f"🛡️ Çarpışma engelleme kontrol başladı")
    
    while True:
        # Güncel pozisyonları al
        current_positions = get_drone_positions(connections, ref_lat, ref_lon)
        
        # Her drone için çarpışma engelleme kuvvetini hesapla ve uygula
        for i, vehicle in enumerate(connections):
            # Çarpışma engelleme kuvveti
            repulsive_force = collision_avoidance.calculate_total_force(i, current_positions)
            
            # Hız sınırlaması ve dinamik hız ayarı
            max_velocity = 3.0  # m/s
            force_magnitude = np.linalg.norm(repulsive_force)
            if force_magnitude > max_velocity:
                repulsive_force = repulsive_force / force_magnitude * max_velocity
            elif force_magnitude < 1.0:  # Dinamik hız azaltma
                repulsive_force = repulsive_force * 0.5
            
            # Hız komutunu gönder
            vx = repulsive_force[0]
            vy = repulsive_force[1]
            send_velocity(vehicle, vx, vy, 0)
        
        # Simülasyon testi için bir kontrol ekle
        if not connections[0].location.global_relative_frame.alt:
            print("⚠️ Simülasyon ortamında test ediliyor...")
        
        time.sleep(0.1)  # 100ms döngü

def main():
    """Ana fonksiyon"""
    print("🚀 Çarpışma Engelleme Drone Sürü Kontrolü")
    
    # 1. Drone'ları arm et
    print("\n1️⃣ Drone'lar arm ediliyor...")
    threads = []
    for conn in connections:
        t = threading.Thread(target=arm, args=(conn,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    time.sleep(2)
    
    # 2. Drone'ları havaya kaldır
    print("\n2️⃣ Drone'lar havaya kaldırılıyor...")
    threads = []
    for conn in connections:
        t = threading.Thread(target=takeoff, args=(conn, 10))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    print("\n🚁 Tüm dronelar havada!")
    
    # 3. Çarpışma engelleme kontrol başlat
    print("\n3️⃣ Çarpışma engelleme kontrol başlatılıyor...")
    collision_avoidance_control(connections)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Görev iptal edildi!")
        # Acil durum: tüm drone'ları durdur
        for vehicle in connections:
            send_velocity(vehicle, 0, 0, 0)
    finally:
        # Bağlantıları kapat
        for vehicle in connections:
            vehicle.close()
        print("🔌 Tüm bağlantılar kapatıldı.")