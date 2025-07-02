from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from pymavlink import mavutil
from scipy.optimize import linear_sum_assignment

NUM_DRONES = 5  # Drone sayısı

# Drone bağlantı noktaları
BASE_PORT = 14550  # Başlangıç port numarası

"""Belirtilen sayıda drone bağlantısı oluşturur"""
connections = []
ports = [f'udp:127.0.0.1:{BASE_PORT + i*10}' for i in range(NUM_DRONES)]

for i, port in enumerate(ports):
    try:
        vehicle = connect(port, wait_ready=True)
        connections.append(vehicle)
        print(f"✓ Bağlandı: Drone {i+1} - {port}")
    except Exception as e:
        print(f"✗ Drone {i+1} bağlantı hatası: {e}")

print(f"Toplam {len(connections)} drone bağlandı.")

class PID:
    def __init__(self, kp, ki, kd, limit=5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def update(self, error):
        dt = time.time() - self.last_time
        if dt <= 0: dt = 0.01
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-self.limit, min(self.limit, output))
        
        self.prev_error = error
        self.last_time = time.time()
        return output

def go(vehicle, target_location, tolerance=1.0, timeout=60):
    """PID ile drone'u hedefe gönder"""
    
    # PID kontrolcüler
    pid_north = PID(0.5, 0.1, 0.2, 3)
    pid_east = PID(0.5, 0.1, 0.2, 3)
    pid_down = PID(0.8, 0.05, 0.02, 2)
    
    start_time = time.time()
    
    while True:
        # Mevcut konum
        curr = vehicle.location.global_relative_frame
        
        # Hatalar (metre)
        north_err = (target_location.lat - curr.lat) * 111000
        east_err = (target_location.lon - curr.lon) * 111000 * math.cos(math.radians(curr.lat))
        down_err = -(target_location.alt - curr.alt)
        
        # Mesafe kontrolü
        distance = math.sqrt(north_err**2 + east_err**2)
        if distance <= tolerance and abs(down_err) <= tolerance:
            send_velocity(vehicle, 0, 0, 0)  # Durdur
            break
            
        # Timeout kontrolü
        if time.time() - start_time > timeout:
            send_velocity(vehicle, 0, 0, 0)
            break
        
        # PID çıkışları
        vx = pid_north.update(north_err)
        vy = pid_east.update(east_err)
        vz = pid_down.update(down_err)
        
        send_velocity(vehicle, vx, vy, vz)
        time.sleep(0.1)

def send_velocity(vehicle, vx, vy, vz):
    """Hız komutu gönder"""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def send_goto_with_pid(vehicle, lat, lon, alt):
    target = LocationGlobalRelative(lat, lon, alt)
    go(vehicle, target)

# ----------------- Genel Fonksiyonlar -----------------

def get_drone_id(vehicle):
    """Drone'un ID'sini döndürür"""
    try:
        return connections.index(vehicle) + 1
    except ValueError:
        return "Bilinmeyen"

def arm(vehicle):
    """Drone'u arm eder"""
    drone_id = get_drone_id(vehicle)
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    vehicle.armed = True
    while not vehicle.armed:
        print(f"⏳ Drone {drone_id} arm edilmeyi bekliyor...")
        time.sleep(1)
    print(f"✅ Drone {drone_id} arm edildi")

def takeoff(vehicle, alt):
    """Drone'u belirtilen yüksekliğe kaldırır"""
    drone_id = get_drone_id(vehicle)
    vehicle.simple_takeoff(alt)
    while vehicle.location.global_relative_frame.alt < alt * 0.95:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f"🚁 Drone {drone_id} yükseklik: {current_alt:.2f}m / {alt}m")
        time.sleep(1)
    print(f"✅ Drone {drone_id} hedef yüksekliğe ulaştı: {alt}m")

def send_goto(vehicle, lat, lon, alt):
    """Drone'u belirtilen koordinatlara gönderir"""
    target_location = LocationGlobalRelative(lat, lon, alt)
    vehicle.simple_goto(target_location)

def get_position(vehicle):
    """Drone'un mevcut konumunu döndürür"""
    return (vehicle.location.global_relative_frame.lat, 
            vehicle.location.global_relative_frame.lon, 
            vehicle.location.global_relative_frame.alt)

def get_swarm_center(vehicles):
    """Sürünün merkez koordinatlarını hesaplar"""
    if not vehicles:
        return None
    
    positions = [get_position(vehicle) for vehicle in vehicles]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    
    print(f"📍 Sürü merkezi: Lat={avg_lat:.8f}, Lon={avg_lon:.8f}, Alt={avg_alt:.2f}")
    return (avg_lat, avg_lon, avg_alt)

# ----------------- Formasyon Fonksiyonları -----------------

def calculate_distance(lat1, lon1, lat2, lon2):
    """İki konum arasındaki mesafeyi hesaplar (metre cinsinden) - Haversine formülü"""
    R = 6371000  # Dünya yarıçapı, metre
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def meters_to_degrees(meters_north, meters_east, lat):
    """Metre cinsinden mesafeyi enlem/boylam derecelerine dönüştürür"""
    # 1 derece enlem = yaklaşık 111000 metre (sabit)
    # 1 derece boylam = 111000 * cos(enlem) metre (enleme bağlı)
    
    lat_deg = meters_north / 111000.0
    lon_deg = meters_east / (111000.0 * math.cos(math.radians(lat)))
    
    return lat_deg, lon_deg

def calculate_cost_matrix(connections, target_positions):
    """Drone'ların mevcut konumlarına göre hedeflere olan mesafe matrisini hesaplar."""
    current_positions = [get_position(conn) for conn in connections]
    num_drones = len(current_positions)
    num_targets = len(target_positions)
    cost_matrix = np.zeros((num_drones, num_targets))
    
    for i, (curr_lat, curr_lon, _) in enumerate(current_positions):
        for j, (tgt_lat, tgt_lon, _) in enumerate(target_positions):
            distance = calculate_distance(curr_lat, curr_lon, tgt_lat, tgt_lon)
            cost_matrix[i][j] = distance
            
    return cost_matrix

def assign_targets(cost_matrix):
    """Hungarian algoritması ile en uygun hedef atamalarını yapar."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def dondur_koordinatlari(x, y, derece):
    radyan = math.radians(derece)
    x_yeni = x * math.cos(radyan) - y * math.sin(radyan)
    y_yeni = x * math.sin(radyan) + y * math.cos(radyan)
    return (x_yeni, y_yeni)

def formation_v(spacing_meter, num_drones, alt, center, donus_acisi_derece=0):
    # V formasyonu için pozisyonları hesapla (metre cinsinden)
    relative_positions_meter = []
    
    # Lider drone'un pozisyonunu hesapla (V'nin önünde)
    # center + 2/3 * (drone_sayisi//2 * dronelar_arası_mesafe * cos(30))
    cos_30 = math.cos(math.radians(30))  # ≈ 0.866
    leader_offset_x = (2/3) * ((num_drones // 2) * spacing_meter * cos_30)
    
    # İlk drone (lider) V'nin önüne
    relative_positions_meter.append((leader_offset_x, 0))
    
    # Diğer dronelar V şeklinde simetrik yerleştirilir
    for i in range(1, num_drones):
        # V formasyonundaki sıra numarası (1, 1, 2, 2, 3, 3, ...)
        row = (i + 1) // 2
        
        # Hangi taraf: tek indexler sol (-), çift indexler sağ (+)
        if i % 2 == 1:  # 1, 3, 5, ... -> sol taraf
            side = -1
        else:  # 2, 4, 6, ... -> sağ taraf
            side = 1
        
        # V formasyonu koordinatları
        # X: geriye doğru (30 derece açı ile)
        x_offset = leader_offset_x - (row * spacing_meter * cos_30)
        # Y: yana doğru (30 derece açı ile)
        y_offset = side * row * spacing_meter * math.sin(math.radians(30))  # sin(30°) = 0.5
        
        relative_positions_meter.append((x_offset, y_offset))
    
    # Döndürme işlemi (metre cinsinden)
    rotated_positions_meter = []
    for x_m, y_m in relative_positions_meter:
        rotated_x, rotated_y = dondur_koordinatlari(x_m, y_m, donus_acisi_derece)
        rotated_positions_meter.append((rotated_x, rotated_y))
    
    # Metre cinsinden pozisyonları enlem/boylam'a çevir
    target_positions = []
    for x_m, y_m in rotated_positions_meter:
        # Merkez konumdan başlayarak offsetleri uygula
        lat_offset, lon_offset = meters_to_degrees(x_m, y_m, center[0])
        target_lat = center[0] + lat_offset
        target_lon = center[1] + lon_offset
        target_positions.append((target_lat, target_lon, alt))
    
    # Debug: pozisyonları yazdır
    print("=== V Formasyonu Pozisyonları ===")
    print(f"Lider drone offset: {leader_offset_x:.2f}m önde")
    for i, (lat, lon, altitude) in enumerate(target_positions):
        x_rel = rotated_positions_meter[i][0]
        y_rel = rotated_positions_meter[i][1]
        print(f"Drone {i+1}: X={x_rel:.2f}m, Y={y_rel:.2f}m (Lat={lat:.8f}, Lon={lon:.8f})")
    
    # Mesafeleri kontrol et
    print("=== Pozisyonlar Arası Mesafeler ===")
    for i in range(len(target_positions)):
        for j in range(i+1, len(target_positions)):
            dist = calculate_distance(target_positions[i][0], target_positions[i][1],
                                    target_positions[j][0], target_positions[j][1])
            print(f"Drone {i+1} - Drone {j+1}: {dist:.2f}m")
    
    # Simetri kontrolü
    print("=== Simetri Kontrolü ===")
    for i in range(1, len(target_positions), 2):  # Tek indexler (sol taraf)
        if i + 1 < len(target_positions):  # Çift index var mı?
            left_y = rotated_positions_meter[i][1]
            right_y = rotated_positions_meter[i+1][1]
            print(f"Sıra {(i+1)//2}: Sol={left_y:.2f}m, Sağ={right_y:.2f}m, Fark={abs(left_y + right_y):.3f}m")
    
    # Hungarian algoritması ile optimum atama
    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    
    print("=== Drone Atamaları ===")
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        current_pos = get_position(conn)
        distance_to_target = calculate_distance(current_pos[0], current_pos[1], target[0], target[1])
        print(f"Drone {i+1} -> Hedef Pozisyon {col_ind[i]+1}: {distance_to_target:.2f}m mesafe")
    
    # Droneları hedeflerine gönder
    threads = []
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        print(f"Drone {i+1} hedefi: {target}")
        t = threading.Thread(target=send_goto_with_pid, args=(conn, *target))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def land_all(connections):
        """Tüm droneları indirir"""
        print("Tüm dronelar indiriliyor...")
        
        for i, vehicle in enumerate(connections):
            print(f"Drone {i+1} indiriliyor...")
            vehicle.mode = VehicleMode("LAND")
def move_swarm_center(current_center, north_meters, east_meters):
    """Sürü merkezini belirtilen metre kadar hareket ettirir"""
    lat_offset, lon_offset = meters_to_degrees(north_meters, east_meters, current_center[0])
    new_lat = current_center[0] + lat_offset
    new_lon = current_center[1] + lon_offset
    new_alt = current_center[2]  # Yükseklik sabit

    return (new_lat, new_lon, new_alt)

# ----------------- Main -----------------

if __name__ == "__main__":
    # Arm işlemini paralel yap
    arm_threads = []
    for conn in connections:
        t = threading.Thread(target=arm, args=(conn,))
        arm_threads.append(t)
        t.start()
    for t in arm_threads:
        t.join()
    print("Tüm dronelar arm edildi.")
    time.sleep(2)

    # Kalkış işlemini paralel yap
    takeoff_threads = []
    for conn in connections:
        t = threading.Thread(target=takeoff, args=(conn, 10))
        takeoff_threads.append(t)
        t.start()
    for t in takeoff_threads:
        t.join()
    print("Kalkış komutları gönderildi.")
    time.sleep(8)
    
    #formasyon_sekli(offset, alt)
    formation_v(5, len(connections), 10, get_swarm_center(connections), 0) 
    time.sleep(8)
    formation_v(5, len(connections), 10, get_swarm_center(connections), 45) 
    time.sleep(8)

    new_center = move_swarm_center(get_swarm_center(connections), 10, 10)

    formation_v(5, len(connections), 10, new_center, 45) 
    time.sleep(8)
    formation_v(5, len(connections), 10, new_center, 315) 
    time.sleep(8)

    new_center = move_swarm_center(get_swarm_center(connections), 10, -10)

    formation_v(5, len(connections), 10, new_center, 315) 
    time.sleep(10)
    formation_v(5, len(connections), 10, new_center, 225) 
    time.sleep(8)

    new_center = move_swarm_center(get_swarm_center(connections), -10, -10)

    formation_v(5, len(connections), 10, new_center, 225) 
    time.sleep(8)

    land_all(connections)