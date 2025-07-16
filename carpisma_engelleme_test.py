from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from pymavlink import mavutil
from scipy.optimize import linear_sum_assignment

NUM_DRONES = 3  # Drone sayısı

# Drone bağlantı noktaları
BASE_PORT = 14550  # Başlangıç port numarası

collision_avoidance_active = False
collision_avoidance_thread = None

# Çarpışma engelleme hızları - drone index'e göre (0-indexli)
collision_velocities = {}
collision_lock = threading.Lock()

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

def get_drone_id(vehicle):
    """Drone'un ID'sini döndürür (0-indexli)"""
    try:
        return connections.index(vehicle)
    except ValueError:
        return -1

def send_velocity(vehicle, vx, vy, vz):
    """Hız komutu gönder"""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def go(vehicle, target_location, tolerance=0.5, timeout=60):
    """PID ile drone'u hedefe gönder + çarpışma engelleme"""
    
    # PID kontrolcüler
    pid_north = PID(0.3, 0.05, 0.2, 3)
    pid_east = PID(0.3, 0.05, 0.2, 3)
    pid_down = PID(0.8, 0.05, 0.02, 2)
    
    drone_idx = get_drone_id(vehicle)
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
        
        # PID çıkışları (m/s)
        vx = pid_north.update(north_err)
        vy = pid_east.update(east_err)
        vz = pid_down.update(down_err)

        # Çarpışma engelleme hızını ekle
        with collision_lock:
            collision_v = collision_velocities.get(drone_idx, (0, 0, 0))
            
        vx += collision_v[0]
        vy += collision_v[1]
        vz += collision_v[2]
        
        # Hız sınırlaması
        max_speed = 3.0
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > max_speed:
            vx = vx / speed * max_speed
            vy = vy / speed * max_speed
            vz = vz / speed * max_speed
        
        send_velocity(vehicle, vx, vy, vz)
        time.sleep(0.05)

def send_goto_with_pid(vehicle, lat, lon, alt):
    target = LocationGlobalRelative(lat, lon, alt)
    go(vehicle, target)

# ----------------- Genel Fonksiyonlar -----------------

def arm(vehicle):
    """Drone'u arm eder"""
    drone_id = get_drone_id(vehicle) + 1  # Display için 1-indexli
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    vehicle.armed = True
    while not vehicle.armed:
        print(f"⏳ Drone {drone_id} arm edilmeyi bekliyor...")
        time.sleep(1)
    print(f"✅ Drone {drone_id} arm edildi")

def takeoff(vehicle, alt):
    """Drone'u belirtilen yüksekliğe kaldırır"""
    drone_id = get_drone_id(vehicle) + 1  # Display için 1-indexli
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
        """
        # Sadece lat/lon kullanarak 2D mesafe hesapla
        lat_diff = (pos_i[0] - pos_j[0]) * 111000  # metre
        lon_diff = (pos_i[1] - pos_j[1]) * 111000 * math.cos(math.radians(pos_i[0]))  # metre
        
        diff = np.array([lat_diff, lon_diff])
        distance = np.linalg.norm(diff)
        
        # Çok yakın durumda sıfıra bölme hatası önleme
        if distance < 0.5:
            distance = 0.5
            # Rastgele bir yön ver
            angle = np.random.uniform(0, 2*np.pi)
            diff = np.array([np.cos(angle), np.sin(angle)]) * 0.5
        
        # Mesafe etkileşim yarıçapından büyükse kuvvet yok
        if distance > self.r_s:
            return np.array([0.0, 0.0])
        
        # Potansiyel fonksiyonu
        force_magnitude = self.a * (np.exp(-self.beta * distance) - np.exp(-self.beta * self.r_s))
        
        # Kuvvet yönü (i'den j'ye doğru, yani itici)
        force_direction = diff / distance
        
        return force_magnitude * force_direction
    
    def calculate_total_force(self, drone_index, positions):
        """
        Bir drone'un diğer tüm drone'lardan aldığı toplam itici kuvvet
        """
        total_force = np.array([0.0, 0.0])  # 2D force (north, east)
        
        for j, pos_j in enumerate(positions):
            if j != drone_index:
                force = self.calculate_repulsive_force(positions[drone_index], pos_j)
                total_force += force
        
        return total_force

def collision_avoidance_control(connections, a, beta, rs):
    """Çarpışma engelleme kontrol döngüsü"""
    collision_avoidance = CollisionAvoidance(a=a, beta=beta, r_s=rs)
    
    print(f"🛡️ Çarpışma engelleme kontrol başladı (a={a}, beta={beta}, rs={rs})")
    
    while collision_avoidance_active:
        try:
            # Güncel pozisyonları al
            current_positions = [get_position(vehicle) for vehicle in connections]
            
            # Her drone için çarpışma engelleme kuvvetini hesapla
            for i, vehicle in enumerate(connections):
                # Çarpışma engelleme kuvveti (metre cinsinden)
                repulsive_force = collision_avoidance.calculate_total_force(i, current_positions)
                
                # Hız sınırlaması
                max_velocity = 2.0  # m/s
                force_magnitude = np.linalg.norm(repulsive_force)
                if force_magnitude > max_velocity:
                    repulsive_force = repulsive_force / force_magnitude * max_velocity
                
                # Hız komutunu hazırla (north/east -> vx/vy m/s cinsinden)
                vx = repulsive_force[0]  # North direction (m/s)
                vy = repulsive_force[1]  # East direction (m/s)
                vz = 0.0  # Altitude sabit
                
                # Thread-safe şekilde collision_velocities'i güncelle
                with collision_lock:
                    collision_velocities[i] = (vx, vy, vz)
            
            # Debug bilgisi
            if any(np.linalg.norm([v[0], v[1]]) > 0.1 for v in collision_velocities.values()):
                print(f"🛡️ Çarpışma engelleme aktif: {[(i, f'{v[0]:.2f}, {v[1]:.2f}') for i, v in collision_velocities.items() if np.linalg.norm([v[0], v[1]]) > 0.1]}")
            
        except Exception as e:
            print(f"🛡️ Çarpışma engelleme hatası: {e}")
        
        time.sleep(0.05)  # 100ms döngü

def start_collision_avoidance(connections, alpha=2.0, beta=0.5, rs=8.0):
    """Çarpışma engelleme sistemini başlat"""
    global collision_avoidance_active, collision_avoidance_thread
    
    if not collision_avoidance_active:
        collision_avoidance_active = True
        
        # collision_velocities'i sıfırla
        with collision_lock:
            collision_velocities.clear()
            for i in range(len(connections)):
                collision_velocities[i] = (0.0, 0.0, 0.0)
        
        collision_avoidance_thread = threading.Thread(
            target=collision_avoidance_control,
            args=(connections, alpha, beta, rs)
        )
        collision_avoidance_thread.daemon = True
        collision_avoidance_thread.start()
        print("🛡️ Çarpışma engelleme sistemi başlatıldı")
    else:
        print("🛡️ Çarpışma engelleme sistemi zaten aktif")

def stop_collision_avoidance():
    """Çarpışma engelleme sistemini durdur"""
    global collision_avoidance_active
    collision_avoidance_active = False
    
    # Çarpışma engelleme hızlarını sıfırla
    with collision_lock:
        for i in range(len(connections)):
            collision_velocities[i] = (0.0, 0.0, 0.0)
    
    if collision_avoidance_thread:
        collision_avoidance_thread.join(timeout=1.0)
    
    print("🛡️ Çarpışma engelleme sistemi durduruldu")

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

def formation_v(spacing_meter, num_drones, alt, donus_acisi_derece=0):
    center = get_swarm_center(connections)
    
    # V formasyonu için pozisyonları hesapla (metre cinsinden)
    relative_positions_meter = []
    
    # Lider drone'un pozisyonunu hesapla (V'nin önünde)
    cos_30 = math.cos(math.radians(30))
    leader_offset_x = (2/3) * ((num_drones // 2) * spacing_meter * cos_30)
    
    # İlk drone (lider) V'nin önüne
    relative_positions_meter.append((leader_offset_x, 0))
    
    # Diğer dronelar V şeklinde simetrik yerleştirilir
    for i in range(1, num_drones):
        row = (i + 1) // 2
        
        if i % 2 == 1:  # sol taraf
            side = -1
        else:  # sağ taraf
            side = 1
        
        x_offset = leader_offset_x - (row * spacing_meter * cos_30)
        y_offset = side * row * spacing_meter * math.sin(math.radians(30))
        
        relative_positions_meter.append((x_offset, y_offset))
    
    # Döndürme işlemi
    rotated_positions_meter = []
    for x_m, y_m in relative_positions_meter:
        rotated_x, rotated_y = dondur_koordinatlari(x_m, y_m, donus_acisi_derece)
        rotated_positions_meter.append((rotated_x, rotated_y))
    
    # Metre cinsinden pozisyonları enlem/boylam'a çevir
    target_positions = []
    for x_m, y_m in rotated_positions_meter:
        lat_offset, lon_offset = meters_to_degrees(x_m, y_m, center[0])
        target_lat = center[0] + lat_offset
        target_lon = center[1] + lon_offset
        target_positions.append((target_lat, target_lon, alt))
    
    # Hungarian algoritması ile optimum atama
    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    
    print("=== V Formasyonu Drone Atamaları ===")
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        current_pos = get_position(conn)
        distance_to_target = calculate_distance(current_pos[0], current_pos[1], target[0], target[1])
        print(f"Drone {i+1} -> Hedef {col_ind[i]+1}: {distance_to_target:.2f}m")
    
    # Droneları hedeflerine gönder
    threads = []
    for i, conn in enumerate(connections):
        target = target_positions[i]
        t = threading.Thread(target=send_goto_with_pid, args=(conn, *target))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def formation_line(spacing_meter, num_drones, alt, donus_acisi_derece=0):
    center = get_swarm_center(connections)
    
    # Line formasyonu için pozisyonları hesapla
    relative_positions_meter = []
    
    total_length = (num_drones - 1) * spacing_meter
    start_offset = -total_length / 2
    
    for i in range(num_drones):
        x_offset = start_offset + (i * spacing_meter)
        y_offset = 0
        relative_positions_meter.append((x_offset, y_offset))
    
    # Döndürme işlemi
    rotated_positions_meter = []
    for x_m, y_m in relative_positions_meter:
        rotated_x, rotated_y = dondur_koordinatlari(x_m, y_m, donus_acisi_derece)
        rotated_positions_meter.append((rotated_x, rotated_y))
    
    # Metre cinsinden pozisyonları enlem/boylam'a çevir
    target_positions = []
    for x_m, y_m in rotated_positions_meter:
        lat_offset, lon_offset = meters_to_degrees(x_m, y_m, center[0])
        target_lat = center[0] + lat_offset
        target_lon = center[1] + lon_offset
        target_positions.append((target_lat, target_lon, alt))
    
    # Hungarian algoritması ile optimum atama
    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    
    print("=== Line Formasyonu Drone Atamaları ===")
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        current_pos = get_position(conn)
        distance_to_target = calculate_distance(current_pos[0], current_pos[1], target[0], target[1])
        print(f"Drone {i+1} -> Hedef {col_ind[i]+1}: {distance_to_target:.2f}m")
    
    # Droneları hedeflerine gönder
    threads = []
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
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

    # Çarpışma engelleme sistemini başlat
    start_collision_avoidance(connections, alpha=200.0, beta=0.5, rs=4)
    time.sleep(2)  # Sistem aktif olana kadar bekle
    
    # Formasyonları test et
    print("=== V Formasyonu Test Ediliyor ===")
    formation_v(4, len(connections), 10, 0) 
    time.sleep(10)
    
    print("=== Line Formasyonu Test Ediliyor ===")
    formation_line(4, len(connections), 10, 0) 
    time.sleep(10)
    
    print("=== Ters V Formasyonu Test Ediliyor ===")
    formation_v(4, len(connections), 10, 180) 
    time.sleep(8)
    
    print("Tüm formasyonlar başarıyla tamamlandı.")
    
    # Çarpışma engelleme sistemini durdur
    stop_collision_avoidance()
    time.sleep(2)

    land_all(connections)