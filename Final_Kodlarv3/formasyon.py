from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from pymavlink import mavutil
from scipy.optimize import linear_sum_assignment

NUM_DRONES = 3
BASE_PORT = 14550  
TARGET_ALTITUDE = 10.0
MAX_SPEED = 3.0 
SAFETY_DISTANCE = 2.0
SPACİNG_METER = 5
ROTATION_ANGLE = 0

collision_avoidance_active = False
collision_avoidance_thread = None

collision_velocities = {}
collision_lock = threading.Lock()

connections = []
ports = [f'udp:127.0.0.1:{BASE_PORT + i*10}' for i in range(NUM_DRONES)]

for i, port in enumerate(ports):
    vehicle = connect(port, wait_ready=True)
    connections.append(vehicle)

# ----------------- PID Kontrolcüsü -----------------

class PID:
    def __init__(self, kp, ki, kd, limit=3):
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

def go(vehicle, lat, lon, alt, tolerance=0.5):
    """PID ile drone'u hedefe gönder + çarpışma engelleme"""
    target_location = LocationGlobalRelative(lat, lon, alt)
    
    # PID kontrolcüler
    pid_north = PID(0.4, 0.02, 0.6, MAX_SPEED)
    pid_east = PID(0.4, 0.02, 0.6, MAX_SPEED)
    
    drone_idx = get_drone_id(vehicle)
    
    while True:
        # Mevcut konum
        curr = vehicle.location.global_relative_frame
        
        # Hedef konuma olan hata
        north_err = (target_location.lat - curr.lat) * 111000
        east_err = (target_location.lon - curr.lon) * 111000 * math.cos(math.radians(curr.lat))
        
        # Mesafe kontrolü
        distance = math.sqrt(north_err**2 + east_err**2)
        if distance <= tolerance:
            send_velocity(vehicle, 0, 0, 0)
            break
        
        # PID çıkışları (m/s)
        vx = pid_north.update(north_err)
        vy = pid_east.update(east_err)

        # Çarpışma engelleme hızını ekle
        with collision_lock:
            collision_v = collision_velocities.get(drone_idx, (0, 0, 0))
            
        vx += collision_v[0]
        vy += collision_v[1]
        
        # Hız sınırlaması
        speed = math.sqrt(vx**2 + vy**2)
        if speed > MAX_SPEED:
            vx = vx / speed * MAX_SPEED
            vy = vy / speed * MAX_SPEED

        send_velocity(vehicle, vx, vy, 0)
        time.sleep(0.05)

# ----------------- Genel Fonksiyonlar -----------------

def arm(vehicle, timeout=10):
    """Drone'u arm etme"""
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)  # Mode geçişinin oturması için kısa bekleme

    vehicle.armed = True

    start_time = time.time()
    while not vehicle.armed:
        if time.time() - start_time > timeout:
            return False
        time.sleep(0.1)

    return True

def takeoff(vehicle, alt, timeout=60):
    """Drone'u belirtilen yüksekliğe kaldırır"""
    start_time = time.time()
    vehicle.simple_takeoff(alt)
    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        if current_alt >= alt * 0.95  or time.time() - start_time > timeout:
            break
        time.sleep(0.2)

def get_position(vehicle):
    """Drone'un mevcut konumunu döndürür"""
    return (vehicle.location.global_relative_frame.lat, 
            vehicle.location.global_relative_frame.lon, 
            vehicle.location.global_relative_frame.alt)

def get_swarm_center(vehicles):
    """Sürünün merkez koordinatlarını hesaplar"""
    positions = [get_position(vehicle) for vehicle in vehicles]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    
    return (avg_lat, avg_lon, avg_alt)

def get_drone_id(vehicle):
    """Drone'un ID'sini döndürür"""
    return connections.index(vehicle)

def send_velocity(vehicle, vx, vy, vz):
    """Hız komutu gönderme"""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def land_all(connections):
    """Tüm droneları indirir"""
    for i, vehicle in enumerate(connections):
        vehicle.mode = VehicleMode("LAND")

# ----------------- Çarpışma Engelleme Fonksiyonları -----------------

class CollisionAvoidance:
    def __init__(self, alpha=500.0, beta=0.5, r_s=SAFETY_DISTANCE):
        """
        Çarpışma engelleme parametreleri
        alpha: Kuvvet büyüklüğü
        beta: Kuvvet azalma hızı
        r_s: Etkileşim yarıçapı (metre)
        """
        self.alpha = alpha
        self.beta = beta
        self.r_s = r_s
    
    def calculate_repulsive_force(self, pos_i, pos_j):
        """İki drone arasındaki itici kuvveti hesaplama"""

        lat_diff = (pos_i[0] - pos_j[0]) * 111000 
        lon_diff = (pos_i[1] - pos_j[1]) * 111000 * math.cos(math.radians(pos_i[0]))
        
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
        force_magnitude = self.alpha * (np.exp(-self.beta * distance) - np.exp(-self.beta * self.r_s))
        
        # Kuvvet yönü (i'den j'ye doğru, yani itici)
        force_direction = diff / distance
        
        return force_magnitude * force_direction
    
    def calculate_total_force(self, drone_index, positions):
        """Bir drone'un diğer tüm drone'lardan aldığı toplam itici kuvvet"""

        total_force = np.array([0.0, 0.0])  
        
        for j, pos_j in enumerate(positions):
            if j != drone_index:
                force = self.calculate_repulsive_force(positions[drone_index], pos_j)
                total_force += force
        
        return total_force

def collision_avoidance_control(connections, alpha, beta, rs):
    """Çarpışma engelleme kontrol döngüsü"""
    collision_avoidance = CollisionAvoidance(alpha=alpha, beta=beta, r_s=rs)
    
    while collision_avoidance_active:
        # Güncel pozisyonları alma
        current_positions = [get_position(vehicle) for vehicle in connections]
        
        # Her drone için çarpışma engelleme kuvvetini hesaplama
        for i, vehicle in enumerate(connections):
            # Çarpışma engelleme kuvveti
            repulsive_force = collision_avoidance.calculate_total_force(i, current_positions)
            
            # Hız sınırlaması
            force_magnitude = np.linalg.norm(repulsive_force)
            if force_magnitude > MAX_SPEED:
                repulsive_force = repulsive_force / force_magnitude * MAX_SPEED
            
            # collision_velocities'i güncelleme
            with collision_lock:
                collision_velocities[i] = (repulsive_force[0], repulsive_force[1], 0.0)
        
        time.sleep(0.1)  

def start_collision_avoidance(connections, alpha=500.0, beta=0.5, rs=SAFETY_DISTANCE):
    """Çarpışma engelleme sistemini başlat"""
    global collision_avoidance_active, collision_avoidance_thread
    
    collision_avoidance_active = True
    
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

def stop_collision_avoidance():
    """Çarpışma engelleme sistemini durdur"""
    global collision_avoidance_active
    collision_avoidance_active = False
    
    with collision_lock:
        for i in range(len(connections)):
            collision_velocities[i] = (0.0, 0.0, 0.0)
    
    if collision_avoidance_thread:
        collision_avoidance_thread.join(timeout=1.0)

#------------------ Formasyon Fonksiyonları -----------------

def calculate_distance(lat1, lon1, lat2, lon2):
    """İki konum arasındaki mesafeyi hesaplama - Haversine formülü"""
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
    """Metre cinsinden mesafeyi enlem/boylam derecelerine dönüştürme"""
    lat_deg = meters_north / 111000.0
    lon_deg = meters_east / (111000.0 * math.cos(math.radians(lat)))
    
    return lat_deg, lon_deg

def calculate_cost_matrix(connections, target_positions):
    """Drone'ların mevcut konumlarına göre hedeflere olan mesafe matrisini hesaplama"""
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
    """Hungarian algoritması ile en uygun hedef atamalarını yapma"""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def rotate_coordinates(x, y, degrees):
    radians = math.radians(degrees)
    x_new = x * math.cos(radians) - y * math.sin(radians)
    y_new = x * math.sin(radians) + y * math.cos(radians)
    return (x_new, y_new)

def formation_v(spacing_meter, num_drones, alt, rotation_angle=0):
    center = get_swarm_center(connections)
    
    relative_positions_meter = []
    
    # Lider drone'un pozisyonunu hesaplama 
    cos_30 = math.cos(math.radians(30))
    leader_offset_x = (2/3) * ((num_drones // 2) * spacing_meter * cos_30)
    relative_positions_meter.append((leader_offset_x, 0))
    
    # Diğer droneların pozisyonlarını hesaplama
    for i in range(1, num_drones):
        row = (i + 1) // 2
        
        if i % 2 == 1:
            side = -1
        else:
            side = 1
        
        x_offset = leader_offset_x - (row * spacing_meter * cos_30)
        y_offset = side * row * spacing_meter * math.sin(math.radians(30))
        
        relative_positions_meter.append((x_offset, y_offset))
    
    # Rotasyon açısına göre döndürme işlemi
    rotated_positions_meter = []
    for x_m, y_m in relative_positions_meter:
        rotated_x, rotated_y = rotate_coordinates(x_m, y_m, rotation_angle)
        rotated_positions_meter.append((rotated_x, rotated_y))
    
    # Metre cinsinden pozisyonları enlem/boylama çevirme
    target_positions = []
    for x_m, y_m in rotated_positions_meter:
        lat_offset, lon_offset = meters_to_degrees(x_m, y_m, center[0])
        target_lat = center[0] + lat_offset
        target_lon = center[1] + lon_offset
        target_positions.append((target_lat, target_lon, alt))
    
    # Macar algoritması ile optimum atama
    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
    
    # Droneları hedeflerine gönderme
    go_threads = []
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        t = threading.Thread(target=go, args=(conn, *target))
        go_threads.append(t)
        t.start()
    for t in go_threads:
        t.join()

def formation_line(spacing_meter, num_drones, alt, rotation_angle=0):
    center = get_swarm_center(connections)
    
    relative_positions_meter = []
    
    # Lider drone'un pozisyonunu hesaplama
    total_length = (num_drones - 1) * spacing_meter
    start_offset = -total_length / 2
    
    # Diğer droneların pozisyonlarını hesaplama
    for i in range(num_drones):
        x_offset = start_offset + (i * spacing_meter)
        y_offset = 0
        relative_positions_meter.append((x_offset, y_offset))
    
    # Rotasyon açısına göre döndürme işlemi
    rotated_positions_meter = []
    for x_m, y_m in relative_positions_meter:
        rotated_x, rotated_y = rotate_coordinates(x_m, y_m, rotation_angle)
        rotated_positions_meter.append((rotated_x, rotated_y))
    
    # Metre cinsinden pozisyonları enlem/boylama çevirme
    target_positions = []
    for x_m, y_m in rotated_positions_meter:
        lat_offset, lon_offset = meters_to_degrees(x_m, y_m, center[0])
        target_lat = center[0] + lat_offset
        target_lon = center[1] + lon_offset
        target_positions.append((target_lat, target_lon, alt))
    
    # Macar algoritması ile optimum atama
    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
    
    # Droneları hedeflerine gönderme
    go_threads = []
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        t = threading.Thread(target=go, args=(conn, *target))
        go_threads.append(t)
        t.start()
    for t in go_threads:
        t.join()

def formation_reverse_v(spacing_meter, num_drones, alt, rotation_angle=0):
    center = get_swarm_center(connections)
    
    relative_positions_meter = []
    
    # Lider drone'un pozisyonunu hesaplama
    cos_30 = math.cos(math.radians(30))
    leader_offset_x = -(2/3) * ((num_drones // 2) * spacing_meter * cos_30)
    relative_positions_meter.append((leader_offset_x, 0))
    
    # Diğer droneların pozisyonlarını hesaplama
    for i in range(1, num_drones):
        row = (i + 1) // 2
        
        if i % 2 == 1:
            side = -1
        else:
            side = 1

        x_offset = leader_offset_x + (row * spacing_meter * cos_30)
        y_offset = side * row * spacing_meter * math.sin(math.radians(30)) 
        
        relative_positions_meter.append((x_offset, y_offset))
    
    # Rotasyon açısına göre döndürme işlemi
    rotated_positions_meter = []
    for x_m, y_m in relative_positions_meter:
        rotated_x, rotated_y = rotate_coordinates(x_m, y_m, rotation_angle)
        rotated_positions_meter.append((rotated_x, rotated_y))
    
    # Metre cinsinden pozisyonları enlem/boylama çevirme
    target_positions = []
    for x_m, y_m in rotated_positions_meter:
        lat_offset, lon_offset = meters_to_degrees(x_m, y_m, center[0])
        target_lat = center[0] + lat_offset
        target_lon = center[1] + lon_offset
        target_positions.append((target_lat, target_lon, alt))
    
    # Macar algoritması ile optimum atama
    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
    
    # Droneları hedeflerine gönderme
    go_threads = []
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        t = threading.Thread(target=go, args=(conn, *target))
        go_threads.append(t)
        t.start()
    for t in go_threads:
        t.join()

if __name__ == "__main__":
    # Arm işlemini paralel yap
    arm_threads = []
    for conn in connections:
        t = threading.Thread(target=arm, args=(conn,))
        arm_threads.append(t)
        t.start()
    for t in arm_threads:
        t.join()
    time.sleep(1)

    # Kalkış işlemini paralel yap
    takeoff_threads = []
    for conn in connections:
        t = threading.Thread(target=takeoff, args=(conn, TARGET_ALTITUDE))
        takeoff_threads.append(t)
        t.start()
    for t in takeoff_threads:
        t.join()
    time.sleep(8)

    # Çarpışma engelleme sistemini başlatma
    start_collision_avoidance(connections, alpha=500.0, beta=0.5, rs=SAFETY_DISTANCE)

    # Formasyon oluşturma adımları
    formation_v(SPACİNG_METER, len(connections), TARGET_ALTITUDE, ROTATION_ANGLE) 
    time.sleep(10)
    
    formation_line(SPACİNG_METER, len(connections), TARGET_ALTITUDE, ROTATION_ANGLE) 
    time.sleep(10)
    
    formation_reverse_v(SPACİNG_METER, len(connections), TARGET_ALTITUDE, ROTATION_ANGLE) 
    time.sleep(10)
    
    # Çarpışma engelleme sistemini durdurma
    stop_collision_avoidance()

    land_all(connections)