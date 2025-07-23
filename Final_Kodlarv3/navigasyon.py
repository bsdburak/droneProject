from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from pymavlink import mavutil

NUM_DRONES = 3
BASE_PORT = 14550  
TARGET_ALTITUDE = 10.0
MAX_SPEED = 3.0 
SAFETY_DISTANCE = 2.0
ROTATION_TIME = 5.0
TARGET_WAYPOINTS = [
        (-35.36319568, 149.16515342, TARGET_ALTITUDE),
        (-35.3631253, 149.16538707, TARGET_ALTITUDE),
        (-35.36309971, 149.16519880, TARGET_ALTITUDE),
    ]

collision_avoidance_active = False
collision_avoidance_thread = None

collision_velocities = {}
collision_lock = threading.Lock()

current_formation_angle = 0.0  
initial_formation_deltas = None  

connections = []
ports = [f'udp:127.0.0.1:{BASE_PORT + i*10}' for i in range(NUM_DRONES)]

for i, port in enumerate(ports):
    vehicle = connect(port, wait_ready=True)
    connections.append(vehicle)

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
    """Drone'u hedef yüksekliğe kaldırır"""
    start_time = time.time()
    vehicle.simple_takeoff(alt)
    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        if current_alt >= alt * 0.95  or time.time() - start_time > timeout:
            break
        time.sleep(0.2)

def send_velocity(vehicle, vx, vy, vz):
    """Hız komutu gönderme"""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def get_position(vehicle):
    """Drone'un mevcut konumunu döndürür"""
    return (vehicle.location.global_relative_frame.lat, 
            vehicle.location.global_relative_frame.lon, 
            vehicle.location.global_relative_frame.alt)

def get_velocity(vehicle):
    """Drone'un mevcut hızını döndürür"""
    return (vehicle.velocity[0], vehicle.velocity[1], vehicle.velocity[2])

def get_drone_id(vehicle):
    """Drone'un ID'sini döndürür"""
    return connections.index(vehicle)

def get_swarm_center(vehicles):
    """Sürünün merkez koordinatlarını hesaplar"""
    positions = [get_position(vehicle) for vehicle in vehicles]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    
    return (avg_lat, avg_lon, avg_alt)

def get_swarm_velocity(vehicles):
    """Sürünün ortalama hızını hesaplar"""
    velocities = [get_velocity(vehicle) for vehicle in vehicles]
    avg_vx = sum(v[0] for v in velocities) / len(velocities)
    avg_vy = sum(v[1] for v in velocities) / len(velocities)
    avg_vz = sum(v[2] for v in velocities) / len(velocities)
    
    return (avg_vx, avg_vy, avg_vz)

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

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    """Lat/lon farkını metre cinsinden döndürme"""
    north_meters = (lat2 - lat1) * 111000
    
    east_meters = (lon2 - lon1) * 111000 * math.cos(math.radians(lat1))
    
    return north_meters, east_meters

def land_all(connections):
    """Tüm droneları indirir"""
    for i, vehicle in enumerate(connections):
        vehicle.mode = VehicleMode("LAND")

# ----------------- Sürü Hareketi Fonksiyonu -----------------

def move_swarm_to_target(vehicles, target_lat, target_lon, kp=0.5, ki=0.02, kd=0.3, tolerance=0.5):
    """
    Sürü formasyonu koruyarak hedef noktaya hareket eder
    PID Formül: u'ᵢ = kₚ(p* - p̄) + kᵢ∫(p* - p̄)dt + kd(- ṗ̄)
    """    
    start_time = time.time()

    # Integral terimleri için birikmiş hata
    integral_north = 0.0
    integral_east = 0.0

     # Önceki döngü zamanı
    prev_time = start_time
    
    while True:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        
        # Mevcut sürü merkezi ve hızını hesaplama
        current_center = get_swarm_center(vehicles)
        current_velocity = get_swarm_velocity(vehicles)
        
        # Hedef ile mevcut merkez arasındaki mesafe
        north_error, east_error = lat_lon_to_meters(
            current_center[0], current_center[1], 
            target_lat, target_lon
        )
        
        # Integral terimlerini güncelleme
        integral_north += north_error * dt
        integral_east += east_error * dt
        
        # Integral windup koruması
        max_integral = 50.0  # maksimum integral değeri
        integral_north = max(-max_integral, min(max_integral, integral_north))
        integral_east = max(-max_integral, min(max_integral, integral_east))

        # Toplam mesafe kontrolü
        horizontal_distance = math.sqrt(north_error**2 + east_error**2)
                
        # Hedefe ulaşma kontrolü
        if horizontal_distance <= tolerance:
            for vehicle in vehicles:
                send_velocity(vehicle, 0, 0, 0)
            break
        
        # PID kontrolü: u'ᵢ = kₚ(p* - p̄) + kᵢ∫(p* - p̄)dt + kd(- ṗ̄)
        # Proportional terimi: kₚ(p* - p̄)
        vx_proportional = kp * north_error
        vy_proportional = kp * east_error

        # Integral terimi: kᵢ∫(p* - p̄)dt
        vx_integral = ki * integral_north
        vy_integral = ki * integral_east
        
        # Derivative terimi: kₐ(- ṗ̄)
        vx_derivative = -kd * current_velocity[0]
        vy_derivative = -kd * current_velocity[1]
        
        # Toplam kontrol sinyali
        control_vx = vx_proportional + vx_integral + vx_derivative
        control_vy = vy_proportional + vy_integral + vy_derivative
        
        # Hız sınırlaması
        min_speed = 0.3  # minimum hız
        slow_down_distance = MAX_SPEED * 2  # yavaşlama başlangıç mesafesi

        if horizontal_distance <= slow_down_distance:
            # Mesafe azaldıkça hızı doğrusal olarak azaltma
            speed_factor = max(min_speed/MAX_SPEED, horizontal_distance / slow_down_distance)
        else:
            speed_factor = 1.0

        # Uyarlanmış maksimum hız
        adapted_max_speed = MAX_SPEED * speed_factor

        current_speed = math.sqrt(control_vx**2 + control_vy**2)
        if current_speed > adapted_max_speed:
            scale = adapted_max_speed / current_speed
            control_vx *= scale
            control_vy *= scale
        
        # Tüm droneları aynı hızla hareket ettirme
        for i, vehicle in enumerate(vehicles):
            with collision_lock:
                collision_v = collision_velocities.get(i, (0, 0, 0))
            send_velocity(vehicle, control_vx + collision_v[0], control_vy + collision_v[1], 0)
        time.sleep(0.1)

# ----------------- Rotasyon Fonksiyonları -----------------

def rotation_matrix(psi_s):
    return np.array([
        [np.cos(psi_s), -np.sin(psi_s)],
        [np.sin(psi_s),  np.cos(psi_s)]
    ])

def derivative_rotation_matrix(psi_s):
    return np.array([
        [-np.sin(psi_s), -np.cos(psi_s)],
        [ np.cos(psi_s), -np.sin(psi_s)]
    ])

class RotationController:
    def __init__(self, psi_start, psi_target, t_start, t_mi):
        self.psi_start = psi_start
        self.psi_target = psi_target
        self.t_start = t_start
        self.t_mi = t_mi
        self.chi1 = psi_start 
        self.chi2 = 0.0
        
    def compute_gamma(self, t):
        t_half = self.t_mi / 2
        if self.t_start <= t < self.t_start + t_half:
            # Hızlanma fazı
            return 4 * (self.psi_target - self.psi_start) / (self.t_mi**2)
        elif self.t_start + t_half <= t < self.t_start + self.t_mi:
            # Yavaşlama fazı
            return -4 * (self.psi_target - self.psi_start) / (self.t_mi**2)
        else:
            # Rotasyon yok
            return 0.0
    
    def update(self, dt, t):
        """Rotasyon açısını güncelleme"""
        gamma = self.compute_gamma(t)
        self.chi2 += gamma * dt  # Hız güncelleme
        self.chi1 += self.chi2 * dt  # Pozisyon güncelleme
        return self.chi1, self.chi2

def initialize_formation_deltas(connections, ref_lat, ref_lon):
    """Başlangıç formasyonu delta değerlerini hesaplama ve kaydetme"""
    global initial_formation_deltas
    
    if initial_formation_deltas is None:
        deltas = []
        for vehicle in connections:
            lat = vehicle.location.global_relative_frame.lat
            lon = vehicle.location.global_relative_frame.lon
            
            # Referans noktaya göre konumu hesaplama
            d_north = (lat - ref_lat) * 111000
            d_east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
            
            deltas.append(np.array([d_north, d_east]))
        
        initial_formation_deltas = deltas
    
    return initial_formation_deltas

def calculate_formation_velocity(i, positions, psi_s, psi_dot_s, deltas, k_p=1.0, k_R=1.0):
    """Formasyonu koruyan hız komutları"""
    N = len(positions)
    u_t = np.zeros(2)
    u_c = np.zeros(2)
    
    R = rotation_matrix(psi_s)
    dR_dpsi = derivative_rotation_matrix(psi_s)
    
    for j in range(N):
        if j == i:
            continue
            
        delta_ji = deltas[j] - deltas[i]
        
        # u_i^t terimi: pozisyon hatası kontrolü
        pos_error = (positions[j] - positions[i]) - R @ delta_ji
        u_t += pos_error
        
        # u_i^c terimi: rotasyon kompanzasyonu  
        u_c += (dR_dpsi @ delta_ji) * psi_dot_s
    
    # Toplam hız komutu
    u_total = k_p * u_t - k_R * u_c
    
    return u_total

def calculate_target_angle(current_lat, current_lon, target_lat, target_lon):
    """Hedef noktaya yönlenme açısını hesaplama"""
    d_north = (target_lat - current_lat) * 111000
    d_east = (target_lon - current_lon) * 111000 * math.cos(math.radians(current_lat))
    
    # Hedef açı (kuzeyden saat yönü)
    target_angle = math.atan2(d_east, d_north)
    return target_angle

def normalize_angle(angle):
    """Açıyı -π ve π aralığına normalize etme"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def rotate_formation_to_target(connections, target_lat, target_lon, t_mi=ROTATION_TIME):
    """Formasyonu hedef noktaya yönlendiren rotasyon kontrolü"""
    global current_formation_angle
    
    # Referans nokta (ilk drone'un konumu)
    ref_lat = connections[0].location.global_relative_frame.lat
    ref_lon = connections[0].location.global_relative_frame.lon
    
    # Başlangıç formasyonu delta değerlerini alma
    formation_deltas = initialize_formation_deltas(connections, ref_lat, ref_lon)
    
    # Hedef açıyı hesaplama
    target_angle = calculate_target_angle(ref_lat, ref_lon, target_lat, target_lon)
    
    # Açı farkını hesaplama
    angle_diff = normalize_angle(target_angle - current_formation_angle)
    final_target_angle = current_formation_angle + angle_diff
    
    # Rotasyon kontrolörü (mevcut açıdan hedef açıya)
    rotation_controller = RotationController(current_formation_angle, final_target_angle, 0.0, t_mi)

    start_time = time.time()
    dt = 0.1
    
    while time.time() - start_time <= t_mi:
        t_now = time.time() - start_time
        
        # Rotasyon açısı ve hızını güncelleme
        psi_s, psi_dot_s = rotation_controller.update(dt, t_now)
        
        current_positions = []
        for vehicle in connections:
            lat = vehicle.location.global_relative_frame.lat
            lon = vehicle.location.global_relative_frame.lon
            d_north = (lat - ref_lat) * 111000
            d_east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
            current_positions.append(np.array([d_north, d_east]))
        
        # Her drone için hız komutunu hesaplama
        for i, vehicle in enumerate(connections):
            velocity = calculate_formation_velocity(
                i, current_positions, psi_s, psi_dot_s, formation_deltas, k_p=0.3, k_R=0.05
            )
            
            vx = velocity[0]
            vy = velocity[1]
            send_velocity(vehicle, vx, vy, 0)
        
        time.sleep(dt)
    
    # Rotasyon bitince durdurma
    for vehicle in connections:
        send_velocity(vehicle, 0, 0, 0)
    
    # Formasyonun yeni yönünü güncelleme
    current_formation_angle = normalize_angle(final_target_angle)
    
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

# ----------------- Main -----------------

if __name__ == "__main__":
    # Arm işlemini paralel yap
    arm_threads = []
    for conn in connections:
        t = threading.Thread(target=arm, args=(conn, TARGET_ALTITUDE))
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

    start_collision_avoidance(connections, alpha=500.0, beta=0.5, rs=SAFETY_DISTANCE)

    for(target_lat, target_lon, target_alt) in TARGET_WAYPOINTS:
        #Sürüyü formasyonu koruyarak hedefe döndürme
        rotate_formation_to_target(connections, target_lat, target_lon, ROTATION_TIME)
        time.sleep(7)
        
        # Sürüyü hedefe gönderme
        move_swarm_to_target(connections, target_lat, target_lon)
        time.sleep(15)

    land_all(connections)