from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from pymavlink import mavutil

#------------------------Drone bağlantı ayarları------------------------

NUM_DRONES = 3  # Drone sayısı

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

# ----------------- Temel Fonksiyonlar -----------------

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

def send_velocity(vehicle, vx, vy, vz):
    """Hız komutu gönder"""
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
    """Drone'un mevcut hızını döndürür (m/s)"""
    return (vehicle.velocity[0], vehicle.velocity[1], vehicle.velocity[2])

def get_swarm_center(vehicles):
    """Sürünün merkez koordinatlarını hesaplar"""
    if not vehicles:
        return None
    
    positions = [get_position(vehicle) for vehicle in vehicles]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    
    return (avg_lat, avg_lon, avg_alt)

def get_swarm_velocity(vehicles):
    """Sürünün ortalama hızını hesaplar"""
    if not vehicles:
        return None
    
    velocities = [get_velocity(vehicle) for vehicle in vehicles]
    avg_vx = sum(v[0] for v in velocities) / len(velocities)
    avg_vy = sum(v[1] for v in velocities) / len(velocities)
    avg_vz = sum(v[2] for v in velocities) / len(velocities)
    
    return (avg_vx, avg_vy, avg_vz)

def calculate_distance(lat1, lon1, lat2, lon2):
    """İki konum arasındaki mesafeyi hesaplar (metre cinsinden)"""
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
    """Lat/lon farkını metre cinsinden North/East olarak döndürür"""
    # North (pozitif = kuzey)
    north_meters = (lat2 - lat1) * 111000
    
    # East (pozitif = doğu)
    east_meters = (lon2 - lon1) * 111000 * math.cos(math.radians(lat1))
    
    return north_meters, east_meters

def land_all(connections):
    """Tüm droneları indirir"""
    print("Tüm dronelar indiriliyor...")
    
    for i, vehicle in enumerate(connections):
        print(f"Drone {i+1} indiriliyor...")
        vehicle.mode = VehicleMode("LAND")

# ----------------- PID -----------------

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

def go(vehicle, target_location, tolerance=0.5, timeout=60):
    """PID ile drone'u hedefe gönder"""
    
    # PID kontrolcüler
    pid_north = PID(0.3, 0.05, 0.5, 3)
    pid_east  = PID(0.3, 0.05, 0.5, 3)
    pid_down  = PID(0.8, 0.05, 0.1, 2)
    
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

# ----------------- Rotasyon -----------------

def rotation_matrix(psi_s):
    """Rotasyon matrisi R(ψ_s)"""
    return np.array([
        [np.cos(psi_s), -np.sin(psi_s)],
        [np.sin(psi_s),  np.cos(psi_s)]
    ])

def derivative_rotation_matrix(psi_s):
    """Rotasyon matrisinin türevi ∂R(ψ_s)/∂ψ_s"""
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
        self.chi1 = psi_start  # ψ_s(t_i)
        self.chi2 = 0.0        # ψ̇_s başlangıç hızı
        
    def compute_gamma(self, t):
        """Rotasyon profili fonksiyonu γ(t)"""
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
        """Rotasyon açısını güncelle"""
        gamma = self.compute_gamma(t)
        self.chi2 += gamma * dt  # Hız güncelleme
        self.chi1 += self.chi2 * dt  # Pozisyon güncelleme
        return self.chi1, self.chi2

def initialize_formation_deltas(connections, ref_lat, ref_lon):
    """Başlangıç formasyonu delta değerlerini hesapla ve kaydet"""
    global initial_formation_deltas
    
    if initial_formation_deltas is None:
        deltas = []
        for vehicle in connections:
            lat = vehicle.location.global_relative_frame.lat
            lon = vehicle.location.global_relative_frame.lon
            
            # Referans noktaya göre konumu hesapla
            d_north = (lat - ref_lat) * 111000
            d_east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
            
            deltas.append(np.array([d_north, d_east]))
        
        initial_formation_deltas = deltas
        print(f"📐 Başlangıç formasyon delta değerleri kaydedildi:")
        for i, delta in enumerate(deltas):
            print(f"   Drone {i+1}: [{delta[0]:.2f}, {delta[1]:.2f}]")
    
    return initial_formation_deltas

def get_initial_formation(connections, ref_lat, ref_lon):
    """Başlangıç formasyonu delta değerlerini hesapla"""
    deltas = []
    for vehicle in connections:
        lat = vehicle.location.global_relative_frame.lat
        lon = vehicle.location.global_relative_frame.lon
        
        # Referans noktaya göre konumu hesapla
        d_north = (lat - ref_lat) * 111000
        d_east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
        
        deltas.append(np.array([d_north, d_east]))
    
    return deltas

def calculate_formation_velocity(i, positions, psi_s, psi_dot_s, deltas, k_p=1.0, k_R=1.0):
    """Formasyonu koruyan hız komutları (u_i^t ve u_i^c)"""
    N = len(positions)
    u_t = np.zeros(2)  # Pozisyon kontrol terimi
    u_c = np.zeros(2)  # Rotasyon kompanzasyon terimi
    
    R = rotation_matrix(psi_s)
    dR_dpsi = derivative_rotation_matrix(psi_s)
    
    for j in range(N):
        if j == i:
            continue
            
        # δ_ji = δ_j - δ_i (drone'lar arası istenen mesafe)
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
    """Hedef noktaya yönlenme açısını hesapla"""
    # Kuzey-doğu koordinatlarında hedef vektörü
    d_north = (target_lat - current_lat) * 111000
    d_east = (target_lon - current_lon) * 111000 * math.cos(math.radians(current_lat))
    
    # Hedef açı (kuzeyden saat yönü)
    target_angle = math.atan2(d_east, d_north)
    return target_angle

def normalize_angle(angle):
    """Açıyı -π ve π aralığına normalize et"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def rotate_formation_to_target(connections, target_lat, target_lon, t_mi=10.0):
    """Formasyonu hedef noktaya yönlendiren rotasyon"""
    global current_formation_angle
    
    # Referans nokta (ilk drone'un konumu)
    ref_lat = connections[0].location.global_relative_frame.lat
    ref_lon = connections[0].location.global_relative_frame.lon
    
    # Başlangıç formasyonu delta değerlerini al (ilk seferinde hesapla)
    formation_deltas = initialize_formation_deltas(connections, ref_lat, ref_lon)
    
    # Hedef açıyı hesapla
    target_angle = calculate_target_angle(ref_lat, ref_lon, target_lat, target_lon)
    
    # Açı farkını hesapla (en kısa rotasyon yolu)
    angle_diff = normalize_angle(target_angle - current_formation_angle)
    final_target_angle = current_formation_angle + angle_diff
    
    # Rotasyon kontrolörü (mevcut açıdan hedef açıya)
    rotation_controller = RotationController(current_formation_angle, final_target_angle, 0.0, t_mi)
    
    print(f"🔄 Rotasyon başlıyor:")
    print(f"   Mevcut açı: {math.degrees(current_formation_angle):.1f}°")
    print(f"   Hedef açı: {math.degrees(target_angle):.1f}°")
    print(f"   Rotasyon miktarı: {math.degrees(angle_diff):.1f}° ({t_mi}s)")
    
    start_time = time.time()
    dt = 0.1
    
    while time.time() - start_time <= t_mi:
        t_now = time.time() - start_time
        
        # Rotasyon açısı ve hızını güncelle
        psi_s, psi_dot_s = rotation_controller.update(dt, t_now)
        
        # Güncel drone pozisyonları
        current_positions = []
        for vehicle in connections:
            lat = vehicle.location.global_relative_frame.lat
            lon = vehicle.location.global_relative_frame.lon
            d_north = (lat - ref_lat) * 111000
            d_east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
            current_positions.append(np.array([d_north, d_east]))
        
        # Her drone için hız komutunu hesapla (sabit formation_deltas kullan)
        for i, vehicle in enumerate(connections):
            velocity = calculate_formation_velocity(
                i, current_positions, psi_s, psi_dot_s, formation_deltas, k_p=0.3, k_R=0.05
            )
            
            vx = velocity[0]
            vy = velocity[1]
            send_velocity(vehicle, vx, vy, 0)
        
        time.sleep(dt)
    
    # Rotasyon bitince durdur
    for vehicle in connections:
        send_velocity(vehicle, 0, 0, 0)
    
    # Formasyonun yeni yönünü güncelle
    current_formation_angle = normalize_angle(final_target_angle)
    
    print(f"✅ Rotasyon tamamlandı.")
    print(f"   Yeni formasyon açısı: {math.degrees(current_formation_angle):.1f}°")

# ----------------- Sürü Hareketi Fonksiyonu -----------------

def move_swarm_to_target(vehicles, target_lat, target_lon, target_alt, kp=0.5, kd=0.3, tolerance=2.0, timeout=120):
    """
    Sürü formasyonu koruyarak hedef noktaya hareket eder
    Formül: u'ᵢ = kₜₚ(p* - p̄) + kₜₐ(- ṗ̄)
    """
    print(f"🎯 Sürü hedef noktaya hareket ediyor: ({target_lat:.8f}, {target_lon:.8f}, {target_alt}m)")
    
    start_time = time.time()
    
    while True:
        # Mevcut sürü merkezi ve hızını hesapla
        current_center = get_swarm_center(vehicles)
        current_velocity = get_swarm_velocity(vehicles)
        
        if not current_center or not current_velocity:
            print("❌ Sürü merkezi hesaplanamadı!")
            break
        
        # Hedef ile mevcut merkez arasındaki mesafe (metre cinsinden)
        north_error, east_error = lat_lon_to_meters(
            current_center[0], current_center[1], 
            target_lat, target_lon
        )
        alt_error = target_alt - current_center[2]
        
        # Toplam mesafe kontrolü
        horizontal_distance = math.sqrt(north_error**2 + east_error**2)
        
        print(f"📍 Merkez: ({current_center[0]:.8f}, {current_center[1]:.8f}, {current_center[2]:.2f})")
        print(f"📏 Mesafe: {horizontal_distance:.2f}m, Yükseklik farkı: {alt_error:.2f}m")
        print(f"🏃 Sürü hızı: ({current_velocity[0]:.2f}, {current_velocity[1]:.2f}, {current_velocity[2]:.2f}) m/s")
        
        # Hedefe ulaştı mı?
        if horizontal_distance <= tolerance and abs(alt_error) <= tolerance:
            print("✅ Sürü hedefe ulaştı!")
            # Tüm droneları durdur
            for vehicle in vehicles:
                send_velocity(vehicle, 0, 0, 0)
            break
        
        # Timeout kontrolü
        if time.time() - start_time > timeout:
            print("⏰ Timeout! Sürü durdurluyor...")
            for vehicle in vehicles:
                send_velocity(vehicle, 0, 0, 0)
            break
        
        # Kontrol formülü: u'ᵢ = kₜₚ(p* - p̄) + kₜₐ(- ṗ̄)
        # Pozisyon terimi: kₜₚ(p* - p̄)
        vx_position = kp * north_error
        vy_position = kp * east_error
        vz_position = kp * alt_error
        
        # Hız sönümleme terimi: kₜₐ(- ṗ̄)
        vx_damping = -kd * current_velocity[0]
        vy_damping = -kd * current_velocity[1]
        vz_damping = -kd * current_velocity[2]
        
        # Toplam kontrol sinyali
        control_vx = vx_position + vx_damping
        control_vy = vy_position + vy_damping
        control_vz = vz_position + vz_damping
        
        # Hız sınırlaması
        max_speed = 5.0  # m/s
        speed_magnitude = math.sqrt(control_vx**2 + control_vy**2 + control_vz**2)
        if speed_magnitude > max_speed:
            scale = max_speed / speed_magnitude
            control_vx *= scale
            control_vy *= scale
            control_vz *= scale
        
        print(f"🎮 Kontrol: Pos=({vx_position:.2f}, {vy_position:.2f}, {vz_position:.2f}), " +
              f"Damp=({vx_damping:.2f}, {vy_damping:.2f}, {vz_damping:.2f})")
        print(f"🚀 Toplam hız: ({control_vx:.2f}, {control_vy:.2f}, {control_vz:.2f}) m/s")
        
        # Tüm droneları aynı hızla hareket ettir
        for i, vehicle in enumerate(vehicles):
            send_velocity(vehicle, control_vx, control_vy, control_vz)
        
        print("-" * 60)
        time.sleep(0.5)  # Kontrol döngüsü frekansı

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
    time.sleep(10)
    
    # Başlangıç formasyonu göster
    print("🔍 Başlangıç formasyonu:")
    for i, conn in enumerate(connections):
        pos = get_position(conn)
        print(f"Drone {i+1}: ({pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.2f})")
    
    start_center = get_swarm_center(connections)
    print(f"📍 Başlangıç merkezi: ({start_center[0]:.8f}, {start_center[1]:.8f}, {start_center[2]:.2f})")
    
    target_waypoints = [
        (-35.36319568, 149.16515342, 10),
        (-35.3631253, 149.16538707, 10),
        (-35.36304971, 149.16519280, 10),
    ]

    for(target_lat, target_lon, target_alt) in target_waypoints:
        #Sürüyü hedefe döndür (formasyonu koruyarak)
        rotate_formation_to_target(connections, target_lat, target_lon, t_mi=5.0)
        time.sleep(7)
        
        # Sürüyü hedefe gönder (formasyonu koruyarak)
        move_swarm_to_target(connections, target_lat, target_lon, target_alt)
        
        print("✅ Hedefe ulaşıldı!")
        time.sleep(15)
    
    print("✅ Sürü hareketi tamamlandı!")
    time.sleep(1)

    land_all(connections)
