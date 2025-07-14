from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from pymavlink import mavutil

NUM_DRONES = 3  # Drone sayısı
BASE_PORT = 14550  # Başlangıç port numarası

# Drone bağlantıları
connections = []
ports = [f'udp:127.0.0.1:{BASE_PORT + i*10}' for i in range(NUM_DRONES)]

# Global formasyon yön değişkeni
current_formation_angle = 0.0  # Formasyonun mevcut bakış yönü
initial_formation_deltas = None  # Başlangıç formasyon delta değerleri

# Formation control thread kontrolü
formation_control_active = False
formation_control_thread = None

# Çift integratör kontrol parametreleri
KP = 0.5  # Pozisyon kazancı
KD = 0.3  # Hız kazancı
MAX_VELOCITY = 3.0  # Maksimum hız limiti (m/s)

# Her drone için son formation control hızları
formation_velocities = {}

for i, port in enumerate(ports):
    try:
        vehicle = connect(port, wait_ready=True)
        connections.append(vehicle)
        print(f"✓ Bağlandı: Drone {i+1} - {port}")
    except Exception as e:
        print(f"✗ Drone {i+1} bağlantı hatası: {e}")

print(f"Toplam {len(connections)} drone bağlandı.")

# PID sınıfı
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

def send_combined_velocity(vehicle, nav_vx=0, nav_vy=0, nav_vz=0, form_vx=0, form_vy=0, form_vz=0):
    """Navigasyon ve formasyon hızlarını birleştirerek gönder"""
    # Toplam hız
    total_vx = nav_vx + form_vx
    total_vy = nav_vy + form_vy
    total_vz = nav_vz + form_vz
    
    # Hız sınırlaması
    total_speed = math.sqrt(total_vx**2 + total_vy**2 + total_vz**2)
    if total_speed > MAX_VELOCITY:
        scale = MAX_VELOCITY / total_speed
        total_vx *= scale
        total_vy *= scale
        total_vz *= scale
    
    send_velocity(vehicle, total_vx, total_vy, total_vz)

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

# ----------------- Çift İntegratör Formasyon Kontrolü -----------------

def get_relative_positions_3d(vehicles):
    """Tüm droneların birbirlerine göre 3D konumlarını hesaplar"""
    N = len(vehicles)
    positions = []
    
    # Her drone'un konumunu al
    for vehicle in vehicles:
        lat = vehicle.location.global_relative_frame.lat
        lon = vehicle.location.global_relative_frame.lon
        alt = vehicle.location.global_relative_frame.alt
        positions.append([lat, lon, alt])
    
    # Birinci drone'u referans al
    ref_lat, ref_lon, ref_alt = positions[0]
    
    # Göreceli konumları hesapla (metre cinsinden)
    relative_positions = []
    for lat, lon, alt in positions:
        north = (lat - ref_lat) * 111000
        east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
        up = alt - ref_alt
        relative_positions.append([north, east, up])
    
    return relative_positions

def get_relative_velocities_3d(vehicles):
    """Tüm droneların birbirlerine göre 3D hızlarını hesaplar"""
    N = len(vehicles)
    velocities = []
    
    # Her drone'un hızını al
    for vehicle in vehicles:
        vx, vy, vz = get_velocity(vehicle)
        velocities.append([vx, vy, vz])
    
    return velocities

def calculate_initial_formation_deltas(vehicles):
    """Başlangıç formasyonu delta değerlerini hesaplar (δ_ji = p_j - p_i)"""
    positions = get_relative_positions_3d(vehicles)
    N = len(vehicles)
    
    # Delta matrisini hesapla
    deltas = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                delta = [
                    positions[j][0] - positions[i][0],  # North
                    positions[j][1] - positions[i][1],  # East
                    positions[j][2] - positions[i][2]   # Up
                ]
                deltas[(j, i)] = delta
    
    return deltas

def double_integrator_control(i, positions, velocities, formation_deltas, kp=KP, kd=KD):
    """
    Çift integratör formasyon kontrol formülü:
    u_i^f = k_p * Σ(p_j - p_i - δ_ji) + k_d * Σ(v_j - v_i)
    """
    N = len(positions)
    
    # Pozisyon hatası terimi: k_p * Σ(p_j - p_i - δ_ji)
    position_term = [0.0, 0.0, 0.0]
    
    # Hız farkı terimi: k_d * Σ(v_j - v_i)
    velocity_term = [0.0, 0.0, 0.0]
    
    for j in range(N):
        if j != i:
            # Pozisyon hatası: (p_j - p_i) - δ_ji
            current_relative_pos = [
                positions[j][0] - positions[i][0],
                positions[j][1] - positions[i][1],
                positions[j][2] - positions[i][2]
            ]
            
            desired_relative_pos = formation_deltas.get((j, i), [0.0, 0.0, 0.0])
            
            position_error = [
                current_relative_pos[0] - desired_relative_pos[0],
                current_relative_pos[1] - desired_relative_pos[1],
                current_relative_pos[2] - desired_relative_pos[2]
            ]
            
            # Pozisyon terimi toplamı
            position_term[0] += position_error[0]
            position_term[1] += position_error[1]
            position_term[2] += position_error[2]
            
            # Hız farkı: v_j - v_i
            velocity_diff = [
                velocities[j][0] - velocities[i][0],
                velocities[j][1] - velocities[i][1],
                velocities[j][2] - velocities[i][2]
            ]
            
            # Hız terimi toplamı
            velocity_term[0] += velocity_diff[0]
            velocity_term[1] += velocity_diff[1]
            velocity_term[2] += velocity_diff[2]
    
    # Toplam kontrol sinyali
    control_signal = [
        kp * position_term[0] + kd * velocity_term[0],
        kp * position_term[1] + kd * velocity_term[1],
        kp * position_term[2] + kd * velocity_term[2]
    ]
    
    return control_signal

def formation_control_loop():
    """Sürekli çalışan formasyon kontrol döngüsü"""
    global formation_control_active, initial_formation_deltas, formation_velocities
    
    print("🔄 Formasyon kontrol döngüsü başlatıldı")
    
    # Başlangıç formasyonu delta değerlerini hesapla
    if initial_formation_deltas is None:
        initial_formation_deltas = calculate_initial_formation_deltas(connections)
        print("📐 Başlangıç formasyon delta değerleri:")
        for (j, i), delta in initial_formation_deltas.items():
            print(f"   δ_{j}{i} = [{delta[0]:.2f}, {delta[1]:.2f}, {delta[2]:.2f}]")
    
    # Formation velocities başlangıç
    for i, vehicle in enumerate(connections):
        drone_id = get_drone_id(vehicle)
        formation_velocities[drone_id] = (0, 0, 0)
    
    while formation_control_active:
        try:
            # Mevcut pozisyon ve hızları al
            positions = get_relative_positions_3d(connections)
            velocities = get_relative_velocities_3d(connections)
            
            # Her drone için kontrol sinyali hesapla
            for i, vehicle in enumerate(connections):
                drone_id = get_drone_id(vehicle)
                
                control_signal = double_integrator_control(
                    i, positions, velocities, initial_formation_deltas, KP, KD
                )
                
                # Hız sınırlaması
                speed = math.sqrt(control_signal[0]**2 + control_signal[1]**2 + control_signal[2]**2)
                if speed > MAX_VELOCITY:
                    scale = MAX_VELOCITY / speed
                    control_signal[0] *= scale
                    control_signal[1] *= scale
                    control_signal[2] *= scale
                
                # Formasyon hızını kaydet (navigasyon hızı ile birleştirilecek)
                formation_velocities[drone_id] = (control_signal[0], control_signal[1], control_signal[2])
                
                # Eğer navigasyon aktif değilse, sadece formasyon hızı gönder
                send_combined_velocity(vehicle, 0, 0, 0, control_signal[0], control_signal[1], control_signal[2])
            
            time.sleep(0.1)  # 10Hz kontrol frekansı
            
        except Exception as e:
            print(f"❌ Formasyon kontrol hatası: {e}")
            time.sleep(0.5)
    
    print("🔄 Formasyon kontrol döngüsü durduruldu")

def start_formation_control():
    """Formasyon kontrol thread'ini başlat"""
    global formation_control_active, formation_control_thread
    
    if not formation_control_active:
        formation_control_active = True
        formation_control_thread = threading.Thread(target=formation_control_loop)
        formation_control_thread.daemon = True
        formation_control_thread.start()
        print("✅ Formasyon kontrol sistemi aktif edildi")

def stop_formation_control():
    """Formasyon kontrol thread'ini durdur"""
    global formation_control_active, formation_control_thread
    
    if formation_control_active:
        formation_control_active = False
        if formation_control_thread:
            formation_control_thread.join(timeout=2.0)
        print("🛑 Formasyon kontrol sistemi durduruldu")

# ----------------- Sürü Hareketi Fonksiyonu -----------------

def move_swarm_to_target(vehicles, target_lat, target_lon, target_alt, kp=0.5, kd=0.3, tolerance=2.0, timeout=120):
    """
    Sürü formasyonu koruyarak hedef noktaya hareket eder
    Formasyon hızı ile navigasyon hızı birleştirilir
    """
    print(f"🎯 Sürü hedef noktaya hareket ediyor: ({target_lat:.8f}, {target_lon:.8f}, {target_alt}m)")
    
    # Formasyon kontrolü aktif kalır
    if not formation_control_active:
        start_formation_control()
    
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
        
        # Hedefe ulaştı mı?
        if horizontal_distance <= tolerance and abs(alt_error) <= tolerance:
            print("✅ Sürü hedefe ulaştı!")
            # Sadece formasyon hızı ile hareket et
            for vehicle in vehicles:
                drone_id = get_drone_id(vehicle)
                form_vel = formation_velocities.get(drone_id, (0, 0, 0))
                send_combined_velocity(vehicle, 0, 0, 0, form_vel[0], form_vel[1], form_vel[2])
            break
        
        # Timeout kontrolü
        if time.time() - start_time > timeout:
            print("⏰ Timeout! Sürü durdurluyor...")
            for vehicle in vehicles:
                drone_id = get_drone_id(vehicle)
                form_vel = formation_velocities.get(drone_id, (0, 0, 0))
                send_combined_velocity(vehicle, 0, 0, 0, form_vel[0], form_vel[1], form_vel[2])
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
        
        # Toplam navigasyon kontrol sinyali
        nav_vx = vx_position + vx_damping
        nav_vy = vy_position + vy_damping
        nav_vz = vz_position + vz_damping
        
        # Navigasyon hız sınırlaması
        nav_speed = math.sqrt(nav_vx**2 + nav_vy**2 + nav_vz**2)
        max_nav_speed = 3.0  # m/s
        if nav_speed > max_nav_speed:
            scale = max_nav_speed / nav_speed
            nav_vx *= scale
            nav_vy *= scale
            nav_vz *= scale
        
        print(f"🎮 Navigasyon hızı: ({nav_vx:.2f}, {nav_vy:.2f}, {nav_vz:.2f}) m/s")
        
        # Tüm droneları navigasyon + formasyon hızı ile hareket ettir
        for i, vehicle in enumerate(vehicles):
            drone_id = get_drone_id(vehicle)
            form_vel = formation_velocities.get(drone_id, (0, 0, 0))
            send_combined_velocity(vehicle, nav_vx, nav_vy, nav_vz, form_vel[0], form_vel[1], form_vel[2])
        
        print("-" * 60)
        time.sleep(0.5)  # Kontrol döngüsü frekansı

# ----------------- Rotasyon (Mevcut kod korundu) -----------------

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

def normalize_angle(angle):
    """Açıyı -π ve π aralığına normalize et"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def calculate_target_angle(current_lat, current_lon, target_lat, target_lon):
    """Hedef noktaya yönlenme açısını hesapla"""
    # Kuzey-doğu koordinatlarında hedef vektörü
    d_north = (target_lat - current_lat) * 111000
    d_east = (target_lon - current_lon) * 111000 * math.cos(math.radians(current_lat))
    
    # Hedef açı (kuzeyden saat yönü)
    target_angle = math.atan2(d_east, d_north)
    return target_angle

def rotate_formation_to_target(connections, target_lat, target_lon, t_mi=10.0):
    """Formasyonu hedef noktaya yönlendiren rotasyon (formasyon kontrolü geçici olarak durdurulur)"""
    global current_formation_angle, initial_formation_deltas
    
    # Formasyon kontrolünü geçici olarak durdur
    was_formation_active = formation_control_active
    if was_formation_active:
        stop_formation_control()
        print("🔄 Rotasyon için formasyon kontrolü geçici olarak durduruldu")
    
    # Referans nokta (ilk drone'un konumu)
    ref_lat = connections[0].location.global_relative_frame.lat
    ref_lon = connections[0].location.global_relative_frame.lon
    
    # Hedef açıyı hesapla
    target_angle = calculate_target_angle(ref_lat, ref_lon, target_lat, target_lon)
    
    # Açı farkını hesapla (en kısa rotasyon yolu)
    angle_diff = normalize_angle(target_angle - current_formation_angle)
    final_target_angle = current_formation_angle + angle_diff
    
    print(f"🔄 Rotasyon başlıyor:")
    print(f"   Mevcut açı: {math.degrees(current_formation_angle):.1f}°")
    print(f"   Hedef açı: {math.degrees(target_angle):.1f}°")
    print(f"   Rotasyon miktarı: {math.degrees(angle_diff):.1f}° ({t_mi}s)")
    
    # Basit rotasyon (tüm dronelar aynı açısal hızla döner)
    angular_velocity = angle_diff / t_mi  # rad/s
    
    start_time = time.time()
    while time.time() - start_time <= t_mi:
        # Her drone için rotasyon hızını hesapla
        for i, vehicle in enumerate(connections):
            # Merkeze göre konumu al
            lat = vehicle.location.global_relative_frame.lat
            lon = vehicle.location.global_relative_frame.lon
            
            d_north = (lat - ref_lat) * 111000
            d_east = (lon - ref_lon) * 111000 * math.cos(math.radians(ref_lat))
            
            # Rotasyon hızı vektörü (v = ω × r)
            rot_vx = -angular_velocity * d_east  # Kuzey yönü
            rot_vy = angular_velocity * d_north   # Doğu yönü
            rot_vz = 0
            
            # Sadece rotasyon hızı gönder (formasyon kontrolü kapalı)
            send_velocity(vehicle, rot_vx, rot_vy, rot_vz)
        
        time.sleep(0.1)
    
    # Rotasyon bitince droneları durdur
    for vehicle in connections:
        send_velocity(vehicle, 0, 0, 0)
    
    # Formasyonun yeni yönünü güncelle
    current_formation_angle = normalize_angle(final_target_angle)
    
    print(f"✅ Rotasyon tamamlandı.")
    print(f"   Yeni formasyon açısı: {math.degrees(current_formation_angle):.1f}°")
    
    # Rotasyon sonrası yeni formasyon delta değerlerini hesapla
    print("🔄 Rotasyon sonrası formasyon delta değerleri yeniden hesaplanıyor...")
    time.sleep(2)  # Droneların durması için bekle
    
    # Yeni formasyon delta değerlerini hesapla
    initial_formation_deltas = calculate_initial_formation_deltas(connections)
    print("📐 Yeni formasyon delta değerleri:")
    for (j, i), delta in initial_formation_deltas.items():
        print(f"   δ_{j}{i} = [{delta[0]:.2f}, {delta[1]:.2f}, {delta[2]:.2f}]")
    
    # Formasyon kontrolünü tekrar başlat
    if was_formation_active:
        start_formation_control()
        print("✅ Formasyon kontrolü tekrar aktif edildi")
        time.sleep(2)  # Formasyon kontrolünün stabilize olması için bekle

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
    
    # Formasyon kontrol sistemini başlat
    start_formation_control()
    
    # Formasyon stabilizasyonu için bekle
    print("⏳ Formasyon stabilizasyonu için 10 saniye bekleniyor...")
    time.sleep(5)
    
    target_waypoints = [
        (-35.36319568, 149.16515342, 10),
        (-35.3631253, 149.16538707, 10),
        (-35.36304971, 149.16519280, 10),
    ]

    for target_lat, target_lon, target_alt in target_waypoints:
        # Sürüyü hedefe döndür (formasyonu koruyarak)
        rotate_formation_to_target(connections, target_lat, target_lon, t_mi=5.0)
        time.sleep(2)
        
        # Sürüyü hedefe gönder (formasyonu koruyarak)
        move_swarm_to_target(connections, target_lat, target_lon, target_alt)
        
        print("✅ Hedefe ulaşıldı!")
        time.sleep(5)
    
    print("✅ Sürü hareketi tamamlandı!")
    
    # Formasyon kontrolünü durdur
    stop_formation_control()
    
    # Tüm droneları durdur
    for vehicle in connections:
        send_velocity(vehicle, 0, 0, 0)
    
    time.sleep(5)
    land_all(connections)