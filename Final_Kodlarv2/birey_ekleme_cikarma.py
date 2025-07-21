#Direkt BEÇ (final olmayan).py ile aynı kod test yapılacak bi de pid eklenecek yazmışız ama pid sınıfı zaten var anlamadım
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math
import threading
import numpy as np

NUM_DRONES = 4  # Drone sayısı
SAFE_DISTANCE = 2.0  # Güvenli mesafe (metre cinsinden)
BASE_PORT = 14550  # Başlangıç port numarası

collision_avoidance_active = False
collision_avoidance_thread = None

# Çarpışma engelleme hızları - drone index'e göre (0-indexli)
collision_velocities = {}
collision_lock = threading.Lock()

# Drone bağlantıları
connections = []
standby_vehicles = []
ports = [f'udp:127.0.0.1:{BASE_PORT + i*10}' for i in range(NUM_DRONES)]

# Global formasyon yön değişkeni
current_formation_angle = 0.0  # Formasyonun mevcut bakış yönü
initial_formation_deltas = None  # Başlangıç formasyon delta değerleri

for i, port in enumerate(ports):
    try:
        if i != NUM_DRONES - 1:  # Son drone standby olarak bağlanacak
            vehicle = connect(port, wait_ready=True)
            connections.append(vehicle)
            print(f"✓ Bağlandı: Drone {i+1} - {port}")
        else:
            vehicle = connect(port, wait_ready=True)
            standby_vehicles.append(vehicle)
            print(f"✓ Bağlandı: Drone {i+1} - {port}")
    except Exception as e:
        print(f"✗ Drone {i+1} bağlantı hatası: {e}")

print(f"Toplam {NUM_DRONES} drone bağlandı.")

#------------------------Drone PID Ayarları------------------------

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
    """PID + çarpışma engelleme ile drone'u hedefe gönder"""
    
    pid_north = PID(0.4, 0.02, 0.6, 4)
    pid_east  = PID(0.4, 0.02, 0.6, 4)
    pid_down  = PID(0.7, 0.03, 0.2, 3)

    start_time = time.time()
    
    while True:
        curr = vehicle.location.global_relative_frame
        
        # Hedef konuma olan hata
        north_err = (target_location.lat - curr.lat) * 111000
        east_err = (target_location.lon - curr.lon) * 111000 * math.cos(math.radians(curr.lat))
        down_err = -(target_location.alt - curr.alt)
        
        distance = math.sqrt(north_err**2 + east_err**2)
        
        # Hedefe ulaşıldı mı?
        if distance <= tolerance and abs(down_err) <= tolerance:
            send_velocity(vehicle, 0, 0, 0)
            print(f"✅ Drone {get_drone_id(vehicle)} hedefe ulaştı.")
            break
        
        # Timeout kontrolü
        if time.time() - start_time > timeout:
            send_velocity(vehicle, 0, 0, 0)
            print(f"⏳ Drone {get_drone_id(vehicle)} timeout! Hedefe ulaşılamadı.")
            break
        
        # PID çıkışları (hedef kuvveti)
        vx = pid_north.update(north_err)
        vy = pid_east.update(east_err)
        vz = pid_down.update(down_err)
        
        # Kalan mesafeye göre hız limiti uygula
        max_speed = 3.0  # m/s
        min_speed = 0.3  # m/s - minimum hız
        slow_down_distance = 4.0  # metre - yavaşlama başlangıç mesafesi
        
        # Mesafeye göre hız faktörü hesapla
        if distance <= slow_down_distance:
            # Mesafe azaldıkça hızı doğrusal olarak azalt
            speed_factor = max(min_speed/max_speed, distance / slow_down_distance)
        else:
            speed_factor = 1.0
        
        # Uyarlanmış maksimum hız
        adapted_max_speed = max_speed * speed_factor
        
        # Mevcut hızı kontrol et ve gerekirse kısıtla
        current_speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if current_speed > adapted_max_speed:
            scale = adapted_max_speed / current_speed
            vx *= scale
            vy *= scale
            vz *= scale
        
        # Debug log
        print(f"🚁 Drone {get_drone_id(vehicle)} -> TargetVel: ({vx:.2f}, {vy:.2f}, {vz:.2f}) Total: ({vx:.2f}, {vy:.2f}, {vz:.2f})")
        
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

def get_position(vehicle):
    """Drone'un mevcut konumunu döndürür"""
    return (vehicle.location.global_relative_frame.lat, 
            vehicle.location.global_relative_frame.lon, 
            vehicle.location.global_relative_frame.alt)

def get_velocity(vehicle):
    """Drone'un mevcut hızını döndürür (m/s)"""
    return (vehicle.velocity[0], vehicle.velocity[1], vehicle.velocity[2])

def get_swarm_velocity(vehicles):
    """Sürünün ortalama hızını hesaplar"""
    if not vehicles:
        return None
    
    velocities = [get_velocity(vehicle) for vehicle in vehicles]
    avg_vx = sum(v[0] for v in velocities) / len(velocities)
    avg_vy = sum(v[1] for v in velocities) / len(velocities)
    avg_vz = sum(v[2] for v in velocities) / len(velocities)
    
    return (avg_vx, avg_vy, avg_vz)

def get_center(connections):
    positions = [get_position(conn) for conn in connections]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    return (avg_lat, avg_lon, avg_alt)

def get_swarm_center(vehicles):
    """Sürünün merkez koordinatlarını hesaplar"""
    if not vehicles:
        return None
    
    positions = [get_position(vehicle) for vehicle in vehicles]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    
    return (avg_lat, avg_lon, avg_alt)

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

def RTL_drone(index):
    connections[index].mode = VehicleMode("RTL")

def append_drone(index, standby_drone, target_position):
    stop_collision_avoidance()

    old_drone = connections[index]
    connections[index] = standby_drone
    standby_vehicles[0] = old_drone

    start_collision_avoidance(connections, alpha=200.0, beta=0.5, rs=SAFE_DISTANCE)
    arm(standby_drone)
    time.sleep(2)
    takeoff(standby_drone, 10 + (SAFE_DISTANCE * 1.5))
    time.sleep(10)

    lat, lon, alt = target_position
    send_goto_with_pid(standby_drone, lat, lon, alt + (SAFE_DISTANCE * 1.5))
    send_goto_with_pid(standby_drone, lat, lon, alt)

def land_all(connections):
    """Tüm droneları indirir"""
    print("Tüm dronelar indiriliyor...")
    
    for i, vehicle in enumerate(connections):
        print(f"Drone {i+1} indiriliyor...")
        vehicle.mode = VehicleMode("LAND")

# ----------------- Navigasyon -----------------

def move_swarm_to_target(vehicles, target_lat, target_lon, target_alt, kp=0.5, ki=0.02, kd=0.3, tolerance=0.5):
    """
    Sürü formasyonu koruyarak hedef noktaya hareket eder
    PID Formül: u'ᵢ = kₚ(p* - p̄) + kᵢ∫(p* - p̄)dt + kₐ(- ṗ̄)
    """
    print(f"🎯 Sürü hedef noktaya hareket ediyor: ({target_lat:.8f}, {target_lon:.8f}, {target_alt}m)")
    
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
        
        # Integral terimlerini güncelle
        integral_north += north_error * dt
        integral_east += east_error * dt
        
        # Integral windup koruması
        max_integral = 50.0  # maksimum integral değeri
        integral_north = max(-max_integral, min(max_integral, integral_north))
        integral_east = max(-max_integral, min(max_integral, integral_east))

        # Toplam mesafe kontrolü
        horizontal_distance = math.sqrt(north_error**2 + east_error**2)
        
        print(f"📍 Merkez: ({current_center[0]:.8f}, {current_center[1]:.8f}, {current_center[2]:.2f})")
        print(f"🏃 Sürü hızı: ({current_velocity[0]:.2f}, {current_velocity[1]:.2f}, {current_velocity[2]:.2f}) m/s")
        
        # Hedefe ulaştı mı?
        if horizontal_distance <= tolerance:
            print("✅ Sürü hedefe ulaştı!")
            # Tüm droneları durdur
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
        max_speed = 4.0  # m/s
        min_speed = 0.3  # m/s - minimum hız
        slow_down_distance = max_speed * 2  # metre - yavaşlama başlangıç mesafesi

        # Mesafeye göre hız faktörü hesapla
        if horizontal_distance <= slow_down_distance:
            # Mesafe azaldıkça hızı doğrusal olarak azalt
            speed_factor = max(min_speed/max_speed, horizontal_distance / slow_down_distance)
        else:
            speed_factor = 1.0

        # Uyarlanmış maksimum hız
        adapted_max_speed = max_speed * speed_factor

        current_speed = math.sqrt(control_vx**2 + control_vy**2)
        if current_speed > adapted_max_speed:
            scale = adapted_max_speed / current_speed
            control_vx *= scale
            control_vy *= scale
        
        # Tüm droneları aynı hızla hareket ettir
        for i, vehicle in enumerate(vehicles):
            with collision_lock:
                collision_v = collision_velocities.get(i, (0, 0, 0))
            send_velocity(vehicle, control_vx + collision_v[0], control_vy + collision_v[1], 0)
        
        print("-" * 60)
        time.sleep(0.1)  # Kontrol döngüsü frekansı

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
        
        time.sleep(0.1)  

def start_collision_avoidance(connections, alpha=2.0, beta=0.5, rs=1.0):
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
    time.sleep(10)  # Drone'ların yükselme süresi için daha uzun bekleme


    # Waypoint listesi (dairesel örnek rota)
    target_waypoints = [
        (-35.36319568, 149.16515342, 10)
    ]

    start_collision_avoidance(connections, alpha=200.0, beta=0.5, rs=SAFE_DISTANCE)

    for(target_lat, target_lon, target_alt) in target_waypoints:
        #Sürüyü hedefe döndür (formasyonu koruyarak)
        rotate_formation_to_target(connections, target_lat, target_lon, t_mi=5.0)
        time.sleep(7)
        
        # Sürüyü hedefe gönder (formasyonu koruyarak)
        move_swarm_to_target(connections, target_lat, target_lon, target_alt)
        
        print("✅ Hedefe ulaşıldı!")
        time.sleep(12)

    k = int(input("Eve dönecek drone indexi: "))
    RTL_drone_position = get_position(connections[k])
    RTL_drone(k)

    time.sleep(15)

    print("Yeni drone ekleniyor.")
    append_drone(k, standby_vehicles[0], RTL_drone_position)

    time.sleep(15)

    # Waypoint listesi (dairesel örnek rota)
    target_waypoints = [
        (-35.3631253, 149.16538707, 10)
    ]

    for(target_lat, target_lon, target_alt) in target_waypoints:
        #Sürüyü hedefe döndür (formasyonu koruyarak)
        rotate_formation_to_target(connections, target_lat, target_lon, t_mi=5.0)
        time.sleep(7)
        
        # Sürüyü hedefe gönder (formasyonu koruyarak)
        move_swarm_to_target(connections, target_lat, target_lon, target_alt)
        
        print("✅ Hedefe ulaşıldı!")
        time.sleep(12)

    print("Land komutu başlatıldı.")

    land_all(connections)
