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

def send_velocity(vehicle, vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

def get_drone_id(vehicle):
    try:
        return connections.index(vehicle) + 1
    except ValueError:
        return "Bilinmeyen"

def arm(vehicle):
    drone_id = get_drone_id(vehicle)
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    vehicle.armed = True
    while not vehicle.armed:
        print(f"⏳ Drone {drone_id} arm edilmeyi bekliyor...")
        time.sleep(1)
    print(f"✅ Drone {drone_id} arm edildi")

def takeoff(vehicle, alt):
    drone_id = get_drone_id(vehicle)
    vehicle.simple_takeoff(alt)
    while vehicle.location.global_relative_frame.alt < alt * 0.95:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f"🚁 Drone {drone_id} yükseklik: {current_alt:.2f}m / {alt}m")
        time.sleep(1)
    print(f"✅ Drone {drone_id} hedef yüksekliğe ulaştı: {alt}m")

# ------------------ ROTASYON KONTROL SİSTEMİ ------------------

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

def rotate_formation_to_target(connections, target_lat, target_lon, t_mi=10.0):
    """Formasyonu hedef noktaya yönlendiren rotasyon"""
    
    # Referans nokta (ilk drone'un konumu)
    ref_lat = connections[0].location.global_relative_frame.lat
    ref_lon = connections[0].location.global_relative_frame.lon
    
    # Başlangıç formasyonu
    initial_deltas = get_initial_formation(connections, ref_lat, ref_lon)
    
    # Hedef açıyı hesapla
    target_angle = calculate_target_angle(ref_lat, ref_lon, target_lat, target_lon)
    
    # Rotasyon kontrolörü (başlangıç: kuzey = 0, hedef: hesaplanan açı)
    rotation_controller = RotationController(0.0, target_angle, 0.0, t_mi)
    
    print(f"🔄 Rotasyon başlıyor: {math.degrees(target_angle):.1f}° ({t_mi}s)")
    
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
        
        # Her drone için hız komutunu hesapla
        for i, vehicle in enumerate(connections):
            velocity = calculate_formation_velocity(
                i, current_positions, psi_s, psi_dot_s, initial_deltas, k_p=0.3, k_R=0.05
            )
            
            vx = velocity[0]
            vy = velocity[1]
            send_velocity(vehicle, vx, vy, 0)
        
        time.sleep(dt)
    
    # Rotasyon bitince durdur
    for vehicle in connections:
        send_velocity(vehicle, 0, 0, 0)
    
    print(f"✅ Rotasyon tamamlandı. Formasyon {math.degrees(target_angle):.1f}° açıya yönlendi.")

def main():
    # Drone'ları arm et
    threads = []
    for conn in connections:
        t = threading.Thread(target=arm, args=(conn,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    time.sleep(2)
    
    # Drone'ları havaya kaldır
    threads = []
    for conn in connections:
        t = threading.Thread(target=takeoff, args=(conn, 10))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("🚁 Tüm dronelar havada, rotasyon için hazır!")
    
    # Hedef koordinatları (örnek)
    target_lat = -35.36368280
    target_lon = 149.16599350
    
    # Rotasyon işlemi
    rotate_formation_to_target(connections, target_lat, target_lon, t_mi=5.0)

if __name__ == "__main__":
    main()