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
    
    # Hedef nokta (mevcut merkezden kuzeye 50 metre)
    target_lat = start_center[0] + (-50 / 111000)  # 50 metre kuzey
    target_lon = start_center[1] + (-30 / 111000)  # Boylam aynı
    target_alt = 10  # 10 metre yükseklik
    
    print(f"🎯 Hedef nokta: ({target_lat:.8f}, {target_lon:.8f}, {target_alt}m)")
    
    # Sürüyü hedefe gönder (formasyonu koruyarak)
    move_swarm_to_target(connections, target_lat, target_lon, target_alt)
    
    print("✅ Sürü hareketi tamamlandı!")
    time.sleep(5)
    
    # Final pozisyonları göster
    print("🔍 Final formasyonu:")
    for i, conn in enumerate(connections):
        pos = get_position(conn)
        print(f"Drone {i+1}: ({pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.2f})")
    
    final_center = get_swarm_center(connections)
    print(f"📍 Final merkezi: ({final_center[0]:.8f}, {final_center[1]:.8f}, {final_center[2]:.2f})")
    
    # Hedefe olan mesafeyi kontrol et
    distance_to_target = calculate_distance(final_center[0], final_center[1], target_lat, target_lon)
    print(f"📏 Hedefe olan mesafe: {distance_to_target:.2f}m")
    
    time.sleep(30)
    land_all(connections)