import time
import math
import threading
import numpy as np
from pymavlink import mavutil
from scipy.optimize import linear_sum_assignment

#------------------------Drone bağlantı ayarları------------------------

NUM_DRONES = 3  # Drone sayısı

# Drone bağlantı noktaları
BASE_PORT = 14550  # Başlangıç port numarası

class MavlinkVehicle:
    """Mavlink drone wrapper sınıfı - Mavlink ile drone durumunu takip eder"""
    def __init__(self, connection, drone_id):
        self.mav = connection
        self.drone_id = drone_id
        self.target_system = 1  # Default target system ID
        self.target_component = 1  # Default component ID
        
        # Son durum bilgileri
        self.lat = 0
        self.lon = 0
        self.alt = 0
        self.relative_alt = 0
        self._armed = False
        self.last_heartbeat = time.time()
        
        # Takip edilen mesajları başlat
        self._start_message_tracking()
    
    def _start_message_tracking(self):
        """Drone mesaj takip thread'i"""
        self._tracking = True
        self._track_thread = threading.Thread(target=self._track_messages)
        self._track_thread.daemon = True
        self._track_thread.start()
    
    def _track_messages(self):
        """Mavlink mesajlarını takip et"""
        while self._tracking:
            try:
                msg = self.mav.recv_match(blocking=True, timeout=1)
                if not msg:
                    continue
                
                msg_type = msg.get_type()
                
                if msg_type == 'HEARTBEAT':
                    self.last_heartbeat = time.time()
                    if hasattr(msg, 'base_mode'):
                        self._armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.alt / 1000.0
                    self.relative_alt = msg.relative_alt / 1000.0
            except Exception as e:
                print(f"Mesaj takip hatası (drone {self.drone_id}): {e}")
                # Kısa bir bekleme ile tekrar dene
                time.sleep(0.1)

    def wait_for_target(self):
        """Hedef sistem ID'sini bekle"""
        print(f"Drone {self.drone_id} için hedef sistem bekleniyor...")
        start_time = time.time()
        while True:
            try:
                msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
                if msg:
                    self.target_system = msg.get_srcSystem()
                    print(f"Drone {self.drone_id} hedef sistem: {self.target_system}")
                    break
                
                # 10 saniye sonra zaman aşımı
                if time.time() - start_time > 10:
                    self.target_system = 1  # Varsayılan değer
                    print(f"Drone {self.drone_id} için hedef sistem bulunamadı, varsayılan: {self.target_system}")
                    break
            except Exception as e:
                print(f"Hedef sistem beklerken hata (drone {self.drone_id}): {e}")
                time.sleep(0.5)

    def motors_armed(self):
        """Motorlar arm durumunu kontrol et"""
        return self._armed
    
    def set_mode(self, mode):
        """Uçuş modunu ayarla"""
        # Mode mapping for common flight modes
        mode_mapping = {
            'STABILIZE': 0,
            'GUIDED': 4,
            'LOITER': 5,
            'RTL': 6,
            'LAND': 9,
            'ALT_HOLD': 2
        }
        
        if mode.upper() in mode_mapping:
            mode_id = mode_mapping[mode.upper()]
        else:
            mode_id = mode  # Doğrudan mode ID gönderilmiş olabilir
        
        self.mav.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,  # Confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
            0, 0, 0, 0, 0  # Not used
        )
        print(f"Drone {self.drone_id} için {mode} modu ayarlandı")
    
    @property
    def location(self):
        """Konum bilgisi"""
        class LocationGlobal:
            def __init__(self, lat, lon, alt, rel_alt):
                self.lat = lat
                self.lon = lon
                self.alt = alt
                self.global_relative_frame = LocationGlobalRelative(lat, lon, rel_alt)
        
        class LocationGlobalRelative:
            def __init__(self, lat, lon, alt):
                self.lat = lat
                self.lon = lon
                self.alt = alt
        
        return LocationGlobal(self.lat, self.lon, self.alt, self.relative_alt)

"""Belirtilen sayıda drone bağlantısı oluşturur"""
connections = []
ports = [f'udpin:127.0.0.1:{BASE_PORT + i*10}' for i in range(NUM_DRONES)]

for i, port in enumerate(ports):
    try:
        mav = mavutil.mavlink_connection(port)
        # Mavlink bağlantısını MavlinkVehicle wrapper'a sar
        vehicle = MavlinkVehicle(mav, i+1)
        vehicle.wait_for_target()  # Hedef sistemin bağlanmasını bekle
        connections.append(vehicle)
        print(f"✓ Bağlandı: Drone {i+1} - {port}")
    except Exception as e:
        print(f"✗ Drone {i+1} bağlantı hatası: {e}")

print(f"Toplam {len(connections)} drone bağlandı.")

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
    
def calculate_repulsion_force(vehicle, safe_distance=2.0, gain=1.5):
    """Çarpışma engelleme kuvvetini hesaplar"""
    vx, vy, vz = 0, 0, 0
    current_pos = get_position(vehicle)
    
    for other in connections:
        if other == vehicle:
            continue
        other_pos = get_position(other)
        
        dx = (current_pos[0] - other_pos[0]) * 111000
        dy = (current_pos[1] - other_pos[1]) * 111000 * math.cos(math.radians(current_pos[0]))
        dz = current_pos[2] - other_pos[2]
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance < safe_distance:
            strength = gain * (safe_distance - distance) / distance
            vx += dx * strength
            vy += dy * strength
            vz += dz * strength
    return vx, vy, vz

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
        
        # Çarpışma engelleme kuvvetini hesapla
        repulsion_vx, repulsion_vy, repulsion_vz = calculate_repulsion_force(vehicle)
        
        # Toplam hız vektörü (hedef + repulsion)
        total_vx = vx + repulsion_vx
        total_vy = vy + repulsion_vy
        total_vz = vz + repulsion_vz
        
        # Kalan mesafeye göre hız limiti uygula
        max_speed = 3.0  # m/s
        min_speed = 0.3  # m/s - minimum hız
        slow_down_distance = 3.0  # metre - yavaşlama başlangıç mesafesi
        
        # Mesafeye göre hız faktörü hesapla
        if distance <= slow_down_distance:
            # Mesafe azaldıkça hızı doğrusal olarak azalt
            speed_factor = max(min_speed/max_speed, distance / slow_down_distance)
        else:
            speed_factor = 1.0
        
        # Uyarlanmış maksimum hız
        adapted_max_speed = max_speed * speed_factor
        
        # Mevcut hızı kontrol et ve gerekirse kısıtla
        current_speed = math.sqrt(total_vx**2 + total_vy**2 + total_vz**2)
        if current_speed > adapted_max_speed:
            scale = adapted_max_speed / current_speed
            total_vx *= scale
            total_vy *= scale
            total_vz *= scale
        
        # Debug log
        print(f"🚁 Drone {get_drone_id(vehicle)} -> TargetVel: ({vx:.2f}, {vy:.2f}, {vz:.2f}) Repulsion: ({repulsion_vx:.2f}, {repulsion_vy:.2f}, {repulsion_vz:.2f}) Total: ({total_vx:.2f}, {total_vy:.2f}, {total_vz:.2f})")
        
        send_velocity(vehicle, total_vx, total_vy, total_vz)
        time.sleep(0.1)

def send_velocity(vehicle, vx, vy, vz):
    """Hız komutu gönder"""
    try:
        # Doğru şekilde velocity komutu gönder (_encode yerine _send)
        vehicle.mav.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # type_mask (only velocities enabled)
            0, 0, 0,  # position
            vx, vy, vz,  # velocity
            0, 0, 0,  # acceleration
            0, 0)  # yaw, yaw_rate
    except Exception as e:
        print(f"Hız komutu gönderme hatası (drone {vehicle.drone_id}): {e}")

def send_goto_with_pid(vehicle, lat, lon, alt):
    """PID ile drone'u belirtilen koordinatlara gönderir (Mavlink uyumlu)"""
    
    # LocationGlobalRelative yerine basit bir sınıf oluştur
    class SimpleLocation:
        def __init__(self, lat, lon, alt):
            self.lat = lat
            self.lon = lon
            self.alt = alt
    
    target = SimpleLocation(lat, lon, alt)
    go(vehicle, target)

def get_location(vehicle):
    """Mavlink üzerinden drone'un mevcut konumunu alır"""
    vehicle.wait_heartbeat()
    msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0
        return (lat, lon, alt)
    return None

# ----------------- Genel Fonksiyonlar -----------------

def get_drone_id(vehicle):
    """Drone'un ID'sini döndürür"""
    return vehicle.drone_id

def arm(vehicle):
    """Drone'u arm eder (Mavlink uyumlu)"""
    drone_id = get_drone_id(vehicle)
    
    # GUIDED moda geçiş
    vehicle.set_mode("GUIDED")
    print(f"⏳ Drone {drone_id} GUIDED moda geçiş yapılıyor...")
    time.sleep(1)
    
    # Arm komutu gönder
    vehicle.mav.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,  # Arm (1: Arm, 0: Disarm)
        0, 0, 0, 0, 0, 0
    )
    
    # Arm durumunu kontrol et
    start_time = time.time()
    while not vehicle.motors_armed():
        print(f"⏳ Drone {drone_id} arm edilmeyi bekliyor...")
        time.sleep(1)
        
        # 10 saniye sonra tekrar arm komutu gönder
        if time.time() - start_time > 10:
            print(f"Drone {drone_id} arm komutu tekrarlanıyor...")
            vehicle.mav.mav.command_long_send(
                vehicle.target_system,
                vehicle.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,  # Arm (1: Arm, 0: Disarm)
                0, 0, 0, 0, 0, 0
            )
            start_time = time.time()
    
    print(f"✅ Drone {drone_id} arm edildi")

def takeoff(vehicle, alt):
    """Drone'u belirtilen yüksekliğe kaldırır (Mavlink uyumlu)"""
    drone_id = get_drone_id(vehicle)
    
    # Kalkış komutu gönder
    vehicle.mav.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,  # Confirmation
        0, 0, 0, 0, 0, 0, alt  # Parameters (altitude is the last parameter)
    )
    
    # Kalkış durumunu kontrol et
    start_time = time.time()
    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f"🚁 Drone {drone_id} yükseklik: {current_alt:.2f}m / {alt}m")
        
        if current_alt >= alt * 0.95:
            break
            
        # 30 saniye sonra zaman aşımı kontrolü
        if time.time() - start_time > 30:
            print(f"⚠️ Drone {drone_id} kalkış zaman aşımı!")
            # Tekrar kalkış komutu gönder
            vehicle.mav.mav.command_long_send(
                vehicle.target_system,
                vehicle.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,  # Confirmation
                0, 0, 0, 0, 0, 0, alt  # Parameters (altitude is the last parameter)
            )
            start_time = time.time()
            
        time.sleep(1)
    
    print(f"✅ Drone {drone_id} hedef yüksekliğe ulaştı: {alt}m")

def send_goto(vehicle, lat, lon, alt):
    """Drone'u belirtilen koordinatlara gönderir (Mavlink uyumlu)"""
    vehicle.mav.mav.mission_item_send(
        vehicle.target_system,
        vehicle.target_component,
        0, # seq
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        2, # current - 2 is guided mode waypoint
        0, # autocontinue
        0, # param1
        0, # param2
        0, # param3
        0, # param4
        lat, # latitude
        lon, # longitude
        alt) # altitude

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

def formation_v(spacing_meter, num_drones, alt, donus_acisi_derece=0):
    center = get_swarm_center(connections)
    
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
        target = target_positions[i]
        print(f"Drone {i+1} hedefi: {target}")
        t = threading.Thread(target=send_goto_with_pid, args=(conn, *target))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def formation_line(spacing_meter, num_drones, alt, donus_acisi_derece=0):
    center = get_swarm_center(connections)
    
    # Line formasyonu için pozisyonları hesapla (metre cinsinden)
    relative_positions_meter = []
    
    # Line formasyonunda dronelar bir çizgi üzerinde eşit aralıklarla yerleşir
    # Merkezi referans alarak simetrik dağılım
    
    # Toplam uzunluk = (drone_sayısı - 1) * aralık
    total_length = (num_drones - 1) * spacing_meter
    start_offset = -total_length / 2  # Başlangıç noktası (merkeze göre)
    
    for i in range(num_drones):
        # X ekseni boyunca eşit aralıklarla yerleştir
        x_offset = start_offset + (i * spacing_meter)
        y_offset = 0  # Line formasyonunda Y ekseni sıfır
        
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
    print("=== Line Formasyonu Pozisyonları ===")
    print(f"Toplam uzunluk: {total_length:.2f}m")
    print(f"Dronelar arası mesafe: {spacing_meter:.2f}m")
    for i, (lat, lon, altitude) in enumerate(target_positions):
        x_rel = rotated_positions_meter[i][0]
        y_rel = rotated_positions_meter[i][1]
        print(f"Drone {i+1}: X={x_rel:.2f}m, Y={y_rel:.2f}m (Lat={lat:.8f}, Lon={lon:.8f})")
    
    # Mesafeleri kontrol et
    print("=== Pozisyonlar Arası Mesafeler ===")
    for i in range(len(target_positions) - 1):
        # Sadece komşu dronelar arasındaki mesafeyi kontrol et
        j = i + 1
        dist = calculate_distance(target_positions[i][0], target_positions[i][1],
                                target_positions[j][0], target_positions[j][1])
        expected_distance = spacing_meter
        print(f"Drone {i+1} - Drone {j+1}: {dist:.2f}m (Beklenen: {expected_distance:.2f}m)")
    
    # Line formasyonunda simetri kontrolü
    print("=== Line Formasyonu Kontrolü ===")
    center_index = num_drones // 2
    if num_drones % 2 == 1:  # Tek sayıda drone
        print(f"Merkez drone: Drone {center_index + 1} (X={rotated_positions_meter[center_index][0]:.2f}m)")
        # Simetrik çiftleri kontrol et
        for i in range(1, center_index + 1):
            left_pos = rotated_positions_meter[center_index - i][0]
            right_pos = rotated_positions_meter[center_index + i][0]
            print(f"Simetrik çift {i}: Sol={left_pos:.2f}m, Sağ={right_pos:.2f}m")
    else:  # Çift sayıda drone
        print("Çift sayıda drone - merkez iki drone arasında")
        # Merkez noktadan eşit uzaklıktaki çiftleri kontrol et
        for i in range(center_index):
            left_pos = rotated_positions_meter[i][0]
            right_pos = rotated_positions_meter[num_drones - 1 - i][0]
            print(f"Simetrik çift {i+1}: Sol={left_pos:.2f}m, Sağ={right_pos:.2f}m")
    
    # Hungarian algoritması ile optimum atama
    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    
    print("=== Drone Atamaları ===")
    total_movement_distance = 0
    for i, conn in enumerate(connections):
        target = target_positions[col_ind[i]]
        current_pos = get_position(conn)
        distance_to_target = calculate_distance(current_pos[0], current_pos[1], target[0], target[1])
        total_movement_distance += distance_to_target
        print(f"Drone {i+1} -> Hedef Pozisyon {col_ind[i]+1}: {distance_to_target:.2f}m mesafe")
    print(f"Toplam hareket mesafesi: {total_movement_distance:.2f}m")
    
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
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,  # Confirmation
            0, 0, 0, 0, 0, 0, 0  # Parameters (unused for LAND command)
        )
            
def formation_reverse_v(spacing_meter, num_drones, alt, donus_acisi_derece=0):
    center = get_swarm_center(connections)
    
    # Ters V formasyonu için pozisyonları hesapla (metre cinsinden)
    relative_positions_meter = []
    
    # Lider drone'un pozisyonunu hesapla (V'nin arkasında)
    cos_30 = math.cos(math.radians(30))  # ≈ 0.866
    leader_offset_x = -(2/3) * ((num_drones // 2) * spacing_meter * cos_30)
    
    # İlk drone (lider) V'nin arkasına
    relative_positions_meter.append((leader_offset_x, 0))
    
    # Diğer dronelar ters V şeklinde simetrik yerleştirilir
    for i in range(1, num_drones):
        # V formasyonundaki sıra numarası (1, 1, 2, 2, 3, 3, ...)
        row = (i + 1) // 2
        
        # Hangi taraf: tek indexler sol (-), çift indexler sağ (+)
        if i % 2 == 1:  # 1, 3, 5, ... -> sol taraf
            side = -1
        else:  # 2, 4, 6, ... -> sağ taraf
            side = 1
        
        # Ters V formasyonu koordinatları
        # X: ileri doğru (30 derece açı ile)
        x_offset = leader_offset_x + (row * spacing_meter * cos_30)
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
    print("=== Ters V Formasyonu Pozisyonları ===")
    print(f"Lider drone offset: {leader_offset_x:.2f}m arkada")
    for i, (lat, lon, altitude) in enumerate(target_positions):
        x_rel = rotated_positions_meter[i][0]
        y_rel = rotated_positions_meter[i][1]
        print(f"Drone {i+1}: X={x_rel:.2f}m, Y={y_rel:.2f}m (Lat={lat:.8f}, Lon={lon:.8f})")
    
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

# ----------------- Main -----------------

if __name__ == "__main__":
    # Sistemde en az 1 drone varsa devam et
    if len(connections) == 0:
        print("Hiçbir drone bağlanamadı! Program sonlandırılıyor.")
        exit(1)
    
    # Arm işlemini paralel yap
    print("Droneları arm etme işlemi başlatılıyor...")
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
    print("Kalkış işlemi başlatılıyor...")
    takeoff_threads = []
    for conn in connections:
        t = threading.Thread(target=takeoff, args=(conn, 10))
        takeoff_threads.append(t)
        t.start()
    for t in takeoff_threads:
        t.join()
    print("Kalkış komutları gönderildi.")
    time.sleep(10)
    
    # Formasyonları daha güvenli şekilde uygula
    try:
        print("V formasyonu uygulanıyor...")
        formation_v(5, len(connections), 15, 0)
        time.sleep(6)
        
        print("Line formasyonu uygulanıyor...")
        formation_line(10, len(connections), 15, 0)
        time.sleep(6)
        
        print("Ters V formasyonu uygulanıyor...")
        formation_reverse_v(10, len(connections), 15, 0)
        time.sleep(6)
        
        print("Tüm formasyonlar başarıyla tamamlandı.")
        print("Son formasyon: Line")
        formation_line(10, len(connections), 15, 0)
        time.sleep(15)
    except Exception as e:
        print(f"Formasyon sırasında hata: {e}")
    finally:
        print("İniş işlemi başlatılıyor...")
        land_all(connections)
        print("Program tamamlandı.")

