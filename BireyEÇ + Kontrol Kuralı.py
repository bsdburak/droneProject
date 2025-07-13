from pymavlink import mavutil
import time
import math
import threading
import numpy as np

# Drone bağlantı noktaları
ports = ['udp:127.0.0.1:14550', 'udp:127.0.0.1:14560', 'udp:127.0.0.1:14570']
connections = [mavutil.mavlink_connection(p) for p in ports]
standby_drone = mavutil.mavlink_connection('udp:127.0.0.1:14580')

# Bağlantı kontrolü
for i, conn in enumerate(connections):
    conn.wait_heartbeat()
    print(f"Bağlandı: Drone {i+1}")
standby_drone.wait_heartbeat()
print(f"standby dronea bağlandı {4}")

# ----------------- Genel Fonksiyonlar -----------------

def arm(conn):
    conn.set_mode_apm("GUIDED")
    time.sleep(1)
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)

def takeoff(conn, alt):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt)

def send_goto(conn, lat, lon, alt):
    conn.mav.set_position_target_global_int_send(
        0, conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b110111111000,
        int(lat * 1e7),
        int(lon * 1e7),
        alt,
        0, 0, 0,
        0, 0, 0,
        0, 0)

def get_position(conn):
    msg = None
    while not msg:
        msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    alt = msg.relative_alt / 1000.0
    return lat, lon, alt

def get_center(connections):
    positions = [get_position(conn) for conn in connections]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    return (avg_lat, avg_lon, avg_alt)

def calculate_heading(from_lat, from_lon, to_lat, to_lon):
    """İki nokta arasındaki yönelim açısını hesaplar (derece cinsinden)"""
    dlon = to_lon - from_lon
    y = math.sin(dlon) * math.cos(to_lat)
    x = math.cos(from_lat) * math.sin(to_lat) - math.sin(from_lat) * math.cos(to_lat) * math.cos(dlon)
    heading = math.atan2(y, x)
    heading = math.degrees(heading)
    heading = (heading + 360) % 360  # 0-360 arası değer
    return heading

def set_heading(conn, heading_deg):
    """Drone'un yönelimini ayarlar"""
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0, heading_deg, 0, 1, 0, 0, 0, 0)
    print(f"Yönelim ayarlandı: {heading_deg:.1f} derece")

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

def is_near_position(current_lat, current_lon, target_lat, target_lon, threshold=5):
    """Drone'un hedef konuma yakın olup olmadığını kontrol eder (metre cinsinden)"""
    distance = calculate_distance(current_lat, current_lon, target_lat, target_lon)
    return distance < threshold

# ----------------- Formasyon Fonksiyonları -----------------

def create_formation(formation_type, center, offset_metre, alt):
    """Belirli bir merkez etrafında formasyon pozisyonları oluşturur"""
    offset = offset_metre / 100000  # GPS koordinatları için ölçekleme
    
    if formation_type == "ucgen":
        return [
            (center[0] + offset, center[1], alt),
            (center[0] - offset * math.cos(math.radians(60)),
             center[1] - offset * math.sin(math.radians(60)), alt),
            (center[0] - offset * math.cos(math.radians(60)),
             center[1] + offset * math.sin(math.radians(60)), alt)
        ]
    elif formation_type == "ters_ucgen":
        return [
            (center[0] - offset, center[1], alt),
            (center[0] + offset * math.cos(math.radians(60)),
             center[1] - offset * math.sin(math.radians(60)), alt),
            (center[0] + offset * math.cos(math.radians(60)),
             center[1] + offset * math.sin(math.radians(60)), alt)
        ]
    elif formation_type == "cizgi":
        return [
            (center[0], center[1] - offset, alt),
            (center[0], center[1], alt),
            (center[0], center[1] + offset, alt)
        ]
    else:
        # Varsayılan olarak üçgen formasyon
        return create_formation("ucgen", center, offset_metre, alt)

def move_formation_to_waypoint(connections, formation_type, offset_metre, waypoint, alt):
    """Sürüyü mevcut formasyonu koruyarak waypoint'e hareket ettirir"""
    # Sürünün mevcut merkezi
    current_center = get_center(connections)
    
    # Hedef merkez (waypoint)
    target_center = waypoint
    
    # Hareket yönünü hesapla
    heading = calculate_heading(current_center[0], current_center[1], target_center[0], target_center[1])
    print(f"Sürü merkezi: {current_center}, Hedef merkez: {target_center}, Yönelim: {heading:.1f} derece")
    
    # Formasyon için hedef pozisyonları oluştur
    target_positions = create_formation(formation_type, target_center, offset_metre, alt)
    time.sleep(5)
    # Tüm drone'ları hedef pozisyona gönder ve hareket yönüne bakacak şekilde ayarla
    threads = []
    for i, conn in enumerate(connections):
        print(f"Drone {i+1} hedefi: {target_positions[i]} - Yönelim: {heading:.1f} derece")
        
        # Önce yönelimi ayarla
        set_heading(conn, heading)
        #time.sleep(0.5)  # Yönelim değişimi için kısa bir bekleme
        
        # Sonra hedef pozisyona gönder
        t = threading.Thread(target=send_goto, args=(conn, *target_positions[i]))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Tüm drone'ların hedef pozisyona ulaşmasını bekle
    all_reached = False
    while not all_reached:
        positions = [get_position(conn) for conn in connections]
        reached = [is_near_position(pos[0], pos[1], target_positions[i][0], target_positions[i][1]) 
                  for i, pos in enumerate(positions)]
        all_reached = all(reached)
        
        if not all_reached:
            time.sleep(1)
            print("Hedefler bekleniyor...")
        else:
            print("Tüm drone'lar hedeflere ulaştı.")

def navigate_waypoints(connections, formation_type, offset_metre, waypoints, alt):
    """Sürüyü belirli bir formasyonda waypoint'ler arasında gezdirir (threading ile)"""
    print(f"Waypoint navigasyonu başlıyor. Formasyon: {formation_type}, Waypoint sayısı: {len(waypoints)}")
    
    def move_to_waypoint(index, waypoint):
        print(f"Waypoint {index+1}/{len(waypoints)} hedefleniyor: {waypoint}")
        move_formation_to_waypoint(connections, formation_type, offset_metre, waypoint, alt)
        print(f"Waypoint {index+1} tamamlandı.")
        time.sleep(2)

    threads = []
    for i, waypoint in enumerate(waypoints):
        t = threading.Thread(target=move_to_waypoint, args=(i, waypoint))
        threads.append(t)

    for t in threads:
        t.start()
        t.join()  # Sırayla çalışmasını sağlıyoruz

    print("Tüm waypoint'ler tamamlandı.")

def RTL_drone(index):
    connections[index].set_mode_apm("RTL")

def append_drone(index, standby_drone, target_position):
    connections[index] = standby_drone
    arm(standby_drone)
    time.sleep(2)
    takeoff(standby_drone, 10)
    time.sleep(10)

    lat, lon, alt = target_position
    send_goto(standby_drone, lat, lon, alt)

def landing_all_drones(conn):
    conn.set_mode_apm("LAND")

def double_integrator_control(p, v, delta, k_p, k_d, i):
    """
    Double integrator kontrol kuralı.
    p: ajan pozisyonları (N x dim)
    v: ajan hızları (N x dim)
    delta: delta matrisi (N x N x dim)
    k_p: pozisyon kazancı
    k_d: hız kazancı
    i: kontrolü hesaplanacak ajan indeksi
    """
    N = len(p)
    pos_term = np.zeros_like(p[i])
    for j in range(N):
        pos_term += (p[j] - p[i] - delta[j][i])
    vel_term = np.zeros_like(v[i])
    for j in range(N):
        vel_term += (v[j] - v[i])
    u_f_i = k_p * pos_term + k_d * vel_term
    return u_f_i

def get_positions_and_velocities(connections):
    """
    Tüm drone'ların pozisyon ve hızlarını alır.
    Pozisyon: [lat, lon, alt]
    Hız: [vx, vy, vz]
    """
    positions = []
    velocities = []
    for conn in connections:
        msg = None
        while not msg:
            msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0
        vx = msg.vx / 100.0 if hasattr(msg, 'vx') else 0.0
        vy = msg.vy / 100.0 if hasattr(msg, 'vy') else 0.0
        vz = msg.vz / 100.0 if hasattr(msg, 'vz') else 0.0
        positions.append(np.array([lat, lon, alt]))
        velocities.append(np.array([vx, vy, vz]))
    return np.array(positions), np.array(velocities)

def formasyon_ucgen(offset_metre, alt):
    offset = offset_metre / 100000
    positions = [get_position(conn) for conn in connections]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    center = (avg_lat, avg_lon, avg_alt)

    print(f"Üçgen Formasyonu Merkezi: {center}")

    triangle = [
        (center[0] + offset, center[1], alt),
        (center[0] - offset * math.cos(math.radians(60)),
         center[1] - offset * math.sin(math.radians(60)), alt),
        (center[0] - offset * math.cos(math.radians(60)),
         center[1] + offset * math.sin(math.radians(60)), alt)
    ]

    # --- Double Integrator Control Uygulaması ---
    p, v = get_positions_and_velocities(connections)
    N = len(p)
    # Basit delta matrisi (örnek: tüm drone'lar için sıfır)
    delta = np.zeros((N, N, 3))
    k_p = 1.0
    k_d = 0.5
    for i in range(N):
        u_f_i = double_integrator_control(p, v, delta, k_p, k_d, i)
        print(f"Drone {i+1} kontrol girdisi (u_f_i): {u_f_i}")

    threads = []
    for i, conn in enumerate(connections):
        print(f"Drone {i+1} hedefi: {triangle[i]}")
        t = threading.Thread(target=send_goto, args=(conn, *triangle[i]))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

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
    time.sleep(15)  # Drone'ların yükselme süresi için daha uzun bekleme

    # İlk merkez koordinatını al
    initial_center = get_center(connections)

    # Waypoint listesi (dairesel örnek rota)
    waypoints = [
        (initial_center[0] + 0.0001, initial_center[1], initial_center[2])  # Kuzey
    ]


    formasyon_ucgen(5, 10) 
    time.sleep(10)

    # Waypoint navigasyonunu ayrı bir thread'de çalıştır
    nav_thread = threading.Thread(target=navigate_waypoints, args=(connections, "ucgen", 5, waypoints, 10))
    nav_thread.start()
    nav_thread.join()

    time.sleep(8)

    k = int(input("Eve dönecek drone indexi: "))
    RTL_drone_position = get_position(connections[k])
    RTL_drone(k)

    time.sleep(40)

    print("Yeni drone ekleniyor.")
    append_drone(k, standby_drone, RTL_drone_position)

    time.sleep(15)

    print("Land komutu başlatıldı.")

    landing_threads = []
    for conn in connections:
        t = threading.Thread(target=landing_all_drones, args=(conn,))
        landing_threads.append(t)
        t.start()
    for t in landing_threads:
        t.join()
