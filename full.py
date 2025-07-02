from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment

# ------------------------------ BAĞLANTI, KALKIŞ VE İNİŞLER ------------------------------

# Drone bağlantı noktaları
ports = ['udp:127.0.0.1:14550', 'udp:127.0.0.1:14560', 'udp:127.0.0.1:14570']
connections = [connect(p, wait_ready=True) for p in ports]

# Bağlantı kontrolü
for i, vehicle in enumerate(connections):
    print(f"Bağlandı: Drone {i+1}")

def arm(vehicle):
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)
    print(f"Drone {vehicle.name} armed")

def takeoff(vehicle, alt):
    vehicle.simple_takeoff(alt)
    while vehicle.location.global_relative_frame.alt < alt * 0.95:
        print(f"Drone {vehicle.name} altitude: {vehicle.location.global_relative_frame.alt}")
        time.sleep(1)
    print(f"Drone {vehicle.name} reached altitude: {alt}")

def landing_all_drones(vehicle):
    vehicle.mode = VehicleMode("LAND")

def RTL_drone(index):
    connections[index].set_mode_apm("RTL")

# ------------------------------ GENEL FONKSİYONLAR ------------------------------

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
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

def send_goto(vehicle, lat, lon, alt):
    target_location = LocationGlobalRelative(lat, lon, alt)
    vehicle.simple_goto(target_location)

def get_position(vehicle):
    return vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon, vehicle.location.global_relative_frame.alt

def get_center(vehicles):
    positions = [get_position(vehicle) for vehicle in vehicles]
    avg_lat = sum(p[0] for p in positions) / len(positions)
    avg_lon = sum(p[1] for p in positions) / len(positions)
    avg_alt = sum(p[2] for p in positions) / len(positions)
    return (avg_lat, avg_lon, avg_alt)



# ------------------------------ FORMASYON GÖREVİ ------------------------------

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

def calculate_cost_matrix(vehicles, target_positions):
    """Drone'ların mevcut konumlarına göre hedeflere olan mesafe matrisini hesaplar."""
    current_positions = [get_position(vehicle) for vehicle in vehicles]
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

def calculate_target_positions(formasyon, center, offset, alt):
    if formasyon == "ucgen":
        return [
            (center[0] + offset, center[1], alt),
            (center[0] - offset * math.cos(math.radians(60)),
             center[1] - offset * math.sin(math.radians(60)), alt),
            (center[0] - offset * math.cos(math.radians(60)),
             center[1] + offset * math.sin(math.radians(60)), alt)
        ]
    elif formasyon == "okbasi":
        return [
            (center[0] - offset, center[1], alt),
            (center[0] + offset * math.cos(math.radians(60)),
             center[1] - offset * math.sin(math.radians(60)), alt),
            (center[0] + offset * math.cos(math.radians(60)),
             center[1] + offset * math.sin(math.radians(60)), alt)
        ]
    elif formasyon == "cizgi":
        return [
            (center[0], center[1] - offset, alt),
            (center[0], center[1], alt),
            (center[0], center[1] + offset, alt)
        ]
    else:
        raise ValueError("Geçersiz formasyon tipi.")

def formation(formation_type, offset_metre, alt):
    offset = offset_metre / 100000
    center = get_center(connections)

    print(f"{formation_type.capitalize()} Formasyonu Merkezi: {center}")
    target_positions = calculate_target_positions(formation_type, center, offset, alt)

    cost_matrix = calculate_cost_matrix(connections, target_positions)
    row_ind, col_ind = assign_targets(cost_matrix)
    print("Maliyet Matrisi:\n", cost_matrix)

    threads = []
    for i, vehicle in enumerate(connections):
        target = target_positions[col_ind[i]]
        print(f"Drone {i+1} hedefi: {target}")
        t = threading.Thread(target=send_goto, args=(vehicle, *target))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

# ------------------------------ BOİDS ALGORİTMASI ------------------------------
# ------------------------------ NAVİGASYON GÖREVİ ------------------------------
# ------------------------------ BİREY EKLEME/ÇIKARMA GÖREVİ ------------------------------
# ------------------------------ ANA FONKSİYON ------------------------------
