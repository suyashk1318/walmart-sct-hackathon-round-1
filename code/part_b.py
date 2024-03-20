import numpy as np
import pandas as pd
from itertools import permutations

# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

# Function to calculate total distance for a given route
def total_distance(route, distances):
    total = 0
    for i in range(len(route) - 1):
        idx1 = int(route[i])
        idx2 = int(route[i + 1])
        total += distances[idx1][idx2]
    return total

# Read input dataset from CSV
def read_input(filename):
    try:
        data = pd.read_csv(filename)
        depot_lat = data['depot_lat'].iloc[0]
        depot_lng = data['depot_lng'].iloc[0]
        locations = [(depot_lat, depot_lng)]  # Start with depot
        order_ids = ['depot']
        for index, row in data.iterrows():
            locations.append((row['lat'], row['lng']))
            order_ids.append(row['order_id'])
        return locations, order_ids
    except Exception as e:
        print("An error occurred:", e)

# Dynamic programming solution for TSP
def tsp_dp(locations, distances):
    n = len(locations)
    
    # Initialize memoization table
    memo = {}
    
    # Recursive function to compute the shortest path
    def dp_mask(mask, last):
        if mask == (1 << n) - 1:
            return distances[last][0]
        if (mask, last) in memo:
            return memo[(mask, last)]
        ans = float('inf')
        for i in range(n):
            if (mask >> i) & 1 == 0:
                ans = min(ans, distances[last][i] + dp_mask(mask | (1 << i), i))
        memo[(mask, last)] = ans
        return ans
    
    return dp_mask(1, 0)

# Orders to vehicles based on vehicle capacity
def assign_orders(order_ids, vehicle_capacity):
    vehicles = []
    current_vehicle = []
    for order_id in order_ids[1:]:  # Exclude depot
        if len(current_vehicle) < vehicle_capacity:
            current_vehicle.append(order_id)
        else:
            vehicles.append(current_vehicle)
            current_vehicle = [order_id]
    if current_vehicle:
        vehicles.append(current_vehicle)
    return vehicles

# Generate routes for each vehicle using dynamic programming
def generate_routes(vehicles, locations, distances, order_ids):
    routes = []
    for vehicle_orders in vehicles:
        vehicle_locations = [(locations[order_ids.index('depot')][0], locations[order_ids.index('depot')][1])]  # Start with depot
        for order_id in vehicle_orders:
            order_index = order_ids.index(order_id)
            vehicle_locations.append(locations[order_index])
        vehicle_distances = np.zeros((len(vehicle_locations), len(vehicle_locations)))
        for i in range(len(vehicle_locations)):
            for j in range(len(vehicle_locations)):
                vehicle_distances[i][j] = haversine(vehicle_locations[i][0], vehicle_locations[i][1], vehicle_locations[j][0], vehicle_locations[j][1])
        route = tsp_dp(vehicle_locations, vehicle_distances)
        routes.append(route)
    return routes

# Write output dataset to CSV
def write_output(filename, order_ids, vehicles, routes, locations):
    data = {'order_id': [], 'lng': [], 'lat': [], 'depot_lat': [], 'depot_lng': [], 'vehicle_num': [], 'dlvr_seq_num': []}
    for i, vehicle_orders in enumerate(vehicles):
        for order_id in vehicle_orders:
            order_index = order_ids.index(order_id)
            data['order_id'].append(order_id)
            data['lng'].append(locations[order_index][1])
            data['lat'].append(locations[order_index][0])
            data['depot_lat'].append(locations[0][0])
            data['depot_lng'].append(locations[0][1])
            data['vehicle_num'].append(i + 1)
            data['dlvr_seq_num'].append(vehicle_orders.index(order_id) + 1)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Main function to run the code
def main(input_file, output_file, vehicle_capacity):
    # Read input dataset
    locations, order_ids = read_input(input_file)
    if locations is None or order_ids is None:
        return
    # Distances between all pairs of locations
    distances = np.zeros((len(locations), len(locations)))
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i != j:
                distances[i][j] = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
    # Orders to vehicles based on vehicle capacity
    vehicles = assign_orders(order_ids, vehicle_capacity)
    # Routes for each vehicle using dynamic programming
    routes = generate_routes(vehicles, locations, distances, order_ids)
    # Write output dataset
    write_output(output_file, order_ids, vehicles, routes, locations)

if __name__ == "__main__":
    input_file = "D:\Walmart\part_b_input_dataset_1.csv"  
    output_file = "D:\Walmart\part_b_input_dataset_1.csv"  
    vehicle_capacity = 20 
    main(input_file, output_file, vehicle_capacity)
