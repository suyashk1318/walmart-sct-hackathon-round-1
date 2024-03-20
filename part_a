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
    
    # Define the recursive function to compute the shortest path
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

# Write output dataset to CSV
def write_output(filename, order_ids, order_route, locations):
    data = {'order_id': order_route, 'lng': [], 'lat': [], 'depot_lat': [], 'depot_lng': []}
    for order_id in order_route:
        if order_id == 'depot':
            data['lng'].append(locations[0][1])
            data['lat'].append(locations[0][0])
        else:
            index = order_ids.index(order_id)
            data['lng'].append(locations[index][1])
            data['lat'].append(locations[index][0])
    data['depot_lat'] = [locations[0][0]] * len(order_route)
    data['depot_lng'] = [locations[0][1]] * len(order_route)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Main function to run the code
def main(input_file, output_file):
    # Read input dataset
    locations, order_ids = read_input(input_file)
    if locations is None or order_ids is None:
        return
    n = len(locations)
    # Calculate distances between all pairs of locations
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            distances[i][j] = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
            print(f"Distance between {locations[i]} and {locations[j]}: {distances[i][j]} kms")
    print("Distances matrix:")
    print(distances)
    # Solve TSP using dynamic programming
    min_distance = tsp_dp(locations, distances)
    print("Minimum distance traveled:", min_distance, "kms")
# Generate the optimal route
    order_route = [[order_ids[i] for i in perm] for perm in permutations(range(1, len(order_ids)))]
    order_route = min(order_route, key=lambda route: total_distance(route, distances))
    print("Optimal route:", order_route)
    # Write output dataset
    write_output(output_file, order_ids, order_route, locations)

# Example usage
if __name__ == "__main__":
    input_file = "D:\Walmart\part_a_input_dataset_2.csv"  # Change to your input filename
    output_file = "D:\Walmart\part_a_output_dataset_2.csv"  # Change to your output filename
    main(input_file, output_file)
