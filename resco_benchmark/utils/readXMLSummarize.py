import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import glob

# Define the metrics in one place
METRICS = {
    'Duration': 'Avg Duration',
    'Waiting Count': 'Avg Waiting Count',
    'CO2 Emissions': 'Total CO2 Emissions',
    'Waiting Time': 'Avg Waiting Time'
    
}

def parse_trip_info(xml_file):
    """Parses a SUMO tripinfo XML file and extracts relevant data."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    
    for trip in root.findall('tripinfo'):
        data.append({
            'Trip ID': trip.get('id'),
            'Duration': float(trip.get('duration')),
            'Waiting Time': float(trip.get('waitingTime')),
            'CO2 Emissions': sum(float(em.get('CO2_abs')) for em in trip.findall('emissions')),
            'Waiting Count': int(trip.get('waitingCount', 0))
        })
    
    return data

def compute_metrics(trip_data):
    """Computes average metrics for the last N results dynamically."""
    df = pd.DataFrame(trip_data)
    #df = df.tail(last_n)
    
    metrics = {name: df[column].mean() if 'Total' not in name else df[column].sum() for column, name in METRICS.items()}
    return metrics

def generate_results_table(log_dir, last_n=10):
    """Generates a summary table by processing last N tripinfo XML files per algorithm folder."""
    results = []
    
    algo_folders = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    for algo in algo_folders:
        algo_path = os.path.join(log_dir, algo)
        trip_files = sorted(glob.glob(os.path.join(algo_path, 'tripinfo_*.xml')), key=os.path.getctime)[-last_n:]
        
        all_trip_data = []
        for file in trip_files:
            trip_data = parse_trip_info(file)
            all_trip_data.extend(trip_data)
        
        if all_trip_data:
            metrics = compute_metrics(all_trip_data)
            results.append([algo] + list(metrics.values()))
    
    df_results = pd.DataFrame(results, columns=['Algorithm'] + list(METRICS.values()))
    return df_results

# Run the function and display results
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results' + os.sep)
df_results = generate_results_table(log_dir, last_n=1)

print(df_results)
df_results.to_excel(log_dir + "results_finished.xlsx", index=False)


df_results = generate_results_table(log_dir, last_n=-1)

print(df_results)
df_results.to_excel(log_dir + "results_started.xlsx", index=False)