import os
import xml.etree.ElementTree as ET

import numpy as np
import sys
from resco_benchmark.config.map_config import map_configs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


log_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results' + os.sep)
print(f"Log dir: {log_dir}")
env_base = '..'+os.sep+'environments'+os.sep
names = [folder for folder in next(os.walk(log_dir))[1]]

# metrics = ['timeLoss', 'duration', 'waitingTime']
metrics = ['duration', 'waitingTime', 'CO2_abs', 'waitingCount']

for metric in metrics:
    output_file = 'avg_{}.py'.format(metric)
    run_avg = dict()

    for name in names:
        split_name = name.split('-')
        print(split_name)
        map_name = split_name[2]
        average_per_episode = []
        for i in range(1, 20): # default (1, 10000)
            trip_file_name = log_dir+name + os.sep + 'tripinfo_'+str(i)+'.xml'
            if not os.path.exists(trip_file_name):
                print('No '+trip_file_name)
                break
            try:
                tree = ET.parse(trip_file_name)
                root = tree.getroot()
                num_trips, total = 0, 0.0
                last_departure_time = 0
                last_depart_id = ''
                for child in root:
                    try:
                        num_trips += 1
                        
                        if metric == 'CO2_abs':
                            for emission in child:
                                total += float(emission.attrib[metric])
                        else:
                            total += float(child.attrib[metric])
                            
                        if metric == 'timeLoss':
                            total += float(child.attrib['departDelay'])
                            depart_time = float(child.attrib['depart'])
                            if depart_time > last_departure_time:
                                last_departure_time = depart_time
                                last_depart_id = child.attrib['id']
                    except Exception as e:
                        #raise e
                        break
                route_file_name = env_base + map_name + os.sep + map_name + '_' + str(i) + '.rou.xml'

                if metric == 'timeLoss':    # Calc. departure delays
                    try:
                        tree = ET.parse(route_file_name)
                    except FileNotFoundError:
                        route_file_name = env_base + map_name + os.sep + map_name + '.rou.xml'
                        tree = ET.parse(route_file_name)
                    root = tree.getroot()
                    last_departure_time = None
                    for child in root:
                        if child.attrib['id'] == last_depart_id:
                            last_departure_time = float(child.attrib['depart'])     # Get the time it was suppose to depart
                    never_departed = []
                    if last_departure_time is None: raise Exception('Wrong trip file')
                    for child in root:
                        if child.tag != 'vehicle': continue
                        depart_time = float(child.attrib['depart'])
                        if depart_time > last_departure_time:
                            never_departed.append(depart_time)
                    never_departed = np.asarray(never_departed)
                    never_departed_delay = np.sum(float(map_configs[map_name]['end_time']) - never_departed)
                    total += never_departed_delay
                    num_trips += len(never_departed)
                    
                average = total / num_trips
                average_per_episode.append(average)
            except ET.ParseError as e:
                #raise e
                break
           
        metricName = '';
        if metric == 'duration':
            metricName = 'Duration'
        elif metric == 'waitingTime':
            metricName = 'Waiting Time'
        elif metric == 'waitingCount':
            metricName = 'Waiting Count'
        elif metric == 'CO2_abs':
            metricName = 'CO2'
        else:
            metricName = 'Not defined'
            
        agentName = split_name[0]
        if 'CO2' in split_name[4]:
            agentName += 'CO2'
            
        #run_name = 'Agent: '+agentName+' | Map: '+split_name[2]+' | Metric: '+metricName + ' | Run: '+split_name[4]
        run_name = f'Agent: {agentName}.{split_name[4]} | {split_name[2]} | {metricName}'
        average_per_episode = np.asarray(average_per_episode)

        if run_name in run_avg:
            run_avg[run_name].append(average_per_episode)
        else:
            run_avg[run_name] = [average_per_episode]


    alg_res = []
    alg_name = []
    for run_name in run_avg:
        list_runs = run_avg[run_name]
        min_len = min([len(run) for run in list_runs])
        list_runs = [run[:min_len] for run in list_runs]
        avg_delays = np.sum(list_runs, 0)/len(list_runs)
        # print('avg_delays = ' + ' '.join(str(e) for e in avg_delays))
        err = np.std(list_runs, axis=0)

        alg_name.append(run_name)
        alg_res.append(avg_delays)

        alg_name.append(run_name+'_yerr')
        alg_res.append(err)

        plt.title(run_name)
        plt.plot(avg_delays)
        plt.show()


    np.set_printoptions(threshold=sys.maxsize)
    with open(output_file, 'a') as out:
        for i, res in enumerate(alg_res):
            out.write("'{}': {},\n".format(alg_name[i], res.tolist()))
