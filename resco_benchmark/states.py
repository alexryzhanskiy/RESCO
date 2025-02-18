import numpy as np

from resco_benchmark.config.mdp_config import mdp_configs


def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'])
            lane_obs.append(signal.full_observation[lane]['total_wait'])
            lane_obs.append(signal.full_observation[lane]['queue'])

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += vehicle['speed']
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'] / 28)
            lane_obs.append(signal.full_observation[lane]['total_wait'] / 28)
            lane_obs.append(signal.full_observation[lane]['queue'] / 28)

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations
    
def mplightCO2(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            total_co2 = 0
            for lane in signal.lane_sets[direction]:
                total_co2 += signal.full_observation[lane]['total_co2']

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    total_co2 -= signal.signals[dwn_signal].full_observation[lane]['total_co2']
            obs.append(total_co2)
        observations[signal_id] = np.asarray(obs)
    return observations

def mplight_Co2Multiple(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0            
            tot_approach = 0
            total_co2 = 0
            awg_speed = 0
            awg_speed_out = 0
            alpha_angle = 0
            total_mass =0
            total_mass_out = 0
            num_lanes = len(signal.lane_sets[direction])
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += (signal.full_observation[lane]['total_wait'] / 28)
                
                #vehicles = signal.full_observation[lane]['vehicles']
                # for vehicle in vehicles:
                #     total_speed += vehicle['speed']
                tot_approach += (signal.full_observation[lane]['approach'] / 28)
                total_co2 += signal.full_observation[lane]['total_co2']
                awg_speed += signal.full_observation[lane]['awg_speed']
                alpha_angle = max(alpha_angle, signal.full_observation[lane]['alpha'])
                total_mass += signal.full_observation[lane]['total_mass']

            # Subtract downstream
            num_lanes_out = len(signal.lane_sets_outbound[direction])
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
                    total_co2 -= signal.signals[dwn_signal].full_observation[lane]['total_co2']
                    awg_speed_out -= signal.signals[dwn_signal].full_observation[lane]['awg_speed']                    
                    total_mass_out += signal.signals[dwn_signal].full_observation[lane]['total_mass']
                    
            obs.append(queue_length)
            obs.append(total_wait)            
            obs.append(tot_approach)
            obs.append(awg_speed/num_lanes if num_lanes else 0)
            obs.append(awg_speed_out/num_lanes_out if num_lanes_out else 0)
            obs.append(total_co2)       
            obs.append(alpha_angle)     
            obs.append(total_mass)
            obs.append(total_mass_out)
        observations[signal_id] = np.asarray(obs)
    return observations


def idqn_Co2Multiple(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'] / 28)
            lane_obs.append(signal.full_observation[lane]['total_wait'] / 28)
            lane_obs.append(signal.full_observation[lane]['queue'] / 28)
 
            lane_obs.append(signal.full_observation[lane]['total_co2'] / 10e3)
            lane_obs.append(signal.full_observation[lane]['awg_speed'] / 50)
            #lane_obs.append(signal.full_observation[lane]['alpha'])
            lane_obs.append(signal.full_observation[lane]['total_mass'] / 1000)

            # total_speed = 0
            # vehicles = signal.full_observation[lane]['vehicles']
            # for vehicle in vehicles:
            #     total_speed += (vehicle['speed'] / 20 / 28)
            # lane_obs.append(total_speed)

            obs.append(lane_obs)
        obs_shaped = np.expand_dims(np.asarray(obs), axis=0)
        observations[signal_id] = obs_shaped
    return observations

def idqn_Co2MultipleLineSets(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        num_directions = len(signal.lane_sets)
        num_features = 9  # queue_length, total_wait, tot_approach, awg_speed, awg_speed_out, total_co2, alpha_angle, total_mass, total_mass_out

        # Initialize an array for normalized observations
        obs = np.zeros((num_directions, num_features), dtype=np.float32)

        for i, direction in enumerate(signal.lane_sets):
            queue_length = 0
            total_wait = 0
            tot_approach = 0
            total_co2 = 0
            awg_speed = 0
            awg_speed_out = 0
            alpha_angle = 0
            total_mass = 0
            total_mass_out = 0

            num_lanes = len(signal.lane_sets[direction])
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += signal.full_observation[lane]['total_wait']  
                tot_approach += signal.full_observation[lane]['approach']  
                total_co2 += signal.full_observation[lane]['total_co2']
                awg_speed += signal.full_observation[lane]['awg_speed']
                alpha_angle = max(alpha_angle, signal.full_observation[lane]['alpha'])
                total_mass += signal.full_observation[lane]['total_mass']

            num_lanes_out = len(signal.lane_sets_outbound[direction])
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
                    total_co2 -= signal.signals[dwn_signal].full_observation[lane]['total_co2']
                    awg_speed_out -= signal.signals[dwn_signal].full_observation[lane]['awg_speed']
                    total_mass_out -= signal.signals[dwn_signal].full_observation[lane]['total_mass']

            #normalize values
            obs[i, 0] = queue_length /28
            obs[i, 1] = total_wait / 28
            obs[i, 2] = tot_approach / 28
            obs[i, 3] = (awg_speed / num_lanes if num_lanes else 0) / 50
            obs[i, 4] = 1 #(awg_speed_out / num_lanes_out if num_lanes_out else 0) / 50
            obs[i, 5] = total_co2 / 10e3
            obs[i, 6] = 1 #alpha_angle
            obs[i, 7] = 1 #total_mass / 1000
            obs[i, 8] = 1 #total_mass_out / 1000

        # Add the active phase as the first channel
        phase_channel = np.full((1, num_directions, num_features), signal.phase, dtype=np.float32)

        # Combine the phase information and lane data
        obs = np.concatenate((phase_channel, np.expand_dims(obs, axis=0)), axis=0)  # (channels, height, width)

        observations[signal_id] = obs

    return observations

 

def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0
            total_speed = 0
            tot_approach = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += (signal.full_observation[lane]['total_wait'] / 28)
                total_speed = 0
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    total_speed += vehicle['speed']
                tot_approach += (signal.full_observation[lane]['approach'] / 28)

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)
    return observations


def wave(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        state = []
        for direction in signal.lane_sets:
            wave_sum = 0
            for lane in signal.lane_sets[direction]:
                wave_sum += signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            state.append(wave_sum)
        observations[signal_id] = np.asarray(state)
    return observations


def ma2c(signals):
    ma2c_config = mdp_configs['MA2C']

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / ma2c_config['norm_wave'], 0, ma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(ma2c_config['coop_gamma'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / ma2c_config['norm_wait'], 0, ma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    fma2c_config = mdp_configs['FMA2C']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    fma2c_config = mdp_configs['FMA2CFull']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)

            waves.append(signal.full_observation[lane]['total_wait'] / 28)
            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations