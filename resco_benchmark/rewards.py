import numpy as np

from resco_benchmark.config.mdp_config import mdp_configs


def wait(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = -total_wait
    return rewards


def wait_norm(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = np.clip(-total_wait/224, -4, 4).astype(np.float32)
    return rewards


def pressure(signals):
    rewards = dict()
    for signal_id in signals:
        queue_length = 0
        for lane in signals[signal_id].lanes:
            queue_length += signals[signal_id].full_observation[lane]['queue']

        for lane in signals[signal_id].outbound_lanes:
            dwn_signal = signals[signal_id].out_lane_to_signalid[lane]
            if dwn_signal in signals[signal_id].signals:
                queue_length -= signals[signal_id].signals[dwn_signal].full_observation[lane]['queue']

        rewards[signal_id] = -queue_length
    return rewards
    
def pressureCO2(signals):
    rewards = dict()
    for signal_id in signals:
        total_co2 = 0
        for lane in signals[signal_id].lanes:
            total_co2 += signals[signal_id].full_observation[lane]['total_co2']

        for lane in signals[signal_id].outbound_lanes:
            dwn_signal = signals[signal_id].out_lane_to_signalid[lane]
            if dwn_signal in signals[signal_id].signals:
                total_co2 -= signals[signal_id].signals[dwn_signal].full_observation[lane]['total_co2']

        rewards[signal_id] = round(-total_co2/100, 1)
    return rewards

def pressure_CO2Multiple(signals):
    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for direction in signal.lane_sets:
            # Initialize variables for inbound metrics
            queue_length = 0
            total_wait = 0
            total_co2 = 0
            total_awg_speed = 0
            num_lanes = len(signal.lane_sets[direction])

            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += signal.full_observation[lane]['total_wait']
                total_co2 += signal.full_observation[lane]['total_co2']
                total_awg_speed += signal.full_observation[lane]['awg_speed']

            # Average speed adjustment
            avg_awg_speed = total_awg_speed / num_lanes if num_lanes else 0

            # Subtract downstream metrics
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
                    total_co2 -= signal.signals[dwn_signal].full_observation[lane]['total_co2']

            # Combine into a reward formula
            reward += (
                -0.5 * queue_length  # Penalize queue length
                -0.3 * total_wait    # Penalize waiting time
                -0.2 * total_co2     # Penalize CO2 emissions
                + 0.4 * avg_awg_speed  # Reward higher average speed
            )

        rewards[signal_id] = reward
    return rewards


def queue_maxwait(signals):
    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['MA2C']['coef'])
        rewards[signal_id] = -reward
    return rewards


def queue_maxwait_neighborhood(signals):
    rewards = queue_maxwait(signals)
    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                sum_reward += (mdp_configs['MA2C']['coop_gamma'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    return neighborhood_rewards


def fma2c(signals):
    fma2c_config = mdp_configs['FMA2C']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    fringe_arrivals = dict()
    liquidity = dict()
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation['arrivals']
        liquidity[manager] += (len(signal.full_observation['departures']) - len(signal.full_observation['arrivals']))
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]['vehicles']:
                    if vehicle['id'] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = dict()
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += (fma2c_config['alpha'] * (fringe_arrivals[neighbor] + liquidity[neighbor]))
        management_neighborhood[manager] = mgr_rew

    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['FMA2C']['coef'])
        rewards[signal_id] = -reward

    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += (fma2c_config['alpha'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards


def fma2c_full(signals):
    fma2c_config = mdp_configs['FMA2CFull']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    fringe_arrivals = dict()
    liquidity = dict()
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation['arrivals']
        liquidity[manager] += (len(signal.full_observation['departures']) - len(signal.full_observation['arrivals']))
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]['vehicles']:
                    if vehicle['id'] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = dict()
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += (fma2c_config['alpha'] * (fringe_arrivals[neighbor] + liquidity[neighbor]))
        management_neighborhood[manager] = mgr_rew

    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['FMA2CFull']['coef'])
        rewards[signal_id] = -reward

    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += (fma2c_config['alpha'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards