import xml.etree.ElementTree as ET 
import sys 
import pprint
import json

def parse_net_xml(file_path): 
    tree = ET.parse(file_path) 
    root = tree.getroot()
    # Mapping from the origin letter and lane index to a turning movement key.
    # For example, for a north approach, index 0 is interpreted as "N-W" (left turn)
    # and index 1 as "N-E" (right turn).
    mapping = {
        'n': {0: 'N-W', 1: 'N-E'},
        's': {0: 'S-E', 1: 'S-W'},
        'e': {0: 'E-N', 1: 'E-S'},
        'w': {0: 'W-S', 1: 'W-N'}
    }

    # Define all desired keys for lane_sets.
    movement_keys = ['S-W', 'S-S', 'S-E',
                    'W-N', 'W-W', 'W-S',
                    'N-E', 'N-N', 'N-W',
                    'E-S', 'E-E', 'E-N']
    
    config = {}

    for junction in root.findall('junction'):
        if junction.get('type') != 'traffic_light':
            continue  # process only traffic light junctions
        
        junc_id = junction.get('id')
        inc_lanes = junction.get('incLanes', '').split()
        
        # Initialize lane_sets with all keys set to empty lists.
        lane_sets = {key: [] for key in movement_keys}
        
        # For each incoming lane, determine its turning movement based on its naming.
        for lane in inc_lanes:
            # Expect lane names of the form: "<origin>_t_<index>"
            parts = lane.split('_')
            if len(parts) < 3:
                continue
            origin = parts[0].lower()
            try:
                index = int(parts[-1])
            except ValueError:
                continue
            if origin in mapping and index in mapping[origin]:
                movement = mapping[origin][index]
                lane_sets[movement].append(lane)
        
        # Ensure lane_sets are ordered according to movement_keys
        #ordered_lane_sets = {key: lane_sets[key] for key in movement_keys}
        ordered_lane_sets = {key: lane_sets.get(key, []) for key in movement_keys}
        
        # Build the configuration for this junction (only lane_sets are included)
        config[junc_id] = {
            'lane_sets': ordered_lane_sets
        }

    return config

if __name__ == '__main__1': 
    if len(sys.argv) < 3: 
        print("Usage: python generate_config.py <net_xml_file> <map_name>") 
        sys.exit(1) 
        file_path = sys.argv[1] 
        map_name = sys.argv[2]

# Parse the net.xml and wrap the junction configs in a map-level dict.

map_name = '1way_single'
file_path = r"C:\Clinical\RESCO\resco_benchmark\environments\1way_single\single-intersection.net.xml"

map_config = {map_name: parse_net_xml(file_path)}

# Pretty-print the generated configuration (which you can copy into signal_config.py)
pprint.pprint(map_config)
print(json.dumps(map_config, indent=4, separators=(',', ': '), default=str))
