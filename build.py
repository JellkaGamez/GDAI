import os, json, base64, gzip, math

def text_after(text, after):
    return text[text.find(after) + len(after):]

def text_before(text, before):
    return text[:text.find(before)]

levels = []

for level in os.listdir("levels/raw"):
    if level.endswith(".gmd"):
        levels.append(level)
        print(f'{level} is a GD level file.')
    else:
        print(f'[WARNING] {level} is not a GD level file!')

SPEED = 8.373

debug = input("Debug mode? (y/n): ")

for level in levels:
    with open(f'levels/raw/{level}', 'r') as f:
        text = f.read()
    
    print(f'Parsing {level}...')

    name = text_after(text, '<s>')
    name = text_before(name, '</s>')

    print(f'Name: {name}') if debug == 'y' else None

    level_data = text_after(text_after(text, '<s>'), '<s>')
    level_data = text_before(level_data, '</s>')

    # decode from base64
    level_data = level_data.encode('utf-8')
    level_data = base64.urlsafe_b64decode(level_data)

    # decompress from gzip
    level_data = gzip.decompress(level_data)
    level_data = level_data.decode('utf-8')

    print(f'Level data snippet: {level_data [:50]}...{level_data[-50:]}') if debug == 'y' else None

    print('Parsing level data...') if debug == 'y' else None

    level_data_raw = level_data
    meta_data = text_before(level_data, ';')

    meta_tokens = meta_data.split(',')

    meta_data = {}
    for i in range(0, len(meta_tokens), 2):
        meta_data[meta_tokens[i]] = meta_tokens[i+1]

    json_meta_data = {}

    colour_tokens = meta_data['kS38'].split('|')
    colour_tokens = [token for token in colour_tokens if token != '']
    colours = []

    for key in meta_data:
        match key:
            case 'kS38':
                for colour in colour_tokens:
                    current_colour_tokens = colour.split('_')
                    colour_data = {
                        'id': current_colour_tokens[0],
                        'r': current_colour_tokens[1],
                        'g': current_colour_tokens[2],
                        'b': current_colour_tokens[3],
                        'a': current_colour_tokens[4]
                    }
                    colours.append(colour_data)
                json_meta_data['colours'] = colours
            case 'kA13':
                json_meta_data['version'] = meta_data[key]
            case 'kA15':
                json_meta_data['official-song'] = meta_data[key]
            case 'kA16':
                json_meta_data['auto'] = meta_data[key]
            case 'kA17':
                json_meta_data['newgrounds-song'] = meta_data[key]
            case 'kA6':
                json_meta_data['demon'] = meta_data[key]
            case 'kA7':
                json_meta_data['difficulty'] = meta_data[key]
            case 'kA17':
                json_meta_data['featured'] = meta_data[key]
            case 'kA18':
                json_meta_data['length'] = meta_data[key]
            case 'kS39':
                json_meta_data['placeholder-1'] = meta_data[key]

    level_code = []
    
    level_data = text_after(level_data, ';')
    level_tokens = level_data.split(';')

    object_conversions = {
        1: 'block',
        2: 'block-grid-edge',
        5: 'block-grid-center',
        8: 'spike-full',
        39: 'spike-half',
        36: 'orb-yellow',
        35: 'pad-yellow',
        10: 'portal-gravity-normal',
        11: 'portal-gravity-reverse',
        12: 'portal-cube',
        13: 'portal-ship',
        3847: 'collectable-star-green'
    }

    for token in level_tokens:
        tokens = token.split(',')
        
        # turn tokens into keys and values
        obj_json = {}
        for i in range(0, len(tokens), 2):
            try:
                obj_json[tokens[i]] = tokens[i+1]
            except:
                continue

        # check if it's a full object
        if '1' in obj_json:
            print("Valid OBJ") if debug == 'y' else None
        else:
            print("Invalid OBJ") if debug == 'y' else None
            continue

        # check if it's supported
        if int(obj_json['1']) not in object_conversions:
            print(f'[WARNING] Object type {obj_json["1"]} is not supported!') if debug == 'y' else None
            continue

        out_json = {}

        for key in obj_json:
            match key:
                case '1':
                    out_json['id'] = object_conversions[int(obj_json[key])]
                case '2':
                    out_json['x'] = round(int(obj_json[key]) / 30, 2)
                case '3':
                    y = int(obj_json[key]) / 30
                    # exceptions
                    if out_json['id'] == 'spike-half':
                        y -= 0.25
                    if out_json['id'] == 'pad-yellow':
                        y -= 0.4
                    out_json['y'] = round(y, 2)
                case '4':
                    out_json['flipX'] = True if int(obj_json[key]) == 1 else False
                case '5':
                    out_json['flipY'] = True if int(obj_json[key]) == 1 else False
                case '6':
                    out_json['rotation'] = int(obj_json[key])

        level_code.append(out_json)

    difficulties = {
        'beginner': 1.5,
        'easy': 2,
        'tricky': 2.5,
        'normal': 3,
        # todo: add rest of the 23 difficulties
    }

    # count spikes
    spike_count = 0
    for obj in level_code:
        if obj['id'] == 'spike-full':
            spike_count += 1
        elif obj['id'] == 'spike-half':
            spike_count += 0.5

    # count pads
    pad_count = 0
    for obj in level_code:
        if obj['id'] == 'pad-yellow':
            pad_count += 1

    # count orbs
    orb_count = 0
    for obj in level_code:
        if obj['id'] == 'orb-yellow':
            orb_count += 1

    # get the last object
    last_obj = level_code[-1]
    length = last_obj['x'] / SPEED

    diff_est = 0
    diff_est += spike_count / (length * 1)
    diff_est += pad_count / (length * 0.5)
    diff_est += orb_count / (length * 0.35)
    diff_est = round(diff_est, 2)

    # get the closest difficulty
    diff = ''
    diff_dist = math.inf

    for key in difficulties:
        if abs(difficulties[key] - diff_est) < diff_dist:
            diff_dist = abs(difficulties[key] - diff_est)
            diff = key

    stats = {
        'spikes': spike_count,
        'spikes-second': round(spike_count / length, 2),
        'orbs': orb_count,
        'orbs-second': round(orb_count / length, 2),
        'pads': pad_count,
        'pads-second': round(pad_count / length, 2),
        'length': round(length, 2),
        'difficulty_est': round(diff_est, 2),
        'difficulty': diff,
        'difficulty-margin': round(diff_dist, 2)
    }

    level_json = json.dumps({
        'name': name,
        'meta': json_meta_data,
        'stats': stats,
        'level': level_code,
        # 'raw': level_data_raw
    })

    # save level json
    with open(f'levels/json/{name}.json', 'w') as f:
        f.write(level_json)

    print(f'Level {name} parsed successfully!')