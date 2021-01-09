import json

INPUT_FILE = 'arxivData.json'
OUTPUT_FILE = 'arxivData_lines.json'

with open(INPUT_FILE) as f:
    data = json.load(f)

    with open(OUTPUT_FILE, 'w') as json_file:
        for entry in data:
            json.dump(entry, json_file)
            json_file.write('\n')
