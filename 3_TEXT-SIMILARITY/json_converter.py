import json

with open('arxivData.json') as f:
    data = json.load(f)

    with open('arxivData_lines.json', 'w') as json_file:
        for entry in data:
            json.dump(entry, json_file)
            json_file.write('\n')
