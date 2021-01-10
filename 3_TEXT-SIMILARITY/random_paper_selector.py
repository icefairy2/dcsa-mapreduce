from datetime import datetime
import json
from random import seed, randint

"""
Use this script to print a random paper from the input file
"""

INPUT_FILE = 'arxivData.json'

# Needed to have seemingly 'true' random values
seed(datetime.now())

with open(INPUT_FILE) as f:
    papers = json.load(f)
    length = len(papers)

    f.close()

    random_index = randint(0, length)
    random_paper = papers[random_index]
    pretty_printed_json = json.dumps(random_paper, indent=2)

    print(pretty_printed_json)
