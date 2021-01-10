# DCSA Project - MapReduce

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Task instructions](#task-instructions)
    1. [IMDB](#1.-imdb)
    2. [Online Retail](#2.-online-retail)
    3. [Similar Paper Recommendations](#3.-similar-paper-recommendations)
    4. [Matrix Multiplication](#4.-matrix-multiplication)

## Prerequisites

As a prerequisite you will need to install [Python 3](https://www.python.org/). The code was tested with
versions `3.7.3` and `3.8.7`. Python 3 comes bundled with the package installer `pip` but if it is not available on your
machine after installing Python 3, install it by following
the [Pip Documentation](https://pip.pypa.io/en/stable/installing/).

You will need the code in this repository and the provided data files. If you cloned the repository, you should have the
corresponding folder structure. Put your data files provided at the course within this folder structure the following
way (only provided data files are listed):

    .
    ├── 1_IMDB
    │   └── title.basics.tsv
    ├── 2_RETAIL
    │   ├── retail0910.csv
    │   └── retail1011.csv
    ├── 3_TEXT-SIMILARITY
    │   └── arxivData.json
    └── 4_MATRIX
        ├── A.txt
        ├── B.txt
        └── C.txt

## Installation

First, a generally good practice when working with Python is to create a virtual environment (venv) to run the code in.
In order to create a virtual environment please refer to
the [official documentation](https://docs.python.org/3.7/tutorial/venv.html).

To install project packages run the following command (make sure your venv is activated):
```
pip install -r requirements.txt
```

## Task instructions

### 1. IMDB

An additional package has to be installed with:
```
python -m spacy download fr_core_news_sm
```

The code corresponding to this category is in folder `1_IMDB`.

Run task 1 with:
```
python imdb_task1.py title.basics.tsv
```

Run task 2 with:
```
python imdb_task2.py title.basics.tsv
```

### 2. Online Retail

The code corresponding to this category is in folder `2_RETAIL`.

Run task 3 with:
```
python retail_task3.py retail0910.csv
python retail_task3.py retail1011.csv
```

At task 4 uncomment the line in `__main__` which you want to run and run it with:
```
python retail_task4.py retail0910.csv retail1011.csv
```

### 3. Similar Paper Recommendations

The code corresponding to this category is in folder `3_TEXT-SIMILARITY`.

Reformat the input JSON file with:
```
python json_converter.py
```

Select a random paper:
```
python random_paper_selector.py
```

Input the random paper into the task script and run it with:
```
python text_similarity_task5.py arxivData_lines.json
```

### 4. Matrix Multiplication

The code corresponding to this category is in folder `4_MATRIX`.

Generate new data files with:
```
python i.py
```

Run the mrjob task with:
```
python matrix_task6.py A.txt B.txt > C_computed.txt
```

Verify the validity of your matrix dot product with:
```
python result_validator.py
```

