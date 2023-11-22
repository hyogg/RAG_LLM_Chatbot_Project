import pandas as pd

def csv_parsing(file_path) :
    df = pd.read_csv(file_path)


def main(file_path) :
    df = csv_parsing(file_path)
    