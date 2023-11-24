import glob
import json
import csv
import time

start = time.time()
from flatten_json import flatten

def convert_to_csv(input_file_path:str = './files/data_v3.jsonl', output_file_path:str = './files/passage_dataset.csv', headers:list[str] = ['title', 'link', 'era', 'info', 'description']) :
    try :
        '''
        input_file_path : Path of JSON file (input)
        output_file_path : Path of CSV file (output)
        headers : In the order -> [title, link, era, info(size, number, category.. etc), description(context-paragraph)]
        '''
        with open (output_file_path, 'w', newline='') as c :
            csvwriter = csv.writer(c, delimiter='|')
            csvwriter.writerow(headers)

        with open(input_file_path, 'r') as f :
            for line in f :
                print('Processing line:', line)
                data = json.loads(line)
                row = flatten(data)
                csvwriter.writerow([row.get(header, '') for header in headers])
        print('Conversion completed successfully.')
        input('Press Enter to exit...')
    except Exception as e :
        print(f'An error occurred : {e}')

