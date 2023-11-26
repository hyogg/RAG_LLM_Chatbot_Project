from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import pandas as pd
import tiktoken
import json

class generate_question :   
    def __init__(self, jsonl_input_path: str = "./files/museum_passage.jsonl", jsonl_output_path: str = "./files/passage_split.jsonl") :
        '''
        jsonl_input_path: 크롤링 데이터 원본을 담고 있는 JSONL 파일의 경로
        jsonl_output_path : 파일을 저장할 경로  
        '''
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_output_path = jsonl_output_path

    def load_data(self):
        '''
        JsonL 파일을 로드해서 리스트로 리턴하는 함수
        '''
        data_list = []
        with open(self.jsonl_input_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                data_list.append(json_data)
        return data_list

    def tiktoken_len(self, text, encoding_name="cl100k_base"):
        '''
        tiktoken library를 활용해 토큰 길이를 계산하는 함수
        1. text : 길이 측정할 텍스트
        2. encoding_name : 토큰으로 변환할 방식 지정
            - cl100k_base : gpt-3.5-turbo, gpt-4, text-embedding-ada-002 모델 사용시
            - p50k_base : text-davinci-002, text-davinci-003 모델 사용시
            - r50k_base (or gpt2) : GPT-3 models like "davinci"
        '''
        tokenizer = tiktoken.get_encoding('cl100k_base')
        # tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo") # 위 코드로 안될 시 모델 이름 넣어서 이걸로 해보기
        tokens = tokenizer.encode(text)
        return len(tokens)

    def tiktoken_split(self, data):
        '''
        원본 크롤링 파일 (.jsonl) 문장에서
        설명이 2000토큰이 넘어갈 경우 Split함
        data : 크롤링 파일
        jsonl_output_path : 파일을 저장할 경로  
        '''
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=100)
        new_lines = []

        for line in data:
            # description 부분만 받아오기 -> 없으면 KeyError 발생
            try :
                text = line['description']
            except KeyError:
                raise KeyError("작품 설명의 key값을 'description'으로 설정하세요.")
            token_len = tiktoken_len(text)

            # 토큰 길이가 2000개 넘을 경우에만 문서 Split
            if token_len > 2000 :
                split_data = splitter.split_text(text)
                for single_split_data in split_data :
                    new_line = line.copy()
                    new_line['description'] = single_split_data
                    new_lines.append(new_line)
                    # print(f'split... -> {single_split_data}')
            else :
                new_lines.append(text)
        
        # new_lines를 jsonl 파일로 저장
        with open(self.json_output_path, 'w', encoding='utf-8') as jsonl_file :
            for dict in new_lines:
                line = json.dump(dict)
                jsonl_file.write(line + '\n')
        return new_lines

    # 실행을 위한 함수
    def process(self):
        try :
            data = self.load_data()
            new_data = self.tiktoken_split(data)
        except Exception as e :
            print('Error 발생.. -> {e}')