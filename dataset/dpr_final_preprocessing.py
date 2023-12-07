import transformers
from datasets import Dataset
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import json
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
class CustomBM25(BM25Okapi):
     def get_top_n(self, query, documents, ctxids, n=10):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_docs_indices = np.argsort(scores)[::-1][:n]
        top_ctx_idx = [ctxids[index] for index in top_docs_indices]
        top_docs = [documents[index] for index in top_docs_indices]
        return top_ctx_idx, top_docs

class preprocess_dataset():
    def __init__(self, file_name, tokenizer):
        self.ctx_ids, self.contexts, self.questions, self.bm25 = [], [], [], None
        self.tokenizer = tokenizer
        self.get_json_lines(file_name, tokenizer)

    def get_json_lines(self, file_name, tokenizer):
        tokenized_corpus = []
        with open(file_name, 'r') as f:
            print('get_json_lines 시작..')
            for data in tqdm(f) :
                line = json.loads(data)
                passage = line['context']

                # ctx_id가 set에 없으면 passage 추가
                if line['ctx_id'] not in self.ctx_ids :
                    self.ctx_ids.append(line['ctx_id'])
                    self.contexts.append(passage)
                    self.questions.append([])
                    tokenized_corpus.append(tokenizer(line['context'])['input_ids'])
                    # tokenized_corpus -> BM25 생성자 input
                    self.bm25 = CustomBM25(tokenized_corpus)
                
                # 현재 context에 매칭된 question list에 해당 줄의 question append하기
                self.questions[-1].append(line['question'])

    # using bm25, return top_doc_index
    def get_negative_passage_bm25(self, question_row, question_column):
        tokenized_question = self.tokenizer(self.questions[question_row][question_column])['input_ids']
        top_ctx_idx, top_docs = self.bm25.get_top_n(query=tokenized_question, documents=self.contexts, ctxids=self.ctx_ids, n=10)
        
        # 디버깅용
        # print("Top 10 Context Indices", top_ctx_idx)
        # print("Top 5 Documents:", top_docs[:4])

        return top_ctx_idx
        
    def make_dataset(self):
        question_data, pos_passage_data, hard_neg_passage_data = [], [], []
        for row, questions in tqdm(enumerate(self.questions), desc='question[row] processing.. in make_dataset'):
            for col, question in enumerate(questions): # 너무 빨라서 tqdm 제거함
                # print(f'{row}번째 리스트 중 {col}번째 질문 데이터 처리중..') # 매우 빠르므로 주석 해제하지 말것..
                top_ctx_ids = self.get_negative_passage_bm25(row, col)
                real_ctx_idx = self.ctx_ids[row] # 해당 question이 속한 list의 index를 찾으면 진짜 ctx_idx도 알 수 있음. [[q1=col1, q2=col2, q3=col3..]=row1, [q4=col1, q5=col2, q6=col3..]=row2...] 이런 구조이므로.

                ''' BM25로 찾았는데 real_ctx_idx가 top_ctx_idx 10개 중에 하나도 안 나오면? -> 쿼리를 버려야 하나? 고민중.. 일단은 가져가는걸로 '''

                # 진짜 context index가 top_ctx_idx에 있으면 리스트에서 제외시켜주기
                if real_ctx_idx in top_ctx_ids :
                    top_ctx_ids.remove(real_ctx_idx)

                # 1. 유사도 높게 나온 context의 ctx_id들의 index를 추출하여 리스트로 만들어주고
                # 2. 같은 index number의 contexts를 negative passage로 넣어줄 예정 (어짜피 self.contexts <-> self.ctx_ids는 페어가 맞춰져 있음.)
                top_ctx_indices = [self.ctx_ids.index(id) for id in top_ctx_ids]
                for idx in top_ctx_indices :
                    question_data.append(question)
                    pos_passage_data.append(self.contexts[row])
                    hard_neg_passage_data.append(self.contexts[idx])

        # shuffle 왜 해줘야하지? 문서 순서대로 넣어줘야 하는거 아닌가
        # index_list = list(range(len(question_data)))
        # random.shuffle(index_list)

        # questions = [question_data[idx] for idx in index_list]
        # pos_passages = [pos_passage_data[idx] for idx in index_list]
        # negative_passages = [negative_passage_data[idx] for idx in index_list]

        return Dataset.from_pandas(pd.DataFrame({'question': question_data,
                                                    'pos_passage': pos_passage_data,
                                                    'hard_neg_passage': hard_neg_passage_data}))


def save_dataset_to_files(dataset, file_prefix):
    dataset.to_csv(f'./museum_dataset/{file_prefix}_dataset.csv', encoding='utf-8-sig')
    dataset.to_json(f'./museum_dataset/{file_prefix}_dataset.jsonl', lines=True, force_ascii=False, orient='records')
    dataset.save_to_disk(f'./museum_dataset/{file_prefix}_dataset')


# sample_dataset_process = preprocess_dataset('./museum_dataset/sample.jsonl', tokenizer)
train_dataset_process = preprocess_dataset('./museum_dataset/train.jsonl', tokenizer)
eval_dataset_process = preprocess_dataset('./museum_dataset/test.jsonl', tokenizer)

# sample_dataset = sample_dataset_process.make_dataset()
train_dataset = train_dataset_process.make_dataset()
eval_dataset = eval_dataset_process.make_dataset()

# save_dataset_to_files(sample_dataset, 'sample')
save_dataset_to_files(train_dataset, 'train')
save_dataset_to_files(eval_dataset, 'eval')