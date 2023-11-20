from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
import re
import json
import asyncio

from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import openai
import tiktoken

def remove_number_dot(text):
    pattern = r"^\d+\.\s+"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text.strip()

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base') # gpt 모델에 인코딩(토크나이징)할 때 쓰이는 임베딩 모델
    tokens = tokenizer.encode(text)
    # print(tokens)
    return len(tokens)

def tiktoken_split(data):
    # tokenizer = tiktoken.get_encoding('cl100k_base') # gpt 모델에 tokenizing
    # tokens = tokenizer.encode(data)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=50, length_function=tiktoken_len
    )
    df = data
    new_rows = []
    for index, row in df.iterrows():
        content = row['content']
        split_data = splitter.create_documents([content])
        # print(split_data)
        for split_content in split_data : # list의 list이므로
            new_row = row.copy()
            new_row['content'] = split_content
            new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv('split_25p.csv', encoding="utf-8-sig")
    new_df.to_json('split_25p.jsonl', orient='records',lines=True,force_ascii=False)
    return new_df

def generate_question(
    title,
    country_and_era,
    size_and_category,
    content,
    model='gpt-3.5-turbo',
    timeout=50,
    max_retries=3) :
    load_dotenv()
    response = openai.ChatCompletion.create(
        model = model,
        messages = [
            {
                'role' : 'system',
                'content' : '너는 박물관 도슨트야. 제시된 작품설명 내에서 나올 수 있는 질문하나를 뽑은 뒤 그 설명을 기반으로만 답변을 생성해서 "질문: {q}, 답변: {a}" 형식으로 만들어줘.'
            },
            {
                'role' : 'user',
                'content' : f'작품설명 [\n 제목: {title} \n 국적/시대 : {country_and_era}, 크기/구분:{size_and_category}, \n 해설:{content}]'
            }
        ]
    )
    retries = 0
    while retries <= max_retries :
        if response.choices :
            try :
                questions = response.choices[0].message['content'].strip()
                return questions
            except Exception as e:
                print(f"{e} question 생성 실패 오류 발생, 재시도 중 ...")
                retries += 1
                if retries > max_retries:
                    print(f"재시도 횟수 초과. 다음 요청으로 넘어갑니다. {e}")
                    return []
                print(f"{retries}/{max_retries}회 시도중.. 에러 : {e}")
        else :
            retries += 1
            if retries > max_retries:
                print(f"TimeOut 재시도 횟수 초과. 다음 요청으로 넘어갑니다. {e}")
                return []
            print(f"TimeOut 재시도중.. {retries}/{max_retries}회 에러 : {e}")

async def main(data):
    qa_pairs = []
    last_processed = 0
    try :
        data = tiktoken_split(data)
    except Exception as tiktoken_e :
        print(f'토큰 Split 에러: {tiktoken_e}')
    try:
        for index, row in tqdm(data.iterrows()):
            print('생성중... 현재상태 ->',row['title'])
            questions = generate_question(
                row["title"], row["country_and_era"], row['size_and_category'], row["content"]
            )
            qa_pairs.append({'question':remove_number_dot(questions), 'title':row["title"], 'index':index, 'content':row['content']})
            last_processed = index

            # 실험용
            # if index >= 2 : 
            #     break
    except Exception as e:
        print(f"에러 발생 (last_processed index = {last_processed}): {e}")
    return qa_pairs, last_processed


data = pd.read_csv("25p.csv")

qa_pairs, last_processed = asyncio.run(main(data))
qa_df = pd.DataFrame(qa_pairs, columns=["question", 'title', 'index', 'content'])
qa_df.to_csv("qa_25p_to_31p.csv", index=False, encoding="utf-8-sig")
qa_df.to_json(
    "qa_25p_to_31p_records.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)