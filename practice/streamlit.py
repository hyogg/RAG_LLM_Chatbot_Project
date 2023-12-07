from dotenv import load_dotenv
load_dotenv() # 파이썬이 환경변수 (.env) 불러와서 쓰게 됨

# Complete Model
# from langchain.llms import OpenAI
# llm = OpenAI()
# result = llm.predict('내가 좋아하는 동물은')

# Chat Model
# from langchain.chat_models import ChatOpenAI
# chat_model = ChatOpenAI()
# content = '코딩'

# result =chat_model.predict(content+'에 대해 20자짜리 시를 써줘.')
# print(result)

import streamlit as st
# st.title('This is a title')
# st.title('_streamlit_ is :blue[cool] :sunglasses:')
# title = st.text_input('시의 주제를 제시해주세요 👇')
# st.write(title)

from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI()
st.title('인공지능 시인과 대화하기')
content = st.text_input('시의 주제를 제시해주세요.')
if st.button('요청하기') :
    with st.spinner('생성 중입니다...'):
        result = chat_model.predict(content+'에 관한 20자짜리 시를 써줘.')
        st.write(result)

# st.write("시의 주제는 ", content)
# st.write('내용은 :', result)