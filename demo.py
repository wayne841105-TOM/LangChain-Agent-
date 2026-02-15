import os
#導入datetime模組以使用日期工具
import datetime
#導入工具裝飾器
from langchain_core.tools import tool
#導入.env檔案的API
from dotenv import load_dotenv
load_dotenv()
#導入GOOGLE向量數據庫及聊天模組
from langchain_google_genai import( GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI)
#導入langchain文字切割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
#導入Chromadb
from langchain_chroma import Chroma
#langchain導入dox檔案
from langchain_community.document_loaders import Docx2txtLoader
#導入langchain提示模板、可運行對象和輸出解析器
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#取得.env中的API
google_gemini_api_key = os.getenv("GOOGLE_API_KEY")
loader = Docx2txtLoader("葉綠宿集團介紹.docx")
ducuments = loader.load()
#使用文字切割器將文本切割成小段落
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每段的最大字數
    chunk_overlap=10,  # 段落之間的重疊字數
    #正則表達式套用標點符號和換行符號作為切割依據
    # separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
ducuments = text_splitter.split_documents(ducuments)
#建立google向量模型
embeddings = GoogleGenerativeAIEmbeddings(api_key=google_gemini_api_key, model= "gemini-embedding-001")
#將切割後的文本轉換為向量並存儲在Chroma向量數據庫中
vector_store = Chroma.from_documents(ducuments, embeddings)
#使用向量數據庫作為檢索器
retriever = vector_store.as_retriever()
#定義一個工具獲取日期
# @tool
# def date_tool():
#     """獲取今天日期"""
#     return datetime.date.today().strftime("%Y年%m月%d日") 


#大模型客戶端
client = ChatGoogleGenerativeAI(
    api_key=google_gemini_api_key, 
    model ="gemini-2.5-flash",
    temperature=0.4
    )
# client_with_tools = client.bind_tools([date_tool])
#建立提示模板
template="""
你是一個專業的助理。請根據下方提供的「檢索內容」來回答用戶的問題。
如果內容中沒有提到，請回答「我不清楚」，不要隨便編造。

檢索內容：
{context}

用戶問題：
{question}

回答：
"""
prompt = ChatPromptTemplate.from_template(template)
#建立鏈
chain = (
    {"context": retriever, "question": RunnablePassthrough(),}
    | prompt
    | client
    | StrOutputParser()
)
user_qusetion = "公司在做什麼的?"
response  = chain.invoke(user_qusetion)

print("AI回答:", response)
