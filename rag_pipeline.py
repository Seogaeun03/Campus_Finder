import os
from dotenv import load_dotenv

# ✅ 변경된 import 경로 반영
from langchain_community.vectorstores import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# ✅ .env 파일 불러오기
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")

if not api_key:
    raise ValueError("❌ Upstage API 키가 설정되지 않았습니다. .env 파일을 확인하세요!")

# ✅ 1. 임베딩 모델 (Solar Embedding)
embedding = UpstageEmbeddings(model="solar-embedding-1-large")

# ✅ 2. 크롤링된 텍스트 파일 불러오기
folder_path = "Result_crawling"
documents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

# ✅ 3. 텍스트를 chunk 단위로 나누기
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = []
for doc in documents:
    texts.extend(splitter.split_text(doc))

# ✅ 4. Chroma DB 생성 및 저장
vectorstore = Chroma.from_texts(texts, embedding=embedding, persist_directory="chroma_db")
print("✅ 벡터 DB 생성 완료!")

# ✅ 5. Solar Pro 모델로 QA Chain 구성
llm = ChatUpstage(model="solar-pro")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ✅ 6. 사용자 입력 받아서 질의응답
print("\n🎓 캠퍼스 파인더 RAG 챗봇 시작!")
while True:
    query = input("\n질문을 입력하세요 (종료하려면 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa_chain.invoke({"query": query})
    print(f"\n💬 답변: {result['result']}")
