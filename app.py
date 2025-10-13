import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ====================================
# 🌟 기본 설정
# ====================================

st.set_page_config(
    page_title="🎓 캠퍼스 파인더 RAG 챗봇",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fc;
    }
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================================
# ⚙️ API 키 로드
# ====================================

load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    st.error("❌ Upstage API 키가 설정되지 않았습니다. .env 파일을 확인하세요!")
    st.stop()

# ====================================
# 🧠 벡터 DB 로드 함수
# ====================================

@st.cache_resource
def load_vectorstore():
    folder_path = "Result_crawling"
    documents = []

    # ✅ 1. 크롤링된 텍스트 읽기
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())

    # ✅ 2. 제목 단위로 분리 (제목 + 본문 묶음) + 길이 초과 방지
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = []

    for doc in documents:
        sections = doc.split("[제목]")
        for section in sections:
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n")
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            combined = f"[제목]{title}\n{body}"

            # 👇 너무 긴 청크는 splitter로 다시 잘라서 추가
            for chunk in splitter.split_text(combined):
                texts.append(chunk)


    # ✅ 3. 벡터 임베딩 생성
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma.from_texts(texts, embedding=embedding, persist_directory="chroma_db")
    return vectorstore

# ====================================
# 🚀 사이드바 UI
# ====================================

st.sidebar.header("⚙️ 설정 및 기능")

# 🔁 DB 다시 생성 버튼
rebuild = st.sidebar.button("🔁 DB 다시 생성하기")
if rebuild:
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        st.sidebar.success("✅ 기존 DB 삭제 완료! 새로 생성 중...")
    load_vectorstore.clear()
    vectorstore = load_vectorstore()
    st.sidebar.success("🎉 새 DB 생성 완료!")

# 💡 예시 질문
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **예시 질문**")
example_questions = [
    "전공 마이크로모듈이란?",
    "전공 마이크로모듈의 구성 방법은?",
    "이수구분이란 무엇인가요?",
    "학과별 전공필수 교과목은 몇 학점인가요?",
]
for q in example_questions:
    if st.sidebar.button(q):
        st.session_state["example_query"] = q

# ====================================
# 💬 메인 챗봇 UI
# ====================================

st.title("🎓 캠퍼스 파인더 RAG 챗봇")
st.markdown("학교 공식 페이지 데이터를 기반으로 정확한 정보를 제공합니다 🏫")

# ✅ 벡터스토어 로드
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 검색 폭 확장
llm = ChatUpstage(model="solar-pro")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
)

# ✅ 대화 히스토리 관리
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ 입력
query = st.chat_input("질문을 입력하세요...")

if "example_query" in st.session_state:
    query = st.session_state.pop("example_query")

if query:
    with st.spinner("답변 생성 중입니다... 🔍"):
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])
        st.session_state.chat_history.append({
            "user": query,
            "bot": answer,
            "sources": [d.metadata.get("source", "(unknown)") for d in sources]
        })

# ✅ 대화 표시
for chat in st.session_state.chat_history:
    st.chat_message("user").markdown(f"**🙋‍♂️ 질문:** {chat['user']}")
    st.chat_message("assistant").markdown(f"**🤖 답변:** {chat['bot']}")
    if chat["sources"]:
        with st.expander("📚 참고 문서"):
            for s in chat["sources"]:
                st.markdown(f"- {s}")

# ✅ 저장 버튼
if st.sidebar.button("💾 대화 내용 저장"):
    with open("chat_history.txt", "w", encoding="utf-8") as f:
        for chat in st.session_state.chat_history:
            f.write(f"[USER] {chat['user']}\n[AI] {chat['bot']}\n\n")
    st.sidebar.success("💾 대화 내용이 'chat_history.txt'로 저장되었습니다!")
