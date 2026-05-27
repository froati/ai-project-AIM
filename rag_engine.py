import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List

load_dotenv()

class RetrievalResponse(BaseModel):
    Reasoning: str = Field(description="검색의 필요유무를 추론하는 과정")
    Retrieve: Literal['Yes', 'No'] = Field(description="검색 필요유무")

class RelevanceResponse(BaseModel):
    Reasoning: str = Field(description="연관문서의 관련성 평가 추론과정")
    ISREL: Literal['Relevant', 'Irrelevant'] = Field(description="관련성 평가 결과")

class GenerationResponse(BaseModel):
    response: str = Field(description="생성된 답변")

class SupportResponse(BaseModel):
    Reasoning: str = Field(description="답변이 문서에 근거하는지 평가")
    ISSUP: Literal['Fully supported', 'Partially supported', 'No support'] = Field(description="지원 평가 결과")

class InvestmentRAGEngine:
    def __init__(self, pdf_paths: List[str]):
        self.pdf_paths = pdf_paths
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        self.vector_db = self._setup_vector_db()
        
    def _setup_vector_db(self):
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for path in self.pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                docs = loader.load_and_split(text_splitter)
                all_docs.extend(docs)
        
        if all_docs:
            return FAISS.from_documents(all_docs, self.embeddings)
        return None

    def process_query(self, query: str):
        # 1. Retrieval Decision
        # (Simplified for now, always search if vector_db exists)
        if not self.vector_db:
            return "분석할 문서가 없습니다.", []

        # 2. Retrieve
        docs = self.vector_db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # 3. Generate with Gemini
        prompt = PromptTemplate.from_template("""
        당신은 삼성전자 투자 전문 비서입니다. 아래 제공된 컨텍스트를 바탕으로 사용자의 질문에 답변하세요.
        답변은 친절하고 전문적이어야 하며, 반드시 제공된 정보에 근거해야 합니다.
        
        질문: {query}
        컨텍스트: {context}
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"query": query, "context": context})
        
        sources = [{"title": doc.metadata.get('source', '알 수 없음'), "page": doc.metadata.get('page', 0)} for doc in docs]
        
        return response.content, sources
