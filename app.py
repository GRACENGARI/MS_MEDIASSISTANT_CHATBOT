from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables (e.g., API keys)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Ensure the API key is set
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Check your .env file.")

app = FastAPI(title="MEDIASSIST CHATBOT API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize components at startup
@app.on_event("startup")
async def startup_db_client():
    app.pdf_path = "C:/Users/grace/Desktop/FABRIC/langachain-cbc-cahatbot/Untitled document-2.pdf"
    
    # Load the document
    loader = PyPDFLoader(app.pdf_path)
    data = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    app.docs = text_splitter.split_documents(data)
    
    # Create embeddings using GoogleGenerativeAIEmbeddings
    app.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Store the document embeddings in FAISS
    app.vectorstore = FAISS.from_documents(app.docs, app.embeddings)
    
    # Set up a retriever for similarity search
    app.retriever = app.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    # Initialize the Google Gemini model for the LLM
    app.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)
    
    # Define system prompt for the chatbot
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    # Create the chat prompt template
    app.prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Initialize memory dict to store conversations by session
    app.conversation_memories = {}


# Request models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    context: Optional[List[Dict[str, Any]]] = None


# Endpoints
@app.post("/query", response_model=ChatResponse)
async def query_endpoint(request: QueryRequest = Body(...)):
    try:
        # Get or create memory for this session
        if request.session_id not in app.conversation_memories:
            app.conversation_memories[request.session_id] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
        
        memory = app.conversation_memories[request.session_id]
        
        # Create the RAG chain
        question_answer_chain = create_stuff_documents_chain(app.llm, app.prompt_template)
        rag_chain = create_retrieval_chain(app.retriever, question_answer_chain)
        
        # Get relevant documents
        docs = app.retriever.get_relevant_documents(request.query)
        
        # Invoke the RAG chain and get the response
        response = rag_chain.invoke({"input": request.query})
        
        # Store conversation history
        memory.save_context({"input": request.query}, {"output": response["answer"]})
        
        # Return the response
        return ChatResponse(
            answer=response["answer"],
            context=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs[:3]]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    if session_id not in app.conversation_memories:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    memory = app.conversation_memories[session_id]
    return {"history": memory.chat_memory.messages}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in app.conversation_memories:
        del app.conversation_memories[session_id]
    return {"status": "success", "message": f"Session {session_id} deleted"}


@app.post("/upload-pdf")
async def upload_pdf():
    # This would be implemented with file upload functionality
    # For now, it's a placeholder
    return {"status": "not implemented", "message": "PDF upload not implemented yet"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)