# RAG-and-LANGCHAIN-Demo
# Custom CBC Chatbot

## Overview
This repository contains a Streamlit-based chatbot powered by Google Gemini AI and FAISS vector search. The chatbot utilizes Retrieval-Augmented Generation (RAG) to answer user queries based on the content of a PDF document.

## Features
- **Document Embedding:** Converts a PDF document into vector embeddings using GoogleGenerativeAIEmbeddings.
- **Vector Search:** Stores document embeddings in FAISS for fast retrieval.
- **Chatbot Interface:** Provides a simple UI using Streamlit.
- **Retrieval-Augmented Generation (RAG):** Uses a retriever to fetch relevant document sections and generate responses.
- **Memory:** Retains conversation history using ConversationBufferMemory.

## Prerequisites
Before running the application, ensure you have the following installed:

- Python 3.8+
- pip (Python package manager)

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the project directory.
   - Add the following line:
     ```sh
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## Dependencies
The following libraries are required:

```sh
pip install streamlit langchain langchain_community langchain_google_genai faiss-cpu python-dotenv
```

## Project Structure
```
|-- app.py                # Main application script
|-- requirements.txt      # Required Python libraries
|-- .env                  # Environment variables
|-- README.md             # Documentation
```

## How It Works
### 1. Load and Process Document
- Loads the specified PDF file using `PyPDFLoader`.
- Splits the document into chunks using `RecursiveCharacterTextSplitter`.

### 2. Create Embeddings and Vector Store
- Converts text chunks into embeddings using `GoogleGenerativeAIEmbeddings`.
- Stores the embeddings in a FAISS vector database for similarity search.

### 3. Initialize Chatbot Components
- Uses `ChatGoogleGenerativeAI` as the main LLM.
- Defines a retrieval function to fetch the most relevant document chunks.
- Implements memory using `ConversationBufferMemory`.

### 4. User Interaction and Response Generation
- The user inputs a query in the Streamlit app.
- The system retrieves relevant document sections and generates a response using `ChatGoogleGenerativeAI`.
- The response is displayed in the UI, and conversation history is retained.

## Running the Application
Run the Streamlit app using:
```sh
streamlit run app.py
```

## Troubleshooting
- Ensure your `.env` file contains a valid `GOOGLE_API_KEY`.
- If FAISS installation fails, try:
  ```sh
  pip install faiss-cpu
  ```
- If `langchain_google_genai` throws an error, ensure you have the latest version installed:
  ```sh
  pip install --upgrade langchain_google_genai
  ```

## Future Improvements
- Implement user authentication.
- Enhance UI/UX.
- Allow multiple document uploads.

## License
This project is licensed under the MIT License.

## Author
Ngari Grace


