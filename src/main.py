import os
import uuid
import pdfplumber
import pytesseract
from PIL import Image
from typing import Dict
from FlagEmbedding import FlagReranker  # reranker model
from flask import Flask, request, jsonify, send_file
from fastapi.responses import HTMLResponse
from pdfminer.pdfparser import PDFSyntaxError
from langchain_community.vectorstores import Chroma
from CustomBGEM3FlagModel import CustomBGEM3FlagModel  # embedding model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

L_chunk_size = 1024
M_chunk_size = 512
S_chunk_size = 256

L_overlap = 384
M_overlap = 192
S_overlap = 96

reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)

loader = DirectoryLoader(
    "src/txtfiles", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

# sort the documents by filename
docs = sorted(docs, key=lambda x: x.metadata["source"].split("/")[-1])

# for i, doc in enumerate(docs):
#     # text = doc.page_content
#     # title = f"article{str(i).zfill(3)}"
#     # doc.metadata["title"] = title
#     doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]

text_splitter_L = RecursiveCharacterTextSplitter(
    chunk_size=L_chunk_size, chunk_overlap=L_overlap)
text_splitter_M = RecursiveCharacterTextSplitter(
    chunk_size=M_chunk_size, chunk_overlap=M_overlap)
text_splitter_S = RecursiveCharacterTextSplitter(
    chunk_size=S_chunk_size, chunk_overlap=S_overlap)
splits_L = text_splitter_L.split_documents(docs)
splits_M = text_splitter_M.split_documents(docs)
splits_S = text_splitter_S.split_documents(docs)

embedding_function = CustomBGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
storedb = "./vectorstore_L"
if os.path.exists(storedb):
    vectorstore_L: Chroma = Chroma(
        embedding_function=embedding_function, persist_directory=storedb,
        collection_metadata={"hnsw:space": "cosine"},
    )
else:
    vectorstore_L = Chroma.from_documents(
        documents=splits_L,
        persist_directory=storedb,
        embedding=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )

storedb = "./vectorstore_M"
if os.path.exists(storedb):
    vectorstore_M: Chroma = Chroma(
        embedding_function=embedding_function, persist_directory=storedb,
        collection_metadata={"hnsw:space": "cosine"},
    )
else:
    vectorstore_M = Chroma.from_documents(
        documents=splits_M,
        persist_directory=storedb,
        embedding=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )
storedb = "./vectorstore_S"
if os.path.exists(storedb):
    vectorstore_S: Chroma = Chroma(
        embedding_function=embedding_function, persist_directory=storedb,
        collection_metadata={"hnsw:space": "cosine"},
    )
else:
    vectorstore_S = Chroma.from_documents(
        documents=splits_S,
        persist_directory=storedb,
        embedding=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'src/uploads'
TEXT_FOLDER = 'src/txtfiles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'src/uploads')
TEXT_FOLDER = os.path.join(os.getcwd(), 'src/txtfiles')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEXT_FOLDER'] = TEXT_FOLDER

@app.route('/similarity_search', methods=['GET'])
def similarity_search() -> Dict:
    query = request.args.get('query')
    # Default to 5 if top_k is not provided
    top_k = int(request.args.get('top_k', 5))

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = search_langchain_article(query, top_k)
    return jsonify(results)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        text = extract_text_from_pdf(filepath)

        if isinstance(text, dict) and 'error' in text:
            return jsonify({'error': text['error']}), 500

        text_filename = file.filename.replace('.pdf', '.txt')
        text_path = os.path.join(app.config['TEXT_FOLDER'], text_filename)

        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        (L, M, S) = load_documents(text_path)
        vectorstore_L.add_documents(L)
        vectorstore_M.add_documents(M)
        vectorstore_S.add_documents(S)

        return jsonify({
            'message': f'File {file.filename} uploaded and processed successfully',
            'text_file': text_path
        }), 200
    else:
        return jsonify({'error': 'Only PDF files are allowed'}), 400

@app.route('/get_full_pdf', methods=['GET'])
def get_full_pdf():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename parameter is required'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    # Serve the PDF as a binary file
    return send_file(file_path, as_attachment=False, mimetype='application/pdf')

def extract_text_from_pdf(pdf_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                
    except PDFSyntaxError as e:
        error_message = f"Error processing PDF file: {e}"
        print(error_message)
        return {"error": error_message}
    except Exception as e:
        error_message = f"Unexpected error: {e}"
        print(error_message)
        return {"error": error_message}
    
    return full_text


def load_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    doc = loader.load()
    print(doc)
    doc = sorted(doc, key=lambda x: x.metadata["source"].split("/")[-1])

    text_splitter_L = RecursiveCharacterTextSplitter(
        chunk_size=L_chunk_size, chunk_overlap=L_overlap)
    text_splitter_M = RecursiveCharacterTextSplitter(
        chunk_size=M_chunk_size, chunk_overlap=M_overlap)
    text_splitter_S = RecursiveCharacterTextSplitter(
        chunk_size=S_chunk_size, chunk_overlap=S_overlap)

    splits_L = text_splitter_L.split_documents(doc)
    splits_M = text_splitter_M.split_documents(doc)
    splits_S = text_splitter_S.split_documents(doc)

    return splits_L, splits_M, splits_S


def search_langchain_article(query: str, top_k: int = 3) -> Dict:
    matches_all_sizes = []
    matches_all_sizes.append(
        vectorstore_L.similarity_search_with_score(query, k=top_k))
    matches_all_sizes.append(
        vectorstore_M.similarity_search_with_score(query, k=top_k))
    matches_all_sizes.append(
        vectorstore_S.similarity_search_with_score(query, k=top_k))
    results = []

    for matches in matches_all_sizes:
        for match in matches:
            obj = dict()
            obj["chunk"] = match[0].page_content
            obj["filename"] = match[0].metadata["source"].split("/")[-1]
            obj["score"] = reranker.compute_score([query, obj["chunk"]])
            
            file_path = os.path.join(app.config['TEXT_FOLDER'], obj["filename"])
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                obj["content"] = content
            
            results.append(obj)
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    results = [result for result in results if result['score'] is not None]

    return {"results": results}


if __name__ == '__main__':
    app.run(debug=True)
