from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

app = Flask(__name__)

# Initialize HuggingFace model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

def load_documents():
    documents = []
    missing_files = []
    
    for i in range(1, 3):
        file_path = f'docs/doc{i}.txt'
        if os.path.exists(file_path):
            try:
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                missing_files.append(f"{file_path} (Error: {str(e)})")
        else:
            print(f"File not found: {file_path}")
            missing_files.append(file_path)
    
    if not documents:
        raise FileNotFoundError(f"No text documents found or could not be loaded. Missing/Error files: {', '.join(missing_files)}")
    
    print(f"Total documents loaded: {len(documents)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    print('hi')
    
    if not texts:
        print('hi ra')
        raise ValueError("No text chunks were created after splitting documents")
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    print('hi2')
    print("Vectorstore type:", type(vectorstore))
    print("Vectorstore class hierarchy:", FAISS.mro())
    return vectorstore

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Load vector store
        try:
            vectorstore = load_documents()
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return jsonify({'error': f'Error loading documents: {str(e)}'}), 500
        
        try:
            # Search for relevant document
            docs = vectorstore.similarity_search(user_message, k=3)
            print(docs)
            if not docs:
                return jsonify({'response': "I don't have enough information to answer that question."}), 200
                
            context = "\n".join([doc.page_content for doc in docs])
            
            # Prepare prompt with context
            prompt = f"Based on the following context, answer the question. If the context is not relevant, say 'I don't have enough information to answer that question.'\n\nContext: {context}\n\nQuestion: {user_message}\nAnswer:"
            
            # Generate response using the model
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(
                **inputs, 
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                no_repeat_ngram_size=2,
        
            )
            
            # Safely decode the output
            if outputs is None or len(outputs) == 0:
                return jsonify({'response': "I apologize, but I couldn't generate a response. Please try again."}), 200
                
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not response.strip():
                return jsonify({'response': "I apologize, but I couldn't generate a meaningful response. Please try again."}), 200
            
            return jsonify({'response': response})
            
        except Exception as e:
            app.logger.error(f"Error generating response: {str(e)}")
            return jsonify({'error': 'An error occurred while processing your request'}), 500
            
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    app.run(debug=True)