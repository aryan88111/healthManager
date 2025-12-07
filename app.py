import os
import numpy as np
import lancedb
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tempfile

load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
# Check for API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment variables. Chat features will not work.")

# --- AI/ML SECTION (Legacy Scikit-Learn Model) ---
train_data = [
    ("I have a headache and blurry vision", "Neurologist"),
    ("My stomach hurts after eating spicy food", "Gastroenterologist"),
    ("I feel sad and anxious all the time", "Psychiatrist"),
    ("My joints are stiff in the morning", "Rheumatologist"),
    ("I have a rash that won't go away", "Dermatologist"),
    ("My heart is racing and I feel dizzy", "Cardiologist"),
    ("I have a persistent cough and fever", "Pulmonologist"),
    ("Doctor A says surgery, Doctor B says meds for my knee", "Orthopedist (Second Opinion)"),
    ("Lost my blood report but I remember high sugar", "Endocrinologist"),
    ("Internet says it's cancer but I just have a mole", "Dermatologist (Screening)"),
    ("Conflicting advice on diet for diabetes", "Dietitian / Endocrinologist"),
    ("Back pain radiating to leg", "Orthopedist / Neurologist"),
    ("Child has high fever and rash", "Pediatrician"),
    ("Toothache and swollen gums", "Dentist"),
    ("Trouble hearing in one ear", "ENT Specialist")
]

X_train = [x[0] for x in train_data]
y_train = [x[1] for x in train_data]

model = make_pipeline(CountVectorizer(), MultinomialNB())
print("Training Basic Health Manager AI Model...")
model.fit(X_train, y_train)
print("Basic Model Trained Successfully!")

# --- LANGCHAIN & LANCEDB SECTION ---
# Initialize LanceDB
db = lancedb.connect("data/lancedb")

# Create dummy knowledge base for RAG
knowledge_base = [
    {"text": "Dr. Smith recommends surgery for ACL tears if the patient is an athlete.", "source": "Dr. Smith's Notes"},
    {"text": "Dr. Jones suggests physical therapy for ACL tears for non-athletes.", "source": "Dr. Jones' Notes"},
    {"text": "Normal fasting blood sugar is between 70 and 100 mg/dL.", "source": "General Medical Guidelines"},
    {"text": "Generic internet advice often confuses tension headaches with migraines.", "source": "Health Blog"},
    {"text": "If you lost your report, check the patient portal or ask the lab for a reprint.", "source": "Admin Guide"},
    {"text": "Conflicting doctors usually require a third neutral opinion or a specialist in that specific sub-field.", "source": "Best Practices"}
]

qa_chain = None
vector_store = None
embeddings = None
init_error = None

def initialize_rag():
    global qa_chain, vector_store, embeddings, init_error
    if GOOGLE_API_KEY:
        try:
            # 1. Embeddings (Switched to Local HuggingFace to avoid Quota limits)
            print("Loading local embedding model (this might take a moment)...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # 2. Create Table in LanceDB
            # We'll use a simple DataFrame loader for this demo
            df = pd.DataFrame(knowledge_base)
            loader = DataFrameLoader(df, page_content_column="text")
            docs = loader.load()
            
            # 3. Create Vector Store
            # LanceDB integration with LangChain
            table_name = "health_knowledge"
            try:
                db.drop_table(table_name)
            except:
                pass
                
            vector_store = LanceDB.from_documents(
                documents=docs,
                embedding=embeddings,
                connection=db,
                table_name=table_name
            )
            
            # 4. LLM
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.3)
            
            # 5. Retrieval Chain
            prompt_template = """You are The Health Manager AI. Your goal is to connect the dots between conflicting medical advice, lost reports, and generic internet information.
            
            Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT}
            )
            print("LangChain + Gemini + LanceDB initialized successfully!")
            init_error = None
            
        except Exception as e:
            print(f"Error initializing LangChain: {e}")
            init_error = str(e)
    else:
        init_error = "GOOGLE_API_KEY not found in environment variables."

initialize_rag()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    user_input = data.get('text', '')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    # Predict using the basic AI model
    prediction = model.predict([user_input])[0]
    probs = model.predict_proba([user_input])[0]
    confidence = np.max(probs) * 100
    
    return jsonify({
        'recommendation': prediction,
        'confidence': f"{confidence:.1f}%",
        'analysis': f"Based on your input '{user_input}', our AI suggests consulting a {prediction}."
    })

@app.route('/chat', methods=['POST'])
def chat():
    if not qa_chain:
        error_msg = init_error if init_error else 'Chat AI not initialized (Unknown Error)'
        return jsonify({'error': f'AI Init Failed: {error_msg}'}), 503
        
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
        
    try:
        response = qa_chain.run(user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load_and_split()
            
            # Add to Vector Store
            if vector_store:
                vector_store.add_documents(pages)
                
                # Update knowledge base list for reference (optional)
                for page in pages:
                    knowledge_base.append({"text": page.page_content, "source": file.filename})
                
                os.unlink(tmp_path) # Clean up
                return jsonify({'message': f'Successfully processed {file.filename}. You can now ask questions about it!'})
            else:
                return jsonify({'error': 'Vector store not initialized'}), 503
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Only PDF files are supported'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8086))
    app.run(debug=True, host='0.0.0.0', port=port)
