"""
Pneumonia Detection + RAG Patient Assistant — Flask app

Two capabilities on one page:
1. Upload a chest X-ray -> model predicts NORMAL / PNEUMONIA + confidence
2. Ask questions about precautions/medications -> RAG retrieves relevant
   context from rag_docs/ and an LLM answers grounded in that context.

Env var required: ANTHROPIC_API_KEY
pip install flask tensorflow anthropic sentence-transformers faiss-cpu pillow
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sentence_transformers import SentenceTransformer
import faiss
import anthropic

app = Flask(__name__)

# ---- Config ----
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
IMG_SIZE = 150
TOP_K = 3  # retrieved chunks per query

# ---- Load prediction model ----
model = load_model("models/pneumonia_model.h5")
class_labels = ["NORMAL", "PNEUMONIA"]

# ---- Load RAG index ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")
rag_index = faiss.read_index("rag_index.faiss")
with open("rag_chunks.pkl", "rb") as f:
    rag_chunks = pickle.load(f)

# ---- LLM client initialization (lazy) ----
SYSTEM_PROMPT = """You are a patient education assistant for a pneumonia care app.
Answer ONLY using the CONTEXT provided below. If the context doesn't cover the
question, say you don't have that information and recommend asking a doctor
or pharmacist.

Rules you must always follow:
- Never provide a diagnosis. You may explain what a result or term means in
  general, educational terms only.
- Never suggest specific dosages or dosage changes. If asked about dosing,
  say to check with the prescribing doctor or pharmacist.
- Always end answers involving medication or symptoms with a brief reminder
  to consult a healthcare professional for their specific situation.
- Keep answers concise and in plain, non-technical language.
"""


def predict_pneumonia(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]  # sigmoid output, 0-1
    if prediction >= 0.5:
        return "PNEUMONIA", float(prediction)
    else:
        return "NORMAL", float(1 - prediction)


def retrieve_context(query, k=TOP_K):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = rag_index.search(query_vec, k)
    retrieved = [rag_chunks[i] for i in indices[0] if i < len(rag_chunks)]
    return retrieved


def answer_with_rag(question):
    retrieved = retrieve_context(question)
    sources = sorted(set(r["source"] for r in retrieved))
    context_text = "\n\n".join(f"[{r['source']}]: {r['text']}" for r in retrieved)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback when ANTHROPIC_API_KEY is not configured
        fallback_text = (
            "*(Retrieved from medical knowledge base — set ANTHROPIC_API_KEY for AI summaries)*\n\n"
            + "\n\n".join(f"• {r['text']}" for r in retrieved)
            + "\n\n*Reminder: Always consult a healthcare professional or doctor regarding your specific situation.*"
        )
        return fallback_text, sources

    try:
        claude = anthropic.Anthropic(api_key=api_key)
        message = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}",
                }
            ],
        )
        answer_text = "".join(block.text for block in message.content if block.type == "text")
        return answer_text, sources
    except Exception as e:
        # If API call fails, return retrieved context gracefully
        fallback_text = (
            f"*(RAG Search Result)*\n\n"
            + "\n\n".join(f"• {r['text']}" for r in retrieved)
            + "\n\n*Reminder: Always consult a healthcare professional regarding your specific situation.*"
        )
        return fallback_text, sources


@app.route("/", methods=["GET", "POST"])
def index():
    result, confidence, file_path = None, None, None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            file_location = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_location)
            result, confidence = predict_pneumonia(file_location)
            file_path = f"/uploads/{file.filename}"

    return render_template(
        "index.html",
        result=result,
        confidence=f"{confidence*100:.2f}%" if confidence else None,
        file_path=file_path,
    )


@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer, sources = answer_with_rag(question)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
