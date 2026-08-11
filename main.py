"""
Pneumonia Detection + RAG Patient Assistant — Flask app

Two capabilities on one page:
1. Upload a chest X-ray -> ONNX model predicts NORMAL / PNEUMONIA + confidence
2. Ask questions about precautions/medications -> RAG retrieves relevant
   context from rag_docs/ and Gemini/Anthropic LLM answers grounded in that context.
"""

import os
import sys
import pickle
import traceback
import numpy as np
from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image

app = Flask(__name__)

# ---- Base Directory for Absolute Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Config ----
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
IMG_SIZE = 150
TOP_K = 3  # retrieved chunks per query

# ---- Absolute Paths for Models and Artifacts ----
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_model.onnx")
H5_MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_model.h5")
FAISS_PATH = os.path.join(BASE_DIR, "rag_index.faiss")
CHUNKS_PATH = os.path.join(BASE_DIR, "rag_chunks.pkl")

# Lazy-loaded globals
_onnx_session = None
_onnx_input_name = None
_onnx_output_name = None
_embedder = None
_rag_index = None
_rag_chunks = None


def get_onnx_session():
    global _onnx_session, _onnx_input_name, _onnx_output_name
    if _onnx_session is None:
        if os.path.exists(ONNX_MODEL_PATH):
            import onnxruntime as ort
            _onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
            _onnx_input_name = _onnx_session.get_inputs()[0].name
            _onnx_output_name = _onnx_session.get_outputs()[0].name
            print(f"ONNX model loaded successfully from {ONNX_MODEL_PATH}")
        elif os.path.exists(H5_MODEL_PATH):
            import tensorflow as tf
            model = tf.keras.models.load_model(H5_MODEL_PATH)
            _onnx_session = model
            print(f"H5 Keras model loaded from {H5_MODEL_PATH}")
        else:
            raise FileNotFoundError(
                f"No model found at {ONNX_MODEL_PATH} or {H5_MODEL_PATH}"
            )
    return _onnx_session


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Sentence embedder loaded")
    return _embedder


def get_rag():
    global _rag_index, _rag_chunks
    if _rag_index is None:
        import faiss
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(f"FAISS index missing at {FAISS_PATH}")
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(f"RAG chunks missing at {CHUNKS_PATH}")

        _rag_index = faiss.read_index(FAISS_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            _rag_chunks = pickle.load(f)
        print("RAG index loaded")
    return _rag_index, _rag_chunks


class_labels = ["NORMAL", "PNEUMONIA"]

# ---- LLM System Prompt ----
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


def vgg16_preprocess(img_array):
    """VGG16 preprocessing: convert RGB to BGR, zero-center by ImageNet mean."""
    img_array = img_array[..., ::-1]  # RGB -> BGR
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_array -= mean
    return img_array


def predict_pneumonia(image_path):
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = vgg16_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 150, 150, 3)

    session = get_onnx_session()

    if _onnx_input_name is not None:
        # ONNX Runtime inference
        outputs = session.run([_onnx_output_name], {_onnx_input_name: img_array})
        prediction = float(outputs[0][0][0])
    else:
        # Keras fallback
        prediction = float(session.predict(img_array)[0][0])

    if prediction >= 0.5:
        return "PNEUMONIA", float(prediction)
    else:
        return "NORMAL", float(1 - prediction)


def retrieve_context(query, k=TOP_K):
    embedder = get_embedder()
    rag_index, rag_chunks = get_rag()
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = rag_index.search(query_vec, k)
    retrieved = [rag_chunks[i] for i in indices[0] if i < len(rag_chunks)]
    return retrieved


def answer_with_rag(question):
    retrieved = retrieve_context(question)
    sources = sorted(set(r["source"] for r in retrieved))
    context_text = "\n\n".join(f"[{r['source']}]: {r['text']}" for r in retrieved)

    # 1. Check for Gemini API key
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"CONTEXT:\n{context_text}\n\nQUESTION: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=600,
                ),
            )
            return response.text, sources
        except Exception as e:
            print(f"Gemini API error: {e}")

    # 2. Check for Anthropic API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            claude = anthropic.Anthropic(api_key=anthropic_key)
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
            print(f"Anthropic API error: {e}")

    # 3. Direct RAG Context Fallback (no API key required)
    fallback_text = (
        "*(Retrieved from medical knowledge base — set GEMINI_API_KEY or ANTHROPIC_API_KEY for AI synthesis)*\n\n"
        + "\n\n".join(f"• {r['text']}" for r in retrieved)
        + "\n\n*Reminder: Always consult a healthcare professional or doctor regarding your specific situation.*"
    )
    return fallback_text, sources


@app.route("/", methods=["GET", "POST"])
def index():
    result, confidence, file_path, error_msg = None, None, None, None
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if file and file.filename != "":
                file_location = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_location)
                result, confidence = predict_pneumonia(file_location)
                file_path = f"/uploads/{file.filename}"
            else:
                error_msg = "Please select an X-ray image file to upload."
        except Exception as e:
            print("Error during image prediction:")
            traceback.print_exc()
            error_msg = f"Prediction Error: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        confidence=f"{confidence*100:.2f}%" if confidence is not None else None,
        file_path=file_path,
        error_msg=error_msg,
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
