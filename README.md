# Pneumonia Detection + RAG Patient Assistant

## Setup order

1. **Get the dataset**: [chest-xray-pneumonia on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
   Unzip so you have `chest_xray/train`, `chest_xray/val`, `chest_xray/test`,
   each with `NORMAL/` and `PNEUMONIA/` subfolders.

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Train the model** (run in Colab/Kaggle with a GPU if possible — CPU will be slow)
   ```
   python train_model.py
   ```
   This saves `models/pneumonia_model.h5`. Expect roughly 90%+ validation
   accuracy with VGG16 transfer learning + fine-tuning (better than the
   original notebook's ~85%, since this version uses class weighting instead
   of throwing away data via undersampling).

4. **Build the RAG index** (from the docs in `rag_docs/`)
   ```
   python build_rag_index.py
   ```
   Add more `.txt` files to `rag_docs/` (WHO/CDC guidance, drug info sheets,
   etc.) and re-run this any time you want to expand what the assistant can
   answer from.

5. **Set your API key**
   ```
   export ANTHROPIC_API_KEY=your_key_here
   ```

6. **Run the app**
   ```
   python main.py
   ```
   Visit `http://127.0.0.1:5000`.

## What's in here
- `train_model.py` — VGG16 transfer-learning pneumonia classifier (modernized
  version of the paultimothymooney Kaggle notebook's approach)
- `build_rag_index.py` — builds a FAISS index over `rag_docs/*.txt`
- `main.py` — Flask app: X-ray upload + prediction, plus a `/chat` endpoint
  that retrieves relevant context and asks an LLM to answer grounded in it
- `templates/index.html` — single-page UI: upload form + chat box
- `rag_docs/` — starter knowledge base (general pneumonia precautions and
  common medication info). Replace/expand with real WHO/CDC sources before
  treating this as more than a demo.

## Known limitations to disclose in your report/demo
- The classifier is a screening aid, not a diagnostic tool — no clinical
  validation was done.
- The RAG knowledge base here is a small starter set, not a comprehensive or
  verified medical corpus — expand it with authoritative sources for a real
  submission.
- The chat assistant is explicitly instructed not to give dosages or
  diagnoses and always defers to a doctor — keep that guardrail if you extend
  the system prompt.
