# Entry point for Render / gunicorn
# Render defaults to "gunicorn app:app", so this file re-exports the Flask app from main.py
from main import app

if __name__ == "__main__":
    app.run()
