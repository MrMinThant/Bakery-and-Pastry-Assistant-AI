# Bakery VLM Voice & Vision Assistant

Flask application: submits a bakery product image and a recorded voice question to Google's Gemini model, which identifies the product and checks its quality (doneness, defects, freshness) and returns a structured PASS / ACTION_REQUIRED report.

## Included
- `app.py` — Flask backend and Gemini integration
- `templates/index.html` — mobile-friendly photo and voice interface
- `knowledge/domain_knowledge.txt` — bakery rules (categories, doneness, defects, spoilage). Edit this to teach the assistant new products
- `knowledge/system_prompt.txt` — instructions and JSON schema; keep the `{domain_knowledge}` placeholder
- `requirements.txt` — Python dependencies
- `.env.example` — safe template; contains no real API key
- `.gitignore` — excludes secrets and local virtual environments

## Security
This ZIP does not include your real Gemini API key. Never upload `.env` or share your key.

## Run on Windows
1. Install Python and extract this ZIP.
2. Open the folder in VS Code and open a terminal.
3. Run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
4. Copy `.env.example` to a new file named `.env`, then put your own key after `GEMINI_API_KEY=`.
5. Run `python app.py` and visit `http://127.0.0.1:5000`.

## Deploy with a new GitHub/Render account
1. Create a repository in the new GitHub account and upload the project files and the `templates` and `knowledge` folders. Do not upload `.env` or `.venv`.
2. Create a Render Web Service from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn --workers 1 --threads 2 --timeout 300 app:app`
5. Add Render environment variable `GEMINI_API_KEY` with your own Gemini API key.
6. Optional: add `GEMINI_MODEL` with value `gemini-3.8-flash`.
7. Deploy. The new service receives a new Render URL.

The ZIP transfers source code only. It does not transfer ownership of your existing GitHub repository, Render service, deployment URL, or Gemini API key.

## Troubleshooting
- Phone microphone requires HTTPS and browser microphone permission.
- HTTP 503 means Gemini is temporarily unavailable; wait and retry.
- HTTP 429 indicates quota or rate limiting.
- Check Render Logs for backend exceptions.
- `/health` returns `{"status":"ok"}` when the Flask app is running.
