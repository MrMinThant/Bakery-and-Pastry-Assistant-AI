import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=API_KEY) if API_KEY else None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")


def load_text(filename: str) -> str:
    """Read a knowledge file fresh on every request so edits apply without a restart."""
    with open(os.path.join(KNOWLEDGE_DIR, filename), "r", encoding="utf-8") as f:
        return f.read().strip()


def build_system_prompt() -> str:
    template = load_text("system_prompt.txt")
    return template.replace("{domain_knowledge}", load_text("domain_knowledge.txt"))


def format_answer(a: dict) -> str:
    """Plain-text version of the analysis (used as fallback and for older clients)."""
    defects = ", ".join(a.get("visible_defects") or []) or "None visible"
    return (
        f"Identified Product: {a.get('identified_product', 'Unknown')}\n"
        f"Category: {a.get('category', 'Unknown')}\n"
        f"Visual Evidence: {a.get('visual_evidence', '')}\n"
        f"Doneness: {a.get('doneness', 'Cannot Tell')}\n"
        f"Defects: {defects}\n"
        f"Freshness: {a.get('freshness_signs', 'Cannot Tell')}\n"
        f"Answer to User: {a.get('answer_to_user', '')}\n"
        f"Recommendation: {a.get('recommendation', '')}\n"
        f"Confidence: {a.get('confidence', 'Low')}\n"
        f"Verdict: {a.get('verdict', '')}"
    )


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/analyze", methods=["POST"])
def analyze():
    if client is None:
        return jsonify({"error": "GEMINI_API_KEY is missing from the server environment."}), 500
    if "image" not in request.files or "audio" not in request.files:
        return jsonify({"error": "Please upload an image and record a voice question."}), 400

    image_file = request.files["image"]
    audio_file = request.files["audio"]
    if not image_file.filename or not audio_file.filename:
        return jsonify({"error": "The image or audio recording is empty."}), 400

    try:
        image_bytes = image_file.read()
        audio_bytes = audio_file.read()
        if not image_bytes or not audio_bytes:
            return jsonify({"error": "The image or audio recording is empty."}), 400

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=image_file.mimetype or "image/jpeg"),
                types.Part.from_bytes(data=audio_bytes, mime_type=audio_file.mimetype or "audio/webm"),
                "Analyze the bakery product in the image and answer the spoken question."
            ],
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(),
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        raw = response.text or ""
        if not raw:
            return jsonify({"error": "The AI returned an empty response."}), 502
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            # Model ignored the JSON format: still show the user its text answer.
            return jsonify({"answer": raw})
        return jsonify({"analysis": analysis, "answer": format_answer(analysis)})

    except Exception as error:
        logger.exception("Gemini analysis failed")
        message = str(error)
        if "503" in message or "UNAVAILABLE" in message:
            return jsonify({"error": "Gemini is temporarily busy (HTTP 503). Please wait and try again."}), 503
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            return jsonify({"error": "Gemini quota or rate limit reached. Please try again later or check your Google AI Studio quota."}), 429
        return jsonify({"error": "AI analysis failed. Check the Render logs for details."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
