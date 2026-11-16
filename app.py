import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

load_dotenv()

from ai_assistant import BakeryAssistant, AssistantError, format_answer  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

assistant = BakeryAssistant()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "backend": assistant.backend})


@app.route("/analyze", methods=["POST"])
def analyze():
    if not assistant.is_ready():
        return jsonify({"error": assistant.not_ready_message()}), 500
    if "image" not in request.files or "audio" not in request.files:
        return jsonify({"error": "Please upload an image and record a voice question."}), 400

    image_file = request.files["image"]
    audio_file = request.files["audio"]
    image_bytes = image_file.read()
    audio_bytes = audio_file.read()
    if not image_bytes or not audio_bytes:
        return jsonify({"error": "The image or audio recording is empty."}), 400

    try:
        analysis = assistant.analyze(
            image_bytes, image_file.mimetype, audio_bytes, audio_file.mimetype
        )
        return jsonify({"analysis": analysis, "answer": format_answer(analysis)})
    except AssistantError as error:
        return jsonify({"error": error.message}), error.status
    except Exception:
        logger.exception("AI analysis failed")
        return jsonify({"error": "AI analysis failed. Check the Render logs for details."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
