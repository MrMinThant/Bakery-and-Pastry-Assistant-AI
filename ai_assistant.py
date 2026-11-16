"""
ai_assistant.py - the AI model assistant for the Bakery & Pastry app.

BakeryAssistant takes a bakery photo and a recorded voice question, asks an AI
model to identify the product and check its quality, and returns a dict:
    identified_product, category, visual_evidence, doneness, visible_defects,
    freshness_signs, answer_to_user, recommendation, confidence, verdict

Two interchangeable backends (choose with the AI_BACKEND environment variable):
    gemini  (default) - Google Gemini in the cloud. Use this on Render.
    local             - Whisper (speech-to-text) + Ollama vision model on your own
                        computer, like the original AI_Visual_Assistant.py.
                        Needs: pip install -r requirements-local.txt
"""
import io
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")


class AssistantError(Exception):
    """Error with a user-friendly message and an HTTP status code."""

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message = message
        self.status = status


def _load_text(filename: str) -> str:
    # Read fresh on every request so knowledge edits apply without a restart.
    with open(os.path.join(KNOWLEDGE_DIR, filename), "r", encoding="utf-8") as f:
        return f.read().strip()


def build_system_prompt() -> str:
    return _load_text("system_prompt.txt").replace(
        "{domain_knowledge}", _load_text("domain_knowledge.txt")
    )


def format_answer(a: dict) -> str:
    """Plain-text version of an analysis (fallback for simple clients)."""
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


class BakeryAssistant:
    def __init__(self, backend: str = None):
        self.backend = (backend or os.environ.get("AI_BACKEND", "gemini")).lower()
        if self.backend not in ("gemini", "local"):
            raise ValueError("AI_BACKEND must be 'gemini' or 'local'")

        if self.backend == "gemini":
            self.model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
            api_key = os.environ.get("GEMINI_API_KEY")
            self._client = None
            if api_key:
                from google import genai  # imported here so local mode doesn't need it
                self._client = genai.Client(api_key=api_key)
        else:
            self.model = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")
            self.whisper_size = os.environ.get("WHISPER_MODEL", "small")
            self._whisper = None  # loaded on first use

    # ------------------------------------------------------------------ public
    def is_ready(self) -> bool:
        return self.backend == "local" or self._client is not None

    def not_ready_message(self) -> str:
        return "GEMINI_API_KEY is missing from the server environment."

    def analyze(self, image_bytes, image_mime, audio_bytes, audio_mime) -> dict:
        """Run the full pipeline and return the analysis dict."""
        raw = (
            self._run_gemini(image_bytes, image_mime, audio_bytes, audio_mime)
            if self.backend == "gemini"
            else self._run_local(image_bytes, audio_bytes)
        )
        if not raw:
            raise AssistantError("The AI returned an empty response.", 502)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Model ignored the JSON format: keep its words so the user still gets an answer.
            return {"answer_to_user": raw, "confidence": "Low", "verdict": "PASS"}

    # ---------------------------------------------------------- Gemini backend
    def _run_gemini(self, image_bytes, image_mime, audio_bytes, audio_mime) -> str:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=image_mime or "image/jpeg"),
                    types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime or "audio/webm"),
                    "Analyze the bakery product in the image and answer the spoken question.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=build_system_prompt(),
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""
        except Exception as error:
            message = str(error)
            if "503" in message or "UNAVAILABLE" in message:
                raise AssistantError("Gemini is temporarily busy (HTTP 503). Please wait and try again.", 503)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                raise AssistantError("Gemini quota or rate limit reached. Please try again later or check your Google AI Studio quota.", 429)
            raise

    # ----------------------------------------------------------- local backend
    def _run_local(self, image_bytes, audio_bytes) -> str:
        try:
            from faster_whisper import WhisperModel
            from ollama import chat
        except ImportError:
            raise AssistantError("Local mode needs: pip install -r requirements-local.txt", 500)

        if self._whisper is None:
            self._whisper = WhisperModel(self.whisper_size, device="cpu", compute_type="int8")

        # Stage 1: voice -> text
        segments, _ = self._whisper.transcribe(io.BytesIO(audio_bytes), language="en")
        question = "".join(s.text for s in segments).strip() or "Analyze this bakery product."

        # Stage 2: image + question -> structured JSON
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": question, "images": [image_bytes]},
            ],
            format="json",
            options={"temperature": 0},
        )
        return response["message"]["content"]
