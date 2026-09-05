import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI


app = Flask(__name__)
CORS(app)


MODEL = "gpt-5.6-luna"


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=api_key)


@app.get("/")
def home():
    return send_from_directory(".", "mobile.html")


@app.get("/health")
def health():
    return jsonify({
        "assistant": "NOVA",
        "status": "online"
    })


@app.post("/chat")
def chat():

    try:
        data = request.get_json(silent=True) or {}

        message = str(
            data.get("message", "")
        ).strip()

        mode = str(
            data.get("mode", "chat")
        ).strip().lower()

        if not message:
            return jsonify({
                "success": False,
                "error": "Message is required."
            }), 400

        instructions = {
            "chat":
                "You are NOVA, a helpful personal AI assistant.",

            "reason":
                "You are NOVA in reasoning mode. Analyze problems carefully and explain clearly.",

            "code":
                "You are NOVA in coding mode. Help write, debug, review and explain code.",

            "math":
                "You are NOVA in math mode. Solve calculations accurately and explain the result."
        }

        instruction = instructions.get(
            mode,
            instructions["chat"]
        )

        response = get_client().responses.create(
            model=MODEL,
            instructions=instruction,
            input=message
        )

        return jsonify({
            "success": True,
            "mode": mode,
            "response": response.output_text
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )