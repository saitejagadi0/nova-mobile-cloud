import os
import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI


app = Flask(__name__)
CORS(app)

MODEL = "gpt-5.6-luna"


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

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
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raw_body = request.get_data(as_text=True).strip()

            if raw_body:
                try:
                    data = json.loads(raw_body)
                except Exception:
                    data = {}
            else:
                data = {}

        if not data and request.form:
            data = request.form.to_dict()

        if not isinstance(data, dict):
            data = {}

        message = str(
            data.get("message", "")
        ).strip()

        mode = str(
            data.get("mode", "chat")
        ).strip().lower()

        history = data.get("history", [])

        if not message:
            return jsonify({
                "success": False,
                "error": "Message is required."
            }), 400

        if not isinstance(history, list):
            history = []

        history = history[-20:]

        instructions = {
            "chat":
                "You are NOVA, a helpful personal AI assistant.",

            "reason":
                "You are NOVA in reasoning mode. Analyze problems carefully and explain clearly.",

            "code":
                "You are NOVA in coding mode. Help write, debug, review and explain code.",

            "math":
                "You are NOVA in math mode. Solve calculations accurately and explain the result.",

            "vision":
                "You are NOVA in vision mode. Analyze visual information carefully when it is provided.",

            "files":
                "You are NOVA in file mode. Help the user understand and work with their files.",

            "web":
                "You are NOVA in web mode. Help answer questions using web information when available.",

            "screen":
                "You are NOVA in screen mode. Help analyze screen content and explain what is visible.",

            "tools":
                "You are NOVA in tools mode. Help the user accomplish tasks using available tools."
        }

        instruction = instructions.get(
            mode,
            instructions["chat"]
        )

        conversation = []

        for item in history:

            if not isinstance(item, dict):
                continue

            role = item.get("role")

            if role not in ("user", "assistant"):
                continue

            text = str(
                item.get("content", "")
            ).strip()

            if not text:
                continue

            conversation.append({
                "role": role,
                "content": text
            })

        conversation.append({
            "role": "user",
            "content": message
        })

        response = get_client().responses.create(
            model=MODEL,
            instructions=instruction,
            input=conversation
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