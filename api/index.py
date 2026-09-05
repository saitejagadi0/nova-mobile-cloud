import os

from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

MODEL = "gpt-5.6-luna"


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured on the server.")

    return OpenAI(api_key=api_key)


def nova_answer(message, mode="chat"):
    instructions = {
        "chat": (
            "You are NOVA, a helpful personal AI assistant. "
            "Answer clearly, naturally and concisely."
        ),

        "reason": (
            "You are NOVA in reasoning mode. "
            "Analyze the user's problem carefully and provide a useful conclusion."
        ),

        "code": (
            "You are NOVA in coding mode. "
            "Help write, debug, explain and improve code. "
            "Give practical code when appropriate."
        ),

        "math": (
            "You are NOVA in math mode. "
            "Solve calculations accurately and explain the result briefly."
        ),
    }

    instruction = instructions.get(mode, instructions["chat"])

    response = get_client().responses.create(
        model=MODEL,
        instructions=instruction,
        input=message,
    )

    return response.output_text


@app.get("/")
def home():
    return jsonify({
        "assistant": "NOVA",
        "status": "online",
        "service": "mobile-cloud"
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "online",
        "assistant": "NOVA"
    })


@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}

        message = str(data.get("message", "")).strip()
        mode = str(data.get("mode", "chat")).strip().lower()

        if not message:
            return jsonify({
                "success": False,
                "error": "Message is required."
            }), 400

        answer = nova_answer(message, mode)

        return jsonify({
            "success": True,
            "mode": mode,
            "response": answer
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500