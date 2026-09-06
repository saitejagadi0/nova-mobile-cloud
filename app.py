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

        # -------------------------------------------------
        # READ REQUEST BODY
        # -------------------------------------------------

        data = request.get_json(
            silent=True
        )

        # If Flask did not parse JSON, manually decode it.
        if not isinstance(data, dict):

            raw_body = request.get_data(
                as_text=True
            ).strip()

            if raw_body:

                try:
                    data = json.loads(
                        raw_body
                    )

                except Exception:
                    data = {}

            else:
                data = {}


        # -------------------------------------------------
        # FALLBACK FOR FORM DATA
        # -------------------------------------------------

        if not data:

            if request.form:

                data = request.form.to_dict()

            else:

                data = {}


        # -------------------------------------------------
        # MESSAGE
        # -------------------------------------------------

        message = str(
            data.get(
                "message",
                ""
            )
        ).strip()


        # -------------------------------------------------
        # MODE
        # -------------------------------------------------

        mode = str(
            data.get(
                "mode",
                "chat"
            )
        ).strip().lower()


        # -------------------------------------------------
        # HISTORY
        # -------------------------------------------------

        history = data.get(
            "history",
            []
        )


        if not isinstance(
            history,
            list
        ):

            history = []


        history = history[-20:]


        # -------------------------------------------------
        # VALIDATE MESSAGE
        # -------------------------------------------------

        if not message:

            return jsonify({
                "success": False,
                "error": "Message is required."
            }), 400


        # -------------------------------------------------
        # MODE INSTRUCTIONS
        # -------------------------------------------------

        instructions = {

            "chat":
                "You are NOVA, a helpful personal AI assistant. "
                "Use the conversation history to remember information "
                "the user has previously told you.",


            "reason":
                "You are NOVA in reasoning mode. "
                "Analyze problems carefully and explain clearly. "
                "Use the conversation history when relevant.",


            "code":
                "You are NOVA in coding mode. "
                "Help write, debug, review and explain code. "
                "Use the conversation history when relevant.",


            "math":
                "You are NOVA in math mode. "
                "Solve calculations accurately and explain the result."
        }


        instruction = instructions.get(
            mode,
            instructions["chat"]
        )


        # -------------------------------------------------
        # BUILD CONVERSATION
        # -------------------------------------------------

        conversation = []


        for item in history:

            if not isinstance(
                item,
                dict
            ):
                continue


            role = item.get(
                "role"
            )


            if role not in (
                "user",
                "assistant"
            ):
                continue


            text = str(
                item.get(
                    "content",
                    ""
                )
            ).strip()


            if not text:
                continue


            conversation.append({

                "role":
                    role,

                "content": [
                    {
                        "type":
                            "input_text",

                        "text":
                            text
                    }
                ]

            })


        # -------------------------------------------------
        # ADD CURRENT MESSAGE
        # -------------------------------------------------

        conversation.append({

            "role":
                "user",

            "content": [
                {
                    "type":
                        "input_text",

                    "text":
                        message
                }
            ]

        })


        # -------------------------------------------------
        # OPENAI
        # -------------------------------------------------

        response = get_client().responses.create(

            model=MODEL,

            instructions=instruction,

            input=conversation

        )


        answer = response.output_text


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "success":
                True,

            "mode":
                mode,

            "response":
                answer

        })


    except Exception as e:

        print(
            "NOVA CHAT ERROR:",
            repr(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ---------------------------------------------------------
# LOCAL DEVELOPMENT
# ---------------------------------------------------------

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