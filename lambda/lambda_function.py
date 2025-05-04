from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import requests
import logging
import json
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (for local testing)
load_dotenv()

# Configuration from environment variables
API_TYPE = os.getenv("API_TYPE", "openai").lower()  # "openai" or "azure"
API_KEY = os.getenv("OPENAI_API_KEY")  # Public or Azure key header
AZURE_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")  # e.g. https://<resource>.openai.azure.com
AZURE_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")  # your deployment/model
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-10-01-preview")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Answer in 50 words or less."
)

# Speak outputs from environment
LAUNCH_MESSAGE = os.getenv("LAUNCH_MESSAGE", "Chat G.P.T. mode activated")
LAUNCH_REPROMPT = os.getenv("LAUNCH_REPROMPT", LAUNCH_MESSAGE)
QUERY_REPROMPT = os.getenv("QUERY_REPROMPT", "Any other questions?")
STOP_MESSAGE = os.getenv("STOP_MESSAGE", "Leaving Chat G.P.T. mode")
ERROR_MESSAGE = os.getenv("ERROR_MESSAGE", "Sorry, I had trouble doing what you asked. Please try again.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []
        return (
            handler_input.response_builder
                .speak(LAUNCH_MESSAGE)
                .ask(LAUNCH_REPROMPT)
                .response
        )

class GptQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        query = handler_input.request_envelope.request.intent.slots["query"].value
        session_attr = handler_input.attributes_manager.session_attributes
        if "chat_history" not in session_attr:
            session_attr["chat_history"] = []
        response = generate_gpt_response(session_attr["chat_history"], query)
        session_attr["chat_history"].append((query, response))
        return (
            handler_input.response_builder
                .speak(response)
                .ask(QUERY_REPROMPT)
                .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (
            ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
            ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input)
        )

    def handle(self, handler_input):
        return (
            handler_input.response_builder
                .speak(STOP_MESSAGE)
                .response
        )

class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        return (
            handler_input.response_builder
                .speak(ERROR_MESSAGE)
                .ask(ERROR_MESSAGE)
                .response
        )


def generate_gpt_response(chat_history, new_question):
    # Choose endpoint & headers based on API_TYPE
    if API_TYPE == "azure":
        if not all([AZURE_API_BASE, AZURE_DEPLOYMENT_ID, API_KEY]):
            return (
                "Azure config incomplete: set AZURE_OPENAI_API_BASE, "
                "AZURE_OPENAI_DEPLOYMENT_ID, and OPENAI_API_KEY"
            )
        url = (
            f"{AZURE_API_BASE}/openai/deployments/"
            f"{AZURE_DEPLOYMENT_ID}/chat/completions?api-version={AZURE_API_VERSION}"
        )
        headers = {"api-key": API_KEY, "Content-Type": "application/json"}
        model_name = AZURE_DEPLOYMENT_ID
    else:
        if not API_KEY:
            return "OpenAI API key is not set. Please set OPENAI_API_KEY."
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        model_name = "gpt-4o-mini"

    # Build message array with system prompt from env
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in chat_history[-10:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": new_question})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.5
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        data = r.json()
        if r.ok:
            return data['choices'][0]['message']['content']
        return f"Error {r.status_code}: {data.get('error', {}).get('message', '')}"
    except Exception as e:
        return f"Error generating response: {e}"

# Register handlers and export
sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())
lambda_handler = sb.lambda_handler()
