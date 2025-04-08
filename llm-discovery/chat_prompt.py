from google import genai
import os
from google.genai import types

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
		raise ValueError("GEMINI_API_KEY variable not set")
client = genai.Client(api_key=GEMINI_API_KEY)

# Setup retry helper
from google.api_core import retry
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

chat = client.chats.create(model='gemini-2.0-flash', history=[])
response = chat.send_message(message="Good morning, I am Jay")
print(response.text)

response = chat.send_message(message="Tell me something interesting about Doctor of Business Administration")
print(response.text)

# Test the history
response= chat.send_message(message="Do you remember my name?")
print(response.text)
