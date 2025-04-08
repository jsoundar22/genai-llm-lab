from google import genai
import os
from google.genai import types

from IPython.display import HTML, Markdown, display

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
		raise ValueError("GEMINI_API_KEY variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

from google.api_core import retry

# Setup retry helper
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Can you tell my personality based on my interaction so far?")

print(response.text)
