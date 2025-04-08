from google import genai
import os
from google.genai import types

from IPython.display import HTML, Markdown, display

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
		raise ValueError("GEMINI_API_KEY variable not set")
client = genai.Client(api_key=GEMINI_API_KEY)
###############################################################
model_config = types.GenerateContentConfig(
    temperature=1,
    top_p=1,
    max_output_tokens=1024)

code_prompt = "Generate a python code to fetch the latest commit by github user jsoundar22, just the code, no explanation"

response = client.models.generate_content(
    model="gemini-2.0-flash",
	config = model_config,
    contents= code_prompt)

print(response.text)
