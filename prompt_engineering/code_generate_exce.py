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
from pprint import pprint
code_config = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
	)

code_prompt = "Generate the first 4 prime numbers and sum them"

response = client.models.generate_content(
    model="gemini-2.0-flash",
	config = code_config,
    contents= code_prompt)
for part in response.candidates[0].content.parts:
	pprint(part.to_json_dict())
	print("-------------------")
