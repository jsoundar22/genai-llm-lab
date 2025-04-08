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
import subprocess

file_contents = subprocess.check_output(
    ["curl", "https://raw.githubusercontent.com/magicmonty/bash-git-prompt/refs/heads/master/gitprompt.sh"],
    text=True
)

explain_prompt = f"""
Can you explain what this file is about at high level, will it be useful to me, and how?
'''
{file_contents}
'''
"""
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents= explain_prompt)

print(response.text)
