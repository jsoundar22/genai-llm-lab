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
# Note that when you use the API, you get the final response from the model, 
# but the thoughts are not captured. To see the intermediate thoughts, try out 
# [the thinking mode model in AI Studio](https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash-thinking-exp-01-21).
###############################################################
import io

response = client.models.generate_content_stream(
	model='gemini-2.0-flash-thinking-exp',
	contents="Who is the youngest Cheif Minister of Karnataka from the Mysore region?",
	#contents='Who was the youngest author listed on the transformers NLP paper?',
	
)

buf = io.StringIO()
for chunk in response:
	buf.write(chunk.text)
	# Display the response as it Streams
	print(chunk.text, end='')
	