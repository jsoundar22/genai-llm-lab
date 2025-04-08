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
    temperature=0.1,
    top_p=1,
    max_output_tokens=5,)

zero_shot_prompt = """
You are a sentiment analysis model.
You are given a sentence and you need to classify it as positive, negative or neutral.
The sentence is: {The team have been working hard, but of late there were many skipping the standup meetings. I see a kind of slackness in the team.
After the slack message on the importance of attending the standup meetings, I see a lot of people attending the meetings.
Hope this continues.}
The sentiment is: {positive, negative or neutral}
"""
response = client.models.generate_content(
    model="gemini-2.0-flash",
	config = model_config,
    contents= zero_shot_prompt)

print(response.text)
