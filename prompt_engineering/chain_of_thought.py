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
chain_of_thought_prompt = "When I was I was 8 year old, my partner was half my age. now I am 48. How old is my partner now? Tell the direct answer.\n\n" 
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents= chain_of_thought_prompt
)
print("++++++++++ Direct answer from the model without chain of thought ++++++++++")
print(response.text)
# ++++++++++ Direct answer from the model without chain of thought ++++++++++
#32

chain_of_thought_prompt = "When I was I was 8 year old, my partner was half my age. now I am 48. How old is my partner now? Lets think step by step. \n\n" 
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents= chain_of_thought_prompt
)
print("++++++++++ Response from the model with chain of thought ++++++++++")
print(response.text)
#++++++++++ Response from the model with chain of thought ++++++++++
#Okay, let's break this down:

#1.  **Find the age difference:** When you were 8, your partner was half your age, meaning they were 8 / 2 = 4 years old.
#2.  **Calculate the age difference:** This means the age difference between you and your partner is 8 - 4 = 4 years.
#3.  **Apply the age difference to your current age:** Since you are now 48, your partner is 48 - 4 = 44 years old.

#So the answer is 44
