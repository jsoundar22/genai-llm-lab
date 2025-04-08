from google import genai
import os
from google.genai import types

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
		raise ValueError("GEMINI_API_KEY variable not set")
client = genai.Client(api_key=GEMINI_API_KEY)

###########################################################
# Control the length of token generation
from google.genai import types

short_config = types.GenerateContentConfig(max_output_tokens=200)

response = client.models.generate_content(
    model = 'gemini-2.0-flash',
	config = short_config,
	contents = 'Write a small poem about Puducherry (India)')
print("++++++++++ Testing the max_output_tokens parameter ++++++++++ ")
print(response.text)



# Controlling the LLM output based on the temparature setting
high_temp_config = types.GenerateContentConfig(temperature=0.0)
# Temperature controls the degree of randomness in token selection. 
# Higher temperatures result in a higher number of candidate tokens 
# from which the next output token is selected, and can produce more
# diverse results, while lower temperatures have the opposite effect, 
# such that a temperature of 0 results in greedy decoding, selecting
# the most probable token at each step. Temperature doesn't provide 
# any guarantees of randomness, but it can be used to "nudge" the output somewhat.
# 
# temperature = 0.0 means the model will give the most likely output
# temperature = 1.0 means the model will give a random output
# temperature = 2.0 means the model will give a very random output
print('++++++++++ Testing the temperature parameter ++++++++++ ')
for _ in range(12):
    # Generate a random month
    # Note: The model is not trained to give a month name, so the output is not deterministic
    # It will give a random month name
    # This is just to show the effect of temperature on the output
    response = client.models.generate_content(
	    model = 'gemini-2.0-flash',
	    config = high_temp_config,
        contents = 'Pick a random month of the gregorian calendar, need single word anaswer')
    if response.text:
            print(response.text)
    else:
            print('No response from the model')

# Top-P parameter
# To control the diversity of the model's output. Top-p determines the probability threshold that once cumulativly exceeded. 
# Top-p = 0, Results in greedy decoding
# Top-p = 1, typically select every possible token from the models vocabulary
# Top-p = 0.9, results in a more diverse output
# Top-p = 0.5, results in a more focused output
# Top-p = 0.1, results in a very focused output

model_config = types.GenerateContentConfig(
	temperature=0.3,
	top_p=0.5)

story_prompt = 'You are a nature lover, write about millipedes in a poetic way'
response = client.models.generate_content(
    model = 'gemini-2.0-flash',
    config = model_config,
    contents = story_prompt)
print('++++++++++ Testing the top_p parameter ++++++++++ ')
print(response.text) 
