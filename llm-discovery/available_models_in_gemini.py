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

# Get the list of models
for models in client.models.list():
	print(models.name)

# Get the details of a models capability
from pprint import pprint

# Get the details of a specific model
for model in client.models.list():
    if model.name == 'models/gemini-2.0-flash':
        print('+++++++++++ Found +++++++++++ ')
        pprint(model.to_json_dict())
        break

