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
few_shot_prompt = """Parse a customer's IceCream order into valid JSON:

EXAMPLE:
I want a small cup of vennila, plain.
JSON Response:
```
{
"size": "small",
"container": "cup",
"falvour": "vennila",
"toppings": ["plain"]
}
```

EXAMPLE:
Can I get a chocolate cone with caremel topping?
JSON Response:
```
{
"size": "medium",
"container": "cone",
"flavour": "chocolate",
"toppings": ["caramel"]
}
```

ORDER:
"""

customer_order = "I want a small cup of chocolate, with sprinkles and nuts."
response = client.models.generate_content(
    model="gemini-2.0-flash",
	config = types.GenerateContentConfig(
		temperature=0.1,
		top_p=1,
		max_output_tokens=250),
        contents= [few_shot_prompt, customer_order],
)
print(response.text)
