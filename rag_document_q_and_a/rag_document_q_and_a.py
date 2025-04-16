# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# In this lab, we will will use the Gemini API to create a vector database, retrieve answers to questions from the database and generate a final answer.
# SETUP
# install -qU "google-genai==1.7.0" "chromadb==0.6.3"
# 
from urllib import response
from google import genai
from google.genai import types
import os
genai.__version__

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
                raise ValueError("GEMINI_API_KEY variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

# Find the model which support embedContent PAI, which we will use it for embedding in tis lab
for m in client.models.list():
        if 'embedContent' in m.supported_actions:
                print(m.name)

# Sample documents for embedding
# These documents are related to the operation of a Googlecar, including climate control, touchscreen display, and shifting gears.
DOCUMENT1 = "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
DOCUMENT2 = 'Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the "Navigation" icon to get directions to your destination or touch the "Music" icon to play your favorite songs.'
DOCUMENT3 = "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

from chromadb import Documents, EmbeddingFunction, Embeddings

from google.api_core import retry
# Define a helper function for retry when the quota is exceeded
is_retriable = lambda e:(isinstance(e, genai.errors.APIError) and e.code in (429, 503))


class GeminiEmbeddingFunction(EmbeddingFunction):
    # flag to select, the embedding is for document or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
            if self.document_mode:
                embedding_task = 'retrieval_document'
            else:
                embedding_task = 'retrieval_query'

            response = client.models.embed_content(
                    model='models/text-embedding-004',
                    contents=input,
                    config=types.EmbedContentConfig(
                            task_type = embedding_task
                    )
            )
            return [e.values for e in response.embeddings]
    
# Create cromadb and populate it with the documents
import chromadb
DB_NAME = "googlecardb"
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# Verify the documents are added
print(f'Count in Croma DB: {db.count()}')
#peak the first document
#print(f'First document: {db.peek(1)}')

#swiatch to query mode when generating embeddings
embed_fn.document_mode = False

# search the Chroma DB using the specified query
query = "What is the function of your touchscreen display?"

# Limiting the retreival to one document for this example
result = db.query(query_texts=[query], n_results=1)
[all_passages] = result['documents']
print(f"Passages: {all_passages}")

#Convert to multi line query in to one liner
query_oneliner = query.replace('\n', ' ')

#prompt, you can specify the format, tone and guardrails
prompt = f"""You are a helpful bot, to answer the customer about the specific questions asked about the Googlecar. 
You need to remain in the context of Googlecar and not provide any other information. You can make use of the reference documents provided to you.
Your response need to be complete, concise, empathetic and conversational. Ignore the reference document if it is not related to the question.
QUESTION : {query_oneliner}
"""
# Enrich the user prompt with the retrieved documents
prompt += f"PASSAGES: {all_passages}"

print(f"Prompt: {prompt}")

#Generate the response using the Gemini API
answer = client.models.generate_content(
       model = "gemini-2.0-flash",
       contents=prompt
)
print(f"\n\nANSWER: {answer.text}")