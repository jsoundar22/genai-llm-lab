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
# In this lab, We have some text with simialr phrases, phrases with spelling mistakes.
# Few unrelted text. Activity is to extract similarity using embedding.
# The task_type is 'semantic_similarity', where as in the previous RAG activty we used 'embedding_task'

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
for model in client.models.list():
        if 'embedContent' in model.supported_actions:
                print(model.name)

texts = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick rbown fox jumps over the lazy dog.',
    'teh fast fox jumps over the slow woofer.',
    'a quick brown fox jmps over lazy dog.',
    'brown fox jumping over dog',
    'fox > dog',
    # Alternative pangram for comparison:
    'The five boxing wizards jump quickly.',
    # Unrelated text, also for comparison:
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et hendrerit massa. Sed pulvinar, nisi a lobortis sagittis, neque risus gravida dolor, in porta dui odio vel purus.',
]

response = client.models.embed_content(
        model='models/text-embedding-004',
        contents = texts,
        config=types.EmbedContentConfig(task_type='semantic_similarity'))

# Helper function to display longr text
def truncate (t: str, limit: int = 50) -> str:
        """Truncate the text to the specified limit"""
        if len(t) > limit:
                return t[:limit-3] + '...'
        else:
                return t
        
truncated_texts = [truncate(t) for t in texts]      

# Display the similatirty in HeatMap - 0.0 (Completelt Dissimilar) to 1.0 (completly Similar)
import pandas as pd
import seaborn as sns

df = pd.DataFrame([e.values for e in response.embeddings], index=truncated_texts)

# Perform the similarity calculations
sim = df @ df.T

# Draw the heatmap
import matplotlib.pyplot as plt

# Use a non-interactive backend to avoid Qt-related issues
plt.switch_backend('Agg')

sns.heatmap(sim, vmin=0, vmax=1, cmap='Greens')

# Save the heatmap to a file instead of displaying it
plt.savefig('heatmap.png')

# Check score for a phrase directly
sim['The quick brown fox jumps over the lazy dog.'].sort_values(ascending=False)
