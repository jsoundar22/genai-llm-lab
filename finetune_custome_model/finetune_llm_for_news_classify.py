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
# In this codelab walks you tuning a model with the API. You will fine-tune a model
# to classify the category a piece of text (a newsgroup post) into the category
# it belongs to (the newsgroup name).
# 
# PREREQUISITE
# !pip uninstall -qqy jupyterlab  # Remove unused conflicting packages
# !pip install -U -q "google-genai==1.7.0"

from google import genai
from google.genai import types
import os
genai.__version__

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
                raise ValueError("GEMINI_API_KEY variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

for m in client.models.list():
        if "createTunedModel" in m.supported_actions:
                print(f"Model which supports tuning: {m.name}")
                # models/gemini-1.5-flash-001-tuning

# Get the newsgroup dataset to train
# The [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) 
# contains 18,000 newsgroups posts on 20 topics divided into training and test sets.
from sklearn.datasets import fetch_20newsgroups

newsgroup_train = fetch_20newsgroups(subset="train")
newsgroup_test = fetch_20newsgroups(subset="test")

# The list of class names in the dataset
newsgroup_train.target_names

#print(newsgroup_train.data[0])

# Prepare the dataset, preprocess using the same methods on lab clssification_ml_through_embedding.py
import email
import re
import pandas as pd

def preprocess_newsgroup_row(data):
    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate the text to fit within the input limits
    text = text[:40000]

    return text

def preprocess_newsgroup_data(newsgroup_dataset):
    # Put data points into dataframe
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data,
         "Label": newsgroup_dataset.target}
    )
    # Clean up the text
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    # Match label to target name index
    # Maps numeric labels (like 4) to string class names (like 'rec.sport.hockey')
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])

    return df

# Apply preprocessing to training and testing
df_train = preprocess_newsgroup_data(newsgroup_train)
df_test = preprocess_newsgroup_data(newsgroup_test)

df_train.head()

# We are using the fine tuning technique called PEFT - Parameter Efficient Fine Tuning
# Will keep only 50 rows for each news category, as that is sufficeitn for PEFT

def sample_data(df, num_samples, classes_to_keep):
        # Sample rows selecting num_samples of each label
        df = (
                df.groupby("Label")[df.columns]
                .apply(lambda x : x.sample(num_samples))
                .reset_index(drop=True)
        )
        df = df[df["Class Name"].str.contains(classes_to_keep)]
        df["Class Name"] = df["Class Name"].astype("category")
        return df

TRAIN_NUM_SAMPLES = 50
TEST_NUM_SAMPLES = 10

# Keep only rec.* and sci.*
CLASSES_TO_KEEP = "^rec|^sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)

# Evaluate baseline performance with a sample row for visual inspection
sample_idx = 1
sample_row = preprocess_newsgroup_row(newsgroup_test.data[sample_idx])
sample_label = newsgroup_test.target_names[newsgroup_test.target[sample_idx]]

print(sample_data)
print('---')
print(f"Label : {sample_label}")

'''
# Call the model with a simple prompt
prompt = "Which news group category does this news belong too?"
response = client.models.generate_content(
        model="gemini-1.5-flash-001", contents = [prompt, sample_row])
print(f"Response: {response.text}")
'''

# Try system prompt to get a succint answer for a direct question
system_instruct = """
You are a classification service, Your input will have a news grouppost. 
your task is to respond with newsgroup the input belongs too.
"""
from google.api_core import retry

# Define retry to handle the token limit
is_retriable = lambda e: isinstance(e, genai.errors.APIError and e.code in {429, 503})

@retry.Retry(predicate=is_retriable)
def predict_label(post: str) -> str:
        response = client.models.generate_content(
                model="gemini-1.5-flash-001", 
                config=types.GenerateContentConfig(
                        system_instruction=system_instruct),
                contents = post)
        rc = response.candidates[0]

        # Mark genereral eeeor on any
        if rc.finish_reason.name != "STOP":
                return "(error)"
        else:
                return response.text.strip()
'''
prediction = predict_label(sample_row)   
print(f"prediction : {prediction}")
print("Correct!" if prediction == sample_data else "Incorrect")
'''
# Try few mre samples and calculate the accuracy
# Add progress bar to the pandas with taqaddam (tqdm)

import tqdm
from tqdm.rich import tqdm as tqdmr
import warnings

tqdmr.pandas()
# Suppress the experimental warning from tqdm
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)

# Sample the test data
df_baseline_eval = sample_data(df_test, 2, ".*")

# Predict using the sample data
df_baseline_eval['Prediction'] = df_baseline_eval['Text'].progress_apply(predict_label)

# Calculate the accuracy
accuracy = (df_baseline_eval["Class Name"] == df_baseline_eval["Prediction"]).sum() / len(df_baseline_eval)
print(f"Accuracy: {accuracy:.2%}") # Calculated accuracy is 18%
#print("----------------------")
#print(f"Inspect the df : {df_baseline_eval}")

# Tune a custome model to make it better with new classification. The data will be 
# the precessed text and the output category. Choose to use the following tuning parameter
# epoch_count: defines how many times to loop through the data,
# batch_size: defines how many rows to process in a single step, and
# learning_rate: defines the scaling factor for updating model weights at each step.

# Convert the data suitable for tuning
input_data = {'examples': df_train[['Text', 'Class Name']].
              rename(columns={'Text': 'textInput', 'Class Name' : 'output'}).
              to_dict(orient='records')
             }

# Give a model ID to identify it while rerunning
model_id = None

# Reuse if any recently trained model
if not model_id:
        queued_model = None
        # Newest model first
        for m in reversed(client.tunings.list()):
                if m.name.startswith('tunedModels/newsgroup-classification-model'):
                        if m.state.name == 'JOB_STATE_SUCCEEDED':
                                model_id = m.name
                                print(f"Found existing model to reuse: {m.name}")
                                break
                        elif m.state.name == 'JOB_STATE_RUNNING' and not queued_model:
                                queued_model= m.name
                else:
                        if queued_model:
                                model_id = queued_model
                                print(f"Found queued model, still eaiting: {model_id}")

if not model_id:
        tuning_op = client.tunings.tune(
                base_model="models/gemini-1.5-flash-001-tuning",
                training_dataset=input_data,
                config=types.CreateTuningJobConfig(
                        tuned_model_display_name="Jay tuned Newgroup Classification model",
                        batch_size=16,
                        epoch_count=2
                )
        )
        print(f"Tuning State : {tuning_op.state}")
print(model_id)        

# Try the newly tuned model for prediction
new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""

response = client.models.generate_content(
        model=model_id, contents=new_text)
print(f"*****Response from newly tuned model: {response.txt}")