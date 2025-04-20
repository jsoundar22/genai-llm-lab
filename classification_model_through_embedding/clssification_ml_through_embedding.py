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
# In this lab, We will use the Embedding from Gemini API to train the Classificaion model.
# This approach will be much efficient and less data than traditional ML classification.
# Newsgroup data is the dataset we will use for this lab.

# SETUP
# install -qU "google-genai==1.7.0" "chromadb==0.6.3"
# 
from google import genai
from google.genai import types
import os
genai.__version__

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
                raise ValueError("GEMINI_API_KEY variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# View the llst of class names for the dataset
#print(f"News Group Class Names: {newsgroups_train.target_names}")

#view a record from the dataset
#print(f"News Group Record: {newsgroups_train.data[0]}")

# Preprocess the data
# The News post is a collection of emails, which will have sensitive information 
# PII (Personally Identifiable information) like email address, phone number and names
# Which is not needed for our classification, the cleaned data can be used in other context as well.

import re
import email
import pandas as pd

def preprocess_newsgroup_row(data):
        msg = email.message_from_string(data)
        text = f"{msg['Subject']}\n\n{msg.get_payload()}"
        # Remove any remaining email addresses
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.+-]+\.[a-zA-Z0-9-.]+", "",text)
        #truncate each entry to 5000 characters
        text = text[:5000]
        return text

def preprocess_newsgroup_data(newsgroup_dataset):
        # Extract data points into DataFrame
        df = pd.DataFrame(
                {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
        )
        # Clean up the Text
        df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
        # Match label to target name index
        df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])
        return df

# Apply preprocessing to training and test data
df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

#print (f"Train data Head: {df_train.head()}")

# Sample the data by taking 100 data points in the training set, and dropping a few 
# of the categories to run thriugh this lab. For our comparison choose Science category

def sample_data(df, num_samples, classes_to_keep):
        df = (df.groupby("Label")[df.columns].
                apply(lambda x: x.sample(num_samples)).
                reset_index(drop=True))
        
        df = df[df["Class Name"].str.contains(classes_to_keep)]
        # We have few categories now, so re-calibrate the label encoding
        df["Class Name"] = df["Class Name"].astype("category")
        df["Encoded Label"] = df["Class Name"].cat.codes

        return df

TRAIN_NUM_SAMPLES = 100
TEST_NUM_SAMPLES = 25
#Class Name should contain "sci" to keep the science categories.
#Try different label from the data - see the newsgroup_train.target_names
CLASSES_TO_KEEP = "sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)
#print(f"Train data Head: {df_train.head()}")

from google.api_core import retry
# tqdm is a library for progress bars
import tqdm
from tqdm.rich import tqdm as tqdm
import warnings

# add pandas to tqdm
tqdm.pandas()
#warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)

# Define a helper fnction to retry when per minute quota is reached
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

@retry.Retry(predicate=is_retriable, timeout=300.0)
def embed_function(test: str) -> list[float]:
        # Since we are performing classification, set the task_type accordingly
        response = client.models.embed_content(
                model="models/text-embedding-004",
                contents=test,
                config=types.EmbedContentConfig(
                        task_type="classification",
                ),
        )
        return response.embeddings[0].values

def create_embeddings(df):
        df["Embeddings"] = df["Text"].progress_apply(embed_function)
        return df

df_train = create_embeddings(df_train)
df_test = create_embeddings(df_test)
print(f"Train data Head: {df_train.head()}")

# Build the classification model
# Simple model with one input later, a hiddden and a output layer
# to specify the class probability. The prediction corresponds to the
# probability of a pience of text in a particular class of news
## Keras will shuffle the data points, metrics calculation and other ML boilerplate

import keras
from keras import layers

def build_classification_model(input_size : int, num_classes: int) -> keras.Model:
        model = keras.Sequential(
                [
                        layers.Input([input_size], name="embedding_inputs"),
                        layers.Dense(input_size, activation='relu', name='hidden'),
                        layers.Dense(num_classes, activation='softmax', name='output_probs'),
                ]
        )
        return model
# Build the model

# Set the embedding size
embedding_size = len(df_train["Embeddings"].iloc[0])

classifier = build_classification_model(
        embedding_size, len(df_train["Class Name"].unique())
)

classifier.summary()

classifier.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics = ["accuracy"]
)

#Train the model
# We will exit early if loss value stabilizes, irrespective of the number of epocs
import numpy as np

NUM_EPOCHS = 20
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 3
early_stopping = keras.callbacks.EarlyStopping(monitor="accuracy", patience=EARLY_STOPPING_PATIENCE)

# Split the X and Y component of the Train and validation subsets

y_train = df_train["Encoded Label"]
x_train = np.stack(df_train["Embeddings"])
y_val = df_test["Encoded Label"]
x_val = np.stack(df_test["Embeddings"])

history = classifier.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        callbacks = [early_stopping],
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
)

# Evaluate model performance
loss, accuracy = classifier.evaluate(x=x_val, y=y_val, return_dict=True)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

## Test with a few samples
def make_prediction(text: str) -> list[float]:
        """Infer the news category from the provided text"""
        # Note that the model takes embedding as input, convert the text to embedding first
        embedded = embed_function(text)
        # Convert the input into batch, as the model expects a batch of inputs
        inpp = np.array([embedded])
        # Make the prediction
        [result] = classifier.predict(inpp)
        return result

# This example avoids any space-specific terminology to see if the model avoids
# biases towards specific jargon.
new_text1 = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""
new_text2 = """
What is the differences between ReLU, Softmax and Sigmoid
"""
# Make the prediction
result = make_prediction(new_text1)

for idx, category in enumerate(df_test["Class Name"].cat.categories):
        print(f"{category}: {result[idx] * 100:0.2f}%")

result = make_prediction(new_text2)

for idx, category in enumerate(df_test["Class Name"].cat.categories):
        print(f"{category}: {result[idx] * 100:0.2f}%")
