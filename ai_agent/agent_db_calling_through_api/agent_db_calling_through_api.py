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
# In this Lab, you will use the Gemini API's automatic function calling to build
#  a chat interface over a local database (SQLite) 
#  This is just a PoC, observe due deligence on safety and security constraints you 
#  would use in a real-world example.
# 
import google.generativeai as genai
import os
genai.__version__

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
                raise ValueError("GEMINI_API_KEY variable not set")

genai.configure(api_key=GEMINI_API_KEY)

## Create a local database
# For this minimal example, you'll create a local SQLite database and add some synthetic data so you have something to query.
# run the create_db.py script to create the database and add some data
import sqlite3

db_file = "retail_store.db"
if not os.path.exists(db_file):
    raise FileNotFoundError(f"Database file {db_file} does not exist. Please run create_db.py to create it.")

db_conn = sqlite3.connect(db_file)

def list_tables()-> list[str]:
    """
    List all tables in the database
    """
    db_cursor = db_conn.cursor()

    db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = db_cursor.fetchall()
    return [t[0] for t in tables]

list_tables()

def describe_table(table_name: str) -> list[tuple[str,str]]:
        """ Lookup the table scema.
        Args:
            table_name (str): The name of the table to describe.
        Returns: 
            list[tuple[str,str]]: A list of tuples containing the column names and their types.
        """
        db_cursor = db_conn.cursor()
        db_cursor.execute(f"PRAGMA table_info({table_name});")
        schema = db_cursor.fetchall()
        return [(col[1], col[2]) for col in schema]

# We have build enough context with available tables and their schema

def execute_query(sql: str) -> list[list[str]]:
    """
    Execute a query and return the results
    Args:
        sq (str): The SQL query to execute.
    Returns:
        list[list[str]]: The results of the query.
    """
    db_cursor = db_conn.cursor()
    db_cursor.execute(sql)
    return db_cursor.fetchall()

# Time to putall this together and create a function to call the database
# Function calling woks by passing specific messages to the chat session.
# The function schema and execute functions are made available to the model
# and a conversation is started, the model may return a function_call instead 
# of the standard text response.
# When we receive a function_call, the client must respond with a function_response 
# with the results of the function_call and the conversation can continue on as normal.

# The work flow with function call is meant for human in the loop kind of conversation. 
# However python SDK supports automatic function calling, where the supplied function 
# will be called automatically 
# --- This is a powerful mechanism use it with caution where you deem it appropriate.

db_tools = [list_tables, describe_table, execute_query]
# The tools are passed to the model in the function_call message

instruction = """You are a helpful chatbot that can interact with an SQL database for a retail
store. You will take the users questions and turn them into SQL queries using the tools
available. Once you have the information you need, you will answer the user's question using
the data returned. Use list_tables to see what tables are present, describe_table to understand
the schema, and execute_query to issue an SQL SELECT query."""

model = genai.GenerativeModel(
       "models/gemini-1.5-flash-latest",
       tools = db_tools,
       system_instruction=instruction, 
)

# Define a retry policy. The model might make multiple consecutive calls automatically
# for a complex query, this ensures the client retries if it hits quota limits.
from google.api_core import retry

retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

# Start a chat with automatic function calling enabled
chat = model.start_chat(enable_automatic_function_calling=True)

# start a coversation
response = chat.send_message("What is the inventory of your products? and the cheapest product?")
# Print the response
print("Response:", response.text)

# You can continue the conversation by sending more messages, if need to reset the chat
# chat.rewind() or start_chat() to start a new chat

response = chat.send_message("Which of the staff is good at cross selling?")
# Print the response
print("Response:", response.text)
# The response for the above query :- Response: There is no information available on staff performance 
# or cross-selling abilities in the database.  To answer your question, I would need a table with 
# information on staff and their sales performance, including metrics related to cross-selling.

# If you are curious about what is happening under the hood, you can check the logs
import textwrap


def print_chat_turns(chat):
    """Prints out each turn in the chat history, including function calls and responses."""
    for event in chat.history:
        print(f"{event.role.capitalize()}:")

        for part in event.parts:
            if txt := part.text:
                print(f'  "{txt}"')
            elif fn := part.function_call:
                args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                print(f"  Function call: {fn.name}({args})")
            elif resp := part.function_response:
                print("  Function response:")
                print(textwrap.indent(str(resp), "    "))

        print()


print_chat_turns(chat)
