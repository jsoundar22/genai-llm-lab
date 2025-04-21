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
# In this Lab, we will be simulating a cafe ordering system called BaristaBot. 
# Chat interface in a loop for thecustmers to order bevarages in natual English language. 
# We will build nodes to represent cafe's live menu and the 'back room' ordering system.
# We will use LangGraph with Gemini PAI for a stateful graph based application
## Pre-requisites
# pip3 install -qU 'langgraph==0.3.21' 'langchain-google-genai==2.1.2' 'langgraph-prebuilt==0.1.7'

'''         +---------+
            |  Start  |
            +----+----+
                 |
                 v
            +---------+
            | Chatbot |--------------------+
            +----+----+                    |
                 ^                         |
                 |                         |
     +-----------+------------+            |
     |           |            |            |
     v           v            v            |
+--------+   +----------+   +--------+     |
| Tools  |   | Ordering |   | Human  |     |
+--------+   +----------+   +--------+     |
                              |            |
                              v            |
                            +----+         |
                            |End |<--------+
                            +----+
'''
import google.generativeai as genai
import os
genai.__version__

# Get the API key variable from the environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
                raise ValueError("GEMINI_API_KEY variable not set")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class OrderState(TypedDict):
    """ State representing the customer's order conversation."""
    # The chat conversation, This preserves the conversation history between nodes.
    # add_message Annotation tells LangGraph that the state is updated by appending
    #  each messaege rather than replacing it.
    messages: Annotated[list, add_messages]

    # In progress orders of the customer
    order: list[str]

    # Flag to mark the completion of the order
    finished: bool

# The system instruction defines the behavior of the chatbot. Rules ore different node 
# transition functions. Guardrails for the conversation

BARISTABOT_SYSINIT =(
    "system", # Indicates the message is a system instruction
    "You are a BaristaBot, an interactive cafe ordering system. A human will talk to you about the "
    "available products you have and you will answer any questions about menu items (and only about "
    "menu items - no off-topic discussion, but you can chat about the products and their history). "
    "The customer will place an order for 1 or more items from the menu, which you will structure "
    "and send to the ordering system after confirming the order with the human. "
    "\n\n"
    "Add items to the customer's order with add_to_order, and reset the order with clear_order. "
    "To see the contents of the order so far, call get_order (this is shown to you, not the user) "
    "Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will "
    "display the order items to the user and returns their response to seeing the list. Their response may contain modifications. "
    "Always verify and respond with drink and modifier names from the MENU before adding them to the order. "
    "If you are unsure a drink or modifier matches those on the MENU, ask a question to clarify or redirect. "
    "You only have the modifiers listed on the menu. "
    "Once the customer has finished ordering items, Call confirm_order to ensure it is correct then make "
    "any necessary updates and then call place_order. Once place_order has returned, thank the user and "
    "say goodbye!"
    "\n\n"
    "If any of the tools are unavailable, you can break the fourth wall and tell the user that "
    "they have not implemented them yet and should keep reading to do so.",
)
# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to the BaristaBot cafe. Type `q` to quit. How may I serve you today?"

# Define a chatbot node that will execute a single turn in a chat conversation using the instructions supplied.
# state = nde(state), The state (a Python dictionary) 

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def chatbot(state: OrderState) -> OrderState:
    """ The chatbot itself, a simple wrapper around the model's own chat interface. """
    message_history = [BARISTABOT_SYSINIT] + state["messages"]
    return {"messages": [llm.invoke(message_history)]}

# Setup the initial graph based on our state definition
graph_builder = StateGraph(OrderState)

# Add the chabot funtion to the app graph as node call 'chatbot'
graph_builder.add_node("chatbot", chatbot)

# Define the chatbot node as the app entrypoint
graph_builder.add_edge(START, "chatbot")

chat_graph = graph_builder.compile()

######
from pprint import pprint

user_msg = "Hello, what can you do?"
state = chat_graph.invoke({"messages": [user_msg]})

# Comment it to remove output clutter after initial understanding
pprint(state)

print ("******** The final two messages of interest ********")
for msg in state["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")

        