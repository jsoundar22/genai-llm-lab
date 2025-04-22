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
"""
user_msg = "Hello, what can you do?"
state = chat_graph.invoke({"messages": [user_msg]})
"""

# Comment it to remove output clutter after initial understanding
#pprint(state)

def print_last_two_msgs(state):
    print ("******** The final two messages of interest ********")
    for msg in state["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")

"""
print_last_two_msgs(state)
# Extend the conversation
user_msg = "Good, what are the breverages do you have?"
state["messages"].append(user_msg)
state = chat_graph.invoke(state)
print_last_two_msgs(state)
"""
# Add a human node
from langchain_core.messages.ai import AIMessage
def human_node(state: OrderState) -> OrderState:
      """Display the last model message to user and receive the user's input"""
      last_msg = state["messages"][-1]
      print (f" Model : {last_msg.content}")

      user_input = input("User: ")

      # Quit the chat, if the user like to
      if user_input in ["q", "quit", "exit", "goodbye"]:
            state['finished'] = True

      return state | {"messages": [("user", user_input)]}

def chatbot_with_welcome_msg(state: OrderState) -> OrderState:
      """Chatbot itself, a wrapper around models own chat interface."""
      if state['messages']:
            # Continue the conversation, if there are message
            new_output = llm.invoke([BARISTABOT_SYSINIT] + state['messages'])
      else:
            # If there are no messages, start with welcome
            new_output = AIMessage(content=WELCOME_MSG)

      return state | {'messages': [new_output]}

# Start building a new graph
graph_builder = StateGraph(OrderState)

# Add the chatbot and the human node to the app graph
graph_builder.add_node("chatbot", chatbot_with_welcome_msg)
graph_builder.add_node("human", human_node)

# Start with the chatbot again, the chat will always go to the human next
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "human")

# If add an edge back from human to chatbot, it will be loop forever. 
# Add a conditional edge to interpret the human wants to quit
from typing import Literal

def check_if_human_want_to_exit(state: OrderState) -> Literal["chatbot", "__end__"]:
    """Route back to the chatbot if the human do not want to exit"""
    if state.get("finished", False):
          return END
    else:
          return "chatbot"
    
graph_builder.add_conditional_edges("human", check_if_human_want_to_exit)
chat_with_human_graph = graph_builder.compile()

# Setting the recursion limit to 100 from the default 25, to play around a bit. 
# Enter q/quit/exit to come out of the chat
config = {"recursion_limit": 100}

#state = chat_with_human_graph.invoke({"messages": []}, config)
#print_last_two_msgs(state)

# Add live menu
from langchain_core.tools import tool

@tool
def get_menu() -> str:
    """Provide the latest menu."""
      # We are hard coding some menu items here, in real world you can connect to 
      # your database, refine with your inventory
    return """
    MENU:
    Coffee Drinks:
    Espresso
    Americano
    Cold Brew

    Coffee Drinks with Milk:
    Latte
    Cappuccino
    Cortado
    Macchiato
    Mocha
    Flat White

    Tea Drinks:
    English Breakfast Tea
    Green Tea
    Earl Grey

    Tea Drinks with Milk:
    Chai Latte
    Matcha Latte
    London Fog

    Other Drinks:
    Steamer
    Hot Chocolate

    Modifiers:
    Milk options: Whole, 2%, Oat, Almond, 2% Lactose Free; Default option: whole
    Espresso shots: Single, Double, Triple, Quadruple; default: Double
    Caffeine: Decaf, Regular; default: Regular
    Hot-Iced: Hot, Iced; Default: Hot
    Sweeteners (option to add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, chocolate sauce, sugar free vanilla sweetener
    Special requests: any reasonable modification that does not involve items not on the menu, for example: 'extra hot', 'one pump', 'half caff', 'extra foam', etc.

    "dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
    "Regular milk" is the same as 'whole milk'.
    "Sweetened" means add some regular sugar, not a sweetener.

    Soy milk has run out of stock today, so soy is not available.
    """

# Add the new tool to the grapgh. The get_menu is wrapped in a ToolNode that handles calling th tool 
# and passing the response as a message through the graph. The tools are bound by a llm object,
# so that the model is ware of it.
from langgraph.prebuilt import ToolNode

# Define the tools and create a ToolNode
tools = [get_menu]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)

def may_route_to_tools(state: OrderState) -> Literal['tools', 'human']:
      """Route between human or tool_nodes, depending if a tool call is made"""
      if not (msgs := state.get("messages", [])):
            raise ValueError(f"No Messages found while parsing state: {state}")
      
      # Only route based on the last message
      msg = msgs[-1]

      #when the chatbot returns tool_calls, route to the 'tools' menu
      if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            return "tools"
      else:
            return "human"
      
def chatbot_with_tools(state: OrderState)->OrderState:
      """The chatbot with tools. A simple wrapper around the models own chat interface"""
      defaults = {"orders": [], "finished" : False}

      if state["messages"]:
            new_output = llm_with_tools.invoke([BARISTABOT_SYSINIT] + state["messages"])
      else:
            new_output = AIMessage(content=WELCOME_MSG)

graph_builder = StateGraph(OrderState)

#Add the nodes including the new tool_node
graph_builder.add_node("chatbot", chatbot_with_welcome_msg)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)

# Chatbot may goto human or tools
graph_builder.add_conditional_edges("chatbot", may_route_to_tools)
# Human may go back to chatbot or exit
graph_builder.add_conditional_edges("human", check_if_human_want_to_exit)

# Tools always route back to chat afterwards
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_menu = graph_builder.compile()

# Test the chatbot, still we don't have the ordering system in place
#state = graph_with_menu.invoke({"messages":[]}, config)
# pprint(state)

#Handle Orders
# We need to update the state to track order, provide tools to update the state. 
# The model shouldn't expose the internal flows.

from collections.abc import Iterable
from random import randint
from langchain_core.messages.tool import ToolMessage

# These functions have no body: LangGraph doesn't allow @tools to update the 
# conversation state, lets impletement a separate noe 'order_node'.
# Using @tools is the convinient way for defining the tools schema, 
# so empty functions have been defined that will be bound to the LLM
# but their implementation is deffered to the order_node.

@tool
def add_to_order(drink: str, modifiers: Iterable[str]) -> str:
      """ Adds the specified drink to the customer's order including any modifiers.
      
      Returns:
        The update order in progress
      """

@tool
def confirm_order() -> str:
      """Asks the customer if the order is correct.
      returns:
      The user's free-text response.
      """

@tool
def get_order()->str:
      """ Returns user's order so far, One item per line"""      

@tool
def clear_order():
      """Removes all orders from user's order"""

@tool
def place_order() -> int:
      """Send the order to barista for the fulfillment
      
      Returns:
      The estiated wait time in terms of number of minutes"""      

@tool
def order_node(state: OrderState) -> OrderState:
      """The ordering node where the order state is manipulated"""      
      tool_msg = state.get("messages",[])[-1]
      order = state.get("order", [])
      outbound_msgs = []
      order_placed = False

      for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_to_order":

            # Each order item is just a string. This is where it assembled as "drink (modifiers, ...)".
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"

            order.append(f'{tool_call["args"]["drink"]} ({modifier_str})')
            response = "\n".join(order)

        elif tool_call["name"] == "confirm_order":

            # We could entrust the LLM to do order confirmation, but it is a good practice to
            # show the user the exact data that comprises their order so that what they confirm
            # precisely matches the order that goes to the kitchen - avoiding hallucination
            # or reality skew.

            # In a real scenario, this is where you would connect your POS screen to show the
            # order to the user.

            print("Your order:")
            if not order:
                print("  (no items)")

            for drink in order:
                print(f"  {drink}")

            response = input("Is this correct? ")

        elif tool_call["name"] == "get_order":

            response = "\n".join(order) if order else "(no order)"

        elif tool_call["name"] == "clear_order":

            order.clear()
            response = None

        elif tool_call["name"] == "place_order":

            order_text = "\n".join(order)
            print("Sending order to kitchen!")
            print(order_text)

            # TODO(you!): Implement cafe.
            order_placed = True
            response = randint(1, 5)  # ETA in minutes

        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
        return {"messages": outbound_msgs, "order": order, "finished": order_placed}
      
def maybe_route_to_tools(state: OrderState) -> str:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    msg = msgs[-1]

    if state.get("finished", False):
        # When an order is placed, exit the app. The system instruction indicates
        # that the chatbot should say thanks and goodbye at this point, so we can exit
        # cleanly.
        return END

    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Route to `tools` node for any automated tool calls first.
        if any(
            tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls
        ):
            return "tools"
        else:
            return "ordering"

    else:
        return "human"      
            
# Time to define the graph, the LLM needs to know about the tools to invoke them.
# We are setting up two sets of tools, one for automated and other for ordering

auto_tools = [get_menu]
tool_node = ToolNode(auto_tools)

#Order-tools will be handled by the order node
order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]

# Specify all the tools for the LLM to know
llm_with_tools = llm.bind_tools(auto_tools + order_tools)

graph_builder = StateGraph(OrderState)

#Nodes
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

# Chatbot -> {ordering, tools, human, END}
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)

# Human -> {chatbot, END}
graph_builder.add_conditional_edges("human", check_if_human_want_to_exit)

# Tools (both the kinds of tools) always route back to chat afterwards
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")

graph_builder.add_edge(START, "chatbot")

graph_with_order_tools = graph_builder.compile()

# Invoke the graph to start the chatbot
state = graph_with_order_tools.invoke({"messages": []}, config)
#state = graph_with_order_tools.invoke({"messages": [START]}, config)

pprint(state["order"])