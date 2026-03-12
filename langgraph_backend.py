from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import os
from google import genai
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    prompt = "\n".join(f"{message.type}: {message.content}" for message in messages)
    response = llm.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return {"messages": [AIMessage(content=response.text or "")]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# Implement Streaming
for message_chunk, metadata in chatbot.stream(
    {"messages": [HumanMessage(content="What is the recipe for chocolate chip cookies?")]},
    stream_mode="messages",
    config= {"configurable":{'thread_id': 'thread_1'}}):

    chunk_text = getattr(message_chunk, "content", str(message_chunk))
    if chunk_text:
        print(chunk_text, end=" ", flush=True)
    