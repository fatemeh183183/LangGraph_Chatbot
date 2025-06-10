#chatbot_core.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_groq import ChatGroq
import os
#1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]
#2. Load Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

tools = [wiki_tool, arxiv_tool]
#3.Load Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
llm_with_tools = llm.bind_tools(tools=tools)
#4. Define Chat Function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
#5. Build Graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")
graph_builder.add_edge("chatbot", "tools")
graph_builder.add_conditional_edges("tools", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
