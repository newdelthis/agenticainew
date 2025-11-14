import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv(override=True)

# Define the state
class CompareState(TypedDict):
    dev_html: str
    deployed_html: str
    differences: str

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the comparison function
def compare_html(state: CompareState) -> CompareState:
    dev_html = state.get("dev_html", "")
    deployed_html = state.get("deployed_html", "")

    prompt = f"""
    Compare the following two HTML files for visual/UI differences.
    Focus on which UI elements, buttons, forms, or sections are missing
    or altered in the second (deployed) version compared to the first (development) version.

    Return a concise bullet-point summary of the differences.

    --- DEVELOPMENT VERSION ---
    {dev_html[:5000]}

    --- DEPLOYED VERSION ---
    {deployed_html[:5000]}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"differences": response.content}

# Build the LangGraph
graph = StateGraph(CompareState)
graph.add_node("compare_html", compare_html)
graph.add_edge(START, "compare_html")
graph.add_edge("compare_html", END)

compiled_graph = graph.compile()

# Load HTML files and run
if __name__ == "__main__":
    dev_file = r"C:\delthis\dev_version.html"
    deployed_file = r"C:\delthis\deployed_version.html"

    with open(dev_file, "r", encoding="utf-8") as f:
        dev_html = f.read()

    with open(deployed_file, "r", encoding="utf-8") as f:
        deployed_html = f.read()

    inputs = {"dev_html": dev_html, "deployed_html": deployed_html}
    result = compiled_graph.invoke(inputs)

    print("\n=== UI Differences Found ===\n")
    print(result["differences"])
