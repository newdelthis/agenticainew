# pip install mailersend chromadb crewai langchain-huggingface langchain-openai gradio

import os
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv
from mailersend import emails
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from crewai import Agent, Task, Crew
from crewai.tools import tool

# Setup
load_dotenv(override=True)

ntfy_topic = os.getenv("NTFY_URGENT_TICKETS_TOPIC")
serp_api_key = os.getenv("SERPAPI_API_KEY")
mailersend_api_key = os.getenv("MAILERSEND_API_KEY")

# Load Chroma VectorDB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    collection_name="customer_support",
    embedding_function=embeddings,
    persist_directory="c://code//agenticai//4_crewai//customer_support_chroma"
)

# Verify database is loaded
try:
    count = vectordb._collection.count()
    print(f"VectorDB loaded successfully: {count} documents")
    if count == 0:
        print("WARNING: VectorDB is empty! Run embedding creation script first.")
except Exception as e:
    print(f"ERROR loading VectorDB: {e}")

df = pd.read_csv("c://code//agenticai//4_crewai//customer_tickets.csv")

# Fixed VectorSearch tool with explicit query parameter
@tool("VectorSearch")
def vector_search(query: str) -> dict:
    """Look up support tickets in the Chroma vector database. ALWAYS pass the user's question as the query parameter."""
    
    if not query or not query.strip():
        return {"answer": None, "priority": None, "error": "Empty query provided"}
    
    try:
        # Perform vector search
        results = vectordb.similarity_search(query.strip(), k=1)
        
        # Check if results exist and have content
        if results and len(results) > 0:
            doc = results[0]
            answer = doc.metadata.get("answer", "No answer available")
            priority = doc.metadata.get("priority", "low")
            
            # Normalize priority to lowercase
            if priority:
                priority = str(priority).strip().lower()
            
            return {
                "answer": answer,
                "priority": priority,
                "matched_text": doc.page_content[:200],
                "success": True
            }
        else:
            return {"answer": None, "priority": None, "error": "No matches found in database", "success": False}
            
    except Exception as e:
        return {"answer": None, "priority": None, "error": f"Search failed: {str(e)}", "success": False}


@tool("SendNTFY")
def send_ntfy(message: str) -> str:
    """Send urgent push notification for high priority tickets"""
    try:
        url = f"https://ntfy.sh/{ntfy_topic}"
        resp = requests.post(url, data=message.encode())
        return f"ntfy push sent (status {resp.status_code})"
    except Exception as e:
        return f"ntfy failed: {str(e)}"


@tool("SendEmail")
def send_test_email(message: str) -> str:
    """Send an email for medium priority tickets"""
    try:
        mailer = emails.NewEmail(mailersend_api_key)
        mail_body = {}

        mail_from = {
            "name": "Chatbot",
            "email": "sender@test-nrw7gymkvkog2k8e.mlsender.net",
        }

        recipients = [{
            "name": "Our Customer",
            "email": "ekahate@gmail.com",
        }]

        reply_to = [{
            "name": "Chatbot",
            "email": "receiver@test-nrw7gymkvkog2k8e.mlsender.net",
        }]

        mailer.set_mail_from(mail_from, mail_body)
        mailer.set_mail_to(recipients, mail_body)
        mailer.set_subject("Support Ticket", mail_body)
        body_text = f"Customer raised a medium priority issue:\n\n{message}"
        mailer.set_html_content(body_text.replace("\n", "<br>"), mail_body)
        mailer.set_plaintext_content(body_text, mail_body)
        mailer.set_reply_to(reply_to, mail_body)

        resp = mailer.send(mail_body)
        return f"Email sent successfully"
    except Exception as e:
        return f"Email send failed: {str(e)}"


@tool("LowPriorityAck")
def low_priority_ack() -> str:
    """Send acknowledgement for low priority tickets"""
    return "We have noted your request and will attend to it within 48 hours."


@tool("SerpSearch")
def serp_fallback(query: str) -> str:
    """Fallback web search using SerpAPI when no vector match found"""
    if not serp_api_key:
        return "SERPAPI_API_KEY not set"

    params = {"q": query, "api_key": serp_api_key, "engine": "google"}
    try:
        resp = requests.get("https://serpapi.com/search", params=params)
        data = resp.json()
        if "organic_results" in data and len(data["organic_results"]) > 0:
            return data["organic_results"][0].get("snippet", "No snippet available")
        return "No results found via SerpAPI"
    except Exception as e:
        return f"SerpAPI request failed: {str(e)}"


# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# CrewAI Agents
router_agent = Agent(
    role="Ticket Router",
    goal="Classify ticket priority and route to appropriate action",
    backstory=(
        "You are a support triage expert. "
        "STEP 1: Call VectorSearch with the user's query text. "
        "STEP 2: Based on priority from VectorSearch: "
        "- high → call SendNTFY "
        "- medium → call SendEmail "
        "- low → call LowPriorityAck "
        "- no match → call SerpSearch"
    ),
    llm=llm,
    tools=[vector_search, send_ntfy, send_test_email, low_priority_ack, serp_fallback],
    verbose=True,
)

review_agent = Agent(
    role="Support Supervisor",
    goal="Review and create final customer response",
    backstory="Senior support manager crafting professional responses",
    llm=llm,
    tools=[],
    verbose=True,
)


# Main handler
def handle_ticket(user_query: str):
    if not user_query:
        return "Please enter a support request."

    routing_task = Task(
        description=(
            f"User query: '{user_query}'\n\n"
            f"Your task:\n"
            f"1. Call VectorSearch with query='{user_query}'\n"
            f"2. Check the priority returned\n"
            f"3. Call the appropriate tool based on priority\n"
            f"4. Return the result"
        ),
        expected_output="Tool execution results and draft response",
        agent=router_agent,
    )

    review_task = Task(
        description="Review the router's actions and create a polished customer response.",
        expected_output="Final customer-friendly support message",
        agent=review_agent,
    )

    crew = Crew(
        agents=[router_agent, review_agent],
        tasks=[routing_task, review_task],
        verbose=True,
    )

    try:
        final_result = crew.kickoff()
        return str(final_result)
    except Exception as e:
        return f"Support flow failed: {str(e)}"


# Gradio UI
with gr.Blocks(title="Customer Support Crew") as demo:
    gr.Markdown("# Customer Support Crew (ChromaDB + CrewAI)")
    gr.Markdown("VectorDB → Router Agent → Supervisor Agent")

    with gr.Row():
        with gr.Column(scale=1):
            user_query = gr.Textbox(
                label="Your Query",
                placeholder="e.g., I cannot log in to my account"
            )
            run_btn = gr.Button("Submit Ticket", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Support Response",
                lines=20,
                max_lines=30,
                show_copy_button=True
            )

    run_btn.click(
        fn=handle_ticket,
        inputs=user_query,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()