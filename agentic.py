import streamlit as st
import ollama
import re
import time
from typing import List, Dict, Tuple, Optional, TypedDict
import textwrap
import logging
from langgraph.graph import StateGraph, END
# Only import ToolNode
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
 
 
# --- Setup Logging ---
 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
 
# --- Configuration and Constants ---
 
MODEL_NAME = "llama3.2:latest"  # Use a valid Ollama model name
MAX_RETRIES = 3
RETRY_DELAY = 2
TEXT_WRAP_WIDTH = 80
 
# --- Error Handling ---
 
class OllamaCommunicationError(Exception):
    pass
 
def ask_question(query: str, context: Optional[List[Dict[str, str]]] = None,
                 model: str = MODEL_NAME, max_retries: int = MAX_RETRIES) -> str:
 
    messages = []
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": query})
    logging.debug(f"ask_question: Query: {query}, Context: {context}")
 
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=model, messages=messages)
            logging.debug(f"ask_question: Raw Response (Attempt {attempt+1}): {response}")
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                error_msg = "Ollama response missing 'message' or 'content' key."
                logging.error(error_msg)
                raise OllamaCommunicationError(error_msg)
        except Exception as e:
            logging.exception(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise OllamaCommunicationError(f"Failed after {max_retries} retries: {e}")
    return "An unexpected error occurred."
 
# --- Prompt Templates ---
 
IDEA_VALIDATION_PROMPT = """
You are a meticulous startup idea evaluator. Analyze the following idea, considering:
1. **Novelty:** Is it truly innovative, or an incremental improvement?
2. **Market Need:** Is there a demonstrable need for this product/service?
3. **Feasibility:** Can it be realistically developed and implemented?
4. **Scalability:** Does it have the potential to grow significantly?
5. **Potential Risks:** What are the major challenges and risks?
Provide a concise, critical, and well-structured assessment.
 
Idea: {query}
"""
 
COMPETITOR_ANALYSIS_PROMPT = """
You are a thorough market research analyst. Identify and analyze the *direct* and *indirect* competitors for the following startup idea.
For each competitor, provide:
1. **Name:**
2. **Strengths:** (Be specific)
3. **Weaknesses:** (Be specific)
4. **Market Share:** (Estimate if precise data isn't available)
5. **Overall Threat Level:** (Low, Medium, High) - Justify your assessment.
 
Idea: {query}
"""
 
LEGAL_ADVISOR_PROMPT = """
You are an expert in startup law and intellectual property. Analyze the following startup idea and user query.
Provide a concise legal assessment, focusing on potential legal challenges, intellectual property considerations (patents, trademarks, copyrights),
regulatory hurdles, and any other relevant legal aspects.  Be objective and highlight potential risks.
 
Startup Idea and User Query:
{user_query}
 
Validation/Analysis from Idea Validator/Competitor Analyst:
{idea_validation_response}
 
Legal Assessment:
"""
 
EXECUTIVE_SUMMARY_PROMPT = """
You are a seasoned business strategist.  Create a comprehensive executive summary for the following startup idea.  Include:
1. **Concept Overview:** A clear, concise description of the idea.
2. **Target Market:**  Define the specific customer segment(s).
3. **Value Proposition:**  What unique value does it offer to the target market?
4. **Business Model:** How will the business generate revenue?
5. **Competitive Landscape:** Briefly summarize the competitive analysis (from previous interaction).
6. **Financial Projections:** (High-level, qualitative: e.g., "Potential for high growth", "Requires significant investment").
7. **Team:** (Placeholder - in a real app, this would come from user input/database).
8. **Call to Action:** What are the next steps (e.g., further research, prototype development)?
 
Idea: {query}
 
Competitor Analysis Summary (from previous turns): {competitor_analysis}
 
Idea Validation Summary (from previous turns) : {idea_validation}
"""
 
CEO_COMMENTS_PROMPT = """
You are the CEO of a venture capital firm. Review the *entire conversation* below and provide your assessment:
 
**Conversation History:**
 
{conversation_history}
 
Provide your overall assessment, including:
 
1. **Investment Potential:** (High, Medium, Low, with justification)
2. **Key Strengths:**
3. **Key Weaknesses/Risks:**
4. **Next Steps:**
5. **Overall Comment:**
"""
 
ROUTER_PROMPT = """
You are an AI assistant specializing in categorizing startup-related queries.  Your ONLY job is to categorize the user's query.
 
First, check if the user's query is a simple greeting (e.g., "Hello", "Hi", "Good morning"). If it is, respond with ONLY the word "greeting".  Do not include anything else.
 
If the query is NOT a greeting, categorize it into *one* of the following categories.  Output ONLY the category name, and nothing else:
 
*   **idea_validation:** The query is about assessing the potential, feasibility, or risks of a startup idea.
*   **competitor_analysis:** The query is about researching existing businesses in the same or related markets.
*   **executive_summary:** The query is about creating a comprehensive business plan overview or summary.
*   **legal_advisor:** The query is about the legal aspects of a startup idea.
*   **unclear:** The query is ambiguous, off-topic, or you cannot determine the category with confidence.
 
User Query: {query}
 
Category:
"""
 
# --- Prompt Formatting ---
 
def format_prompt(prompt_template: str, **kwargs) -> str:
    return prompt_template.format(**kwargs)
 
# --- Function Definitions (Tools) ---
 
def idea_validator(query: str, context: List[Dict[str, str]]) -> str:
    prompt = format_prompt(IDEA_VALIDATION_PROMPT, query=query)
    logging.info(f"Calling idea_validator with query: {query}")
    return ask_question(prompt, context)
 
def competitor_analysis(query: str, context: List[Dict[str, str]]) -> str:
    prompt = format_prompt(COMPETITOR_ANALYSIS_PROMPT, query=query)
    logging.info(f"Calling competitor_analysis with query: {query}")
    return ask_question(prompt, context)
 
def executive(query: str, context: List[Dict[str, str]], idea_validation: str = "", competitor_analysis: str = "") -> str:
    prompt = format_prompt(EXECUTIVE_SUMMARY_PROMPT, query=query, idea_validation=idea_validation, competitor_analysis=competitor_analysis)
    logging.info(f"Calling executive")
    return ask_question(prompt, context)
 
def legal_advisor(user_query: str, idea_validation_response: str, context: List[Dict[str, str]]) -> str:
    prompt = format_prompt(LEGAL_ADVISOR_PROMPT, user_query=user_query, idea_validation_response=idea_validation_response)
    logging.info(f"Calling legal_advisor")
    return ask_question(prompt, context)
 
def ceo_comments(context: List[Dict[str, str]]) -> str:
    conversation_history = ""
    for message in context:
        conversation_history += f"{message['role'].upper()}: {message['content']}\n\n"
    prompt = format_prompt(CEO_COMMENTS_PROMPT, conversation_history=conversation_history)
    logging.info("Calling ceo_comments")
    return ask_question(prompt, context=None)
 
def router(query: str, context: List[Dict[str, str]]) -> str:
    prompt = format_prompt(ROUTER_PROMPT, query=query)
    response_text = ask_question(prompt, context).strip().lower()
    logging.info(f"Calling router with query: {query}, initial response: {response_text}")
 
    if response_text == "greeting":
        return "greeting"
    elif "idea_validation" in response_text:
        return "idea_validation"
    elif "competitor_analysis" in response_text:
        return "competitor_analysis"
    elif "executive_summary" in response_text:
        return "executive_summary"
    elif "legal_advisor" in response_text:
        return "legal_advisor"
    else:
        return "unclear"
 
# --- UI Helper ---
 
def display_wrapped_message(message: Dict[str, str]):
    with st.chat_message(message['role']):
        st.write(textwrap.fill(message['content'], width=TEXT_WRAP_WIDTH))
 
# --- LangGraph Agentic Flow ---
 
# 1. Define the tools
tools = [
    StructuredTool.from_function(
        func=idea_validator,
        name="idea_validation",
        description="Assess the potential, feasibility, and risks of a startup idea.",
    ),
    StructuredTool.from_function(
        func=competitor_analysis,
        name="competitor_analysis",
        description="Research existing businesses in the same or related markets.",
    ),
    StructuredTool.from_function(
        func=executive,
        name="executive_summary",
        description="Create a comprehensive business plan overview or summary.",
    ),
    StructuredTool.from_function(
        func=legal_advisor,
        name="legal_advisor",
        description="Analyze the legal aspects of a startup idea.",
    ),
    StructuredTool.from_function(
        func=ceo_comments,
        name="ceo_comments",
        description="Provides CEO comments based on the conversation history.",
    ),
]
 
 
# 2. Define the agent state
class AgentState(TypedDict):
    input: str  # User's input
    chat_history: List[Dict[str, str]]  # Conversation history
    next_action: str # "tool", "return"  # This is not used directly, but good for clarity
    tool_name: Optional[str]  # Name of the tool to use, if applicable. Not directly used, but helpful for understanding
    tool_input: Optional[Dict]  # Input to the tool, if applicable.  Not directly used, but helpful for understanding
    agent_outcome: Optional[str]  # Final response, after tool use
 
# 3. Define the nodes (functions that will be executed)
def route_node(state: AgentState) -> str:
    """Routes the flow based on user input and conversation history."""
    if not state["chat_history"]:
        category = router(state["input"], [])
    else:
        last_message = state["chat_history"][-1]
        if "content" in last_message:
            content = last_message["content"]
            if "ceo comments" in content.lower():
                return "ceo_comments"
            elif "(Category:" in content:
                category_match = re.search(r"\(Category: (.*?)\)", content)
                category = category_match.group(1).strip().lower() if category_match else "unclear"
            else:
                category = "unclear"
        else:
            category = "unclear"
 
    logging.info(f"Route Node Category: {category}")
    return category  # Return the category string directly
 
 
def idea_validation_node(state: AgentState) -> Dict:
    response = idea_validator(state["input"], state["chat_history"])
    return {"agent_outcome": response, "chat_history": state["chat_history"] + [{"role": "assistant", "content": response}]}
 
def competitor_analysis_node(state: AgentState) -> Dict:
    response = competitor_analysis(state["input"], state["chat_history"])
    return {"agent_outcome": response, "chat_history": state["chat_history"] + [{"role": "assistant", "content": response}]}
 
def executive_summary_node(state: AgentState) -> Dict:
    idea_val = next((msg["content"] for msg in reversed(state["chat_history"]) if "(Category: idea_validation)" in msg["content"]), "")
    comp_anal = next((msg["content"] for msg in reversed(state["chat_history"]) if "(Category: competitor_analysis)" in msg["content"]), "")
    response = executive(state["input"], state["chat_history"], idea_val, comp_anal)
    return {"agent_outcome": response, "chat_history": state["chat_history"] + [{"role": "assistant", "content": response}]}
 
def legal_advisor_node(state: AgentState) -> Dict:
    idea_val_response = next((msg["content"] for msg in reversed(state["chat_history"]) if "(Category: idea_validation)" in msg["content"]), "")
    response = legal_advisor(state["input"], idea_val_response, state["chat_history"])
    return {"agent_outcome": response, "chat_history": state["chat_history"] + [{"role": "assistant", "content": response}]}
 
def ceo_comments_node(state: AgentState) -> Dict:
    response = ceo_comments(state["chat_history"])
    return {"agent_outcome": response, "chat_history": state["chat_history"] + [{"role": "assistant", "content": response}]}
 
def greet_node(state: AgentState) -> Dict:
    return {"agent_outcome": "Hello there!", "chat_history": state["chat_history"] + [{"role": "assistant", "content": "Hello there!"}]}
 
def unclear_node(state: AgentState) -> Dict:
    return {"agent_outcome": "I couldn't understand your request. Please rephrase it.", "chat_history": state["chat_history"] + [{"role": "assistant", "content": "I couldn't understand your request. Please rephrase it."}]}
 
# 4. Build the graph
workflow = StateGraph(AgentState)
 
workflow.add_node("router", route_node)
workflow.add_node("idea_validation", ToolNode(tools[0]))
workflow.add_node("competitor_analysis", ToolNode(tools[1]))
workflow.add_node("executive_summary", ToolNode(executive_summary_node))
workflow.add_node("legal_advisor", ToolNode(legal_advisor_node))
workflow.add_node("greet", greet_node)
workflow.add_node("unclear", unclear_node)
workflow.add_node("ceo_comments", ToolNode(ceo_comments_node))
 
workflow.set_entry_point("router")
 
workflow.add_edge("idea_validation", "router")
workflow.add_edge("competitor_analysis", "router")
workflow.add_edge("executive_summary", "router")
workflow.add_edge("legal_advisor", "router")
workflow.add_edge("greet", "router")
workflow.add_edge("unclear", "router")
workflow.add_edge("ceo_comments", END)
 
workflow.add_conditional_edges(
    "router",
    route_node,
    {
        "greeting": "greet",
        "idea_validation": "idea_validation",
        "competitor_analysis": "competitor_analysis",
        "executive_summary": "executive_summary",
        "legal_advisor": "legal_advisor",
        "ceo_comments": "ceo_comments",
        "unclear": "unclear",
    }
)
workflow.add_edge("router", "router")
 
 
app = workflow.compile()
 
def main():
    st.set_page_config(page_title="Startup Idea Analyzer", page_icon=":bulb:", layout="wide")
 
    with st.sidebar:
        st.title("About")
        st.markdown("This app analyzes startup ideas...")
        st.markdown(f"**Model:** {MODEL_NAME}")  # Ensure MODEL_NAME is defined
 
    st.title("Startup Idea Analyzer")
 
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
 
    user_query = st.chat_input("Enter your startup idea or question:")
 
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
 
        for message in st.session_state.chat_history:
            display_wrapped_message(message)  # Assuming display_wrapped_message is defined
 
        try:
            # Run the LangGraph agent
            inputs = {"input": user_query, "chat_history": st.session_state.chat_history}
            for output in app.stream(inputs, config={"recursion_limit": 50}):
                if "__end__" not in output:  # Stream until the end
                    # Determine the key for the output (node name)
                    node_name = list(output.keys())[0]
 
                    # Extract the agent's outcome and update chat history
                    st.session_state.chat_history = output[node_name]["chat_history"]
 
        except OllamaCommunicationError as e:  # Assuming OllamaCommunicationError is defined
            st.error(f"LLM Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        for message in st.session_state.chat_history:
            display_wrapped_message(message)
 
 
 
if __name__ == "__main__":
    main()
 