import streamlit as st
import ollama
import re
import time
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
import textwrap
import logging
# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# --- Configuration and Constants ---
MODEL_NAME = "llama3.2:latest"
MAX_RETRIES = 3
RETRY_DELAY = 2
TEXT_WRAP_WIDTH = 80
OPENAI_BASE_URL = "https://integrate.api.nvidia.com"  # Or your OpenAI base URL if not using NVIDIA
OPENAI_API_KEY = "nv"
OPENAI_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct" # or a good general purpose model
# --- Error Handling ---
class OllamaCommunicationError(Exception):
    pass
def ask_question(query: str, context: Optional[List[Dict[str, str]]] = None,
                 model: str = MODEL_NAME, max_retries: int = MAX_RETRIES) -> str:
    messages = []
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": query})
    logging.debug(f"ask_question: Query: {query}, Context: {context}")  # Log input
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=model, messages=messages)
            logging.debug(f"ask_question: Raw Response (Attempt {attempt+1}): {response}")  # Log raw response
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                error_msg = "Ollama response missing 'message' or 'content' key."
                logging.error(error_msg)  # Log error
                raise OllamaCommunicationError(error_msg)
        except Exception as e:
            logging.exception(f"Attempt {attempt+1} failed: {e}")  # Log exception with traceback
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise OllamaCommunicationError(f"Failed after {max_retries} retries: {e}")
    return "An unexpected error occurred."
# --- Specialized Prompt Templates ---
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
 
ROUTER_PROMPT = """

You are an AI assistant specializing in startup-related queries, but you also understand common greetings. Your **ONLY** Job is to categorize the users query.
Strictly give output only in idea_validation, 
First, check if the user's query is a simple greeting (e.g., "Hello", "Hi", "Good morning"). If it is, respond appropriately with a similar greeting.
 
If the query is NOT a greeting, categorize it into *one* of the following:
 
*   **idea_validation:** Assessing potential, feasibility, or risks.

*   **competitor_analysis:** Researching existing businesses.

*   **executive_summary:** Comprehensive business plan overview.

*   **unclear:** Ambiguous, off-topic, or doesn't fit the above.
 
Output ONLY the category name (e.g., 'idea_validation') OR the greeting response.  Do not output both.
 
User Query: {query}
 
"""
# --- Prompt Formatting Functions ---
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
# --- REVISED Router Prompt ---
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
# --- Function Definitions ---
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
def router(query: str, context: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    try:
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)  # Create OpenAI client inside the function
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": format_prompt(ROUTER_PROMPT, query=query)}], # Use formatted prompt
            temperature=0.0,  # Set temperature to 0 for deterministic categorization
            max_tokens=20, # Limit tokens for category name
        )
 
        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
 
        response_text = response_text.strip().lower()
 
        logging.info(f"Router response: {response_text}")
 
        if response_text == "greeting":
            greeting_response = "Hello there!"
            context.append({"role": "assistant", "content": greeting_response})
            return "greeting", context
 
        category = "unclear"
        if response_text in ("idea_validation", "competitor_analysis", "executive_summary", "legal_advisor"):
            category = response_text
 
        logging.info(f"Router classified query as: {category}")
        context.append({"role": "assistant", "content": f"(Category: {category})"})
        return category, context
 
    except Exception as e:
        logging.error(f"Error in router: {e}")
        context.append({"role": "assistant", "content": f"(Error in categorization)"})  # Add error to context
        return "unclear", context  # Return unclear to handle the error gracefully
# --- UI Helper ---

def display_wrapped_message(message: Dict[str, str]):

    with st.chat_message(message['role']):

        st.write(textwrap.fill(message['content'], width=TEXT_WRAP_WIDTH))
 
# --- Streamlit App ---

def main():

    st.set_page_config(page_title="Startup Idea Analyzer", page_icon=":bulb:", layout="wide")
 
    # --- CUSTOM CSS STYLES ---
 
    st.markdown(

        """
<style>

/* General Styles */

body {

    background-color: #F5F5F5;

    color: #333333;

    font-family: 'Open Sans', sans-serif; /* Replace with your chosen font */

}
 
/* Headings */

h1, h2, h3 {

    color: #2A7886;

}
 
/* Buttons */

.stButton > button {

    background-color: #2A7886;

    color: white;

    border: none;

    border-radius: 5px; /* Slightly rounded corners */

}

.stButton > button:hover {

    background-color: #E0A161; /* Hover state */

    color: white;

}
 
/* Chat Message Styling (Example - adjust as needed) */

.stChatMessage[data-baseweb="chat-message"]:nth-child(odd) {

    background-color: #F5F5F5; /* User message background */

}

.stChatMessage[data-baseweb="chat-message"]:nth-child(even) {

    background-color: #E8E8E8; /* Assistant message background (slightly darker) */

}

.stChatMessage[data-baseweb="chat-message"] p {

    color: #333333;

}
 
/* Input field */

.stTextInput input {

   border-radius: 5px;

   border-color: #A9A9A9;

}
</style>

        """,

        unsafe_allow_html=True,

    )
 
    # --- Sidebar Content ---
 
    with st.sidebar:

        st.title("About")

        st.markdown("This app analyzes startup ideas...")

        st.markdown(f"**Model:** {MODEL_NAME}")
 
    st.title("Startup Idea Analyzer")
 
 
    if 'chat_history' not in st.session_state:

        st.session_state.chat_history = []
 
    user_query = st.chat_input("Enter your startup idea or question:")
 
    if user_query:

        st.session_state.chat_history.append({"role": "user", "content": user_query})
 
        for message in st.session_state.chat_history:

            display_wrapped_message(message)
 
        try:

            category, updated_context = router(user_query, st.session_state.chat_history)

            st.session_state.chat_history = updated_context
 
            if category == "greeting":

                pass  # Greeting already handled in router
 
            elif category in ("idea_validation", "competitor_analysis", "executive_summary", "legal_advisor"):

                if category == "idea_validation":

                    response = idea_validator(user_query, st.session_state.chat_history)

                elif category == "competitor_analysis":

                    response = competitor_analysis(user_query, st.session_state.chat_history)

                elif category == "executive_summary":

                    idea_val = idea_validator(user_query, st.session_state.chat_history)

                    comp_anal = competitor_analysis(user_query, st.session_state.chat_history)

                    response = executive(user_query, st.session_state.chat_history, idea_val, comp_anal)

                elif category == "legal_advisor":

                    idea_val = idea_validator(user_query, st.session_state.chat_history)

                    response = legal_advisor(user_query, idea_val, st.session_state.chat_history)
 
                st.session_state.chat_history.append({"role": "assistant", "content": response})
 
                if category != "greeting" and category != "executive_summary":

                    ceo_response = ceo_comments(st.session_state.chat_history)

                    st.session_state.chat_history.append({"role": "assistant", "content": f"CEO Comments:\n{ceo_response}"})
 
            elif category == "unclear":

                response = "I couldn't understand your request. Please rephrase it, or be more specific."

                st.session_state.chat_history.append({"role": "assistant", "content": response})

            else:

                response = f"Unexpected category: {category}"  # Shouldn't happen

                st.session_state.chat_history.append({"role": "assistant", "content": response})

            st.rerun()  # Update the UI
 
        except OllamaCommunicationError as e:

            st.error(f"LLM Error: {e}")
 
        except Exception as e:

            st.error(f"An unexpected error occurred: {e}")

    else: #added else

        for message in st.session_state.chat_history:

            display_wrapped_message(message)
 
if __name__ == "__main__":

    main()

 
