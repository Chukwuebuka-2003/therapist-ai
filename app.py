import streamlit as st
import yaml
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()



@st.cache_data
def load_config(filepath="therapist_chatbot_prompt.yaml"):
    """
    Loads the chatbot configuration from a YAML file.
    The use of @st.cache_data ensures the file is loaded only once.
    """
    try:
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Fatal Error: The configuration file `therapist_chatbot_prompt.yaml` was not found.")
        st.info("Please make sure the configuration file is in the same directory as the app.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Fatal Error: Could not parse the YAML configuration file. Please check its formatting. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the configuration: {e}")
        st.stop()

def construct_system_prompt(config):
    """
    Constructs the full system prompt string from the loaded configuration dictionary.
    This prompt provides the LLM with its persona, capabilities, limitations, and safety guardrails.
    """
    # Start with the core system prompt
    system_prompt_text = config.get('system_prompt', [{}])[0].get('content', '')

    # Append Persona and Identity details
    persona_config = config.get('persona', {})
    identity_parts = [
        f"\n--- PERSONA & IDENTITY ---\n"
        f"Your name is {persona_config.get('name', 'AI')}, and you are an {persona_config.get('role', 'assistant')}. "
        f"{persona_config.get('description', '')}"
    ]

    identity_config = config.get('identity', [])
    for item in identity_config:
        if 'capabilities' in item:
            identity_parts.append("\nYour capabilities are:\n- " + "\n- ".join(item['capabilities']))
        if 'limitations' in item:
            identity_parts.append("\nYour non-negotiable limitations are:\n- " + "\n- ".join(item['limitations']))

    # Append Guardrail details
    guardrail_parts = ["\n\n--- SAFETY GUARDRAILS (Strictly Follow These Rules) ---"]
    for i, rule in enumerate(config.get('guardrails', [])):
        rule_text = f"\n# Guardrail Rule {i+1}: {rule.get('rule', 'No title')}\n"
        rule_text += f"Description: {rule.get('description', '')}\n"

        if 'example_user_prompt' in rule:
            rule_text += f"Example Trigger: A user might say something like, '{rule['example_user_prompt']}'\n"
            logic = rule.get('correct_response_logic', [])
            rule_text += "Correct Response Logic:\n" + "\n".join([f"  - {step}" for step in logic]) + "\n"
        if 'response' in rule:
            rule_text += "If this rule is triggered, you MUST immediately stop any other line of conversation and use this exact response:\n" + "\n".join(rule['response']) + "\n"
        if 'response_template' in rule:
            rule_text += f"When this rule is triggered, you must use this response template: '{rule['response_template']}'\n"

        guardrail_parts.append(rule_text)

    # Append Response Style Guide
    style_guide_parts = ["\n\n--- RESPONSE STYLE GUIDE ---"]
    for item in config.get('response_style', []):
        if 'tone' in item:
            style_guide_parts.append(f"Tone: {item['tone']}")
        if 'length' in item:
            style_guide_parts.append(f"Length: {item['length']}")
        if 'techniques' in item:
            style_guide_parts.append("Techniques to use:\n- " + "\n- ".join(item['techniques']))


    full_prompt = "\n".join([system_prompt_text] + identity_parts + guardrail_parts + style_guide_parts)
    return full_prompt




st.set_page_config(page_title="Serene", page_icon="🧘‍♀️", layout="centered")

config = load_config()
persona_name = config.get('persona', {}).get('name', 'Chatbot')
st.title(f"{persona_name}: Your AI Wellness Companion")
st.write(config.get('persona', {}).get('description', 'This is a safe space to talk.'))
st.markdown("---")



with st.sidebar:
    st.header("Configuration")
    st.markdown("To use this chatbot, you need a Google API Key.")
    # Try getting the key from Streamlit Cloud, then .env, then manual input.
    api_key = None

    # 1. Try Streamlit Cloud secrets (for deployment)
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key loaded from Streamlit Cloud.")
    # 2. Fallback to .env file (for local development)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("API Key loaded from local .env file.")

    # 3. If still no key, ask for manual input as a last resort
    if not api_key:
        st.warning("API Key not found. Please provide your key to continue.")
        api_key = st.text_input("Enter your Google API Key", type="password", help="You can get a key from Google AI Studio.")

if not api_key:
    st.info("Please enter your Google API Key in the sidebar to start the chat.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',

        safety_settings={
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
    )
except Exception as e:
    st.error(f"Failed to configure the AI model. Please check your API key. Error: {e}")
    st.stop()


if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = construct_system_prompt(config)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    # This includes the system prompt and the entire conversation history
    # The system prompt acts as a permanent instruction at the start of the conversation
    api_history = [
        {"role": "user", "parts": [st.session_state.system_prompt]},
        {"role": "model", "parts": ["Understood. I am Serene. I will adhere to all my instructions and begin the conversation with the user."]},
    ]
    for msg in st.session_state.messages:
        role = "model" if msg["role"] == "assistant" else "user"
        api_history.append({"role": role, "parts": [msg["content"]]})

    with st.spinner("Thinking..."):
        try:
            # Call the Gemini API
            response = model.generate_content(api_history)

            # Extract text from response and handle potential errors
            if response.parts:
                bot_response = response.text
            else:
                bot_response = "I'm sorry, I couldn't generate a response. Please try again."

            # Add AI response to session state and display it
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
