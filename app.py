import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_name = "nutkung1/Mitr_Phol"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
generation_config = GenerationConfig.from_pretrained(model_name)

# Streamlit app header
st.title("Hugging Face Chatbot")

# Create a text input box for user messages
user_input = st.text_input("You:", "")

# Create a chat history area to display the conversation
chat_history = st.empty()

def generate_chatbot_response(user_input, model, tokenizer):
    # Tokenize user input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response from the model
    with torch.no_grad():
        output = model.generate(input_ids)

    # Decode the model's response
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    return bot_response

if user_input:
    # Generate a response from the chatbot
    bot_response = generate_chatbot_response(user_input, model, tokenizer)

    # Append the user input and chatbot response to the chat history
    chat_history.text(f"You: {user_input}")
    chat_history.text(f"Chatbot: {bot_response}")


