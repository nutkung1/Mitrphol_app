import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from auto_gptq import AutoGPTQForCausalLM
import torch

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
# Streamlit app header
st.title("Mitr Phol Chatbot")

# Create a text input box for user messages
user_input = st.text_input("You:", "")

# Create a chat history area to display the conversation
chat_history = st.empty()

access_token = "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"
model_name = "nutkung1/Mitr_Phol"
offload_folder = "E:/app_mitrphol/Mitrphol_app-main/offload_folder"  # Specify the offload folder

# Initialize the tokenizer and model with offload_folder
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, offload_folder=offload_folder)
model = AutoGPTQForCausalLM.from_quantized(model_name, token=access_token, offload_folder=offload_folder)
generation_config = GenerationConfig.from_pretrained(model_name)

def generate_chatbot_response(user_input, model, tokenizer):
    # Tokenize user input
    input_ids = tokenizer.encode(user_input, return_tensors="pt").input_ids.to("cuda:0")

    # Generate a response from the model
    with torch.no_grad():
        output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)

    # Decode the model's response
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    return bot_response

if user_input:
    # Generate a response from the chatbot
    bot_response = generate_chatbot_response(user_input, model, tokenizer)

    # Append the user input and chatbot response to the chat history
    chat_history.text(f"You: {user_input}")
    chat_history.text(f"Chatbot: {bot_response}")


