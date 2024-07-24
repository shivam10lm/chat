import os
from dotenv import load_dotenv
import gradio as gr
import transformers
import torch

load_dotenv()
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Define a function to handle chatbot interactions
def chat_with_bot(user_input, chat_history):
    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Generate the response
    inputs = [{"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"}] + chat_history
    outputs = pipeline(inputs, max_new_tokens=256)
    
    # Extract the generated text
    generated_text = outputs[0]["generated_text"][-1]
    
    # Append chatbot response to chat history
    chat_history.append({"role": "assistant", "content": generated_text})
    
    # Return the updated chat history
    return chat_history, generated_text

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>Llama3.1 Pirate Chatbot</h1>")
    
    with gr.Row():
        chatbot = gr.Chatbot(height=450, label="Chatbot")
    
    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your message and press Enter", scale=7, label="User Message")
        clear = gr.Button("Clear Chat")
    
    # Define the interactions
    user_input.submit(chat_with_bot, [user_input, chatbot], [chatbot, user_input], queue=False)
    clear.click(lambda: ([], ""), None, [chatbot, user_input], queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(debug=True)
