from dotenv import load_dotenv
import gradio as gr
import transformers
import torch

load_dotenv()
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {
        "role":"system",
        "content":"You are a pirate chatbot who always responds in pirate speak"
    },
    {
        "role":"user",
        "content":"Who are you?"
    }
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens = 256,
    eos_token_id = terminators,
    do_sample = True,
    temperature = 0.6,
    top_p = 0.9,
)

print(outputs[0]["generated_text"][len(prompt):])

# Define a function to handle chatbot interactions
def chat_function(message, history, system_prompt, max_new_tokens, temperature):
    messages = [{"role":"system","content":system_prompt},
                {"role":"user", "content":message}]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,)
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(
        prompt,
        max_new_tokens = max_new_tokens,
        eos_token_id = terminators,
        do_sample = True,
        temperature = temperature + 0.1,
        top_p = 0.9,)
    return outputs[0]["generated_text"][len(prompt):]

gr.ChatInterface(
    chat_function,
    textbox=gr.Textbox(placeholder="Enter message here", container=False, scale = 7),
    chatbot=gr.Chatbot(height=400),
    additional_inputs=[
        gr.Textbox("You are helpful AI", label="System Prompt"),
        gr.Slider(500,4000, label="Max New Tokens"),
        gr.Slider(0,1, label="Temperature")
    ]
    ).launch()
