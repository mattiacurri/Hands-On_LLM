from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", # change if you don't have access to Llama 3.2
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)

# To use do_sample False
model.generation_config.temperature = 1.0
model.generation_config.top_p = 1.0

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # the prompt will nto be returned but merely the output of the model
    max_new_tokens=1024,
    do_sample=False # sampling strategy, with False it will use greedy decoding (the most likely token)
)

messages = [
    {"role": "user", "content": "How do I make a cake?"},
]

output = generator(messages)
print(output[0]["generated_text"])