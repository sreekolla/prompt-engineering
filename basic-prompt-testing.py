from transformers import AutoTokenizer, AutoModelForCausalLM

local_path = "C:/Users/Laptop/Desktop/llm-tunning/Prompt-Engineering/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path)

# ✅ Basic test
prompt = "Patient presents with chest pain and shortness of breath. What is the likely diagnosis?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id  # explicitly set padding token
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated_text[len(prompt):].strip()
print(response)


