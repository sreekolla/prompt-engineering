from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

local_path = "C:/Users/Laptop/Desktop/llm-tunning/Prompt-Engineering/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path)
model.eval()

clinical_case = """
A 58-year-old woman presents with fatigue, pallor, and shortness of breath. 
Lab results show Hb = 7.8 g/dL, MCV = 70 fL, ferritin = 5 ng/mL.
What is the likely diagnosis, and what should be the next step in management?
"""

prompt_variants = [
    "You are an expert clinician. Diagnose the patient:\n" + clinical_case,
    "Summarize the most likely diagnosis and recommended treatment:\n" + clinical_case,
    "List differential diagnoses with explanations:\n" + clinical_case,
    "Generate a concise clinical note for this case:\n" + clinical_case
]

def generate_response(prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt repetition if needed
    return text[len(prompt):].strip()

for i, prompt in enumerate(prompt_variants):
    print(f"\n=== Prompt Variant {i+1} ===")
    print(prompt)
    print("\n--- Model Output ---")
    print(generate_response(prompt))
