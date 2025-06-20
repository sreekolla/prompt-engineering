# Clinical Prompt Engineering with Falcon-RW-1B

This project demonstrates **iterative prompt engineering** for clinical decision support using the **Falcon-RW-1B** language model loaded locally.

It runs multiple variants of a prompt to evaluate which phrasing yields the most useful output from the model. This is especially useful for clinical applications such as diagnosis generation, treatment planning, and differential diagnosis.

---

## 🧠 Use Case

Given a clinical scenario, the script generates responses from the LLM to simulate:
- Diagnostic reasoning
- Treatment suggestions
- SOAP notes
- Differential diagnosis explanations

---


---

## 🛠️ Requirements

Install dependencies:

```bash
pip install transformers torch
