# app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model & tokenizer (on startup)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response[len(user_input):].strip()})
