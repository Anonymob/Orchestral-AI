from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Classification Model(distill mode)
classifier = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli",  
    device=0 if torch.cuda.is_available() else -1
)

# candidate labels
candidate_labels = [
    "image recognition", 
    "text analysis", 
    "code generation", 
    "natural language processing", 
    "speech recognition", 
    "sentiment analysis", 
    "data visualization", 
    "machine learning", 
    "web development", 
    "game development",
    "financial analysis",
    "social media analysis",
    "healthcare analytics",
    "mathematics" 
]

# Optimized Checkpoint(NLP model)
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16).to(device)
model.eval()

#maths model
math_model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
math_tokenizer = AutoTokenizer.from_pretrained(math_model_name)
math_model = AutoModelForCausalLM.from_pretrained(
    math_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
math_model.eval()

def generate_response(user_prompt, max_new_tokens=150, use_math_model=False):
    """
    Generates a response for the given input text using the preloaded model and tokenizer.
    If 'use_math_model' is True, it uses the math-specific model for response generation.
    """
    if use_math_model:
        # Tokenize and generate response using the math model
        inputs = math_tokenizer(user_prompt, return_tensors="pt", truncation=True).to(device)
        output_ids = math_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = math_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()

    else:
        # Tokenize and generate response using the default NLP model
        inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=min(max_new_tokens, tokenizer.model_max_length - inputs.input_ids.shape[1]),
            temperature=0.3,  
            top_p=0.85,       
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(user_prompt):].strip()

print("Interactive Program. Press Enter on empty input to exit.")
while True:
    user_prompt = input("Please enter a prompt for classification: ").strip()

    if not user_prompt:  
        print("Exiting program. Goodbye!")
        break

    # Classify the prompt
    result = classifier(user_prompt, candidate_labels=candidate_labels)
    top_label = result["labels"][0]
    print(f"Classification Result: {top_label}")

   
    if top_label == "natural language processing" or top_label in [
        "speech recognition", 
        "sentiment analysis"
    ]:
        try:
            
            response = generate_response(user_prompt)
            print(f"Generated Response: {response}")
        except RuntimeError as e:
            print(f"Error generating response: {e}")
    elif top_label == "mathematics":
        try:
            
            response = generate_response(user_prompt, use_math_model=True)
            print(f"Mathematics Response: {response}")
        except RuntimeError as e:
            print(f"Error generating response: {e}")
    else:
        print(f"No suitable model/action found for: {top_label}")
