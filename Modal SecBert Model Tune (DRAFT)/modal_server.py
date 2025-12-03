import modal

# Define container image with dependencies
# Note: FastAPI/uvicorn are automatically included with @modal.fastapi_endpoint
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers", "torch", "accelerate", "bitsandbytes", "huggingface_hub")
    .pip_install("fastapi", "uvicorn")
)

app = modal.App("Foundation-Sec-8B", image=image)

# Download model during build
# Note: Add secrets=[modal.Secret.from_name("huggingface-secret")] if model requires authentication
@app.function(
    image=image,
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download("fdtn-ai/Foundation-Sec-8B")

# Deploy inference class with GPU
# Note: Add secrets=[modal.Secret.from_name("huggingface-secret")] if model requires authentication
@app.cls(
    gpu="A10G",  # or "A100", "H100"
    scaledown_window=300,  # Fixed: renamed from container_idle_timeout
)
class Model:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "fdtn-ai/Foundation-Sec-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
    
    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, conversation_history: list = None):
        import torch
        
        # Handle conversation history if provided
        if conversation_history:
            # Format conversation for chat
            messages = conversation_history + [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Single prompt without history
            formatted_prompt = prompt
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return generated_text.strip()

# Create web endpoint
@app.function()
@modal.fastapi_endpoint(method="POST")
def api_endpoint(data: dict):
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    conversation_history = data.get("conversation_history", None)
    
    model = Model()
    response = model.generate.remote(prompt, max_tokens, temperature, conversation_history)
    
    return {"response": response}