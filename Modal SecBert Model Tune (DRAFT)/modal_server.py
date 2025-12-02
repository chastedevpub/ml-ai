import modal

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "transformers", "torch")
)

app = modal.App("foundation-model", image=image)

# Download model during build
@app.function(
    image=image,
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download("jackaduma/SecBERT")

# Deploy inference class with GPU
@app.cls(
    gpu="A10G",  # or "A100", "H100"
    secrets=[modal.Secret.from_name("huggingface-secret")],  # if model requires auth
    # --- 🟢 THE FIX IS HERE ---
    scaledown_window=300, # Changed from container_idle_timeout=300
    # --------------------------
)
class Model:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        self.llm = LLM("jackaduma/SecBERT")
    
    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 100):
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        result = self.llm.generate(prompt, params)
        return result[0].outputs[0].text

# Create web endpoint
@app.function()
@modal.web_endpoint(method="POST")
def api_endpoint(data: dict):
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    
    # Note: For production use, you might use Model.local() or Model.remote()
    # directly on the class if you didn't need to pass a prompt/max_tokens.
    # The current pattern is fine for simple usage.
    model = Model()
    response = model.generate.remote(prompt, max_tokens)
    
    return {"response": response}