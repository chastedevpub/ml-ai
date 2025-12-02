# Lanflow Custom Modul for Hugging Face Foundation-Secure (Chat Model)
from langflow.base.io.chat import ChatComponent
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message
from huggingface_hub import InferenceClient
import os

class HFTextGeneration(ChatComponent):
    display_name = "HF Foundation-Secure (Chat Model)"
    description = "fdtn-ai/Foundation-Sec-8B-Instruct – proper chat endpoint"
    icon = "HuggingFace"

    inputs = [
        MessageTextInput(name="prompt", display_name="Prompt"),
        MessageTextInput(
            name="model",
            display_name="Model ID",
            value="fdtn-ai/Foundation-Sec-8B-Instruct",
        ),
    ]

    outputs = [Output(display_name="Response", name="response", method="generate_response")]

    def generate_response(self) -> Message:
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            return Message(text="Error: Missing HF token")

        if not self.prompt or self.prompt.strip() == "":
            return Message(text="")

        client = InferenceClient(token=token)

        # ← CORRECT METHOD FOR CHAT MODELS
        result = client.chat_completion(
            messages=[{"role": "user", "content": self.prompt}],
            model=self.model or "fdtn-ai/Foundation-Sec-8B-Instruct",
            max_tokens=512,
            temperature=0.7,
        )

        return Message(text=result.choices[0].message.content)