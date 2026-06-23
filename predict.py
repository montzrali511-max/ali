import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

class Predictor:
    def setup(self):
        """تحميل الموديل عند بدء التشغيل"""
        # تحميل أوزان GLM-5.2 من Hugging Face
        model_path = snapshot_download(repo_id="zai-org/GLM-5.2")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """تشغيل التوقع بناءً على النص المدخل"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
