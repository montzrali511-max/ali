from cog import BasePredictor, Input
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

class Predictor(BasePredictor):
    def setup(self):
        """تحميل الموديل عند بدء التشغيل"""
        model_id = "zai-org/GLM-5.2"
        # تحميل الأوزان من Hugging Face
        model_path = snapshot_download(repo_id=model_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def predict(
        self,
        prompt: str = Input(description="النص المدخل للموديل"),
        max_tokens: int = Input(description="أقصى عدد من الكلمات الناتجة", default=512),
        temperature: float = Input(description="درجة العشوائية", default=0.7),
    ) -> str:
        """تشغيل التوقع"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
