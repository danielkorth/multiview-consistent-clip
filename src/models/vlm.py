import torch
import torch.nn as nn
from typing import List

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, AutoModel, AutoProcessor, AutoTokenizer

class VLM(nn.Module):
    def __init__(self, device, vlm_name: str = 'clip'):
        self.device = device
        super().__init__()
        if vlm_name == 'clip':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        elif vlm_name == 'siglip':
            self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
            self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        else:
            raise Exception("Provide a valid VLM name [clip | siglip]")
        
    
    @torch.no_grad()
    def forward_image(self, x):
        inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs)

    @torch.no_grad()
    def forward_text(self, x: List[str]):
        tokens = self.tokenizer(x, padding="max_length", return_tensors="pt").to(self.device)
        return self.model.get_text_features(**tokens)

    @torch.no_grad() 
    def forward(self, texts: List[str], image):
        inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return {
            "text_embed": outputs['text_embeds'],
            "image_embed": outputs['image_embeds']
        }
