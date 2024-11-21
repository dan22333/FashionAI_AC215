from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
model = CLIPModel.from_pretrained("weiyueli7/fashionclip")
processor = CLIPProcessor.from_pretrained("weiyueli7/fashionclip")

def get_clip_vector(input_data, is_image=False):
    if is_image:
        inputs = processor(images=input_data, return_tensors="pt", padding=True)
        outputs = model.get_image_features(**inputs)
    else:
        inputs = processor(text=[input_data], return_tensors="pt", padding=True)
        outputs = model.get_text_features(**inputs)
    return outputs.detach().numpy().flatten()
