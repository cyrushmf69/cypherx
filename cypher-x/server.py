from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io, torch
from transformers import BlipForConditionalGeneration
from my_processor import BeastProcessingInfo, BeastProcessor

app = FastAPI()

# Pick a small multimodal model (runs on free GPUs)
MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = BeastProcessor(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()


@app.post("/generate")
async def generate(text: str = Form("Describe this image:"), image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_obj = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = {
        BeastProcessingInfo.TEXT_KEY: text,
        BeastProcessingInfo.IMAGE_KEY: image_obj,
    }

    processed = processor.process(BeastProcessingInfo, inputs)
    for k, v in processed.items():
        processed[k] = v

    output_ids = model.generate(**processed, max_length=64)
    resp = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"response": resp}
