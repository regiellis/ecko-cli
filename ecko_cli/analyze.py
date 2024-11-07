import os
from functools import lru_cache
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
)
from typing import Optional
from PIL import Image

import torch
import torchvision.transforms.functional as TVF
import numpy as np
import onnxruntime as ort
import pandas as pd
import warnings

from rich.progress import Progress
from enum import Enum

from .helpers import (
    feedback_message,
    get_models_dir,
    make_square,
    smart_resize,
)

CAPTION_MODEL = "MiaoshouAI/Florence-2-base-PromptGen-v2.0" #"microsoft/Florence-2-large" #"MiaoshouAI/Florence-2-large-PromptGen-v2.0" 
JOYCAP_MODEL = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
IMAGE_SIZE_JOYCAP = (384, 384)

# Suppress the FutureWarning
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)


def delete_training_image(training_image: str) -> None:
    """
    Delete the training image.
    """
    try:
        os.remove(training_image)
    except FileNotFoundError:
        feedback_message(f"File not found: {training_image}", "warning")


@lru_cache(maxsize=256)
def load_models():
    """
    Load Florence model and processor.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        CAPTION_MODEL,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL, trust_remote_code=True)

    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, "model.onnx")

    # ONNX Runtime providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        ort_session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"Error creating ONNX session: {e}")
        ort_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    return model, processor, ort_session


def preprocess_image_for_joycap(image_path: str) -> torch.Tensor:
    """
    Preprocess the image for Joycaption.
    Resize to (384, 384), normalize and convert to tensor.
    """
    image = Image.open(image_path).convert("RGB")
    if image.size != IMAGE_SIZE_JOYCAP:
        image = image.resize(IMAGE_SIZE_JOYCAP, Image.LANCZOS)

    # Convert image to tensor and normalize
    pixel_values = TVF.pil_to_tensor(image).float() / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    pixel_values = pixel_values.unsqueeze(0)

    return pixel_values


@lru_cache(maxsize=256)
def load_joycap():
    """
    Load the JoyCaption model and tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        JOYCAP_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    llava_model.tie_weights()
    llava_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(JOYCAP_MODEL, use_fast=True)

    return llava_model.to(device), tokenizer


def generate_joycap_caption(image_path: str, prompt: str) -> str:
    """
    Generate a caption using the JoyCaption model, given an image.
    """
    try:
        llava_model, tokenizer = load_joycap()

        # Preprocess image
        image = Image.open(image_path).convert("RGB").resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0).float() / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])

        device = next(llava_model.parameters()).device
        dtype = next(llava_model.parameters()).dtype
        pixel_values = pixel_values.to(device).to(dtype)

        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]

        convo_string = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the conversation
        input_ids = tokenizer.encode(convo_string, return_tensors="pt").to(device)

        # Generate caption
        with torch.no_grad():
            generated_ids = llava_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=256,
                do_sample=True,
                use_cache=True,
            )

        caption = tokenizer.decode(
            generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        return caption

    except Exception as e:
        print(f"Error generating JoyCap caption: {e}")
        return "Error generating caption"


def analyze_image(
    image_path: str,
    task: int,
    progress: Progress,
    trigger,
    use_joycap,
    is_anime,
    is_object,
    is_style,
) -> Optional[str]:
    try:
        if use_joycap:
            prompt = "Write a 256 stable diffusion prompt for this image."
            caption = generate_joycap_caption(image_path, prompt)
        else:

            model, processor, _ = load_models()

            image = Image.open(image_path)

            # Generate Image description using Florence model
            caption = generate_florence_description(
                image, model, processor, is_object, is_anime, is_style
            )

        # WD14 tagging
        _, _, ort_session = load_models()

        # Get model input details
        input_shape = ort_session.get_inputs()[0].shape
        input_name = ort_session.get_inputs()[0].name
        height = input_shape[2]

        # Load and preprocess the image for WD14
        image = Image.open(image_path)  # Reload image for WD14 tagging
        image_np = np.array(image.convert("RGB"))
        image_np = image_np[:, :, ::-1]  # RGB to BGR

        image_np = make_square(image_np, height)
        image_np = smart_resize(image_np, height)
        image_np = image_np.astype(np.float32)
        image_np = np.expand_dims(image_np, 0)

        # Run inference
        ort_inputs = {input_name: image_np}
        confidence = ort_session.run(None, ort_inputs)[0]

        # Load tags from CSV
        models_dir = get_models_dir()
        if is_anime:
            csv_path = os.path.join(models_dir, "anime.csv")
        else:
            csv_path = os.path.join(models_dir, "realistic.csv")

        df_tags = pd.read_csv(csv_path)

        # Get probabilities and create tag-confidence pairs
        tag_confidence = list(zip(df_tags["name"], confidence[0]))

        # Filter and sort tags
        threshold = 0.55
        filtered_tags = [
            (tag, conf)
            for tag, conf in tag_confidence
            if conf > threshold
            and df_tags.loc[df_tags["name"] == tag, "category"].iloc[0] != 9
        ]

        # Keep track of all tags
        wd14_tags = [tag for tag, _ in filtered_tags]  # Collect filtered tags

        # final output
        trigger_word = f"{trigger} " if trigger else ""
        result = f"{trigger_word}{caption} {', '.join(wd14_tags)}"

        delete_training_image(image_path)

        progress.update(task, advance=1)
        return result

    except Exception as e:
        print(f"Error processing image: {e}")
        progress.update(task, advance=1)
        return None


class CaptionType(Enum):
    DETAILED = "<MORE_DETAILED_CAPTION>"
    OBJECT_DETECTION = "<OD>"
    GENERATE_TAGS = "<GENERATE_TAGS>"
    CAPTION = "<CAPTION>"


def get_task_prompt(is_object: bool, is_anime: bool, is_style: bool) -> str:
    if is_object:
        return CaptionType.OBJECT_DETECTION.value
    elif is_anime:
        return CaptionType.GENERATE_TAGS.value
    elif is_style:
        return CaptionType.CAPTION.value
    else:
        return CaptionType.DETAILED.value  # Default


def generate_florence_description(
    image: Image.Image,
    model: AutoModelForCausalLM,
    processor,
    is_anime=False,
    is_object=False,
    is_style=False,
) -> str:
    """
    Generate a description of the image using the Florence model.
    """
    try:
        device = next(model.parameters()).device

        # Get task prompt as per input flags
        task_prompt = get_task_prompt(is_object, is_anime, is_style)

        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

            if key == "pixel_values" and torch.cuda.is_available():
                inputs[key] = inputs[key].to(torch.float16)

        # Ensure that input_ids are Long
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)

        if not inputs or "input_ids" not in inputs or "pixel_values" not in inputs:
            return "Error: Invalid inputs"

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text or "No description generated"

    except Exception as e:
        print(f"Error in generate_florence_description: {e}")
        return f"Error generating description: {str(e)}"
