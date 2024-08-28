import os
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional, Dict, Final
from PIL import Image

import torch
import numpy as np
import onnxruntime as ort
import pandas as pd
import warnings

from rich.progress import Progress

from .helpers import (
    feedback_message,
    get_models_dir,
    make_square,
    smart_resize,
)

KAOMOJIS: Final = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def delete_training_image(training_image: str) -> None:
    """
    Delete the training image
    """
    try:
        os.remove(training_image)
    except FileNotFoundError:
        feedback_message(f"File not found: {training_image}", "warning")


# Suppress the FutureWarning
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)


@lru_cache(maxsize=None)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    )

    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, "model.onnx")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Create ONNX Runtime session with the selected providers
    try:
        ort_session = ort.InferenceSession(model_path, providers=providers)
        # print(f"Using ONNX Runtime with providers: {ort_session.get_providers()}")
    except Exception as e:
        print(f"Error creating ONNX session: {e}")
        print("Falling back to CPU only")
        ort_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    return model, processor, ort_session


def analyze_image(
    image_path: str, task: int, progress: Progress, trigger, is_object
) -> Optional[Dict[str, str]]:
    try:
        model, processor, ort_session = load_models()

        # Load and process the image
        image = Image.open(image_path)

        # Generate image description using Florence-2
        florence_description = generate_florence_description(
            image, model, processor, is_object
        )

        # WD14 tagging
        # Get model input details
        input_shape = ort_session.get_inputs()[0].shape
        input_name = ort_session.get_inputs()[0].name
        height = input_shape[2]

        # Load and preprocess the image for WD14
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
        csv_path = os.path.join(models_dir, "selected_tags.csv")
        df_tags = pd.read_csv(csv_path)

        # Get probabilities and create tag-confidence pairs
        tag_confidence = list(zip(df_tags["name"], confidence[0]))

        # Filter and sort tags
        threshold = 0.3  # Adjust this threshold as needed
        filtered_tags = [
            (tag, conf)
            for tag, conf in tag_confidence
            if conf > threshold
            and df_tags.loc[df_tags["name"] == tag, "category"].iloc[0] != 9
        ]

        # TODO: Filter out tags that are in the KAOMOJIS list
        filtered_tags.sort(key=lambda x: x[1], reverse=True)

        # Extract all tags for output without limiting
        wd14_tags = [tag for tag, _ in filtered_tags]  # No limit on tags

        # Combine results
        # result = {
        #     "florence_description": florence_description,
        #     "wd14_tags": ", ".join(wd14_tags)
        # }
        trigger_word: str = "{0} ".format(trigger) if trigger else ""
        result = f"{trigger_word}{florence_description} {', '.join(wd14_tags)}"
        delete_training_image(image_path)

        progress.update(task, advance=1)
        return result
    except Exception as e:
        print(
            f"Error processing image: {str(e)}"
        )  # Using print instead of feedback_message
        progress.update(task, advance=1)
        return None


def generate_florence_description(image, model, processor, is_object=False):
    try:
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        # Set the task token as the only input
        if is_object:
            task_prompt = "<OD>"
        else:
            task_prompt = "<MORE_DETAILED_CAPTION>"

        # Process the inputs with only the task token
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(
            device
        )

        # Validate inputs
        if not inputs or "input_ids" not in inputs or "pixel_values" not in inputs:
            return "Error generating description"

        # Move inputs to the correct device and dtype
        inputs = {
            k: v.to(
                device=device,
                dtype=model_dtype if v.dtype == torch.float32 else v.dtype,
            )
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

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

        # Post-process the generated text
        florence_description = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        if not florence_description:
            florence_description = "No description generated"

        return florence_description.get("<MORE_DETAILED_CAPTION>")
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return "Error generating description"
