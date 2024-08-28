import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.io import read_image, write_png
from torch.nn.functional import interpolate
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path


class ImageProcessor:
    def __init__(self, output_size=512):
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_image(
        self, image_sizes: list, image_path: str, output_path: str, output_dir: str
    ) -> None:
        # Using the ImageProcessor class from images.py
        # process a single image into 1024x1024 pixels, 768x768 pixels, and 512x512 pixels
        # and save the processed images in the output directory

        for size in image_sizes:
            output_filename = (
                f"{Path(output_path).stem}_{size}{Path(output_path).suffixes[0]}"
            )
            # print(output_filename)
            self._resize_image(image_path, output_filename, output_dir)

    def _resize_image(self, image_path, output_path, output_dir):
        # Read the image
        img = (
            read_image(image_path).to(self.device).float()
        )  # Convert to float for better precision

        # Find minimum dimension
        min_dimension = min(img.shape[1:])

        # Crop image
        img_cropped = self._center_crop(img, min_dimension)

        # Determine if we're upscaling or downscaling
        current_size = img_cropped.shape[-1]  # Assuming square image after cropping
        if self.output_size > current_size:
            # Upscaling: use bicubic
            mode = "bicubic"
            align_corners = False
        else:
            # Downscaling: use area
            mode = "area"
            align_corners = None  # 'area' mode doesn't use align_corners

        # Resize image
        img_resized = F.interpolate(
            img_cropped.unsqueeze(0),
            size=(self.output_size, self.output_size),
            mode=mode,
            align_corners=align_corners,
        ).squeeze(0)

        # Clamp values to valid range and convert back to uint8
        img_resized = img_resized.clamp(0, 255).to(torch.uint8)

        # Save the processed image
        final_output_path = f"{output_dir}/{output_path}"
        write_png(img_resized.cpu(), final_output_path)

    def _center_crop(self, img, min_dimension):
        center = [dim // 2 for dim in img.shape[1:]]
        crop_start = [c - min_dimension // 2 for c in center]
        crop_end = [c + min_dimension // 2 for c in center]
        return img[:, crop_start[0] : crop_end[0], crop_start[1] : crop_end[1]]
