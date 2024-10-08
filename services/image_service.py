import os

import numpy as np
import torch
from PIL import Image

from config import app_config
from enums import MediaType
from models.requests import SAMRequest
from providers import storage
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from .video_service import segment_video


async def segment_media(payload: SAMRequest):
    if payload.media_type == MediaType.Image:
        return await segment_image(payload)
    return await segment_video(payload)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
async def segment_image(payload: SAMRequest):
    try:
        file_path = await storage.download_file(payload.media_url, f"{app_config.paths.tmp_file_dir}inputs/")
        model = _build_model(payload.model.get_config(), payload.model.get_checkpoint())
        print(f"Built SAM model: {payload.model.get_config()} config and {payload.model.get_checkpoint()} checkpoint")
        predictor = SAM2ImagePredictor(model)
        input_image = _get_image(file_path)
        predictor.set_image(input_image)

        input_point = np.array([[[pointer.x, pointer.y]] for pointer in payload.pointers])
        input_label = np.array([[pointer.label] for pointer in payload.pointers])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        print(f"Image Segmentation successful.")
        mask_paths = storage.save_masks(os.path.basename(payload.media_url), input_image, masks)
        mask_urls = []
        for path in mask_paths:
            url = await storage.upload_file(
                path,
                app_config.storage.s3.bucket,
                f"stg/SAM/image_masks/{os.path.basename(path)}")
            mask_urls.append(url)
            await storage.delete_file(path)
        print(f"Masked upload successful")
        await storage.delete_file(file_path)
        return {
            "media_url": payload.media_url,
            "masks": mask_urls,
        }
    except Exception as e:
        raise e


def _build_model(config, checkpoint):
    return build_sam2(config, ckpt_path=checkpoint, device="cuda")


def _get_image(path):
    image = Image.open(path)
    return np.array(image.convert("RGB"))
