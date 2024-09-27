import os

import cv2
import numpy as np
import torch

from config import app_config
from models.requests import SAMRequest
from providers import storage
from sam2.build_sam import build_sam2_video_predictor


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
async def segment_video(payload: SAMRequest):
    try:
        name = os.path.basename(payload.media_url).split(".")[0]
        video_path = await storage.download_file(payload.media_url, f"{app_config.paths.tmp_video_dir}inputs/")
        video_dir = f"{app_config.paths.tmp_video_dir}/frames/{name}/"
        fps = _extract_frames(video_path, video_dir)
        predictor = _build_video_model(payload.model.get_config(), payload.model.get_checkpoint())
        inference_state = predictor.init_state(video_path=video_dir)

        obj_id = 1
        input_point = np.array([[[pointer.x, pointer.y]] for pointer in payload.pointers])
        input_label = np.array([[pointer.label] for pointer in payload.pointers])

        print(f"Identifying object in from {payload.frame_idx}")
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=payload.frame_idx,
            obj_id=obj_id,
            points=input_point,
            labels=input_label,
        )
        print("Processing masklets...")
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for frame_idx, obj_ids, masklets in predictor.propagate_in_video(
                inference_state
        ):
            per_obj_output_mask = {
                obj_id: (masklets[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }
            video_segments[frame_idx] = per_obj_output_mask

        print("Saving masklets...")
        mask_dir = f"{app_config.paths.tmp_video_dir}outputs/{name}/"
        mask_paths = storage.save_masklets(video_segments, name)
        output_video = f"{app_config.paths.tmp_video_dir}outputs/{name}.mp4"
        print("Creating mask video...")
        _combine_frames(mask_dir, output_video, fps)
        print("Uploading video...")
        masklet_url = await storage.upload_file(
            output_video,
            app_config.storage.s3.bucket,
            f"stg/SAM/video_masks/{name}.mp4"
        )
        for path in mask_paths:
            await storage.delete_file(path, False)
        await storage.delete_file(video_path)
        return {
            "media_url": payload.media_url,
            "masks": [
                masklet_url
            ]
        }
    except Exception as e:
        raise e


def _build_video_model(config, checkpoint):
    return build_sam2_video_predictor(config, ckpt_path=checkpoint, device="cuda")


def _extract_frames(video_path, output_dir, quality=95):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0000
    os.makedirs(output_dir, exist_ok=True)
    print("Extracting fps...")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Extracting frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"{frame_count}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame_count += 1

    cap.release()
    print(f"Total frames extracted: {frame_count}")
    return fps


def _combine_frames(frames_dir, output_path, fps=30.0):
    # Get all image files in the specified folder
    images = [img for img in os.listdir(frames_dir)]

    # Sort images by filename to maintain order
    images.sort()

    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the width and height
    first_image_path = os.path.join(frames_dir, images[0])
    frame = cv2.imread(first_image_path)

    if frame is None:
        print(f"Error reading image: {first_image_path}")
        return

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(frames_dir, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error reading image: {image_path}")
            continue

        # Resize frame if necessary (optional)
        frame = cv2.resize(frame, (width, height))

        # Write the frame to the video
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved at: {output_path}")
