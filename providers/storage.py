import asyncio
import os

import aioboto3
import aiohttp
import cv2
import numpy as np
from PIL import Image
from botocore.exceptions import ClientError

from config import app_config
from exceptions import DownloadError, DeleteError, UploadError


async def download_file(url, dest_path):
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        file_path = os.path.join(dest_path, os.path.basename(url))
        async with session.get(url) as response:
            if response.status == 200:
                with open(file_path, 'wb') as file:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        file.write(chunk)
                print(f"Downloaded file: {file_path}")
                return file_path
            else:
                print(f"Error downloading {file_path}: {response.status}")
                raise DownloadError("failed to download")


async def delete_file(file_path, log_enabled=True):
    try:
        await asyncio.to_thread(os.remove, file_path)
        if log_enabled:
            print(f"Deleted file: {file_path}")
    except Exception as e:
        raise DeleteError(f"Failed to delete file {e.args}")


async def upload_file(file_path, bucket_name, object_name):
    async with aioboto3.Session().client('s3') as s3_client:
        try:
            await s3_client.upload_file(file_path, bucket_name, object_name)
            print(f"Uploaded file: {file_path} to bucket: {bucket_name} as {object_name}")
            return f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        except ClientError as e:
            print(f"Error uploading {file_path} to S3: {e}")
            raise UploadError("Failed to upload file to S3")


def save_masks(name, input_image, masks):
    n = name.split(".")[0]
    paths = []
    os.makedirs(f"{app_config.paths.tmp_image_dir}outputs/", exist_ok=True)
    for i in range(masks.shape[0]):
        m = masks[i]
        mask = (m * 255).astype(np.uint8)
        img = cv2.bitwise_and(input_image, input_image, mask = mask)
        image = Image.fromarray(img)
        paths.append(f"{app_config.paths.tmp_image_dir}outputs/{n}_{i}.png")
        image.save(f"{app_config.paths.tmp_image_dir}outputs/{n}_{i}.png")
    return paths


def save_masklets(segments, mask_dir):
    paths = []
    os.makedirs(mask_dir, exist_ok=True)
    for frame in segments.items():
        for obj in frame[1].items():
            image = Image.fromarray((obj[1][0] * 255).astype(np.uint8))
            paths.append(f"{mask_dir}/{frame[0]}.png")
            image.save(f"{mask_dir}/{frame[0]}.png")
    return paths
