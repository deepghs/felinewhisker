from typing import Tuple

from PIL import Image
from imgutils.data import load_image, ImageTyping


def padding_align(image: ImageTyping, size: Tuple[int, int], color: str = 'white') -> Image.Image:
    width, height = size
    image = load_image(image, force_background=None, mode='RGBA')
    r = min(width / image.width, height / image.height)
    resized = image.resize((int(image.width * r), int(image.height * r)))

    new_image = Image.new('RGBA', (width, height), color)
    left, top = int((new_image.width - resized.width) // 2), int((new_image.height - resized.height) // 2)
    new_image.paste(resized, (left, top, left + resized.width, top + resized.height), resized)
    return new_image
