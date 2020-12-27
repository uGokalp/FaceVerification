"""
This module contains helpful utilities for recognition package.
"""

import PIL
from typing import Union
import numpy as np

from PIL import PngImagePlugin, JpegImagePlugin, Image

IMAGE_TYPES = (
    Image.Image,
    PngImagePlugin.PngImageFile,
    JpegImagePlugin.JpegImageFile,
)


def type_check(argument, types: tuple) -> None:
    """Used to assert right types are passes to the function"""
    if type(argument) not in types:
        raise TypeError(
            f"Wrong type ({type(argument)}) passed. Expected one of {types}"
        )


def check_image_channels(img: Union[np.ndarray, Image.Image], num_channels=3) -> None:
    """Checks image channels to be equal to num_channels"""
    types = (
        np.ndarray,
        Image.Image,
        PIL.PngImagePlugin.PngImageFile,
    )  # Last option is a workaround.
    type_check(img, types)

    if isinstance(img, np.ndarray):
        assert (
            img.ndim == num_channels
        ), f"Expected image to have ndim <= {num_channels}, but got {img.ndim}"
    if isinstance(img, Image.Image):
        if type(img) != Image.Image:
            img = img.convert("RGB")
        img_array = np.array(img)
        check_image_channels(img_array)
