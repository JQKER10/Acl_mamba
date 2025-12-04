import SimpleITK as sitk
import numpy as np
from typing import Tuple, List, Union, Optional, Callable

def crop_center(img: sitk.Image, crop_size: Tuple[int, int, int]) -> sitk.Image:
    """
    Crop the center region of a 3D image.

    Args:
        img (sitk.Image): Input 3D image.
        crop_size (Tuple[int, int, int]): Desired crop size (depth, height, width).

    Returns:
        sitk.Image: Cropped center region of the image.
    """
    img_size = img.GetSize()
    start = [(img_size[i] - crop_size[i]) // 2 for i in range(3)]
    end = [start[i] + crop_size[i] for i in range(3)]
    
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(crop_size)
    roi_filter.SetIndex(start)
    
    cropped_img = roi_filter.Execute(img)
    return cropped_img

def crop_random(img: sitk.Image, crop_size: Tuple[int, int, int]) -> sitk.Image:
    """
    Randomly crop a region from a 3D image.

    Args:
        img (sitk.Image): Input 3D image.
        crop_size (Tuple[int, int, int]): Desired crop size (depth, height, width).

    Returns:
        sitk.Image: Randomly cropped region of the image.
    """
    img_size = img.GetSize()
    start = [np.random.randint(0, img_size[i] - crop_size[i] + 1) for i in range(3)]
    
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(crop_size)
    roi_filter.SetIndex(start)
    
    cropped_img = roi_filter.Execute(img)
    return cropped_img

def crop_fixed(img: sitk.Image, start: Tuple[int, int, int], crop_size: Tuple[int, int, int]) -> sitk.Image:
    """
    Crop a fixed region from a 3D image.

    Args:
        img (sitk.Image): Input 3D image.
        start (Tuple[int, int, int]): Starting index for the crop (depth, height, width).
        crop_size (Tuple[int, int, int]): Desired crop size (depth, height, width).

    Returns:
        sitk.Image: Cropped region of the image.
    """
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(crop_size)
    roi_filter.SetIndex(start)
    
    cropped_img = roi_filter.Execute(img)
    return cropped_img

