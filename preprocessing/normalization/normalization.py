from monai.transforms import NormalizeIntensityd, ScaleIntensityRanged
import SimpleITK as sitk
from typing import Tuple, Union, Optional, Callable
import numpy as np

def normalize_intensity(img: sitk.Image,
                        method: str = 'minmax',
                        min_max: Optional[Tuple[float, float]] = None,
                        mean_std: Optional[Tuple[float, float]] = None) -> sitk.Image:
    """
    Normalize the intensity of a 3D image using specified method.

    Args:
        img (sitk.Image): Input 3D image.
        method (str): Normalization method ('minmax' or 'zscore').
        min_max (Optional[Tuple[float, float]]): Min and max values for min-max normalization.
        mean_std (Optional[Tuple[float, float]]): Mean and std values for z-score normalization.

    Returns:
        sitk.Image: Normalized 3D image.
    """
    img_array = sitk.GetArrayFromImage(img).astype(np.float32)

    if method == 'minmax':
        if min_max is None:
            min_val, max_val = np.min(img_array), np.max(img_array)
        else:
            min_val, max_val = min_max
        img_array = (img_array - min_val) / (max_val - min_val)
    elif method == 'zscore':
        if mean_std is None:
            mean_val, std_val = np.mean(img_array), np.std(img_array)
        else:
            mean_val, std_val = mean_std
        img_array = (img_array - mean_val) / std_val
    else:
        raise ValueError("Unsupported normalization method. Use 'minmax' or 'zscore'.")

    normalized_img = sitk.GetImageFromArray(img_array)
    normalized_img.CopyInformation(img)
    return normalized_img

def normalize_intensity_dict(data: dict,
                             keys: list,
                             method: str = 'minmax',
                             min_max: Optional[Tuple[float, float]] = None,
                             mean_std: Optional[Tuple[float, float]] = None) -> dict:
    """
    Normalize the intensity of images in a dictionary using specified method.

    Args:
        data (dict): Dictionary containing images to be normalized.
        keys (list): List of keys in the dictionary corresponding to images.
        method (str): Normalization method ('minmax' or 'zscore').
        min_max (Optional[Tuple[float, float]]): Min and max values for min-max normalization.
        mean_std (Optional[Tuple[float, float]]): Mean and std values for z-score normalization.

    Returns:
        dict: Dictionary with normalized images.
    """
    for key in keys:
        img = data[key]
        normalized_img = normalize_intensity(img, method, min_max, mean_std)
        data[key] = normalized_img
    return data

