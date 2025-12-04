import SimpleITK as sitk
import os
import tqdm
import numpy as np
import json
from copy import deepcopy
import shutil
import glob
from light_training.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from light_training.normalization.default_normalization_schemes import ZScoreNormalization
from light_training.preprocessing.cropping.cropping import crop_to_nonzero
from .default_preprocessor import DefaultPreprocessor
from batchgenerators.utilities.file_and_folder_operations import *

class Promise12Preprocessor(DefaultPreprocessor):
    def __init__(self, 
                 base_dir,
                 image_dir,
                 data_filenames=[],
                 seg_filename="",
                 ):
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.data_filenames = data_filenames
        self.seg_filename = seg_filename

    def get_iterable_list(self):
        all_cases = os.listdir(os.path.join(self.base_dir, self.image_dir))
        return all_cases

    def _normalize(self, data: np.ndarray, seg: np.ndarray,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            normalizer_class = ZScoreNormalization
            normalizer = normalizer_class(use_mask_for_norm=False,
                                          intensityproperties=foreground_intensity_properties_per_channel)
            data[c] = normalizer.run(data[c], seg[0])
        return data
    
    def read_data(self, case_name):
        ## only for CT dataset
        assert len(self.data_filenames) != 0
        data = []
        for dfname in self.data_filenames:
            d = sitk.ReadImage(os.path.join(self.base_dir, self.image_dir, case_name, dfname))
            spacing = d.GetSpacing()
            data.append(sitk.GetArrayFromImage(d).astype(np.float32)[None,])
        
        data = np.concatenate(data, axis=0)

        seg_arr = None
        
    
        if self.seg_filename != "":
            seg = sitk.ReadImage(os.path.join(self.base_dir, self.image_dir, case_name, self.seg_filename))
            seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)[None,]
        
        return data, seg_arr