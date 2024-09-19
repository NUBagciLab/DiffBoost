""""
Write the image back for visualization
For 2D image, write to png format
For 3D image, write to nii.gz for mat
"""

import os
import numpy as np
from skimage import io as skio
import SimpleITK as sitk

def write_2d_image(image_np:np.array, out_dir:str, 
                   case_name:str, meta_info=None):
    num_classes = meta_info["meta"]['num_classes']
    skio.imsave(os.path.join(out_dir, case_name+'.png'),
                np.round(255*image_np/num_classes).astype(np.uint8))

def write_3d_image(image_np:np.array, out_dir:str, 
                   case_name:str, meta_info=None):
    image = sitk.GetImageFromArray(image_np)
    image.SetSpacing(meta_info["spacing"][::-1])
    image_meta = meta_info["meta"]
    for key in image_meta.keys():
        image.SetMetaData(key, image_meta[key])
    
    sitk.WriteImage(sitk.Cast(image, sitk.sitkUInt8),
                    os.path.join(out_dir, case_name+'.nii.gz'))