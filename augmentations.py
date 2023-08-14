import albumentations as A
from albumentations.pytorch import transforms as A_torch
from albumentations.augmentations.functional import _maybe_process_in_chunks, preserve_shape
import torch
from scipy.ndimage import gaussian_filter


def build_train_augmentations(in_channels = 2):
    """augmentations utilizadas pelo campeão do XView3. No caso do ScaleRotate, foi utilizada reflexão ao invés do padding com NaN""" 
    transforms = A.Compose(
		[
			#UnclippedRandomBrightnessContrast(brightness_limit=(-1,1), contrast_limit=0.1, image_in_log_space=False, p=0.25),
			#UnclippedGaussNoise(image_in_log_space=False, var_limit=(0.0001, 0.005), mean=0, per_channel=True, p=0.5),
			A.HorizontalFlip(p=0.2),
			A.VerticalFlip(p=0.2),
			#ElasticTransform(alpha=(10,100), p=0.1),
			#A.ShiftScaleRotate(scale_limit=0, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
			#RandomGridShuffle(p=0.2),
			#A.MedianBlur(blur_limit=5, p=0.05),
			#A.GaussianBlur(blur_limit=(3,5),p=0.05),
            A.Normalize(
                mean=in_channels*[0.0],
                std=in_channels*[1.0],
                max_pixel_value=255.0,
            ),
            A.Resize(height = 512, width = 512),
			A_torch.ToTensorV2()
		]
    )
    return transforms

def build_test_augmentations(in_channels = 2):
    transforms = A.Compose(
		[
            A.Normalize(
                mean=in_channels*[0.0],
                std=in_channels*[1.0],
                max_pixel_value=255.0,
            ),
            A.Resize(height = 512, width = 512),
			A_torch.ToTensorV2()
		]
    )
    return transforms

def apply_TTA(image, DEVICE):
    h_image = torch.Tensor(image.cpu().numpy()[:,:,:,::-1].copy())
    v_image = torch.Tensor(image.cpu().numpy()[:,:,::-1,:].copy())
    #r_image = torch.Tensor(image.cpu().numpy()[:,:,::-1,::-1].copy())
    r_image = torch.Tensor(gaussian_filter(image.cpu().numpy(), sigma=7).copy())
    
    return h_image, v_image, r_image

def revert_TTA(h_mask, v_mask, r_mask, DEVICE):
    h_output = torch.Tensor(h_mask.cpu().numpy()[:,:,:,::-1].copy())
    v_output = torch.Tensor(v_mask.cpu().numpy()[:,:,::-1,:].copy())
    #r_output = torch.Tensor(r_mask.cpu().numpy()[:,:,::-1,::-1].copy())
    r_output = torch.Tensor(r_mask.cpu().numpy().copy())
    
    return h_output, v_output, r_output

def get_final_bin_mask2(orig_bin_mask, h_bin_mask, v_bin_mask, r_bin_mask):
    sum_bin_mask = orig_bin_mask + h_bin_mask + v_bin_mask + r_bin_mask
    final_bin_mask = sum_bin_mask >= 3
    return final_bin_mask

def get_final_bin_mask(orig_bin_mask, h_bin_mask, v_bin_mask):
    sum_bin_mask = orig_bin_mask + h_bin_mask + v_bin_mask
    final_bin_mask = sum_bin_mask >= 2
    return final_bin_mask