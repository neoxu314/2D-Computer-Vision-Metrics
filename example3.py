'''
This is an example for computing the IoU of masks for image instance segmentation.
'''
from lib import instance_segmentation_evaluation as ise
import numpy as np
import cv2


# Create a matrix for ground-truth masks, whose shape is [height, width, instances] where height and width are the
# dimension of the corresponding image and instances is the number of all masks (objects). The matrix is a Boolean
# matrix where True represents the area has masks and False represents the area has no mask.
Gt_masks = np.zeros((500, 500, 2), dtype=bool)
Gt_masks[300:400, 300:400, 0] = True
Gt_masks[50:200, 50:200, 1]= True
# Visualize the defined masks
# cv2.imshow('Ground-truth mask 1', Gt_masks[:, :, 0])
# cv2.imshow('Ground-truth mask 2', Gt_masks[:, :, 1])

# Create a matrix for predicted masks, which has the same properties with the ground-truth masks.
Pr_masks = np.zeros((500, 500, 2), dtype=bool)
Pr_masks[290:390, 300:400, 0] = True
Pr_masks[40:200, 30:210, 1] = True

# Compute the IoU for the ground-truth masks and the predicted masks.
IoUs = ise.compute_iou_of_masks(Gt_masks, Pr_masks)
print(IoUs)


cv2.waitKey(0)
