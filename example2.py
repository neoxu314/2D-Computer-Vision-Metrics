'''
An example for using BoundingBox class, plotting precision-recall curve, plotting BoundingBox objects on images, and
computing PASCAL-VOC-style mAP (11-points-interpolation method and all-points-interpolation method).
'''

import cv2
import os
from lib import object_detection_evaluation as ode
from lib import BoundingBox as bb
import numpy as np
from lib import utils


CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


################################################################
# Prepare for data (An example for using the BoundingBox class)#
################################################################
DATA_DIR = 'data/'

# Three images used for evaluation
img_filename_list = os.listdir(DATA_DIR)
img_path_list = []
for filename in img_filename_list:
    img_path_list.append(os.path.join(DATA_DIR, filename))

# Create ground truth bounding boxes and predicted bounding boxes for the images and store them in the lists
gt_bb_list = []
pr_bb_list = []
# Define the ground truth bounding box and the predicted bounding box for img1
img1_gt_bb1 = bb.BoundingBox(class_id='object', img_path='data/001.jpg', x1=72, y1=133, x2=881, y2=575, bb_type='gt')
img1_pr_bb1 = bb.BoundingBox(class_id='object', img_path='data/001.jpg', x1=60, y1=120, x2=850, y2=500, bb_type='pr', \
                           prediction_confidence=0.8)
gt_bb_list.append(img1_gt_bb1)
pr_bb_list.append(img1_pr_bb1)
# Define the ground truth bounding box and the predicted bounding box for img2
img2_gt_bb1 = bb.BoundingBox(class_id='object', img_path='data/002.jpg', x1=4, y1=160, x2=251, y2=313, bb_type='gt')
img2_gt_bb2 = bb.BoundingBox(class_id='object', img_path='data/002.jpg', x1=294, y1=10, x2=458, y2=309, bb_type='gt')
img2_pr_bb1 = bb.BoundingBox(class_id='object', img_path='data/002.jpg', x1=98, y1=134, x2=318, y2=317, bb_type='pr', \
                           prediction_confidence=0.77)
gt_bb_list.append(img2_gt_bb1)
gt_bb_list.append(img2_gt_bb2)
pr_bb_list.append(img2_pr_bb1)
# Define the ground truth bounding box and the predicted bounding box for img3
img3 = cv2.imread('data/003.jpg')
img3_height, img3_width = img3.shape[:2]
img3_gt_bb1 = bb.BoundingBox(class_id='object', img_path='data/003.jpg', x1=26, y1=23, x2=317, y2=img3_height-1,
                             bb_type='gt')
img3_gt_bb2 = bb.BoundingBox(class_id='object', img_path='data/003.jpg', x1=258, y1=0, x2=575, y2=396, bb_type='gt')
img3_pr_bb1 = bb.BoundingBox(class_id='object', img_path='data/003.jpg', x1=10, y1=10, x2=210, y2=310, bb_type='pr', \
                           prediction_confidence=0.70)
img3_pr_bb2 = bb.BoundingBox(class_id='object', img_path='data/003.jpg', x1=266, y1=47, x2=580, y2=329, bb_type='pr', \
                           prediction_confidence=0.66)
gt_bb_list.append(img3_gt_bb1)
gt_bb_list.append(img3_gt_bb2)
pr_bb_list.append(img3_pr_bb1)
pr_bb_list.append(img3_pr_bb2)
# Define the ground truth bounding box and the predicted bounding box for img4
img4_gt_bb1 = bb.BoundingBox(class_id='object', img_path='data/004.jpg', x1=52, y1=77, x2=252, y2=184, bb_type='gt')
img4_gt_bb2 = bb.BoundingBox(class_id='object', img_path='data/004.jpg', x1=266, y1=87, x2=410, y2=187, bb_type='gt')
img4_pr_bb1 = bb.BoundingBox(class_id='object', img_path='data/004.jpg', x1=30, y1=60, x2=106, y2=104, bb_type='pr', \
                           prediction_confidence=0.53)
img4_pr_bb2 = bb.BoundingBox(class_id='object', img_path='data/004.jpg', x1=210, y1=72, x2=410, y2=194, bb_type='pr', \
                           prediction_confidence=0.59)
img4_pr_bb3 = bb.BoundingBox(class_id='object', img_path='data/004.jpg', x1=390, y1=160, x2=420, y2=190, bb_type='pr', \
                           prediction_confidence=0.40)
gt_bb_list.append(img4_gt_bb1)
gt_bb_list.append(img4_gt_bb2)
pr_bb_list.append(img4_pr_bb1)
pr_bb_list.append(img4_pr_bb2)
pr_bb_list.append(img4_pr_bb3)
# Define the ground truth bounding box and the predicted bounding box for img5
img5_gt_bb1 = bb.BoundingBox(class_id='object', img_path='data/005.jpg', x1=1, y1=127, x2=266, y2=473, bb_type='gt')
img5_gt_bb2 = bb.BoundingBox(class_id='object', img_path='data/005.jpg', x1=209, y1=96, x2=472, y2=396, bb_type='gt')
img5_pr_bb1 = bb.BoundingBox(class_id='object', img_path='data/005.jpg', x1=6, y1=115, x2=277, y2=475, bb_type='pr', \
                           prediction_confidence=0.97)
img5_pr_bb2 = bb.BoundingBox(class_id='object', img_path='data/005.jpg', x1=180, y1=300, x2=470, y2=390, bb_type='pr', \
                           prediction_confidence=0.61)
gt_bb_list.append(img5_gt_bb1)
gt_bb_list.append(img5_gt_bb2)
pr_bb_list.append(img5_pr_bb1)
pr_bb_list.append(img5_pr_bb2)


###################################################################################################################
# Test evaluation methods (plot precision-recall curve, compute mAP by the 11-points-interpolation method and the #
# all-points-interpolation method of the PASCAL VOC)                                                              #
###################################################################################################################
# Get the dictionary list for evaluated predicted bounding boxes, whose item contains a predicted bounding box and a
pr_bb_dict_list = ode.eval_predicted_bb_list(gt_bb_list, pr_bb_list)
class_id_list = list(set([bb_dict['bb'].get_class_id() for bb_dict in pr_bb_dict_list]))
ap_by_11_points_list = []
ap_by_all_points_list = []

for class_id in class_id_list:
    class_pr_bb_dict_list = \
        [pr_bb_dict for pr_bb_dict in pr_bb_dict_list if pr_bb_dict['bb'].get_class_id() == class_id]
    class_gt_bb_list = [gt_bb for gt_bb in gt_bb_list if gt_bb.get_class_id() == class_id]
    acc_precision_recall_dict_list = \
        ode.draw_precision_recall_curve(class_pr_bb_dict_list, len(class_gt_bb_list), plot=True)
    ap_by_11_points = ode.get_ap_by_11_points_interpolation(acc_precision_recall_dict_list)
    ap_by_11_points_list.append(ap_by_11_points)
    ap_by_all_points = ode.get_ap_by_all_points_interpolation(acc_precision_recall_dict_list)
    ap_by_all_points_list.append(ap_by_all_points)

map_by_11_points = np.mean(ap_by_11_points_list)
map_by_all_points = np.mean(ap_by_all_points_list)

print('The mAP calculated by the 11-points-interpolation method of PASCAL VOC is: ', map_by_11_points)
print('The mAP calculated by the 11-points-interpolation method of PASCAL VOC is: ', map_by_all_points)


############################################################
# An example for Visualizing images with corresponding bbs #
############################################################
# for img_path in img_path_list:
#     img_gt_bb_list = [gt_bb for gt_bb in gt_bb_list if gt_bb.get_img_path() == img_path]
#     # img_pr_bb_list = [pr_bb for pr_bb in pr_bb_list if pr_bb.get_img_path() == img_path]
#     img_pr_bb_dict_list = [pr_bb_dict for pr_bb_dict in pr_bb_dict_list if pr_bb_dict['bb'].get_img_path() == img_path]
#     img_with_bbs = cv2.imread(img_path)
#     for img_gt_bb in img_gt_bb_list:
#         utils.draw_bb_on_img(img_gt_bb, img_with_bbs)
#     for img_pr_bb_dict in img_pr_bb_dict_list:
#         utils.draw_bb_on_img(img_pr_bb_dict['bb'], img_with_bbs)
#         upper_left_corner = img_pr_bb_dict['bb'].get_upper_left_corner()
#         cv2.putText(img_with_bbs, str(img_pr_bb_dict['TP']), upper_left_corner, CV2_FONT, 0.5, \
#                     (0, 0, 0), 1, cv2.LINE_AA)
#     cv2.imshow(img_path, img_with_bbs)


cv2.waitKey(0)


