'''
This is the example for using BoundingBox class and computing the IoU between two bounding boxes.
'''
import cv2
from lib import object_detection_evaluation as cve
from lib import BoundingBox as bb
from lib import utils


def main():
    # Get the example image
    img_path = 'data/001.jpg'
    img = cv2.imread(img_path)
    # Define the ground-truth bounding box and the predicted bounding box.
    gt_bb = bb.BoundingBox(class_id='car', img_path=img_path, x1=72, y1=133, x2=881, y2=575, bb_type='gt')
    pr_bb = bb.BoundingBox(class_id='car', img_path=img_path, x1=60, y1=120, x2=850, y2=500, bb_type='pr', \
                           prediction_confidence=0.8)

    # Get the iou based on the two bounding boxes above
    iou, intersection_area, union_area = cve.get_iou_from_2_bbs(gt_bb, pr_bb)
    # Draw the ground-truth bounding box and the predicted bounding box on the image
    img_with_bbs = utils.draw_bb_on_img(gt_bb, img)
    img_with_bbs = utils.draw_bb_on_img(pr_bb, img_with_bbs)

    # Show the result
    cv2.putText(img_with_bbs, "IoU: {:.4f}".format(iou), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Car detection', img_with_bbs)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()