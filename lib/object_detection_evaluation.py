import sys
import numpy as np
import matplotlib.pyplot as plt


def get_iou_from_2_bbs(bb1, bb2):
    """ Get the IoU score from two input bounding boxes.

    Args:
        bb1: A BoundingBox object for one of the input bounding boxes.
        bb2: Another BoundingBox object for another input bounding box.

    Returns:
        iou: The IoU score of the given two bounding boxes.
        intersection_area: The area of the intersection between the two input bounding boxes.
        union_area: The area of the union between the two input bounding boxes.
    """
    # Get the upper-left corner and the lower-right corner of the intersection rectangle area if the intersection exists
    x1 = max(bb1._x1, bb2._x1)
    y1 = max(bb1._y1, bb2._y1)
    x2 = min(bb1._x2, bb2._x2)
    y2 = min(bb1._y2, bb2._y2)

    # If the intersection does not exist, the length of at least one side will be 0
    intersection_area = max(0, x2-x1+1) * max(0, y2-y1+1)
    union_area = (bb1._x2-bb1._x1+1)*(bb1._y2-bb1._y1+1) + ((bb2._x2-bb2._x1+1)*(bb2._y2-bb2._y1+1)) - intersection_area
    iou = intersection_area/union_area

    return iou, intersection_area, union_area


def eval_predicted_bb_list(gt_bb_list, pr_bb_list, iou_threshold = 0.3):
    """ Evaluate the predicted bounding box list based on the corresponding ground-truth bounding box list and the IoU
    threshold. For each image, first sort its predicted bounding boxes by the decreasing order of prediction confidence.
    Then find the corresponding ground-truth bounding box for each predicted bounding box based on the rule that: 1.
    process the predicted bounding box from the one which has the largest prediction confidence; 2. if a prediction
    bounding box is overlapping over multiple ground-truth bounding boxes, the selected corresponding ground truth
    bounding box is the one which has the largest IoU (greater than IoU threshold) with the predicted bounding box;
    3. after looping over all the predicted bounding boxes, if one predicted bounding box has a corresponding
    ground-truth bounding box, it is a TP, otherwise, it is a FP.

    Args:
        gt_bb_list: A list of BoundingBox objects for all the ground truth bounding boxes.
        pr_bb_list: A list of BoundingBox objects for all the corresponding predicted bounding boxes of gt_bbs
        iou_threshold: If a prediction bounding box is overlapping over multiple ground-truth bounding boxes, the
        selected corresponding ground truth bounding box is the one which has the largest IoU (greater than IoU
        threshold) with the predicted bounding box.

    Returns:
        evaluated_predicted_bb_dict_list: A sorted dictionary list of all the input predicted bounding boxes, whose each
        item contains a BoundingBox object for a predicted bounding box and a corresponding TP flag. If TP is True,
        the corresponding predicted bounding box is a TP; otherwise, it is a FP. The list is sorted by the decreasing
        order of the prediction confidence of each predicted bounding box.
    """
    evaluated_predicted_bb_dict_list = []

    # Get image path list from the ground truth bounding boxes for current class
    img_path_list = list(set([gt_bb.get_img_path() for gt_bb in gt_bb_list]))

    # Count TP, FP, FN for each images
    for img_path in img_path_list:
        # print('Count TP and FP for', img_path)

        # Get the ground truth bounding box list for current processing image
        img_gt_bb_list = [gt_bb for gt_bb in gt_bb_list if gt_bb.get_img_path() == img_path]
        # Create a dictionary list for gt_bb whose item contains a ground truth bounding box and a flag 'assigned'
        # used for recording whether the ground truth bounding box is assigned to a valid predicted bounding box.
        img_gt_bb_dict_list = []
        for gt_bb in img_gt_bb_list:
            item = {
                'bb': gt_bb,
                'assigned': False
            }
            img_gt_bb_dict_list.append(item)

        # Get the predicted bounding box list of current processing image
        img_pr_bb_list = [pr_bb for pr_bb in pr_bb_list if pr_bb.get_img_path() == img_path]
        # Create a dictionary list for predicted bounding boxes whose item contains a predicted bounding box object
        # and a TP flag.
        img_pr_bb_dict_list = []
        for pr_bb in img_pr_bb_list:
            item = {
                'bb': pr_bb,
                'TP': False
            }
            img_pr_bb_dict_list.append(item)

        # Sort the img_pr_bb_list by the decreasing order of prediction confidence
        sorted_img_pr_bb_dict_list = sorted(img_pr_bb_dict_list,
                                            key=lambda bb_dict: bb_dict['bb'].get_prediction_confidence(),
                                            reverse=True)

        # Loop over all the predicted bounding boxes and compare them to ground-turth bounding boxes to count TP
        # and FP for the current image
        for pr_bb_dict in sorted_img_pr_bb_dict_list:
            iou_max = sys.float_info.min
            bb_id_max = 0

            # Loop over all ground-truth bounding boxes to get the ground-truth bounding box which has the
            # largest IoU with the current predicted bounding box from un-assigned ground-truth bounding boxes
            for bb_id in range(len(img_gt_bb_dict_list)):
                # If the current ground-truth bounding box is not assigned to any predicted bounding box, then
                # compute the iou between it and the predicted bounding box
                if not img_gt_bb_dict_list[bb_id]['assigned']:
                    iou, _, _ = get_iou_from_2_bbs(pr_bb_dict['bb'], img_gt_bb_dict_list[bb_id]['bb'])
                    if iou > iou_max:
                        iou_max = iou
                        bb_id_max = bb_id

            # If the largest IoU is greater than or equal to the IoU threshold, then assign the corresponding
            # ground-truth bounding box to the current predicted bounding box and set the corresponding predicted
            # bounding box dictionary item's TP attribute to True.
            if iou_max >= iou_threshold:
                img_gt_bb_dict_list[bb_id_max]['assigned'] = True
                pr_bb_dict['TP'] = True

        evaluated_predicted_bb_dict_list.extend(sorted_img_pr_bb_dict_list)

    evaluated_predicted_bb_dict_list = sorted(evaluated_predicted_bb_dict_list,
                                              key=lambda bb_dict: bb_dict['bb'].get_prediction_confidence(),
                                              reverse=True)
    return evaluated_predicted_bb_dict_list


def draw_precision_recall_curve(evaluated_predicted_bb_dict_list, ground_truth_bounding_box_num, plot = True):
    """Draw the precision-recall curve based on the evaluated predicted bounding box list. The precision-recall curve is
    generated based on the accumulated precision list and the accumulated recall list. To get the accumulated precision
    list and the accumulated recall list: 1. loop over the evaluated predicted bounding box list from the bounding box
    which has the largest confidence value; for each bounding box, if it is a TP then add 1 to the accumulated TP, if it
    is a FP then add 1 to the accumulated FP; then calculate the current precision and recall based on the current
    value of the accumulated TP and the accumulated FP, and append the current precision-recall pair to the resulting
    list.

    Args:
        evaluated_predicted_bb_dict_list: A dictionary list of evaluated predicted bounding boxes. Each item contains a
        BoundingBox object for a predicted bounding box and a TP flag (True for TP, False for FP).
        ground_truth_bounding_box_num:
        plot: A flag for determining whether this function plotting the precision-recall curve.

    Returns:
        acc_precision_recall_dict_list: The dictionary list whose each item contains a pair of precision and recall
        values.
    """
    acc_precision_recall_dict_list = []

    acc_TP = 0
    acc_FP = 0

    for bb_dict in evaluated_predicted_bb_dict_list:
        # print(bb_dict['bb'].get_prediction_confidence())
        # print(bb_dict['TP'])

        if bb_dict['TP']:
            acc_TP = acc_TP + 1
        else:
            acc_FP = acc_FP + 1
        # print('acc TP:', acc_TP)
        # print('acc FP:', acc_FP)

        precision = acc_TP / (acc_TP + acc_FP)
        recall = acc_TP / ground_truth_bounding_box_num

        item = {
            'precision': precision,
            'recall': recall
        }
        acc_precision_recall_dict_list.append(item)
        # print('precision: ', precision)
        # print('recall: ', recall)

    if plot:
        precision_list = [item['precision'] for item in acc_precision_recall_dict_list]
        recall_list = [item['recall'] for item in acc_precision_recall_dict_list]
        plt.plot(recall_list, precision_list)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

    return acc_precision_recall_dict_list


def get_ap_by_11_points_interpolation(acc_precision_recall_dict_list):
    """The core of the 11-points-interpolation method of PASCAL VOC is that in the precision-recall curve, choose 11
    equally spaced recall levels [0, 0.1, 0.2, ..., 1]. For each recall level r_i, find the corresponding maximum
    precision. The corresponding largest precision at each recall level r_i is got by finding the maximum precision
    value in the precision-recall curve in the range of [r_i, 1]. Once all the 11 maximum precision values are found,
    the AP can be got by calculating the mean of these values.

    Args:
        acc_precision_recall_dict_list: The input dictionary list whose each item is a pair of precision and recall.
        values. The list must be sorted by the increasing order of recall value.

    Returns:
        ap: The AP calculated by using 11-points-interpolation method of PASCAL VOC.
    """
    # Sort the acc_precision_recall_list
    sorted_acc_precision_recall_dict_list = sorted(acc_precision_recall_dict_list, key=lambda item: item['recall'])
    recall_max = sorted_acc_precision_recall_dict_list[len(sorted_acc_precision_recall_dict_list)-1]['recall']
    precision_max_list = []

    for i in np.arange(0,1.1,0.1):
        # For each interpolation recall point i in the precision-recall curve, find the corresponding precision
        # From recall i to recall_max, find the largest precision in the precision-recall curve
        if i <= recall_max:
            precision_max = sys.float_info.min
            for item in sorted_acc_precision_recall_dict_list:
                if item['recall'] >= i and item['precision'] >= precision_max:
                    precision_max = item['precision']
            precision_max_list.append(precision_max)
        else:
            precision_max_list.append(0)

    ap = np.mean(precision_max_list)
    return ap


def get_ap_by_all_points_interpolation(acc_precision_recall_dict_list):
    """The core of the all-points-interpolation method of PASCAL VOC is that in the precision-recall curve, for each
    point (r_1, p_1), find the corresponding vertex (r_2, p_2) where r_1 <= r_2 <= r_max and p_2 is the largest
    precision value of the precision-recall curve in the range of [r_1,r_max]. Once all vertices, (vr_1, vp_1),
    (vr_2, vp_2), ..., (vr_n, vp_n) are found, then computing the mAP of the precision-recall curve by calculating the
    sum of (vr_(j+1) - vr_j)*vp_(j+1), where 0 <= j <= 1.
    (It is better to use an graphical precision-recall curve to understand this algorithm.)

    Args:
        acc_precision_recall_dict_list: The input dictionary list whose each item is a pair of precision and recall.
        values. The list must be sorted by the increasing order of recall value.

    Returns:
        ap: The AP calculated by using all-points-interpolation method of PASCAL VOC.
    """
    # Sort the accumulated accuracy-precision list by the increasing order of recall
    sorted_acc_precision_recall_dict_list = sorted(acc_precision_recall_dict_list, key=lambda item: item['recall'])

    vertex_list = []

    i = 0
    while i < len(sorted_acc_precision_recall_dict_list) - 1:
        # Start with the least recall, for each point (r_i, p_i) in the precision-recall curve, find the corresponding
        # vertex.
        precision_max = sorted_acc_precision_recall_dict_list[i]['precision']
        j_max = i
        found_vertex = False
        for j in range(i, len(sorted_acc_precision_recall_dict_list)):
            if sorted_acc_precision_recall_dict_list[j]['precision'] >= precision_max:
                precision_max = sorted_acc_precision_recall_dict_list[j]['precision']
                j_max = j
                found_vertex = True

        if not found_vertex:
            # If no vertex was found, then break the while loop.
            break
        else:
            # If one vertex was found, then append the vertex to the vertex list.
            vertex_list.append(sorted_acc_precision_recall_dict_list[j_max])
            i = j_max
            # Increasing i to continue the loop: if p_(i+1) == p_i for the points (r_(j_max), p_(j_max)) and
            # (r_(i+1), p_(i+1)), then continue increasing i till p_(i+1) != p_(j_max).
            while True:
                if (i + 1) < len(sorted_acc_precision_recall_dict_list):
                    if sorted_acc_precision_recall_dict_list[i+1]['recall'] == \
                            sorted_acc_precision_recall_dict_list[j_max]['recall']:
                        i = i + 1
                    else:
                        break
                else:
                    break

    # Firstly, check if the found vertices contains the points where recall == 0 or recall == 1
    found_recall_0 = False
    found_recall_1 = False
    for vertex in vertex_list:
        if vertex['recall'] == 0:
            found_recall_0 = True
        elif vertex['recall'] == 1:
            found_recall_1 = True

    # If there is no vertex whose recall is 0, then insert (0, 0) to vertex_list[0].
    if not found_recall_0:
        vertex_0_0 = {
            'precision': 0,
            'recall': 0
        }
        vertex_list.insert(0, vertex_0_0)
    # If there is no vertex whose recall is 1, then append (1, 0) to last of the vertex_list.
    if not found_recall_1:
        vertex_1_0 = {
            'precision': 0,
            'recall': 1
        }
        vertex_list.append(vertex_1_0)

    # print('Vertices are:')
    # for vertex in vertex_list:
    #     print(vertex)

    # Computing the mAP
    ap = 0
    for i in range(len(vertex_list) - 1):
        ap += (vertex_list[i+1]['recall'] - vertex_list[i]['recall']) * vertex_list[i+1]['precision']

    return ap
