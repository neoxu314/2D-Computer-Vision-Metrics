from BoundingBox import BoundingBox
import

gt_bb1 = BoundingBox(class_id='car', x1=72, y1=133, x2=881, y2=575, bb_type='gt')
gt_bb1 = BoundingBox(class_id='car', x1=60, y1=120, x2=850, y2=500, bb_type='pr')

precision_list = [1, 0.5, 0.6666, 0.5, 0.4, 0.3333, 0.2857, 0.25, 0.2222, 0.3, 0.2727, 0.3333, 0.3846, 0.4285, 0.4,
                  0.375, 0.3529, 0.3333, 0.3157, 0.3, 0.2857, 0.2727, 0.3043, 0.2916]
recall_list = [0.0666, 0.0666, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.2, 0.2, 0.2666, 0.3333, 0.4,
               0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4666, 0.4666]

# plt.plot(recall_list, precision_list)
# plt.show()

acc_precision_recall_dict_list = []
for i in range(len(precision_list)):
    item = {
        'precision': precision_list[i],
        'recall': recall_list[i]
    }
    acc_precision_recall_dict_list.append(item)

# ap = cve.get_ap_by_11_points_interpolation(acc_precision_recall_dict_list)
ode.get_ap_by_all_points_interpolation(acc_precision_recall_dict_list)
# print(ap)