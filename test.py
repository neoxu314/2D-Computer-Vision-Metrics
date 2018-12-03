import numpy as np
from lib import instance_segmentation_evaluation as ise

print('******* Mask List 1 ********')
Mask_list1 = np.zeros((4, 4, 2), dtype=bool)
Mask_list1[1:3, 2, 0] = True
Mask_list1[0:2, 0:2, 1] = True
# print(Mask_list1[:, :, 0])
# print(Mask_list1[:, :, 1])

Mask_list1 = np.reshape(Mask_list1, (-1, Mask_list1.shape[-1])).astype(np.float32)
# print(Mask_list1)
area1 = np.sum(Mask_list1, axis=0)
# print(area1.shape)

print('******* Mask List 2 ********')
Mask_list2 = np.zeros((4, 4, 2), dtype=bool)
Mask_list2[2, 2, 0] = True
Mask_list2[1:3, 0:2, 1] = True
# print(Mask_list2[:, :, 0])
# print(Mask_list2[:, :, 1])
Mask_list2 = np.reshape(Mask_list2, (-1, Mask_list2.shape[-1])).astype(np.float32)
# print(Mask_list2)
area2 = np.sum(Mask_list2, axis=0)


print('******* Intersection ********')
Intersections = np.dot(Mask_list1.T, Mask_list2)
print(Intersections)

print('******* Union ********')
print(area1)
print(area1.shape)
print(area1[:, None])
print(area1[:, None].shape)

print(area2.shape)
print(area2[np.newaxis, :].shape)

print(area1[:, np.newaxis] + area2[None, :])