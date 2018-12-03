import numpy as np
np.set_printoptions(threshold=np.nan)


def compute_iou_of_masks(Masks1, Masks2):
    """Computes IoU Overlaps between two sets of masks.

    Args:
        Masks1: The shape is [Height, Width, instances], where height, width are the dimension of the corresponding
        image, and instances are the number of the sub-masks (objects). This is a Boolean matrix where True represents
        the area that has masks and False represents the area that has no mask.
        Masks2: Same with Mask1

    Returns:
        IoUs: The shape is [instances1, instances2] where instances1 and instances are the instance number of Masks1 and
        Masks2, respectively. IoUs[i,j] is the IoU between instance i of Mask1 and instance j of Mask2.
    """
    # flatten masks and compute their areas
    Masks1 = np.reshape(Masks1 > .5, (-1, Masks1.shape[-1])).astype(np.float32)
    Masks2 = np.reshape(Masks2 > .5, (-1, Masks2.shape[-1])).astype(np.float32)

    # Compute the area for each sub-mask
    area1 = np.sum(Masks1, axis=0)
    area2 = np.sum(Masks2, axis=0)

    # Intersections and Union
    Intersections = np.dot(Masks1.T, Masks2)
    Unions = area1[:, None] + area2[None, :] - Intersections
    IoUs = Intersections / Unions

    return IoUs