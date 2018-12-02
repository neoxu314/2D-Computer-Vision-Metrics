import cv2


def draw_bb_on_img(bb, img):
    if bb._bb_type == 'gt':
        bb_color = (255, 0, 0)
    elif bb._bb_type == 'pr':
        bb_color = (0, 255, 255)

    cv2.rectangle(img, (bb._x1, bb._y1), (bb._x2, bb._y2), bb_color, 2)
    # return img
