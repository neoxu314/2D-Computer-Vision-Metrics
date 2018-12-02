import warnings


class BoundingBox:
    def __init__(self, class_id, img_path, x1, y1, x2, y2, bb_type="gt", prediction_confidence=None):
        self._class_id = class_id
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._bb_type = bb_type
        self._img_path = img_path

        if self._bb_type == "pr" and prediction_confidence is None:
            raise IOError('The parameter "prediction_confidence" is necessary for predicted bounding boxes')
        elif self._bb_type == "pr" and prediction_confidence is not None:
            self._prediction_confidence = prediction_confidence

    def get_class_id(self):
        return self._class_id

    def get_bb_type(self):
        return self._bb_type

    def get_img_path(self):
        return self._img_path

    def get_prediction_confidence(self):
        if self._bb_type == 'gt':
            warnings.warn("This is a ground-truth bounding box and does not have a prediction confidence value")
            return None
        elif  self._bb_type == 'pr':
            return self._prediction_confidence

    def get_upper_left_corner(self):
        return self._x1, self._y1

    

