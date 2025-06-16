from typing import Tuple

import cv2
import numpy as np


class DataPreProcess:
    @staticmethod
    def preprocess_image(image: np.ndarray, size: Tuple[float, float]) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, size)
        cv2.imwrite("debug.png", image)
        return image
