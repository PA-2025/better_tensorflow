import cv2
import numpy as np


class DataPreProcess:
    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50, 37))
        cv2.imwrite("debug.png", image)
        return image
