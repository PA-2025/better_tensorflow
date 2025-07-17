from typing import Tuple

import cv2
import numpy as np
import os


class DataPreProcess:
    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        size: Tuple[float, float],
        is_experimental: bool = os.environ.get("EXPERIMENTAL", "False") == "true",
    ) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, size)
        cv2.imwrite("debug.png", image)
        return (
            image
            if not is_experimental
            else DataPreProcess.test_preprocess_image_statistique(image)
        )

    @staticmethod
    def test_preprocess_image_statistique(image: np.ndarray) -> np.ndarray:
        number_full_pixels_white = 0
        number_full_pixels_black = 0
        max_height_pixel_white = 0
        max_height_pixel_black = 0

        pixel_values = image.flatten()
        mean_intensity = np.mean(pixel_values)
        std_intensity = np.std(pixel_values)
        energy = np.sum(np.square(pixel_values))

        mid_pixel_count = np.sum((image >= 100) & (image <= 150))
        white_pixel_positions = np.argwhere(image >= 220)

        if white_pixel_positions.size > 0:
            cog_white_y = int(np.mean(white_pixel_positions[:, 0]))
        else:
            cog_white_y = 0

        ratio_white_black = (
            number_full_pixels_white / number_full_pixels_black
            if number_full_pixels_black > 0
            else 0
        )

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] >= 200:
                    number_full_pixels_white += 1
                    max_height_pixel_white = max(max_height_pixel_white, i)
                elif image[i, j] <= 50:
                    number_full_pixels_black += 1
                    max_height_pixel_black = max(max_height_pixel_black, i)

        image_data = np.zeros((3, 4), dtype=np.uint8)

        image_data[0, 0] = number_full_pixels_white
        image_data[0, 1] = number_full_pixels_black
        image_data[0, 2] = max_height_pixel_white
        image_data[0, 3] = max_height_pixel_black

        image_data[1, 0] = int(mean_intensity)
        image_data[1, 1] = int(std_intensity)
        image_data[1, 2] = int(energy / 10000)
        image_data[1, 3] = int(mid_pixel_count)

        image_data[2, 0] = int(cog_white_y)
        image_data[2, 1] = int(ratio_white_black)
        image_data[2, 2] = int(np.max(image))
        image_data[2, 3] = int(np.min(image))

        return image_data
