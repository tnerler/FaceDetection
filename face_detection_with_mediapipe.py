from typing import Tuple, Union
import math
import cv2
import numpy as np 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10 
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)

def _normalized_to_pixel_coordinates(
        normalized_x : float, normalized_y: float, image_width: int,
        image_height: int
) -> Union[None, Tuple[int, int]] : 
    """converts normalized value pair to pixel coordinates"""

    # Check if the float value between 0 and 1.

    def is_valid_normalized_value(value: float) -> bool: 
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
    
    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        
        # TODO: Draw Coordinates even if it's outside of the image bounds
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
        image,
        detection_result
) -> np.ndarray:
    """
    Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image
        detection_result: The list of all 'Detection' entities to be visualize.
    Returns:
        Image with bounding boxes
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections: 
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = start_point[0] + bbox.width, start_point[1] + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, FONT_THICKNESS)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)

            color, thickness, radius = (0, 0, 255), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
        
        category = detection.categories[0]
        category_name = category.category_name
        category_name = "" if category_name is None else category_name

        probability = round(category.score, 2)
        result_text = category_name + "(" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)

        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)

    image_np = mp_image.numpy_view()
    annotated_image = visualize(image_np, detection_result)

    cv2.imshow("Camera Face Detection", annotated_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
