import cv2
import numpy as np
from typing import List, Tuple, Union
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_hands_cv(
    image: Union[str, Image.Image],
    visualize: bool = False,
    model_path: str = 'YOUR_MODEL_PATH_HERE/hand_landmarker.task',
    max_num_hands: int = 2
) -> List[List[Tuple[int, int]]]:
    """
    Detect hand landmarks in an image using the MediaPipe Tasks hand detection model.
    
    Args:
        image: Image file path or PIL.Image.Image object.
        visualize: Whether to display the visualization window.
        model_path: Path to the hand recognition model (.task).
        max_num_hands: Maximum number of hands to detect.

    Returns:
        A list of landmark coordinates (pixel coordinates) for each detected hand.
    """
    # 1. Load image as mediapipe.Image
    if isinstance(image, str):
        img_cv = cv2.imread(image)
        if img_cv is None:
            raise ValueError(f"Cannot read image file: {image}")
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    else:
        raise TypeError("image must be a file path string or a PIL.Image object")

    # 2. Create HandLandmarker object
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=max_num_hands
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 3. Detect hand landmarks
    detection_result = detector.detect(image_mp)
    image_height, image_width = img_cv.shape[:2]
    hand_coords = []

    # 4. Extract landmark pixel coordinates
    if detection_result.hand_landmarks:
        for hand in detection_result.hand_landmarks:
            coords = []
            for i, lm in enumerate(hand):
                x_px = int(lm.x * image_width)
                y_px = int(lm.y * image_height)
                coords.append((x_px, y_px))
                if visualize:
                    cv2.putText(img_cv, str(i), (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(img_cv, (x_px, y_px), 3, (255, 0, 0), -1)
            hand_coords.append(coords)

    if visualize:
        cv2.imshow("Hand Detection", img_cv)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return hand_coords

if __name__ == '__main__':
    # Example usage for demonstration purposes only.
    # Requires valid paths to run successfully.
    
    # sample_image_path = "path/to/your/dataset/test_image.png"
    
    # coords = detect_hands_cv(
    #     image=sample_image_path,
    #     visualize=True
    # )
    
    # for i, hand in enumerate(coords):
    #     print(f"Hand {i+1} landmark coordinates:")
    #     for j, point in enumerate(hand):
    #         print(f"  Point {j}: {point}")
    pass