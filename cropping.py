import cv2
import numpy as np

def crop(bounding_box, path):
    output_image_path = []
    img = cv2.imread(path)

    bounding_boxes = np.array(bounding_box)

    for i, bbox in enumerate(bounding_boxes):
        x1, y1, x2, y2 = bbox
        cropped_image = img[y1:y2, x1:x2]
        image_path = f"Bin/cropped_image_{i + 1}.jpg"
        cv2.imwrite(image_path, cropped_image)
        output_image_path.append(image_path)

        return output_image_path, cropped_image
