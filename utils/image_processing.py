import cv2
import random

def draw_boxes(image, results):
    """
    Draw bounding boxes and labels on the image.
    """
    for *box, conf, cls in results.xyxy[0]:
        class_id = int(cls)
        label = f"{results.names[class_id]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, box)

        # Generate a random color for each class
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image
