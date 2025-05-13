
import numpy as np
import yaml
import cv2

def get_corners(cx, cy, win_w, win_h, image_w, image_h):
    # Calculate the search window coordinates, ensuring they are within image bounds
    x1 = max(0, int(cx - win_w / 2))
    y1 = max(0, int(cy - win_h / 2))
    x2 = min(image_w, x1 + win_w)
    y2 = min(image_h, y1 + win_h)
    return x1, x2, y1, y2

def to_surface_coordinates(u, v, H):
    """
    Converts pixel coordinates (u, v) to surface coordinates using a homography matrix H.
    
    - Ensures u and v are NumPy arrays.
    - If u or v is a single float or int, converts them into a NumPy array.
    - If the result is a single point, returns x and y as scalars.
    """
    # Ensure u and v are NumPy arrays
    u = np.array([u]) if isinstance(u, (int, float)) else np.asarray(u)
    v = np.array([v]) if isinstance(v, (int, float)) else np.asarray(v)

    # Create homogeneous coordinates
    pixels = np.array([u, v, np.ones_like(u)])  # Shape (3, N)

    # Apply homography transformation
    points = H @ pixels  # Matrix multiplication (3,3) @ (3,N) -> (3,N)

    # Normalize x, y by w
    x, y = points[:2] / points[-1]  # Normalize x and y by the last row (homogeneous coordinate)

    # If x and y are single-element arrays, unpack them to return scalars
    if len(x) == 1 and len(y) == 1:
        return x[0], y[0]

    return x, y


def read_transform_config(filepath):
    """
    Reads calibration data from a YAML file and returns individual variables, including
    values from a nested dictionary.

    Args:
    filepath (str): Path to the YAML file containing calibration data.

    Returns:
    tuple: Returns individual variables extracted from the YAML file.
    """
    # Load the YAML data from the specified file
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    # Use eval to convert homography string to list of lists
    homography_str = data.get('homography', None)
    H = np.array(eval(homography_str))
    return H
    

def draw_box(image, im_canny, corners, color=(0, 255, 0), thickness = 2):

    x1, x2, y1, y2 = corners

    # Draw the box on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)

    # Extract the patch from the grayscale image
    patch = cv2.cvtColor(im_canny, cv2.COLOR_GRAY2RGB)[y1:y2, x1:x2, :]

    # Replace the corresponding region in the RGB image with the patch
    image[y1:y2, x1:x2, :] = patch

    return image

def parse_predictions(predictions, class_ids = [0]):

    """
    Process the model predictions and create a mask image for the specified class IDs.

    Parameters:
    - predictions (list): A list containing prediction results, typically including bounding boxes, masks, and class labels.
    - class_ids (list): List of class IDs to be included in the mask (default is [0] for center line).

    Returns:
    - bool: True if at least one valid mask is found, False otherwise.
    - numpy.ndarray or None: The final mask image resized to the original input size, or None if no masks match the specified class IDs.
    """

    # We only process one image at a time (batch size = 1)
    if len(predictions) < 1:
        return False, None

    p = predictions[0]

    ids = np.array(p.class_ids)  # Class IDs e.g., center line (0), stop sign (1)
    masks = np.array(p.masks)  # Masks for each detected object

    # Create a mask for detections that match our target class IDs
    cls_mask = np.isin(ids, class_ids)

    # If none of the detections match the desired class IDs, exit early
    if not cls_mask.any():
        return False, None

    # Keep only the masks and IDs that match the class of interest
    ids = ids[cls_mask]
    masks = masks[cls_mask]

    # Create an empty output image to store our final mask
    output = np.zeros_like(masks[0], dtype=np.uint8)

    for i, mask in enumerate(masks):
        # We expect only one left and one right lane line (or other relevant objects)
        # If there are multiple detections, we combine them into one mask
        # (Alternatively, we could keep only the detection with the highest confidence)
        output[mask == 1] = ids[i] + 1  # Add +1 to avoid zero (background) value

    return True, output

def get_base(mask, N = 100):
    y, x = np.nonzero(mask)
    xs = x[np.argsort(y, )][-N:]
    ys = y[np.argsort(y)][-N:]

    cx, cy = np.mean([xs, ys], axis = 1)

    return cx, cy

def draw_circle(image, x, y, radius=5, color=(0, 255, 0), thickness=-1):
    center = (int(x), int(y))  # Create center tuple from x and y
    cv2.circle(image, center, radius, color, thickness)
    return image

def detect_bbox_center(predictions, target_id):

    # Check if there are any predictions
    if len(predictions) == 0:
        return False, None, None

    p = predictions[0]  # Get the first prediction

    all_boxes = np.array(p.boxes)  # Access the bounding boxes
    ids = np.array(p.class_ids) # Class IDs for each detected object
    confidences = np.array(p.scores)  # Confidence scores for each detection

    # Check if the target class ID is present in the predictions
    if target_id not in ids:
        return False, None, None

    # Filter the boxes with the target ID
    boxes = all_boxes[ids == target_id]

    # Extract the center and size (xyxy) of the first box
    x1, y1, x2, y2 = boxes[0]

    # Calculate the center X-coordinate
    center_x = (x1 + x2) /2

    return True, float(center_x), float(y2)  # Return the bottom center coordinates
