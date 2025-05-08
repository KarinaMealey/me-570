import numpy as np
import yaml
import cv2

CAR_ID_NUM = 0

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

    # Normalize x, y, and z by w
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


def get_car_loc(predictions, class_ids = [0]):
        """
        Process the model predictions and create a mask image.

        Parameters:
        - predictions (list): A list containing prediction results like bounding boxes, masks, and class labels.

        Returns:
        - numpy.ndarray or None: The final mask image resized to the original input size, or None if no masks were found.
        """

        # We only process one image at a time (batch size = 1)
        p = predictions[0]

        bboxs = p.boxes
        ids = bboxs.cls.cpu().numpy()        # Class IDs (e.g., left line, right line)
        confidences = bboxs.conf.cpu().numpy() # Confidence scores (not used here)
        
        # Create a mask for detections that match our target classes (we're only interested in the center line)
        cls_mask = np.isin(ids, class_ids)
        # If none of the detections match the desired class_ids, exit early
        if not cls_mask.any():
            return False, None
         # Keep only the masks and IDs that match our class of interest
        ids = ids[cls_mask]
        confidences = confidences[cls_mask]
        if len(confidences) < 1:
            return False, None
        
        # find highest value confidence
        max_con = confidences[0]
        i = 0     
        j = 0   
        for con_val in confidences:
            if con_val > max_con:
                max_con = con_val
                j=i
            i += 1
        bbox = bboxs[j]
        xyxy = bbox.xyxy.cpu().numpy()
        if len(xyxy) < 4:
            return False, None
        # x value is the midpoint of the bounding box
        x_val = round((xyxy[0] + xyxy[2])/2.0)
        y_val = round(xyxy[3])
        return True, (x_val, y_val)


def parse_onnx_predictions(predictions, class_ids = [0]):
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

def parse_predictions(predictions, class_ids = [0]):
    """
    Process the model predictions and create a mask image.

    Parameters:
    - predictions (list): A list containing prediction results like bounding boxes, masks, and class labels.

    Returns:
    - numpy.ndarray or None: The final mask image resized to the original input size, or None if no masks were found.
    """

    # We only process one image at a time (batch size = 1)
    if len(predictions) < 1:
      return False, None

    p = predictions[0]

    bboxs = p.boxes
    ids = bboxs.cls.cpu().numpy()        # Class IDs e.g., center(0), stop(1)
    confidences = bboxs.conf.cpu().numpy() # Confidence scores (not used here)

    masks = p.masks
    if masks is None:
        return False, None

    # Create a mask for detections that match our target classes (we're only interested in the center line)
    cls_mask = np.isin(ids, class_ids)

    # If none of the detections match the desired class_ids, exit early
    if not cls_mask.any():
        return False, None

    shape = masks.orig_shape
    (height, width) = shape

    # Each detected object has its own mask
    data = masks.data.cpu().numpy()  # Shape: (N, W, H) — N = number of masks

    # Keep only the masks and IDs that match our class of interest
    ids = ids[cls_mask]
    data = data[cls_mask]

    # Create an empty output image to store our final mask
    output = np.zeros(shape=shape, dtype=np.uint8)

    for i, mask in enumerate(data):
        # Resize the mask to match the original image size
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # We expect only one left and one right lane line
        # If there are multiple detections, we combine them into one mask
        # (Another option would be to keep only the detection with the highest confidence)
        output[mask == 1] = ids[i] + 1  # We add +1 because background is 0

    return True, output
def get_base(mask, N = 100):
    y, x = np.nonzero(mask)
    xs = x[np.argsort(y, )][-N:]
    ys = y[np.argsort(y)][-N:]

    cx, cy = np.mean([xs, ys], axis = 1)

    return cx, cy

"""# Apply preprocessing (Canny edge detection)
im_canny = self.preprocess_image(image)

# Find the line center position
success, self.win_x = self.find_center(self.win_x, im_canny)

# Plot and publish result if debug is Trze
if self.debug:
    corners = self.get_corners(self.win_x)
    result = draw_box(image, im_canny, corners)
    im_msg = np_to_image(result)
    self.im_publisher.publish(im_msg)

if success:
    
    # Transform from pixel to world coordinates
    x, y = self.to_surface_coordinates(self.win_x, self.win_y)

    # Extract the timestamp from the incoming image message
    timestamp = msg.header.stamp

    # Publish waypoint as Pose message
    pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
    self.publisher.publish(pose_msg)
    self.get_logger().info(f"{self.win_x=}")

else:
    # self.get_logger().info(f"{self.win_x}")
    self.get_logger().info("Lost track!")"""
"""def preprocess_image(self, image, filter_method = "Gaussian"):
# Convert to grayscale
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image
if filter_method == "Gaussian":
    im_burred = cv2.GaussianBlur(im_gray, self.kernel, 0)
elif filter_method == "Median":
    im_burred = cv2.medianBlur(im_gray, self.k)
else:
    raise ValueError('filter_method must only be "Gaussian" or "Median"')
# Apply Canny edge detection
im_canny = cv2.Canny(im_burred, self.canny_min, self.canny_max) # 0-255

return im_canny

def find_center(self, win_x, im_canny):

corners = self.get_corners(win_x)
x1, x2, y1, y2 = corners

patch = im_canny[y1:y2, x1:x2]

# Find all non-zero (white) pixels in the image
ys, xs = np.nonzero(patch)

if len(xs) == 0:
    print("Lost track!")
    return False, self.win_x

self.win_x = x1 + int(np.mean(xs))

#offset = im_center - win_x
#print(offset)
# return: A tuple (success flag, updated center).
return True, self.win_x"""