import numpy as np
import cv2
from skimage.filters import threshold_otsu

def median(img):
    # Calculate the median value of the input image or array.
    return np.median(img)

def mad(img, med):
    # Calculate the median absolute deviation (MAD) of the input image or array from the given median value.
    # More robust to outliers than the standard deviation.

    return np.median(np.abs(img - med))

def calculate_threshold(img):
    # Calculate the threshold value for the input image or array based on the
    # given formula: T = MED + 3 * 1.4826 * MAD, where MED is the median value and MAD is the median absolute deviation.
    med_value = median(img)
    mad_value = mad(img, med_value)
    threshold = med_value + 3 * 1.4826 * mad_value
    
    # Ensure the threshold value is within the valid range [0, 255]
    threshold = np.clip(threshold, 0, 255)
    
    return int(threshold)


def smallest_adjacent_white_area(images):
    smallest_area = float('inf')  # Initialize with infinity

    for image in images:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image to obtain binary mask of white pixels
        _, binary_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
        
        # Find contours of white regions
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate through each contour and calculate area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < smallest_area:
                smallest_area = area

    return smallest_area



def register_images(image1, image2):
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Match keypoints between the two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate transformation (homography) between the images using RANSAC
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image1 onto image2
    registered_image = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))

    return H, registered_image



def building_change_detection(image1, image2):
    
    gray_image1=  image1
    gray_image2 = image2
    
    # Compute absolute difference between the two images
    abs_diff = cv2.absdiff(np.uint8(gray_image1*255), np.uint8(gray_image2*255))
    
    # Apply Otsu's thresholding to obtain binary mask
    thresh = threshold_otsu(abs_diff)
    binary_mask = (abs_diff > thresh).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the binary mask (optional)
    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Identify contours of the detected changes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours (noise)
    min_contour_area = 100  # Adjust as needed
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Draw bounding boxes around detected changes (buildings)
    result_image = image1.copy()
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return result_image


def extract_buildings(img):
    # draw gray box around image to detect edge buildings
    h,w = img.shape[:2]
    cv2.rectangle(img,(0,0),(w-1,h-1), (50,50,50),1)

    # convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define color ranges
    low_yellow = (0,28,0)
    high_yellow = (27,255,255)

    low_gray = (0,0,0)
    high_gray = (179,255,233)

    # create masks
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow )
    gray_mask = cv2.inRange(hsv, low_gray, high_gray)

    # combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, gray_mask)
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE,kernel)

    # findcontours
    contours, hier = cv2.findContours(combined_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find and draw buildings
    for x in range(len(contours)):
            # if a contour has not contours inside of it, draw the shape filled
            c = hier[0][x][2]
            if c == -1:
                    cv2.drawContours(img,[contours[x]],0,(0,0,255),-1)

    # draw the outline of all contours
    for cnt in contours:
            cv2.drawContours(img,[cnt],0,(0,255,0),2)

    # display result
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    
def compute_iou(image, label):
    intersection = np.logical_and(image, label)
    union = np.logical_or(image, label)
    
    if np.sum(union) == 0:
        return 1.0
    
    iou = np.sum(intersection) / np.sum(union)
    return iou