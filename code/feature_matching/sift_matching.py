import cv2 as cv
import os

if __name__ == "__main__":
    # Define paths for sequence and annotations
    sequence_path = "sequences/uav0000169_00000_s/"
    annotation = "annotations/uav0000169_00000_s.txt"

    # Read the first image in the sequence
    images = []
    for filename in sorted(os.listdir(sequence_path)):
        img = cv.imread(os.path.join(sequence_path, filename))
        if img is not None:
            images.append(img)
            break  # Only load the first frame for this example

    if not images:
        print("No images found in sequence.")
        exit()

    frame = images[0]

    # Read annotations
    with open(annotation, "r") as f:
        annotations = f.readlines()

    if not annotations:
        print("No annotations found.")
        exit()

    # Extract the first annotation
    try:
        x, y, w, h = map(int, annotations[0].split(","))
    except ValueError:
        print("Error reading annotations. Ensure they are in the correct format.")
        exit()

    # Extract the region of interest (ROI)
    roi = frame[y:y + h, x:x + w]

    # Initialize SIFT and compute keypoints/descriptors
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(roi, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    if not kp1 or not kp2:
        print("No keypoints found.")
        exit()

    # Match keypoints using BFMatcher
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda match: match.distance)

    # Draw circles on matched points in the second image (frame)
    for match in matches:
        pt = kp2[match.trainIdx].pt  # Get the coordinates of the matched point
        cv.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)  # Draw a circle

    # Take average of top 5 matches for x and y coordinates
    x, y = 0, 0
    for match in matches[:5]:
        pt = kp2[match.trainIdx].pt
        x += pt[0]
        y += pt[1]

    x /= 5
    y /= 5

    # Draw a rectangle around the matched points
    cv.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

    # Display the frame with matched points
    cv.imshow("Matched Points in Frame", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
