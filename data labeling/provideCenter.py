import cv2
import os
import pandas as pd

# Directory containing images
image_folder = "volume-26"

# Get all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# Function to capture clicks
clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked at: {x}, {y}")
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click on points, press 'n' for next, 'q' to quit", param)
        
# Iterate over images
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    
    clicked_points = []  # Reset points
    cv2.imshow("Click on points, press 'n' for next, 'q' to quit", image)
    cv2.setMouseCallback("Click on points, press 'n' for next, 'q' to quit", click_event, image)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # Move to next image
            break
        elif key == ord('q'):  # Quit the program
            cv2.destroyAllWindows()
            exit()
    
    # Save clicked points
    if clicked_points:
        csv_path = os.path.join(image_folder, image_file.replace(".png", ".csv"))
        pd.DataFrame(clicked_points).to_csv(csv_path, index=False, header=False)
        print(f"Saved points for {image_file} at {csv_path}")

cv2.destroyAllWindows()
