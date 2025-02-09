import logging
import os
import numpy as np
import torch
import cv2
from PIL import Image
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from models import UNet, NuClick_NN
from config import DefaultConfig
from utils.process import post_processing, gen_instance_map
from utils.misc import get_coords_from_csv, get_clickmap_boundingbox, get_output_filename, readImageAndGetClicks
from utils.guiding_signals import get_patches_and_signals
import csv

# Hardcoded paths
MODEL_NAME = "nuclick"  # Replace with "unet" if needed
PRETRAINED_WEIGHTS_PATH = r"C:\Users\New\Downloads\tp\RL projet\nuclick_torch-master\checkpoints\NuClick_Nuclick_40xAll.pth"  # Update this path
OUTPUT_DIR = "output_directory"  # Directory to save instance maps
MASK_THRESHOLD = DefaultConfig.mask_thresh
SCALE_FACTOR = DefaultConfig.img_scale
GPU_ID = None  # Set GPU ID here (e.g., "0" for GPU 0)

def predict_img(net,
                full_img,
                device,
                points_csv,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    # Get click coordinates from CSV file
    cx, cy = get_coords_from_csv(points_csv)
    imgWidth = full_img.width
    imgHeight = full_img.height
    # Get click map and bounding box
    clickMap, boundingBoxes = get_clickmap_boundingbox(cx, cy, imgHeight, imgWidth)
    # Convert full_img to numpy array (3, imgHeight, imgWidth)
    image = np.asarray(full_img)[:, :, :3]
    image = np.moveaxis(image, 2, 0)
    # Generate patches, inclusion and exclusion maps
    patchs, nucPoints, otherPoints = get_patches_and_signals(image, clickMap, boundingBoxes, cx, cy, imgHeight, imgWidth)
    # Divide patches by 255
    patchs = patchs / 255
    # Concatenate input to model
    input = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
    input = torch.from_numpy(input)
    input = input.to(device=device, dtype=torch.float32)
    # Predict
    with torch.no_grad():
        output = net(input)  # (no.patchs, 1, 128, 128)
        output = torch.sigmoid(output)
        output = torch.squeeze(output, 1)  # (no.patchs, 128, 128)
        preds = output.cpu().numpy()

    masks = post_processing(preds, thresh=out_threshold, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)
    
    # Generate instanceMap
    instanceMap = gen_instance_map(masks, boundingBoxes, imgHeight, imgWidth)
    return instanceMap

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Setting GPUs
if GPU_ID is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    print(f"Using GPU: {GPU_ID}")

# Load the image and get clicks
img, cx, cy, imgPath = readImageAndGetClicks()
csv_path = "temp_points.csv"  # Temporary CSV file to store clicked points
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for x, y in zip(cx, cy):
        writer.writerow([x, y])

# Load the model
if MODEL_NAME.lower() == 'nuclick':
    net = NuClick_NN(n_channels=5, n_classes=1)
elif MODEL_NAME.lower() == 'unet':
    net = UNet(n_channels=5, n_classes=1)
else:
    raise ValueError('Invalid model type. Acceptable networks are UNet or NuClick')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Loading model {MODEL_NAME}')
logging.info(f'Using device {device}')
net.to(device=device)
net.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device))
logging.info('Model loaded!')

# Perform prediction
logging.info(f'\nPredicting image {imgPath} ...')
img_pil = Image.fromarray(img[:, :, ::-1])  # Convert RGB to PIL Image
instanceMap = predict_img(net=net,
                            full_img=img_pil,
                            scale_factor=SCALE_FACTOR,
                            out_threshold=MASK_THRESHOLD,
                            points_csv=csv_path,
                            device=device)

# Save instance map
out_filename = get_output_filename(imgPath, OUTPUT_DIR)
if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
cv2.imwrite(out_filename, instanceMap)
logging.info(f'Instance map saved as {out_filename}')


# Get unique object labels (excluding background 0)
unique_instances = np.unique(instanceMap)
unique_instances = unique_instances[unique_instances > 0]  # Remove background (0)

# Generate **unique** grayscale values for each object in range [70, 255]
num_objects = len(unique_instances)
grayscale_values = np.linspace(70, 255, num_objects, dtype=np.uint8)  # Unique values

# Shuffle to randomize intensity assignment
np.random.shuffle(grayscale_values)

# Create a grayscale mask with the same shape as instance_map
grayscale_mask = np.zeros_like(instanceMap, dtype=np.uint8)

# Assign a unique grayscale intensity to each object
for label, intensity in zip(unique_instances, grayscale_values):
    grayscale_mask[instanceMap == label] = intensity

# Display the grayscale mask
plt.imshow(grayscale_mask, cmap="gray")
plt.title("Unique Grayscale Instance Mask")
plt.axis("off")
plt.show()

# Save the grayscale mask
cv2.imwrite("unique_grayscale_instance_mask.png", grayscale_mask)


# Visualize instance map
logging.info(f'Visualizing results for image {imgPath}, close to continue...')
instanceMap_RGB = label2rgb(instanceMap, image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
plt.figure(), plt.imshow(instanceMap_RGB)
# Save the figure to a file
output_visualization_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(imgPath))[0]}_visualization.png")
plt.savefig(output_visualization_path, bbox_inches='tight', pad_inches=0, dpi=300)
logging.info(f"Visualization saved as {output_visualization_path}")

plt.show()
