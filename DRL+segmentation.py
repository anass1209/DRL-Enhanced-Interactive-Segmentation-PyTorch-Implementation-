import logging
import os
import numpy as np
from PIL import Image
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2
import glob
import csv
from models import UNet, NuClick_NN
from config import DefaultConfig
from utils.process import post_processing, gen_instance_map
from utils.misc import get_coords_from_csv, get_clickmap_boundingbox, get_output_filename, readImageAndGetClicks
from utils.guiding_signals import get_patches_and_signals
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

#############################################
# Phase 0 : Utility Functions
#############################################

def read_centers_from_csv(csv_path):
    """
    Reads all centers (x,y) from the CSV.
    Each line in the CSV must contain 2 values, e.g. "123,45".
    Returns a list of tuples [(x1, y1), (x2, y2), ...].
    """
    print("   [Phase 0.2] Reading centers from CSV:", csv_path)
    centers = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                x = float(row[0])
                y = float(row[1])
                centers.append((int(x), int(y)))
            except Exception as e:
                print("     [Warning] Incorrect CSV line:", row, e)
    if len(centers) == 0:
        raise ValueError("No valid center found in the CSV")
    return centers

def save_center_to_csv(csv_path, center):
    """
    Saves a center (x,y) into a CSV file on a single line.
    """
    print("   [Phase 0.3] Saving center to CSV:", csv_path)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(center)

def get_random_start_point(roi_center, image_shape, mask, offset_range=15, max_attempts=100):
    """
    Generates a random starting point near the ROI center.
    The generated point is roi_center plus a random offset in [-offset_range, offset_range].
    It checks that the pixel value in the mask at that point equals the pixel value at the center.
    If no valid point is found after max_attempts, the ROI center is returned.
    """
    cx, cy = roi_center
    height, width = image_shape
    center_intensity = mask[cy, cx]
    
    for attempt in range(max_attempts):
        offset_x = random.randint(-offset_range, offset_range)
        offset_y = random.randint(-offset_range, offset_range)
        init_x = max(0, min(width - 1, cx + offset_x))
        init_y = max(0, min(height - 1, cy + offset_y))
        if mask[init_y, init_x] == center_intensity:
            print(f"      [Phase 0.4] Valid start point found after {attempt+1} attempt(s): ({init_x}, {init_y})")
            return (init_x, init_y)
    print("      [Phase 0.4] No valid start point found, using ROI center as start point")
    return roi_center

#############################################
# Phase 1 : Agent Environment (BiomedicalEnv)
#############################################
class BiomedicalEnv:
    def __init__(self, image, roi_center, init_point, max_steps=50, patch_size=128, history_length=3):
        self.image = image
        self.roi_center = roi_center
        self.patch_size = patch_size
        self.max_steps = max_steps
        self.history_length = history_length
        self.height, self.width = self.image.shape
        self.x, self.y = init_point
        self.init_point = init_point
        self.current_step = 0
        self.done = False
        self.state_history = deque(maxlen=self.history_length)
        self.proximity_threshold = 3

    def reset(self, init_point=None):
        self.current_step = 0
        self.done = False
        if init_point:
            self.x, self.y = init_point
            self.init_point = init_point
        self.state_history.clear()
        state = self.get_state()
        for _ in range(self.history_length):
            self.state_history.append(state)
        return self.get_stacked_state()

    def get_state(self):
        half_size = self.patch_size // 2
        x_min = max(self.x - half_size, 0)
        x_max = min(self.x + half_size, self.width)
        y_min = max(self.y - half_size, 0)
        y_max = min(self.y + half_size, self.height)
        patch = self.image[y_min:y_max, x_min:x_max]
        padded_patch = np.zeros((self.patch_size, self.patch_size))
        padded_patch[:(y_max - y_min), :(x_max - x_min)] = patch
        padded_patch = padded_patch[np.newaxis, :, :]
        return torch.tensor(padded_patch, dtype=torch.float32) / 255.0  

    def get_stacked_state(self):
        return torch.cat(list(self.state_history), dim=0)

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def step(self, action):
        step_size = 1
        if action == 0:
            self.y = max(0, self.y - step_size)
        elif action == 1:
            self.y = min(self.height - 1, self.y + step_size)
        elif action == 2:
            self.x = max(0, self.x - step_size)
        elif action == 3:
            self.x = min(self.width - 1, self.x + step_size)
        self.current_step += 1
        ed1 = self.euclidean_distance((self.x, self.y), self.init_point)
        ed2 = self.euclidean_distance((self.x, self.y), self.roi_center)
        reward = ed1 - ed2
        if self.current_step >= self.max_steps:
            self.done = True
        self.state_history.append(self.get_state())
        return self.get_stacked_state(), reward, self.done

#############################################
# Phase 1b : Test Environment (TestEnv)
#############################################
class TestEnv(BiomedicalEnv):
    """
    A testing version of the environment where the ROI center is unknown.
    We do not use reward or proximity-based termination. Instead, we run for a fixed
    number of steps and take the final position as the predicted ROI center.
    """
    def __init__(self, image, init_point, max_steps=50, patch_size=128, history_length=3):
        dummy_roi = (-1, -1)
        super().__init__(image, dummy_roi, init_point, max_steps, patch_size, history_length)
    
    def step(self, action):
        step_size = 1
        if action == 0:
            self.y = max(0, self.y - step_size)
        elif action == 1:
            self.y = min(self.height - 1, self.y + step_size)
        elif action == 2:
            self.x = max(0, self.x - step_size)
        elif action == 3:
            self.x = min(self.width - 1, self.x + step_size)
        self.current_step += 1
        self.state_history.append(self.get_state())
        done = (self.current_step >= self.max_steps)
        reward = 0.0
        return self.get_stacked_state(), reward, done

#############################################
# Phase 2 : Neural Network (QNetwork)
#############################################
class QNetwork(nn.Module):
    def __init__(self, in_channels=3, num_actions=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

#############################################
# Phase 3 : DDQN Agent
#############################################
class DDQNAgent:
    def __init__(self, in_channels=3, num_actions=4, gamma=0.99, lr=0.0005, buffer_capacity=300000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(in_channels, num_actions).to(self.device)
        self.target_net = QNetwork(in_channels, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0).to(self.device))
                return q_values.argmax(dim=1).item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0] * (1 - dones)
        target = rewards + self.gamma * next_q_values

        loss = F.mse_loss(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#############################################
# Phase 4 : Main Script
#############################################
if __name__ == "__main__":
    print("=== Start of Code Execution ===")

    ##############################
    # Training Phase
    ##############################
    # Directory paths for training (using relative paths)
    image_dir = "training_images"
    result_dir = os.path.join(image_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    mask_dir = os.path.join(image_dir, "masks")
    model_save_path = os.path.join(result_dir, "ddqn_agent_model.pth")

    # List of training images (assumed to be .png)
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
    print("Phase 4.1 - Found training images:", all_images)

    # Training parameters
    num_episodes = 40     # Number of episodes per image
    max_steps = 25

    # Create a DDQN agent
    agent = DDQNAgent()

    # If a saved model exists, load it and skip training.
    if os.path.exists(model_save_path):
        print(f"Model found at {model_save_path}. Loading model and skipping training.")
        agent.policy_net.load_state_dict(torch.load(model_save_path, map_location=agent.device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    else:
        print("No saved model found. Starting training...")
        # Loop over training images (no epoch loop)
        for image_file in all_images:
            print("\n--------------------------------------------")
            print(f"Phase 4.3 - Processing training image: {image_file}")
            image_path = os.path.join(image_dir, image_file)
            
            # File paths based on the image name
            base_name = os.path.splitext(image_file)[0]
            roi_csv_path = os.path.join(image_dir, base_name + ".csv")
            predicted_csv_path = os.path.join(image_dir, base_name + "_predicted.csv")
            mask_path = os.path.join(mask_dir, base_name + "_grayscale_mask.png")
            
            # Retrieve the ROI center from CSV; throw error if not available.
            if os.path.exists(roi_csv_path):
                try:
                    centers = read_centers_from_csv(roi_csv_path)
                    print(f"Phase 4.3.1 - Centers from CSV: {centers}")
                except Exception as e:
                    print(f"Phase 4.3.1 - [ERROR] {e}. Skipping image {image_file}.")
                    continue
            else:
                print(f"Phase 4.3.1 - [ERROR] No CSV file found for ROI center for {image_file}. Skipping this image.")
                continue

            # Load the grayscale training image
            print("Phase 4.3.3 - Loading grayscale training image")
            pil_image = Image.open(image_path).convert("L")
            image_array = np.array(pil_image).astype(np.float32)
            height, width = image_array.shape

            # Read the mask from the subdirectory "masks"; do not compute it manually.
            if os.path.exists(mask_path):
                print("Phase 4.3.3.1 - Loading mask:", mask_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (height, width):
                    print("      [ERROR] The loaded mask is invalid or has incorrect dimensions. Skipping this image.")
                    continue
            else:
                print(f"      [ERROR] No mask file found for {image_file} in '{mask_dir}'. Skipping this image.")
                continue

            # For each episode, re-select a random ROI center and generate a new random start point
            for episode in range(num_episodes):
                current_roi = random.choice(centers)
                current_init = get_random_start_point(current_roi, image_array.shape, mask, offset_range=15)
                print(f"Phase 4.3.4 - Episode {episode+1}: Selected ROI center: {current_roi} | New start point: {current_init}")

                env = BiomedicalEnv(image_array, current_roi, current_init, max_steps=max_steps)
                state = env.reset(current_init)
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    agent.replay_buffer.append((state, action, reward, next_state, done))
                    agent.update()
                    state = next_state

                predicted_center = (env.x, env.y)
                print(f"Phase 4.3.6 - Episode {episode+1}: Predicted center by agent: {predicted_center}")
                save_center_to_csv(predicted_csv_path, predicted_center)

                print("Phase 4.3.8 - Saving training result figure")
                plt.figure(figsize=(6, 6))
                plt.imshow(image_array / 255.0, cmap='gray')
                plt.scatter([current_roi[0]], [current_roi[1]], c='green', s=40, label='ROI Center (CSV/Detected)')
                plt.scatter([current_init[0]], [current_init[1]], c='blue', s=40, label='Start Point')
                plt.scatter([predicted_center[0]], [predicted_center[1]], c='red', s=40, label='Predicted Center')
                plt.title(f"Result for {image_file} (Episode {episode+1})")
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=5)
                result_image_path = os.path.join(result_dir, f"{base_name}_result_episode{episode+1}.png")
                plt.savefig(result_image_path, bbox_inches='tight')
                plt.close()
                print(f"Phase 4.3.9 - Figure saved in {result_image_path}")

        torch.save(agent.policy_net.state_dict(), model_save_path)
        print(f"Phase 4.4 - Training complete. Model saved in {model_save_path}")

    ##############################
    # Testing Phase
    ##############################
    print("\n=== Starting Testing Phase ===")
    test_dir = "test_images"  # Folder with test images
    test_result_dir = "DRL_test_results"
    os.makedirs(test_result_dir, exist_ok=True)
    centers_dir = os.path.join(test_dir, "centers")  # Subdirectory containing true centers CSV files
    
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(".png")]
    print("Testing Phase - Found test images:", test_images)
    
    for test_image_file in test_images:
        print("\n--------------------------------------------")
        print(f"Testing Phase - Processing test image: {test_image_file}")
        test_image_path = os.path.join(test_dir, test_image_file)
        base_name = os.path.splitext(test_image_file)[0]
        test_csv_path = os.path.join(test_dir, base_name + ".csv")
        
        # Load the test image in grayscale
        pil_image = Image.open(test_image_path).convert("L")
        image_array = np.array(pil_image).astype(np.float32)
        height, width = image_array.shape
        
        # Read the initial click (starting point) from the CSV file
        if os.path.exists(test_csv_path):
            try:
                with open(test_csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    row = next(reader)
                    init_x = int(float(row[0]))
                    init_y = int(float(row[1]))
                    test_init_point = (init_x, init_y)
                    print(f"Testing Phase - Initial click point from CSV: {test_init_point}")
            except Exception as e:
                print(f"Testing Phase - [ERROR] Could not read initial click from {test_csv_path}: {e}")
                continue
        else:
            print(f"Testing Phase - [ERROR] No CSV file found for {test_image_file}. Skipping this image.")
            continue
        
        # Create a TestEnv which ignores ROI information.
        test_env = TestEnv(image_array, test_init_point, max_steps=max_steps)
        state = test_env.reset(test_init_point)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = test_env.step(action)
            state = next_state

        predicted_center = (test_env.x, test_env.y)
        print(f"Testing Phase - Predicted center: {predicted_center}")
        
        # Save the adjusted (predicted) center to a CSV file in the test directory.
        adjusted_csv_path = os.path.join(test_dir, base_name + "_adjusted.csv")
        save_center_to_csv(adjusted_csv_path, predicted_center)
        
        # Read the true center from the centers subdirectory.
        true_csv_path = os.path.join(centers_dir, base_name + ".csv")
        if os.path.exists(true_csv_path):
            try:
                true_center = read_centers_from_csv(true_csv_path)[0]
                print(f"Testing Phase - True center: {true_center}")
                dist_init_true = test_env.euclidean_distance(test_init_point, true_center)
                dist_pred_true = test_env.euclidean_distance(predicted_center, true_center)
                print(f"Distance between initial click and true center: {dist_init_true:.2f}")
                print(f"Distance between adjusted click and true center: {dist_pred_true:.2f}")
                
                # Create a figure to compare the distances
                plt.figure(figsize=(4, 4))
                labels = ["Initial vs True", "Adjusted vs True"]
                values = [dist_init_true, dist_pred_true]
                plt.bar(labels, values, color=['blue', 'red'])
                plt.ylabel("Euclidean Distance")
                plt.title(f"Distance Comparison for {test_image_file}")
                distance_result_path = os.path.join(test_result_dir, f"{base_name}_distance_comparison.png")
                plt.savefig(distance_result_path, bbox_inches='tight')
                plt.close()
                print(f"Testing Phase - Distance comparison figure saved in {distance_result_path}")
            except Exception as e:
                print(f"Testing Phase - [ERROR] Could not read true center from {true_csv_path}: {e}")
        else:
            print(f"Testing Phase - [ERROR] No true center CSV found in {centers_dir} for {test_image_file}.")
        
        # Plot and save the test result figure
        plt.figure(figsize=(6, 6))
        plt.imshow(image_array / 255.0, cmap='gray')
        plt.scatter([test_init_point[0]], [test_init_point[1]], c='blue', s=40, label='Initial Click')
        plt.scatter([predicted_center[0]], [predicted_center[1]], c='red', s=40, label='Predicted Center')
        plt.title(f"Test Result for {test_image_file}")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=5)
        test_result_path = os.path.join(test_result_dir, f"{base_name}_result.png")
        plt.savefig(test_result_path, bbox_inches='tight')
        plt.close()
        print(f"Testing Phase - Figure saved in {test_result_path}")
        
    print("=== End of DRL Code Execution ===")


# Hardcoded paths and parameters
MODEL_NAME = "nuclick"  # Replace with "unet" if needed
PRETRAINED_WEIGHTS_PATH = r"checkpoints\NuClick_Nuclick_40xAll.pth"  # Update this path
TRAINING_IMAGES_DIR = "test_images"  # Directory containing images and CSV files
OUTPUT_DIR = "segmentation_test_results"  # Directory to save instance maps and figures
MASK_THRESHOLD = DefaultConfig.mask_thresh
SCALE_FACTOR = DefaultConfig.img_scale
GPU_ID = None  # Set GPU ID here (e.g., "0" for GPU 0)
GT_CIRCLE_RADIUS = 5  # Radius (in pixels) for drawing ground truth centers

def predict_img(net,
                full_img,
                device,
                points_csv,
                scale_factor=1,
                out_threshold=0.5):
    """
    Given an input image and a CSV file with coordinates,
    generate a predicted instance map.
    """
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
    # Normalize patches
    patchs = patchs / 255
    # Concatenate inputs for the model
    inp = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
    inp = torch.from_numpy(inp)
    inp = inp.to(device=device, dtype=torch.float32)
    # Predict
    with torch.no_grad():
        output = net(inp)  # (num_patches, 1, 128, 128)
        output = torch.sigmoid(output)
        output = torch.squeeze(output, 1)  # (num_patches, 128, 128)
        preds = output.cpu().numpy()
    masks = post_processing(preds, thresh=out_threshold, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)
    
    # Generate instance map
    instanceMap = gen_instance_map(masks, boundingBoxes, imgHeight, imgWidth)
    return instanceMap

def dice_score(gt_mask, pred_mask):
    """
    Compute Dice coefficient between two binary masks.
    """
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)
    intersection = np.sum(gt_bin * pred_bin)
    dice = (2. * intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-7)
    return dice

def generate_ground_truth_mask(full_img, centers_csv, radius=GT_CIRCLE_RADIUS):
    """
    Generate a binary mask from a centers CSV. The CSV is assumed to contain
    coordinates (x,y). For each center, a filled circle of a given radius is drawn.
    """
    width, height = full_img.width, full_img.height
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    cx, cy = get_coords_from_csv(centers_csv)
    for x, y in zip(cx, cy):
        cv2.circle(gt_mask, (int(x), int(y)), radius, 1, -1)
    return gt_mask

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set GPU if specified
if GPU_ID is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    print(f"Using GPU: {GPU_ID}")

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

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Containers for Dice scores
dice_scores_original = []
dice_scores_adjusted = []
image_names_for_dice = []  # Used only for counting images

# Process all images in the training_images directory
image_files = glob.glob(os.path.join(TRAINING_IMAGES_DIR, "*.png")) + glob.glob(os.path.join(TRAINING_IMAGES_DIR, "*.jpg"))
for img_file in image_files:
    # Get the base name of the image (without extension)
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    
    # Load the image using cv2 and convert to PIL (BGR -> RGB)
    img = cv2.imread(img_file)
    if img is None:
        logging.warning(f"Could not load image {img_file}. Skipping...")
        continue
    img_pil = Image.fromarray(img[:, :, ::-1])
    
    # Dictionary to store predictions for each variant
    predictions = {}
    
    for variant, suffix in zip(['original', 'adjusted'], ["", "_adjusted"]):
        csv_file = os.path.join(TRAINING_IMAGES_DIR, f"{base_name}{suffix}.csv")
        if not os.path.exists(csv_file):
            logging.warning(f"CSV file {csv_file} not found for image {img_file} ({variant} pass). Skipping this variant...")
            continue

        logging.info(f'\nPredicting image {img_file} using {variant} CSV: {csv_file} ...')
        instanceMap = predict_img(net=net,
                                  full_img=img_pil,
                                  scale_factor=SCALE_FACTOR,
                                  out_threshold=MASK_THRESHOLD,
                                  points_csv=csv_file,
                                  device=device)
        
        # Save instance map if needed (currently commented out)
        # out_filename = get_output_filename(img_file, OUTPUT_DIR, suffix=suffix)
        # cv2.imwrite(out_filename, instanceMap)
        # logging.info(f'Instance map saved as {out_filename}')

        # Generate unique grayscale mask
        unique_instances = np.unique(instanceMap)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background (0)
        num_objects = len(unique_instances)
        grayscale_values = np.linspace(70, 255, num_objects, dtype=np.uint8)
        np.random.shuffle(grayscale_values)
        grayscale_mask = np.zeros_like(instanceMap, dtype=np.uint8)
        for label, intensity in zip(unique_instances, grayscale_values):
            grayscale_mask[instanceMap == label] = intensity

        grayscale_out_filename = os.path.join(OUTPUT_DIR, f"{base_name}{suffix}_grayscale_mask.png")
        cv2.imwrite(grayscale_out_filename, grayscale_mask)
        logging.info(f'Grayscale mask saved as {grayscale_out_filename}')

        # Visualize instance map (overlay on original image)
        instanceMap_RGB = label2rgb(instanceMap, image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
        visualization_out_filename = os.path.join(OUTPUT_DIR, f"{base_name}{suffix}_visualization.png")
        plt.imsave(visualization_out_filename, instanceMap_RGB)
        logging.info(f"Visualization saved as {visualization_out_filename}")

        # Save binary mask for Dice computation
        predictions[variant] = (instanceMap > 0).astype(np.uint8)
    
    # Process ground truth from the centers subdirectory
    gt_csv = os.path.join(TRAINING_IMAGES_DIR, "centers", f"{base_name}.csv")
    if not os.path.exists(gt_csv):
        logging.warning(f"Ground truth CSV {gt_csv} not found for image {img_file}. Skipping Dice computation for this image.")
        continue

    gt_mask = generate_ground_truth_mask(img_pil, gt_csv, radius=GT_CIRCLE_RADIUS)
    
    # Compute Dice scores if prediction(s) exist
    dice_original = None
    dice_adjusted = None
    if 'original' in predictions:
        dice_original = dice_score(gt_mask, predictions['original'])
        dice_scores_original.append(dice_original)
    if 'adjusted' in predictions:
        dice_adjusted = dice_score(gt_mask, predictions['adjusted'])
        dice_scores_adjusted.append(dice_adjusted)
    
    # Save a numeric label for this image (e.g., image number)
    image_names_for_dice.append(base_name)
    logging.info(f"Dice scores for {base_name} -- Original: {dice_original}, Adjusted: {dice_adjusted}")

# After processing all images, create a summary figure for Dice scores.
plt.figure(figsize=(10, 6))
indices = np.arange(len(image_names_for_dice))

# Create bar plots for original and adjusted dice scores if available.
width = 0.35  # width of the bars
if dice_scores_original and dice_scores_adjusted:
    plt.bar(indices - width/2, dice_scores_original, width, label='Original')
    plt.bar(indices + width/2, dice_scores_adjusted, width, label='Adjusted')
elif dice_scores_original:
    plt.bar(indices, dice_scores_original, width, label='Original')
elif dice_scores_adjusted:
    plt.bar(indices, dice_scores_adjusted, width, label='Adjusted')

plt.xlabel('Image Number')
plt.ylabel('Dice Score')
plt.title('Dice Score Comparison: Original vs Adjusted Predictions')
# Use numbers (1, 2, 3, ...) as x-axis tick labels.
numeric_labels = [str(i + 1) for i in indices]
plt.xticks(indices, numeric_labels)
plt.legend()
plt.tight_layout()

summary_fig_filename = os.path.join(OUTPUT_DIR, "dice_comparison.png")
plt.savefig(summary_fig_filename)
plt.close()
logging.info(f"Dice comparison figure saved as {summary_fig_filename}")
logging.info("Processing completed for all images.")
print("=== End of Segmentation Code Execution ===")

