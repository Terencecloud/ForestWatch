import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import psycopg2

# Database connection
DATABASE_URL = 'postgresql://postgres:Smallholder19@localhost/forestwatch'

# Define threshold for deforestation
deforestation_threshold = 0.2

# Define folder path for images to analyze
images_folder_path = 'data/training'

# Define output images folder path
output_images_folder_path = 'output_images'

# Create output images folder if it doesn't exist
if not os.path.exists(output_images_folder_path):
    os.makedirs(output_images_folder_path)

# Load reference image
reference_image_path = 'data/reference image/reference_image.png'
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# Define custom dataset class
class DeforestationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# Define data transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load labeled training data
image_paths = []
labels = []
for filename in os.listdir(images_folder_path):
    image_path = os.path.join(images_folder_path, filename)

    # Extract label from filename assuming it's the first part of the filename
    parts = filename.split('_')
    label = int(parts[0]) if parts[0].isdigit() and parts[0] in ['0', '1'] else 0

    image_paths.append(image_path)
    labels.append(label)

# Create dataset and data loader
dataset = DeforestationDataset(image_paths, labels, transform=data_transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define CNN model
class DeforestationModel(nn.Module):
    def __init__(self):
        super(DeforestationModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(512, 2)  # Adjust for binary classification

    def forward(self, x):
        x = self.resnet(x)
        return x

model = DeforestationModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(4):
    running_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader):.4f}')

# Evaluate model
model.eval()
deforestation_probability = []
labels_list = []
with torch.no_grad():
    for images, labels in data_loader:
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
        deforestation_probability.extend(probabilities.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

# Calculate evaluation metrics
predicted_labels = np.array(deforestation_probability) > 0.5
accuracy = accuracy_score(labels_list, predicted_labels)
precision = precision_score(labels_list, predicted_labels)
recall = recall_score(labels_list, predicted_labels)
conf_mat = confusion_matrix(labels_list, predicted_labels)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("Confusion Matrix:")
print(conf_mat)

# Analyze images using combined model
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

for filename in os.listdir(images_folder_path):
    image_path = os.path.join(images_folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (reference_image.shape[1], reference_image.shape[0]))

    # Calculate change in NDVI
    ndvi_change = reference_image.astype(float) - image_resized.astype(float)
    # Threshold NDVI change to detect deforestation
    deforestation_mask = (ndvi_change > deforestation_threshold) & (reference_image > 0.5)

    # Calculate deforestation percentage
    deforested_pixels = np.sum(deforestation_mask)
    total_pixels = reference_image.size

    # Calculate areas based on total area
    total_area_m2 = 200000  # Total area in square meters
    pixels_per_m2 = total_pixels / total_area_m2
    deforested_area_m2 = deforested_pixels / pixels_per_m2
    remaining_area_m2 = total_area_m2 - deforested_area_m2
    deforestation_percentage = (deforested_area_m2 / total_area_m2) * 100
    remaining_forest_percentage = (remaining_area_m2 / total_area_m2) * 100

    # Calculate NDVI statistics
    ndvi_highest = np.max(ndvi_change)
    ndvi_mean = np.mean(ndvi_change)
    ndvi_lowest = np.min(ndvi_change)

    # Load image for classification
    image_color = cv2.imread(image_path)
    image_color = cv2.resize(image_color, (224, 224))
    image_color = Image.fromarray(image_color)
    image_color = data_transform(image_color)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        image_color = image_color.unsqueeze(0)
        output = model(image_color)
        probability = torch.nn.functional.softmax(output, dim=1)[:, 1].item()

    # Print and store results
    print(f'Image: {filename}')
    print(f'Deforestation Percentage: {deforestation_percentage:.2f}%')
    print(f'Deforested Area: {deforested_area_m2:.2f} m² ({deforested_area_m2/10000:.2f} ha)')
    print(f'Remaining Forest Area: {remaining_area_m2:.2f} m² ({remaining_area_m2/10000:.2f} ha)')
    print('NDVI Statistics:')
    print(f'  Highest: {ndvi_highest:.2f}')
    print(f'  Mean: {ndvi_mean:.2f}')
    print(f'  Lowest: {ndvi_lowest:.2f}')
    print('---')

    # Save results to database
    try:
        cur.execute("""
            INSERT INTO deforestation_results (
                image_name,
                deforestation_percentage,
                deforested_area_m2,
                remaining_area_m2,
                remaining_forest_percentage,
                ndvi_highest,
                ndvi_mean,
                ndvi_lowest
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            filename,
            deforestation_percentage,
            deforested_area_m2,
            remaining_area_m2,
            remaining_forest_percentage,
            ndvi_highest,
            ndvi_mean,
            ndvi_lowest
        ))
        conn.commit()
        print(f"Data inserted successfully for {filename}")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        conn.rollback()

    # Save original image
    original_image_path = os.path.join(output_images_folder_path, f"{filename}_original.png")
    if not os.path.exists(original_image_path):
        original_image = cv2.imread(image_path)
        cv2.imwrite(original_image_path, original_image)
        print(f"Saved original image: {original_image_path}")
    else:
        print(f"Original image already exists: {original_image_path}")

    # Save image after time (resized)
    image_after_time_path = os.path.join(output_images_folder_path, f"{filename}_after_time.png")
    if not os.path.exists(image_after_time_path):
        image_resized_color = cv2.resize(cv2.imread(image_path), (reference_image.shape[1], reference_image.shape[0]))
        cv2.imwrite(image_after_time_path, image_resized_color)
        print(f"Saved image after time: {image_after_time_path}")
    else:
        print(f"Image after time already exists: {image_after_time_path}")

# Save output image (deforestation mask)
output_image_path = os.path.join(output_images_folder_path, f"{filename}_deforestation.png")
if not os.path.exists(output_image_path):
    deforestation_image = (deforestation_mask.astype(int) * 255)
    cv2.imwrite(output_image_path, deforestation_image)
    print(f"Saved output image: {output_image_path}")
else:
    print(f"Output image already exists: {output_image_path}")

conn.commit()
cur.close()
conn.close()

# Visualize results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Reference Image')
plt.imshow(reference_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Image After Time')
plt.imshow(image_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Detected Deforestation')
plt.imshow(deforestation_mask.astype(int) * 255, cmap='hot')
plt.axis('off')
plt.tight_layout()
plt.show()

# Optional: Save visualization as an image
visualization_output_path = "deforestation_visualization.png"
plt.savefig(visualization_output_path)
print(f"Visualization saved to: {visualization_output_path}")

print("Deforestation analysis completed.")