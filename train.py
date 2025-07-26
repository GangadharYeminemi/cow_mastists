import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Custom dataset class
class MastitisDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = r"E:\MASTITS\Data"
mastitis_dir = os.path.join(data_dir, 'mastitis')
no_mastitis_dir = os.path.join(data_dir, 'normal_teats')

# Collect image paths and labels
image_paths = []
labels = []

if os.path.exists(mastitis_dir):
    for img_name in os.listdir(mastitis_dir):
        img_path = os.path.join(mastitis_dir, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(img_path)
            labels.append(1)
else:
    raise FileNotFoundError(f"Directory not found: {mastitis_dir}")

if os.path.exists(no_mastitis_dir):
    for img_name in os.listdir(no_mastitis_dir):
        img_path = os.path.join(no_mastitis_dir, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(img_path)
            labels.append(0)
else:
    raise FileNotFoundError(f"Directory not found: {no_mastitis_dir}")

# Split dataset
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Create datasets and dataloaders
train_dataset = MastitisDataset(train_paths, train_labels, transform=transform)
test_dataset = MastitisDataset(test_paths, test_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MViTv2 model for feature extraction and direct prediction
mvit_model = timm.create_model('mvitv2_small', pretrained=True, num_classes=0)  # Feature extraction
mvit_classifier = nn.Linear(768, 1).to(device)  # MViTv2_small outputs 768-dim features
mvit_model = mvit_model.to(device)

# ResNet50 model for feature extraction and direct prediction
resnet_model = timm.create_model('resnet50', pretrained=True, num_classes=0)  # Feature extraction
resnet_classifier = nn.Linear(2048, 1).to(device)  # ResNet50 outputs 2048-dim features
resnet_model = resnet_model.to(device)

# Loss and optimizer for MViTv2 and ResNet50
criterion = nn.BCEWithLogitsLoss()
mvit_optimizer = torch.optim.Adam(list(mvit_model.parameters()) + list(mvit_classifier.parameters()), lr=0.0001)
resnet_optimizer = torch.optim.Adam(list(resnet_model.parameters()) + list(resnet_classifier.parameters()), lr=0.0001)

# Training loop for MViTv2 and ResNet50
num_epochs = 10
for epoch in range(num_epochs):
    mvit_model.train()
    mvit_classifier.train()
    resnet_model.train()
    resnet_classifier.train()
    mvit_loss = 0.0
    resnet_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)

        # Train MViTv2
        mvit_optimizer.zero_grad()
        mvit_features = mvit_model(images)
        mvit_outputs = mvit_classifier(mvit_features).squeeze()
        mvit_loss_batch = criterion(mvit_outputs, labels)
        mvit_loss_batch.backward()
        mvit_optimizer.step()
        mvit_loss += mvit_loss_batch.item()

        # Train ResNet50
        resnet_optimizer.zero_grad()
        resnet_features = resnet_model(images)
        resnet_outputs = resnet_classifier(resnet_features).squeeze()
        resnet_loss_batch = criterion(resnet_outputs, labels)
        resnet_loss_batch.backward()
        resnet_optimizer.step()
        resnet_loss += resnet_loss_batch.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], MViTv2 Loss: {mvit_loss/len(train_loader):.4f}, ResNet50 Loss: {resnet_loss/len(train_loader):.4f}')

# Feature extraction for SVM
mvit_model.eval()
resnet_model.eval()

def extract_features(loader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = model(images).cpu().numpy()
            features.append(feats)
            labels.extend(lbls.numpy())
    return np.vstack(features), np.array(labels)

# Extract features for SVM training
train_mvit_features, train_labels = extract_features(train_loader, mvit_model)
train_resnet_features, _ = extract_features(train_loader, resnet_model)
test_mvit_features, test_labels = extract_features(test_loader, mvit_model)
test_resnet_features, _ = extract_features(test_loader, resnet_model)

# Combine features
train_features = np.hstack((train_mvit_features, train_resnet_features))
test_features = np.hstack((test_mvit_features, test_resnet_features))

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
train_features, train_labels = smote.fit_resample(train_features, train_labels)

# Standardize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Apply PCA
pca = PCA(n_components=50, random_state=42)
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)

# Train SVM
class_weights = {0: len(train_labels) / (2 * np.bincount(train_labels)[0]), 
                 1: len(train_labels) / (2 * np.bincount(train_labels)[1])}
svm = SVC(kernel='rbf', class_weight=class_weights, probability=True, random_state=42)
svm.fit(train_features, train_labels)

# Get predictions for the test set
mvit_predictions = []
resnet_predictions = []
svm_predictions = []
ensemble_predictions = []
actual_labels = test_labels

mvit_classifier.eval()
resnet_classifier.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        # MViTv2 predictions
        mvit_features = mvit_model(images)
        mvit_outputs = mvit_classifier(mvit_features).squeeze()
        mvit_probs = torch.sigmoid(mvit_outputs).cpu().numpy()
        mvit_predictions.extend(mvit_probs)

        # ResNet50 predictions
        resnet_features = resnet_model(images)
        resnet_outputs = resnet_classifier(resnet_features).squeeze()
        resnet_probs = torch.sigmoid(resnet_outputs).cpu().numpy()
        resnet_predictions.extend(resnet_probs)

# SVM predictions
svm_probs = svm.predict_proba(test_features)[:, 1]
svm_predictions = svm_probs

# Ensemble predictions (average of probabilities)
ensemble_predictions = (np.array(mvit_predictions) + np.array(resnet_predictions) + np.array(svm_predictions)) / 3

# Convert probabilities to binary predictions
mvit_binary_predictions = (np.array(mvit_predictions) > 0.5).astype(int)
resnet_binary_predictions = (np.array(resnet_predictions) > 0.5).astype(int)
svm_binary_predictions = (np.array(svm_predictions) > 0.5).astype(int)
ensemble_binary_predictions = (np.array(ensemble_predictions) > 0.5).astype(int)

# Generate classification reports
mvit_report = classification_report(actual_labels, mvit_binary_predictions, target_names=['Normal', 'Mastitis'])
resnet_report = classification_report(actual_labels, resnet_binary_predictions, target_names=['Normal', 'Mastitis'])
svm_report = classification_report(actual_labels, svm_binary_predictions, target_names=['Normal', 'Mastitis'])
ensemble_report = classification_report(actual_labels, ensemble_binary_predictions, target_names=['Normal', 'Mastitis'])

# Calculate accuracies
mvit_accuracy = accuracy_score(actual_labels, mvit_binary_predictions)
resnet_accuracy = accuracy_score(actual_labels, resnet_binary_predictions)
svm_accuracy = accuracy_score(actual_labels, svm_binary_predictions)
ensemble_accuracy = accuracy_score(actual_labels, ensemble_binary_predictions)

# Save classification reports
with open('classification_reports.txt', 'w') as f:
    f.write("Classification Report for MViTv2 Model\n")
    f.write("=" * 80 + "\n\n")
    f.write(mvit_report + "\n\n")
    f.write("Classification Report for ResNet50 Model\n")
    f.write("=" * 80 + "\n\n")
    f.write(resnet_report + "\n\n")
    f.write("Classification Report for SVM Model\n")
    f.write("=" * 80 + "\n\n")
    f.write(svm_report + "\n\n")
    f.write("Classification Report for Ensemble Model (MViTv2 + ResNet50 + SVM with PCA)\n")
    f.write("=" * 80 + "\n\n")
    f.write(ensemble_report)

# Plot accuracy comparison
models = ['MViTv2', 'ResNet50', 'SVM', 'Ensemble']
accuracies = [mvit_accuracy, resnet_accuracy, svm_accuracy, ensemble_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'magenta', 'cyan'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of MViTv2, ResNet50, SVM, and Ensemble Models')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
plt.grid(True, axis='y')
plt.savefig('accuracy_comparison.png')
plt.close()

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(actual_labels, 'b-o', label='Actual', linestyle='solid')
plt.plot(mvit_predictions, 'g--', label='MViTv2 Prediction', marker='^')
plt.plot(resnet_predictions, 'r--', label='ResNet50 Prediction', marker='s')
plt.plot(svm_predictions, 'm--', label='SVM Prediction', marker='d')
plt.plot(ensemble_predictions, 'c--', label='Ensemble Prediction', marker='o')
plt.xlabel('Sample Index')
plt.ylabel('Probability of Mastitis')
plt.title('Mastitis Prediction: Actual vs MViTv2, ResNet50, SVM & Ensemble')
plt.legend()
plt.grid(True)
plt.savefig('prediction_comparison.png')
plt.close()

# Save models
os.makedirs('models', exist_ok=True)
torch.save(mvit_model.state_dict(), 'models/mastitis_mvitv2.pth')
torch.save(mvit_classifier.state_dict(), 'models/mvit_classifier.pth')
torch.save(resnet_model.state_dict(), 'models/resnet50_features.pth')
torch.save(resnet_classifier.state_dict(), 'models/resnet_classifier.pth')
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)