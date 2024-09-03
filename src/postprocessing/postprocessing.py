import torch
import numpy as np
import cv2

def median_blurring(predictions, kernel_size=15):
    predictions_np = predictions.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    
    blurred_predictions = [cv2.medianBlur(pred, kernel_size) for pred in predictions_np]
    return torch.tensor(blurred_predictions, device=predictions.device)

def gaussian_blurring(predictions, kernel_size=21, sigma=5):
    predictions_np = predictions.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    
    blurred_predictions = [cv2.GaussianBlur(pred, (kernel_size, kernel_size), sigma) for pred in predictions_np]
    blurred_predictions_np = np.stack(blurred_predictions, axis=0)
    
    return torch.tensor(blurred_predictions_np, device=predictions.device)

def remove_small_connected_objects(predictions, min_size=10):
    predictions_np = predictions.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    cleaned_predictions = []
    for pred in predictions_np:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=4)
        sizes = stats[1:, -1]
        clean_pred = np.zeros(labels.shape, dtype=np.uint8)
        for i in range(0, num_labels - 1):
            if sizes[i] >= min_size:
                clean_pred[labels == i + 1] = 255
        cleaned_predictions.append(clean_pred)
    return torch.tensor(cleaned_predictions, device=predictions.device)

def small_kernel_erosion(predictions, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    predictions_np = predictions.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    eroded_predictions = [cv2.erode(pred, kernel, iterations=1) for pred in predictions_np]
    return torch.tensor(eroded_predictions, device=predictions.device)