import torch
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from PIL import Image
import os


def get_mul_f1_score(models, model_weights, val_loader, threshold = 0.5):
    for model in models:
        model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():  # Disable gradient computation
        for images, masks in val_loader:
            images = images.to('cuda')
            masks = masks.to('cuda')
            predictions = torch.sum(torch.stack([model(torch.stack([images])) * model_weights[i] for i, model in enumerate(models)]), 2)
            
            mask = masks[0].cpu().numpy().flatten()
            prediction = (predictions[0] > threshold).cpu().numpy().flatten().astype(np.uint8)
            mask = (mask > 0.5).astype(np.uint8)
            y_true.extend(mask)
            y_pred.extend(prediction)
    
    return f1_score(y_true, y_pred)

def get_f1_score(model, val_loader, post_process = None, threshold = 0.5):
    '''
    Compute the F1 score of the model on the validation set.
    Args:
        model: The model to evaluate
        val_loader: The validation loader
        post_process: A function to apply to the model's predictions before thresholding
        threshold: The threshold to apply to the model's predictions
    Returns:
        The F1 score of the model on the validation set
    '''
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad(): 
        for images, masks in val_loader:
            images = images.to('cuda')
            masks = masks.to('cuda')
            predictions = model(images)
            if post_process is not None:
                predictions = post_process(predictions)

            for i in range(images.size(0)):
                mask = (masks[i].cpu().numpy().flatten() > 0.5).astype(np.uint8)
                prediction = (predictions[i] > threshold).cpu().numpy().flatten().astype(np.uint8)
                y_true.extend(mask)
                y_pred.extend(prediction)
                
    return f1_score(y_true, y_pred)

def visualize_predictions(model, val_loader, post_process = None, num_images=5):
    '''
    Visualize the model's predictions on the validation set.
    Args:
        model: The model to visualize
        val_loader: The validation loader
        post_process: A function to apply to the model's predictions before thresholding
        num_images: The number of images to visualize    
    '''
    model.eval()  # Set model to evaluation mode
    image_width = 3 if post_process is None else 4
    fig, axes = plt.subplots(num_images, image_width, figsize=(image_width * 5, num_images * 5))
    
    with torch.no_grad():  # Disable gradient computation
        for i, (images, masks) in enumerate(val_loader):
            images = images.to('cuda')
            masks = masks.to('cuda')
            predictions = model(images)
            if post_process is not None:
                post_predictions = post_process(predictions)
            
            for j in range(min(num_images, images.size(0))):
                image = images[j].cpu().permute(1, 2, 0).numpy()
                mask = masks[j].cpu().permute(1, 2, 0).numpy()
                prediction = (predictions[j]).permute(1, 2, 0).cpu().numpy()
                if post_process is not None:
                    post_prediction = (post_predictions[j]).cpu().numpy().astype(np.uint8)
                
                # Plot original image
                axes[j, 0].imshow(image)
                axes[j, 0].set_title("Original Image")
                axes[j, 0].axis('off')
                
                # Plot ground truth mask
                axes[j, 1].imshow(mask, cmap='gray')
                axes[j, 1].set_title("Ground Truth Mask")
                axes[j, 1].axis('off')
                
                # Plot predicted mask
                axes[j, 2].imshow(prediction, cmap='gray')
                axes[j, 2].set_title("Predicted Mask")
                axes[j, 2].axis('off')
                
                # Plot post-processed mask
                if post_process is not None:  
                    axes[j, 3].imshow(post_prediction, cmap='gray')
                    axes[j, 3].set_title("Post-Processed Mask")
                    axes[j, 3].axis('off')
            
            if i + 1 >= num_images // images.size(0):
                break
            
def get_preprocessed_data(images_path: str, transform, masks_path = None):
    '''
    Function to get preprocessed data from the images and masks path
    Args:
        images_path: path to the images
        masks_path: path to the masks
    Returns:
        images: tensor of images
        masks: tensor of masks (is None if masks_path is None)
    '''
    images = [transform(Image.open(os.path.join(images_path, img)).convert('RGB')) for img in sorted(os.listdir(images_path))]
    if(masks_path == None):
        return images, None
    masks = [transform(Image.open(os.path.join(masks_path, mask)).convert("L")) for mask in sorted(os.listdir(masks_path))]
    return torch.stack(images).to(torch.device('cuda')), torch.stack(masks).to(torch.device('cuda'))