import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU, Precision, Recall, Fscore


class PSPNet():
    """
    PSPNet (Pyramid Scene Parsing Network) is a semantic segmentation model that uses a pyramid pooling module
    to capture multi-scale contextual information. It is based on the ResNet-152 encoder and uses sigmoid activation
    for binary segmentation.

    Args:
        device (torch.device): The device to run the model on.

    Attributes:
        device (torch.device): The device the model is running on.
        model (torch.nn.Module): The PSPNet model.

    """

    def __init__(self, device):
        """Initialize the PSPNet model.

        Args:
            device (torch.device): The device to use for computation.
        """
        self.device = device
        self.model = smp.PSPNet(
            encoder_name="resnet152",
            encoder_weights="imagenet",
            classes=1,
            activation='sigmoid'
        ).to(device)

    def train(self, train_loader, val_loader, epochs=50, save_path=None):
        """
        Trains the model using the provided train and validation data loaders for the specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for the training set.
            val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
            epochs (int, optional): The number of epochs to train the model. Defaults to 50.
            save_path (str, optional): The file path to save the trained model. Defaults to None.
        """
        loss = DiceLoss()
        metrics = [
            IoU(threshold=0.5),
            Precision(threshold=0.5),
            Recall(threshold=0.5),
            Fscore(threshold=0.5),
        ]
        optimizer = optim.Adam(self.model.parameters(), lr=0.0008, weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-6,
        )
        
        train_epoch = smp.utils.train.TrainEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=self.device,
            verbose=True,
        )
        
        valid_epoch = smp.utils.train.ValidEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            device=self.device,
            verbose=True,
        )
        
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []
        
        for i in range(0, epochs):
            print('\nEpoch: {}'.format(i+1))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)
            
            # Save model if a better val F1 score is obtained
            if save_path and best_iou_score < valid_logs['fscore']:
                best_iou_score = valid_logs['fscore']
                torch.save(self.model, save_path)
                print('Model saved!')
                
                
    def predict(self, images):
        """
        Perform prediction on the given images.

        Args:
            images (torch.Tensor): A tensor containing the input images.

        Returns:
            list: A list of numpy arrays representing the predicted masks for each image.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images.to(self))
            predictions = [pred.cpu().permute(1, 2, 0).numpy() for pred in predictions]
            predictions = (predictions >= 0.5).astype('uint8')
        return predictions