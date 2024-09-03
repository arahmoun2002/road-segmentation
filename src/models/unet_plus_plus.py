import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU, Precision, Recall, Fscore

class Unetplus():
    """
    A class to represent the Unet++ model for image segmentation.  Remenber that your images need to have the correct shape
    
    Attributes
    ----------
    device : torch.device
        The device (CPU or GPU) on which the model will be run.
    model : smp.UnetPlusPlus
        The Unet++ model from the segmentation_models_pytorch library.
    """
    
    def __init__(self, device):
        """
        Constructs all the necessary attributes for the Unetplus object.

        Parameters
        ----------
        device : torch.device
            The device (CPU or GPU) on which the model will be run.
        """
        self.device = device
     
        # Initialize the Unet++ model with a ResNet34 encoder pre-trained on ImageNet
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights='imagenet',
            classes=1, 
            activation='sigmoid'  
        ).to(device)
    
    def train(self, train_loader, val_loader, epochs=50, save_path=None):
        """
        Trains the Unet++ model.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training data.
        val_loader : DataLoader
            DataLoader for the validation data.
        epochs : int, optional
            Number of epochs to train the model (default is 50).
        save_path : str, optional
            Path to save the trained model (default is None).
        """
        
        # Define loss function
        loss = DiceLoss()

        # Define metrics
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Precision(threshold=0.5),
            smp.utils.metrics.Recall(threshold=0.5),
        ]

        # Define optimizer
        optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=0.00008),
        ])

    

        # Training and validation epochs
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
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(self.model, './mode_unet++.pth')
                print('Model saved!')

                
                
