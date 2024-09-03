import os
import numpy as np
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate
from tensorflow.keras.applications import DenseNet121
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random

class DenseUnet:
    def __init__(self, root_path, target_size=(256, 256), convert_to_grayscale=False):
        """
        Initialize the DenseUnet class with data loading and preprocessing.
        
        Args:
            root_path (str): Path to the root directory containing images and masks.
            target_size (tuple): Desired size of the output images.
            convert_to_grayscale (bool): If True, convert images to grayscale.
        """
        self.target_size = target_size
        self.root_path = root_path
        self.convert_to_grayscale = convert_to_grayscale

        self.images = self.load_all_from_path(os.path.join(root_path, 'images'))
        self.masks = self.load_all_from_path(os.path.join(root_path, 'groundtruth'), convert_to_grayscale=True)

        if len(self.masks.shape) == 3:  # Ensure masks have the correct shape
            self.masks = np.expand_dims(self.masks, axis=-1)

        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(
            self.images, self.masks, test_size=0.2, random_state=42)

    def load_all_from_path(self, path, convert_to_grayscale=False):
        """
        Load and resize all images from the given directory.
        
        Args:
            path (str): Path to the directory containing images.
            convert_to_grayscale (bool): If True, convert images to grayscale.
        
        Returns:
            np.ndarray: Array of images.
        """
        images = []
        for f in sorted(glob(os.path.join(path, '*.png'))):
            img = Image.open(f)
            if convert_to_grayscale:
                img = img.convert('L')  # Convert to grayscale
            else:
                img = img.convert('RGB')
            img = img.resize(self.target_size, Image.ANTIALIAS)
            images.append(np.array(img))

        images = np.stack(images).astype(np.float32) / 255.

        if convert_to_grayscale:
            images = np.expand_dims(images, axis=-1)  # Add channel dimension for grayscale

        return images

    def preprocess(self, image, mask):
        """
        Preprocess the images and masks: resize and ensure correct channels.
        
        Args:
            image (np.ndarray): Image to preprocess.
            mask (np.ndarray): Mask to preprocess.
        
        Returns:
            Tuple of preprocessed image and mask.
        """
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=-1)

        image = tf.image.resize(image, (256, 256))
        mask = tf.image.resize(mask, (256, 256))
        return image, mask

    def load_dataset(self, images, masks):
        """
        Create a TensorFlow dataset from images and masks.
        
        Args:
            images (np.ndarray): Array of images.
            masks (np.ndarray): Array of masks.
        
        Returns:
            tf.data.Dataset: Preprocessed dataset.
        """
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        masks = tf.convert_to_tensor(masks, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        dataset = dataset.map(lambda x, y: self.preprocess(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(8).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def unet_model(self, input_size=(256, 256, 3)):
        """
        Build the U-Net model using DenseNet121 as the encoder.
        
        Args:
            input_size (tuple): Size of the input image.
        
        Returns:
            tf.keras.Model: U-Net model.
        """
        inputs = Input(input_size)
        encoder = DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
        skip1 = encoder.get_layer("conv1/relu").output  # 128x128
        skip2 = encoder.get_layer("pool2_relu").output  # 64x64
        skip3 = encoder.get_layer("pool3_relu").output  # 32x32
        skip4 = encoder.get_layer("pool4_relu").output  # 16x16
        bottleneck = encoder.get_layer("relu").output  # 8x8

        up1 = UpSampling2D((2, 2))(bottleneck)
        up1 = Conv2D(512, (3, 3), padding="same", activation="relu")(up1)
        up1 = concatenate([up1, skip4])

        up2 = UpSampling2D((2, 2))(up1)
        up2 = Conv2D(256, (3, 3), padding="same", activation="relu")(up2)
        up2 = concatenate([up2, skip3])

        up3 = UpSampling2D((2, 2))(up2)
        up3 = Conv2D(128, (3, 3), padding="same", activation="relu")(up3)
        up3 = concatenate([up3, skip2])

        up4 = UpSampling2D((2, 2))(up3)
        up4 = Conv2D(64, (3, 3), padding="same", activation="relu")(up4)
        up4 = concatenate([up4, skip1])

        up5 = UpSampling2D((2, 2))(up4)
        up5 = Conv2D(32, (3, 3), padding="same", activation="relu")(up5)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(up5)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, epochs=25):
        """
        Train the U-Net model.
        
        Args:
            epochs (int): Number of epochs to train the model.
        
        Returns:
            tf.keras.Model: Trained U-Net model.
            tf.keras.callbacks.History: History object containing training history.
        """
        train_dataset = self.load_dataset(self.train_images, self.train_masks)
        val_dataset = self.load_dataset(self.val_images, self.val_masks)
        
        model = self.unet_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        
        model.save('/content/drive/MyDrive/Road-Segmentation-Comp/unet_model.h5')
        return model, history

    def plot_metrics(self, history):
        """
        Plot the training and validation metrics.
        
        Args:
            history (tf.keras.callbacks.History): History object containing training history.
        """
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    def predict_and_evaluate(self):
        """
        Evaluate the model and make predictions on the validation dataset.
        
        Returns:
            float: F1 score on the validation dataset.
        """
        val_dataset = self.load_dataset(self.val_images, self.val_masks)
        model = tf.keras.models.load_model('/content/drive/MyDrive/Road-Segmentation-Comp/unet_model.h5')

        y_true, y_pred = [], []

        for images, masks in val_dataset:
            predictions = model.predict(images)
            predictions = (predictions > 0.5).astype(np.uint8)

            y_true.extend((masks.numpy() > 0.5).astype(np.uint8).flatten())
            y_pred.extend(predictions.flatten())

        f1 = f1_score(y_true, y_pred)
        print("F1 Score: ", f1)

        val_images, val_masks = [], []
        for images, masks in val_dataset:
            val_images.extend(images.numpy())
            val_masks.extend(masks.numpy())

        indices = random.sample(range(len(val_images)), 3)
        predicted_masks = model.predict(np.array([val_images[i] for i in indices]))
        predicted_masks = (predicted_masks > 0.5).astype(np.uint8)

        for i, idx in enumerate(indices):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(val_images[idx])
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(val_masks[idx].squeeze(), cmap='gray')
            plt.title('True Mask')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(predicted_masks[i].squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.show()

        return f1

# Example usage:
# denseunet = DenseUnet("/content/drive/MyDrive/Road-Segmentation-Comp/sampled/")
# model, history = denseunet.train(epochs=25)
# denseunet.plot_metrics(history)
# f1 = denseunet.predict_and_evaluate()
