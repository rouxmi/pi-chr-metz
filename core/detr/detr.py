import numpy as np
import pytorch_lightning as pl
import torch
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from PIL import Image
import os

from core.usefull.detr import convert_boxes



class Detr(pl.LightningModule):
    """
    Implementation of the DETR (DEtection TRansformer) model for object detection.
    """

    def __init__(self, logger):
        """
        Initializes the DETR model.

        Args:
            logger (logging.Logger): Logger object.
        """
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("model/detr/",
                                                            ignore_mismatched_sizes=True)
        logger.info("Loaded DETR model")

    def predict(self, image_path, conf=0.5, imgsz=800, logger=None):
        """
        Predicts bounding boxes for a given image.

        Args:
            image_path (str): Path to the image.
            conf (float): Minimum confidence for a bounding box to be considered.
            imgsz (int): Size of the image.

        Returns:
            list: List of bounding boxes.
        """
        try:
            feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
            
            image = Image.open(image_path)
            image = image.convert("RGB")  # Convert image to RGB if it's not already
            image = np.array(image)  # Convert image to numpy array
            image = np.expand_dims(image, axis=0)  # Add an extra dimension to the image
            
            encoding = feature_extractor(images=torch.tensor(image), return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze() 
            
            pixel_values = pixel_values.unsqueeze(0) 
            logger.info(f"Preprocessed image {image_path}")
            outputs = self.model(pixel_values=pixel_values, pixel_mask=None)
            
            bboxes = convert_boxes(image.shape[2],image.shape[1], outputs, threshold=conf, keep_highest_scoring_bbox=False)
            logger.info(f"Predicted {len(bboxes)} bounding boxes for {image_path}")
            for box in bboxes:
                logger.info(f"Bounding box: conf={box['conf']}, x={box['x']}, y={box['y']}, w={box['w']}, h={box['h']}")
            return bboxes
        except Exception as e:
            logger.error(f"An error occurred while predicting bounding boxes for {image_path}: {str(e)}")
            return []