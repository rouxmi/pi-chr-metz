import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import configparser
from PIL import Image

from monai.networks.nets import UNet
import monai.transforms as mt
from monai.networks.layers import Norm, Act

class Unet(pl.LightningModule):
    """
    Implementation of the Unet model for object detection.
    """

    def __init__(self, logger):
        """
        Initializes the Unet model.

        Args:
            logger (logging.Logger): Logger object.
        """
        super().__init__()
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 512, 1024, 1024),
            strides=(2, 2, 2, 2, 2, 2),
            num_res_units=3,
            norm=Norm.INSTANCE_NVFUSER,
            act=Act.SOFTMAX
        )
        self.model.load_state_dict(torch.load("model/Unet.pt"), strict=False)
        logger.info("Loaded Unet model")

    def process_image(self, img_pil, model=None):
        """
        Perform image segmentation prediction using a given model.

        Args:
            img_pil (PIL.Image.Image): The input image in PIL format.
            seg_pil (PIL.Image.Image, optional): The ground truth segmentation mask in PIL format. Defaults to None.
            model (torch.nn.Module, optional): The segmentation model. Defaults to the global model.
            img_id (str, optional): The ID of the image. Defaults to "0".

        Returns:
            None
        """
        global overlay
        
        config = configparser.ConfigParser()
        config.read('config.ini')

        transforms = mt.compose.Compose(
            [
                mt.NormalizeIntensity(
                    nonzero=True,
                    channel_wise=True
                ),
                mt.Resize(
                    spatial_size=(int(config["DEFAULT"]["image_height"]), int(config["DEFAULT"]["image_width"])),
                    mode="bilinear"
                ),
                mt.ToTensor(
                    dtype=torch.float32
                )
            ]
        )

        img = np.asarray(img_pil, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img_preproc = transforms(img)
        pred = model(img_preproc.unsqueeze(0))
        # Create a binary mask from the predicted segmentation
        pred = pred.argmax(dim=1).float()
        pred = pred.squeeze(0)
        pred = pred.squeeze(0)
        pred = pred.cpu().detach().numpy()
        pred = (pred * 255).astype(np.uint8)

        return pred

    def predict(self, image_path, logger=None, conf=0.5, imgsz=800):
        """
        Process the image and find bounding boxes.

        Args:
            image_path (str): Path to the image.

        Returns:
            list: List of bounding boxes.
        """
        logger.info("Processing image with Unet: %s", image_path)

        # Get the image and the ground truth
        img = Image.open(image_path).convert("L")
        prediction_image = self.process_image(img, self.model)
        logger.info("Predicted mask for %s", image_path)
        _, binary_image = cv2.threshold(prediction_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logger.info("Found %d contours", len(contours))
        # Iterate through contours and find bounding boxes
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": 1.0
            })

        return bounding_boxes

