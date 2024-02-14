from ultralytics import YOLO

class yolo_model:
    """
    Class representing a YOLO model for object detection.
    """

    def __init__(self, logger):
        """
        Initializes the YOLO model.

        Args:
            logger: The logger object for logging messages.
        """
        self.model = YOLO('model/yolo.pt')
        logger.info("YOLO model loaded")
        
    def predict(self, path, imgsz=1024, conf=0.5, logger=None):
        """
        Performs object detection on an image.

        Args:
            path: The path to the dicom image file.

        Returns:
            A list of dictionaries representing the predicted bounding boxes.
        """
        try:
            results = self.model.predict(path, imgsz=imgsz, conf=conf, save_conf=True, verbose=False,save_txt=False)
            
            boxes = []
            for i in range(len(results[0].boxes)):
                boxes.append({"conf":float(results[0].boxes.conf[i].tolist()),"x":float(results[0].boxes.xyxy[i][0].tolist()),"y":float(results[0].boxes.xyxy[i][1].tolist()),"w":float(results[0].boxes.xywh[i][2].tolist()),"h":float(results[0].boxes.xywh[i][3].tolist())})
            logger.info(f"Predicted {len(boxes)} bounding boxes for {path}")
            for box in boxes:
                logger.info(f"Bounding box: conf={box['conf']}, x={box['x']}, y={box['y']}, w={box['w']}, h={box['h']}")
            return boxes
        except Exception as e:
            logger.error(f"An error occurred while predicting bounding boxes for {path}: {str(e)}")
            return []
