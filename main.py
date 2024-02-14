import logging
import configparser
import os

from core.yolo.yolo import yolo_model
from core.detr.detr import Detr
from core.Unet.unet import Unet

from core.usefull.script import clear_tmp_files, check_directory, show_results_popup
from core.convertion.convert import Dicom_to_png
from core.dicom.bbox_to_gsps import create_gsps
from core.dicom.push_dicom import send_dicom_to_pacs


logging.basicConfig(filename='log/logfile_'+ str(len(os.listdir("log"))+1) + '.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    
    check_directory("input",logger,create=False)
    check_directory("tmp",logger)
    
    #read config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # add console handler if debug mode is enabled
    if config["DEFAULT"]["debug_mode"] == "True":
        logger.info("Debug mode enabled")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        
    
    if config["DEFAULT"]["model"] == "yolo":
        model = yolo_model(logger)
    elif config["DEFAULT"]["model"] == "detr":
        model = Detr(logger)
    elif config["DEFAULT"]["model"] == "unet":
        model = Unet(logger)
    else:
        logger.error("Invalid model specified in config.ini")
        exit()
    
    for file in os.listdir("input"):
        try:
            os.rename("input/"+file,"tmp/"+file)
            logger.info(f"File {file} moved to temporary directory")
            
            image_width, image_height = Dicom_to_png("tmp/"+file, logger, width=int(config["DEFAULT"]["image_width"]), height=int(config["DEFAULT"]["image_height"]), uint=int(config["DEFAULT"]["uint"]))
            logger.info(f"Converted DICOM file {file} to PNG")
            
            bboxes = model.predict("tmp/tmp.png",conf=float(config["DEFAULT"]["min_confidence"]),imgsz=int(config["DEFAULT"]["image_width"]),logger=logger)
            logger.info(f"Predicted {len(bboxes)} bounding boxes for {file}")
            
            if len(bboxes) == 0:
                logger.info(f"No bounding boxes found for {file}")
            
            list_rectangle =[]
            for box in bboxes:
                x,y,w,h = box["x"],box["y"],box["w"],box["h"]
                ratio_width, ratio_height = 2048/ int(config["DEFAULT"]["image_width"])   , 1024/ int(config["DEFAULT"]["image_height"]) 
                x,y,w,h = (x*ratio_width)+((image_width - 2048) / 2),(y*ratio_height)+((image_height - 1024)/2),w*ratio_width,h*ratio_height
                list_rectangle.append([x,y,w,h])
            
            gsps_dataset_confidence = create_gsps("tmp/"+file,"output/"+file.replace(".dcm","_confidence.dcm"),list_rectangle,[box["conf"] for box in bboxes],logger)
            logger.info(f"Created GSPS with confidence shown for {file}")
            gsps_dataset_no_confidence = create_gsps("tmp/"+file,"output/"+file.replace(".dcm","_no_confidence.dcm"),list_rectangle,[box["conf"] for box in bboxes],logger,show_confidence=False)
            logger.info(f"Created GSPS without confidence shown for {file}")

            if config["DEFAULT"]["debug_mode"] == "True":
                show_results_popup(bboxes)
                logger.info(f"Displayed results for {file}")

            status = send_dicom_to_pacs(gsps_dataset_confidence,config["DEFAULT"]["pacs_ip"],int(config["DEFAULT"]["pacs_port"]),config["DEFAULT"]["aetitle"],config["DEFAULT"]["pacs_aetitle"],logger)
            logger.info(f"Done sending GSPS with confidence shown for {file}") if status else None

            status = send_dicom_to_pacs(gsps_dataset_no_confidence,config["DEFAULT"]["pacs_ip"],int(config["DEFAULT"]["pacs_port"]),config["DEFAULT"]["aetitle"],config["DEFAULT"]["pacs_aetitle"],logger)
            logger.info(f"Done sending GSPS without confidence shown for {file}") if status else None

            clear_tmp_files(logger)
            
        except Exception as e:
            clear_tmp_files(logger)
            logger.warning("Failed to process file {}: {}".format(file, str(e)))
    
    logger.info("Processing completed")
