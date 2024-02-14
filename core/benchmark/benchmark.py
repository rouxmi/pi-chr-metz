import logging
import configparser
import os
import time

from core.yolo.yolo import yolo_model

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
    else:
        logger.error("Invalid model specified in config.ini")
        exit()
        
    png_conversion_time_dict = {}
    prediction_time_dict = {}
    gsps_creation_time_dict = {}
    gsps_sending_time_dict = {}
    tmp_files_clearing_time_dict = {}
    stats = {}
    counter = 0
    
    for file in os.listdir("input"):
        try:
            os.rename("input/"+file,"tmp/"+file)
            logger.info(f"File {file} moved to temporary directory")
            
            start_time = time.time()
            image_width, image_height = Dicom_to_png("tmp/"+file, logger, uint=int(config["DEFAULT"]["uint"]))
            logger.info(f"Converted DICOM file {file} to PNG")
            png_conversion_time = time.time() - start_time
            
            start_time = time.time()
            bboxes = model.predict("tmp/tmp.png",conf=float(config["DEFAULT"]["min_confidence"]),imgsz=int(config["DEFAULT"]["image_width"]))
            logger.info(f"Predicted bounding boxes for {file}")
            prediction_time = time.time() - start_time
            
            if len(bboxes) == 0:
                logger.info(f"No bounding boxes found for {file}")
            
            list_rectangle =[]
            for box in bboxes:
                x,y,w,h = box["x"],box["y"],box["w"],box["h"]
                x,y,w,h = (x*2)+((image_width - 2048) / 2),(y*2)+((image_height - 1024)/2),w*2,h*2
                list_rectangle.append([x,y,w,h])
            
            start_time = time.time()
            gsps_dataset_confidence = create_gsps("tmp/"+file,"output/"+file.replace(".dcm","_confidence.dcm"),list_rectangle,[box["conf"] for box in bboxes],logger)
            logger.info(f"Created GSPS with confidence shown for {file}")
            gsps_creation_time = time.time() - start_time
            
            start_time = time.time()
            gsps_dataset_no_confidence = create_gsps("tmp/"+file,"output/"+file.replace(".dcm","_no_confidence.dcm"),list_rectangle,[box["conf"] for box in bboxes],logger,show_confidence=False)
            logger.info(f"Created GSPS without confidence shown for {file}")
            gsps_creation_time += time.time() - start_time
            
            if config["DEFAULT"]["debug_mode"] == "True":
                start_time = time.time()
                show_results_popup(bboxes)
                logger.info(f"Displayed results for {file}")
                popup_display_time = time.time() - start_time
            
            start_time = time.time()
            status = send_dicom_to_pacs(gsps_dataset_confidence,config["DEFAULT"]["pacs_ip"],int(config["DEFAULT"]["pacs_port"]),config["DEFAULT"]["aetitle"],config["DEFAULT"]["pacs_aetitle"],logger)
            logger.info(f"Done sending GSPS with confidence shown for {file}") if status else None
            gsps_sending_time = time.time() - start_time
            
            start_time = time.time()
            status = send_dicom_to_pacs(gsps_dataset_no_confidence,config["DEFAULT"]["pacs_ip"],int(config["DEFAULT"]["pacs_port"]),config["DEFAULT"]["aetitle"],config["DEFAULT"]["pacs_aetitle"],logger)
            logger.info(f"Done sending GSPS without confidence shown for {file}") if status else None
            gsps_sending_time += time.time() - start_time
            
            counter += 1
            stats[file] = {"nb_bboxes":len(bboxes),"size":os.path.getsize("tmp/"+file),"position":counter}
            
            start_time = time.time()
            clear_tmp_files(logger)
            tmp_files_clearing_time = time.time() - start_time
            
            logger.info(f"Processing completed for {file}")
            logger.info(f"Time taken for PNG conversion: {png_conversion_time} seconds")
            logger.info(f"Time taken for prediction: {prediction_time} seconds")
            logger.info(f"Time taken for GSPS creation: {gsps_creation_time} seconds")
            logger.info(f"Time taken for GSPS sending: {gsps_sending_time} seconds")
            logger.info(f"Time taken for clearing temporary files: {tmp_files_clearing_time} seconds")
            
            png_conversion_time_dict[file] = png_conversion_time
            prediction_time_dict[file] = prediction_time
            gsps_creation_time_dict[file] = gsps_creation_time
            gsps_sending_time_dict[file] = gsps_sending_time
            tmp_files_clearing_time_dict[file] = tmp_files_clearing_time
            
            #clear the log file to not influence the next benchmark by removing all the lines 
            with open('log/logfile_'+ str(len(os.listdir("log"))) + '.log', 'w') as f:
                f.write("")
                
            
        except Exception as e:
            clear_tmp_files(logger)
            logger.warning("Failed to process file {}: {}".format(file, str(e)))
    
    logger.info("Benchmark completed")
    # save the data in a csv file
    with open('benchmark.csv', 'w') as f:
        f.write("file,nb_bboxes,size,position,png_conversion_time,prediction_time,gsps_creation_time,gsps_sending_time,tmp_files_clearing_time\n")
        for key in stats.keys():
            f.write(f"{key},{stats[key]['nb_bboxes']},{stats[key]['size']},{stats[key]['position']},{png_conversion_time_dict[key]},{prediction_time_dict[key]},{gsps_creation_time_dict[key]},{gsps_sending_time_dict[key]},{tmp_files_clearing_time_dict[key]}\n")
            
    logger.info("Benchmark results saved in benchmark.csv")