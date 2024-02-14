import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def clear_tmp_files(logger):
    """
    Clears all temporary files in the 'tmp' directory.

    This function iterates over all files in the 'tmp' directory and attempts to remove them.
    If a file cannot be removed, a warning message is logged.

    """
    for file in os.listdir("tmp"):
        try:
            os.remove("tmp/"+file)
            logger.info("Temporary file {} cleared".format(file))
        except Exception as e:
            logger.warning("Failed to clear temporary file {}: {}".format(file, str(e)))
            
            
def check_directory(path,logger,create=True):
    """
    Checks if a directory exists.

    Args:
        path (str): The path to the directory.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    if not os.path.exists(path):
        if create:
            try:
                os.mkdir(path)
                logger.info("Created directory {}".format(path))
            except Exception as e:
                logger.warning("Failed to create directory {}: {}".format(path, str(e)))
        else:
            logger.warning("Directory {} does not exist".format(path))
        return False
    return True

def show_results_popup(bboxes):
    """
    Shows a popup window with the results of the object detection.

    Args:
        results (list): A list of dictionaries representing the predicted bounding boxes.

    Returns:
        None
    """
    img = mpimg.imread('tmp/tmp.png')
    imgplot = plt.imshow(img)
    for box in bboxes:
        x1 = box["x"]
        y1 = box["y"]
        x2 = box["x"]+box["w"]
        y2 = box["y"]+box["h"]
        conf = round(box["conf"],2)
        plt.text(x1,y1,str(conf),color="white")
        plt.plot([x1,x2],[y1,y1],color="red")
        plt.plot([x1,x1],[y1,y2],color="red")
        plt.plot([x1,x2],[y2,y2],color="red")
        plt.plot([x2,x2],[y1,y2],color="red")
    #plt.savefig("test.png")
    plt.show()