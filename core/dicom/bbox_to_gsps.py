import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from pydicom._storage_sopclass_uids import GrayscaleSoftcopyPresentationStateStorage
import datetime


def create_gsps(dicom_file_path, gsps_file_path, list_rectangle_coordinates, list_indic, logger, show_confidence=True):
    """
    Create a Grayscale Softcopy Presentation State (GSPS) from a DICOM file.

    Args:
        dicom_file_path (str): The path to the original DICOM file.
        gsps_file_path (str): The path to save the GSPS file.
        list_rectangle_coordinates (list): A list of rectangle coordinates (x, y, width, height).
        list_indic (list): The list of confidence for each rectangle.
        logger: The logger object for logging any errors.
        precision (bool): If True, the confidence is displayed on the GSPS

    Raises:
        Exception: If an error occurs while creating the GSPS.

    Returns:
        gsps_dataset: The GSPS dataset.
    """
    
    try:
        # Lecture du fichier dicom d'origine
        dicom_dataset = pydicom.dcmread(dicom_file_path)

        # Creation du dataset du GSPS
        gsps_dataset = Dataset()
        gsps_dataset.is_little_endian = True 
        gsps_dataset.is_implicit_VR = True

        # Patient
        gsps_dataset.PatientName = dicom_dataset.PatientName
        gsps_dataset.PatientID = dicom_dataset.PatientID
        gsps_dataset.PatientAge = dicom_dataset.PatientAge
        gsps_dataset.PatientSex = dicom_dataset.PatientSex

        # General Study
        gsps_dataset.StudyDate = dicom_dataset.StudyDate
        gsps_dataset.StudyTime = dicom_dataset.StudyTime
        gsps_dataset.AccessionNumber = dicom_dataset.AccessionNumber
        gsps_dataset.StudyDescription = dicom_dataset.StudyDescription
        gsps_dataset.StudyInstanceUID = dicom_dataset.StudyInstanceUID
        gsps_dataset.StudyID = dicom_dataset.StudyID

        # General Series
        today = pydicom.valuerep.DA(datetime.date.today())
        now = pydicom.valuerep.TM(datetime.datetime.now().time())
        gsps_dataset.SeriesDate = today
        gsps_dataset.SeriesTime = now
        gsps_dataset.Modality = "PR"
        gsps_dataset.SeriesInstanceUID = generate_uid()
        gsps_dataset.SeriesNumber = dicom_dataset.SeriesNumber + 1  # Increment the series number

        # Presentation State Identification 
        gsps_dataset.InstanceNumber = dicom_dataset.InstanceNumber + 1  # Increment the instance number
        gsps_dataset.ContentLabel = f"LESION" # {indic}"
        gsps_dataset.PresentationCreationDate = today
        gsps_dataset.PresentationCreationTime = now
        gsps_dataset.ContentCreatorName = 'OPTIMOTO'

        # Presentation State Relationship
        gsps_dataset.ReferencedSeriesSequence = [Dataset()]
        gsps_dataset.ReferencedSeriesSequence[0].ReferencedImageSequence = [Dataset()]
        gsps_dataset.ReferencedSeriesSequence[0].SeriesInstanceUID = dicom_dataset.SeriesInstanceUID
        gsps_dataset.ReferencedSeriesSequence[0].ReferencedImageSequence[0].ReferencedSOPClassUID = dicom_dataset.SOPClassUID
        gsps_dataset.ReferencedSeriesSequence[0].ReferencedImageSequence[0].ReferencedSOPInstanceUID = dicom_dataset.SOPInstanceUID

        # Displayed Area
        gsps_dataset.DisplayedAreaSelectionSequence = [Dataset()]
        gsps_dataset.DisplayedAreaSelectionSequence[0].DisplayedAreaTopLeftHandCorner = [0, 0]
        gsps_dataset.DisplayedAreaSelectionSequence[0].DisplayedAreaBottomRightHandCorner = [dicom_dataset.Columns, dicom_dataset.Rows]
        gsps_dataset.DisplayedAreaSelectionSequence[0].PresentationSizeMode = 'SCALE TO FIT'
        gsps_dataset.DisplayedAreaSelectionSequence[0].PresentationPixelAspectRatio = [1, 1]


        # Graphic Annotations - 1 per rectangle
        nb_rectangles = len(list_rectangle_coordinates)
        for i in range(nb_rectangles):
            # on traite le i-ème rectangle avec son indicateur 
            rectangle_coordinates = list_rectangle_coordinates[i]
            indic = list_indic[i]

            if i == 0:
                gsps_dataset.GraphicAnnotationSequence = [Dataset()]
            else :
                gsps_dataset.GraphicAnnotationSequence += [Dataset()]
            gsps_dataset.GraphicAnnotationSequence[i].GraphicLayer = 'ANALYSIS LAYER'
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence = [Dataset()]
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].GraphicAnnotationUnits = 'PIXEL'
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].GraphicDimensions = 2
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].NumberOfGraphicPoints = 4
            x, y, w, h = rectangle_coordinates
            x1 = int(x)
            x2 = int(x + w)
            x3 = int(x + w)
            x4 = int(x)
            y1 = int(y)
            y2 = int(y)
            y3 = int(y + h)
            y4 = int(y + h)
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].GraphicData = [x1, y1, x2, y2, x3, y3, x4, y4, x1, y1]
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].GraphicType = "POLYLINE"
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].GraphicFilled = "N"
            
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence = [Dataset()]
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].ShadowStyle = 'OFF'
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].ShadowOffsetX = 0
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].ShadowOffsetY = 0
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].LineThickness = 2
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].LineDashingStyle = 'SOLID'
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].ShadowOpacity = 0
            gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].PatternOnOpacity = 1
            # COULEUR
            if indic > 0.8:
                # l'annotation est VERTE
                interpol = int((1-indic)/0.2*0xAAAA)
                gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].PatternOnColorCIELabValue = [0xFFFF, interpol, 0xFFFF]
            elif indic > 0.65:
                # l'annotation est JAUNE
                # take a value linearly between 0xAAAA if indic = 0.8 and 0xE000 if indic=0.65
                interpol1 = int((0.8-indic)/0.15*0x3556+0xAAAA)
                interpol2 = int((0.8-indic)/0.15*(-4095)+0xFFFF)
                gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].PatternOnColorCIELabValue = [0xFFFF, interpol1, interpol2]
            else : 
                # l'annotation est ROUGE
                interpol1 = int((0.65-indic)/0.15*(-4095)+0xFFFF)
                interpol2 = int((0.65-indic)/0.15*(-8191)+0xFFFF)
                gsps_dataset.GraphicAnnotationSequence[i].GraphicObjectSequence[0].LineStyleSequence[0].PatternOnColorCIELabValue = [interpol1, interpol2, 0xF000]

            # [0xFFFF, 0x0000, 0xFFFF] : VERT
            # [0xF000, 0xFFFF, 0xF000] : ROUGE
            # [0xFFFF, 0xF000, 0xF000] : ORANGE
            # [0xFFFF, 0xAAAA, 0xFFFF] : JAUNE

            # On rajoute une annotation texte indiquant le niveau de précision
            if show_confidence:
                gsps_dataset.GraphicAnnotationSequence[i].TextObjectSequence = [Dataset()]
                gsps_dataset.GraphicAnnotationSequence[i].TextObjectSequence[0].AnchorPointAnnotationUnits = 'PIXEL'
                gsps_dataset.GraphicAnnotationSequence[i].TextObjectSequence[0].UnformattedTextValue = f"Précision: {int(round(indic,2)*100)}%"
                gsps_dataset.GraphicAnnotationSequence[i].TextObjectSequence[0].AnchorPoint = [x ,y1 - 30]
                gsps_dataset.GraphicAnnotationSequence[i].TextObjectSequence[0].AnchorPointVisibility = 'N'

        # Graphic Layer
        gsps_dataset.GraphicLayerSequence = [Dataset()]
        gsps_dataset.GraphicLayerSequence[0].GraphicLayer = 'ANALYSIS LAYER'
        gsps_dataset.GraphicLayerSequence[0].GraphicLayerOrder = 0

        # Softocopy Presentation LUP 
        gsps_dataset.PresentationLUTShape = 'IDENTITY'
        
        # file meta information
        gsps_dataset.file_meta = Dataset()
        
        

        # SOP Common
        gsps_dataset.InstanceCreationDate = today
        gsps_dataset.InstanceCreationTime = now
        gsps_dataset.SOPClassUID = GrayscaleSoftcopyPresentationStateStorage
        gsps_dataset.SOPInstanceUID = generate_uid()  # unique UID pour le GSPS qu'on cree
        
        return gsps_dataset
    except Exception as e:
        logger.warning(f"An error occurred while creating GSPS: {str(e)}")




