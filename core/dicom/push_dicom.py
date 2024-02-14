from pynetdicom import AE, StoragePresentationContexts
from pynetdicom.sop_class import (
    CTImageStorage, MRImageStorage, SecondaryCaptureImageStorage,
    OphthalmicPhotography8BitImageStorage, DigitalXRayImageStorageForPresentation
)
from pydicom.uid import (
    ExplicitVRBigEndian, ImplicitVRLittleEndian, ExplicitVRLittleEndian, JPEGBaseline8Bit
)

def send_dicom_to_pacs(dicom_dataset, pacs_ip, pacs_port, aetitle, pacs_aetitle, logger):

    # AE Configuration (Application Entity)
    ae = AE(ae_title=aetitle)

    supported_contexts = [
        CTImageStorage,
        MRImageStorage,
        SecondaryCaptureImageStorage,
        OphthalmicPhotography8BitImageStorage,
        DigitalXRayImageStorageForPresentation,
    ]
    
    for sop_class in supported_contexts:
        ae.add_requested_context(sop_class, [ImplicitVRLittleEndian,ExplicitVRLittleEndian,JPEGBaseline8Bit,ExplicitVRBigEndian])

    dicom_dataset.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    # Connection to PACS
    assoc = ae.associate(pacs_ip, pacs_port, StoragePresentationContexts, ae_title=pacs_aetitle)
    logger.info(f"Trying to connect to PACS: {pacs_ip}:{pacs_port}")

    if assoc.is_established:
        
        logger.info(f"Connected to PACS: {pacs_ip}:{pacs_port}")
        
        status = assoc.send_c_store(dicom_dataset)

        if status:
            assoc.release()
            logger.info(f"Sent GSPS to PACS: {pacs_ip}:{pacs_port} with status: {status}")
            return True
        else:
            assoc.release()
            logger.error(f"Failed to send GSPS to PACS: {pacs_ip}:{pacs_port}")
            return False
    else:
        logger.error(f"Failed to connect to PACS: {pacs_ip}:{pacs_port}")
        return False