from matplotlib import pyplot as plt
import torch

def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box coordinates from (center_x, center_y, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        x (torch.Tensor): Bounding box coordinates in (center_x, center_y, width, height) format.

    Returns:
        torch.Tensor: Bounding box coordinates in (x_min, y_min, x_max, y_max) format.
    """
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    """
    Rescales the bounding boxes based on the given output bounding box and image size.

    Args:
        out_bbox (torch.Tensor): The output bounding box.
        size (tuple): The size of the image in the format (width, height).

    Returns:
        torch.Tensor: The rescaled bounding boxes.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def convert_boxes(width,height, outputs, threshold=0.9, keep_highest_scoring_bbox=False):
    """
    Visualizes the predictions on an image.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        outputs (torch.Tensor): The model's output tensor.
        id2label (dict): A dictionary mapping class IDs to labels.
        threshold (float, optional): The confidence threshold for keeping predictions. Defaults to 0.9.
        keep_highest_scoring_bbox (bool, optional): Whether to keep only the highest scoring bounding box. Defaults to False.
    """
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold 
    if keep_highest_scoring_bbox:
        keep = probas.max(-1).values.argmax()
        keep = torch.tensor([keep])

    # convert predicted boxes from [0; 1] to image scales
    print(outputs.pred_boxes[0, keep].cpu())
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), (width, height))
    # add the score of each predicted box
    scores = probas[keep].cpu().max(-1).values
    bboxes_scaled = bboxes_scaled.tolist()
    for i in range(len(bboxes_scaled)):
        bboxes_scaled[i].append(scores[i].item())
    bboxes = []
    for box in bboxes_scaled:
        bboxes.append({"x":box[0],"y":box[1],"w":box[2]-box[0],"h":box[3]-box[1],"conf":box[4]})
    print(bboxes)
    return bboxes

