import os
import torch
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from cropping import crop
from model_ResNet import super_class_input

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl", confidence=0.85)

def _Labels(path):
    output_image_name = "Segmented_Image.jpeg"

    results, output = ins.segmentImage(
        path, show_bboxes=True, output_image_name=output_image_name
    )

    bounding_box = results["boxes"]
    classes = results["class_names"]
    count = results["object_counts"]

    cropped_image_path, crop_img = crop(bounding_box,path)

    super_classes = super_class_input(cropped_image_path)

    label = [x + y for x, y in zip(classes, super_classes)]
    result_string = " ,".join(label)

    return result_string,count

    
