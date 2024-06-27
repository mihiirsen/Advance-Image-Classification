import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

model = tf.keras.applications.ResNet50(weights="imagenet")


def super_class_input(path):
    result_list = []

    for img_path in path:
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        imagenet_id, label, score = decoded_predictions[0]

        if float(score) > 0.3:
            result_list.append("->"+label)
        else:
            result_list.append(" ")

    return result_list
