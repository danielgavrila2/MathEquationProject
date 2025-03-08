import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def binarize_image(image_to_be_processed):

    image_to_be_processed = image.img_to_array(image_to_be_processed, dtype='uint8')
    binarized_image = np.expand_dims(cv2.adaptiveThreshold(image_to_be_processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                           cv2.THRESH_BINARY, 11, 2), -1)

    return binarized_image

data_directory = 'data/extracted_images'
batch_size = 32
img_height = 45
img_width = 45

class_names = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'times', 'y']

def detect_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def detect_contours(image_path):

    #import the grayscale image
    input_image = cv2.imread(image_path, 0)

    copy_image = input_image.copy()

    #we convert the image into binary form, to be able to process it using the grayscale tempo
    binarized_image = cv2.adaptiveThreshold(copy_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    invered_image = ~binarized_image

    #In this step, we will find the contours, and we will keep track of the hierarchy of them
    contours_list, hierarchy = cv2.findContours(invered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frames = []
    for contour in contours_list:
        x, y, w, h = cv2.boundingRect(contour)
        frames.append([x, y, w, h])

    frames_copy = frames.copy()
    keep = []

    while len(frames_copy) > 0:

        current_x, current_y, current_w, current_h = frames_copy.pop(0)

        #we will skip the very small boxes
        if current_w * current_h < 20:
            continue

        throw = []

        for i, (x, y, w, h) in enumerate(frames_copy):
            current_interval = [current_x, current_w + current_x]
            next_interval = [x, x + w]

            #Now we check if these intervals have pixels that overlap
            if detect_overlap(current_interval, next_interval) > 1:
                new_interval_x = [min(current_x, x), max(current_x + current_w, x + w)]
                new_interval_y = [min(current_y, y), max(current_y + current_h, y + h)]

                x2, y2 = new_interval_x[0], new_interval_y[0]
                w2, h2 = new_interval_x[1] - new_interval_x[0], new_interval_y[1] - new_interval_y[0]

                current_x, current_y, current_w, current_h = x2, y2, w2, h2

                #we can throw this box, because it is already merged
                throw.append(i)

        for i in sorted(throw, reverse=True):
            frames_copy.pop(i)

        # we will keep the current box to compare with the other one
        keep.append([current_x, current_y, current_w, current_h])

    return keep

def resize_image(img, size, padColor = 255):

    h, w = img.shape[:2]
    sh, sw = size

    #we use the interpolation method to evaluate the data
    if h > sh or w > sw:
        #shrink the image
        interpolation = cv2.INTER_AREA
    else:
        #stractching the image
        interpolation = cv2.INTER_CUBIC

    aspect_ratio = w / h

    #here we will compute the scaling and the pad sizing
    if aspect_ratio > 1: #in this case we have a horizontal image
        #so, we have to modify the image in the top and bottom
        w2 = sw
        h2 = np.round(w2 / aspect_ratio).astype(int)
        pad_vertical = (sh - h2) / 2
        pad_top, pad_bottom = np.floor(pad_vertical).astype(int), np.ceil(pad_vertical).astype(int)
        pad_left, pad_right = 0, 0

    elif aspect_ratio < 1 : #vertical image
        h2 = sh
        w2 = np.round(h2 * aspect_ratio).astype(int)
        pad_horizontal = (sw - w2) / 2
        pad_left, pad_right = np.floor(pad_horizontal).astype(int), np.ceil(pad_horizontal).astype(int)
        pad_top, pad_bottom = 0, 0

    else : #square image
        #no adjustments needed
        h2 = sh
        w2 = sw
        pad_left, pad_right = 0, 0
        pad_top, pad_bottom = 0, 0

    #setting the pad and the coloring
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    #scaling and padding
    scaled_image = cv2.resize(img, (w2, h2), interpolation=interpolation)
    scaled_image = cv2.copyMakeBorder(scaled_image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_image

def put_raise_power_symbol(string):
    l = list(string)
    i = 0
    while i < len(l) - 1:
        if l[i].isalpha():
            if l[i + 1].isdigit():
                l.insert(i + 1, '**')
                i = i + 1
        i += 1
    new_string = ''.join(l)
    return new_string

def put_multiply_symbol(string):
    l = list(string)
    i = 0
    while i < len(l) - 1:
        if l[i].isdigit() and l[i +1].isalpha():
            l.insert(i + 1, '*')
        i = i + 1
    new_string = ''.join(l)
    return new_string

model = tf.keras.models.load_model('equation-detection-model-v1', compile=False)

def solve_equation(image_path):
    print(image_path.split('\\')[-1])  # Fixed line
    IMAGE = image_path.split('\\')[-1]

    image_path = "static/" + IMAGE
    image_directory = "static/"

    print(image_path)
    input_image = cv2.imread(image_path, 0)  # Ensure grayscale
    input_image_copy = input_image.copy()
    keep = detect_contours(image_directory + IMAGE)

    eq_list = []
    inverted_binary_image = binarize_image(input_image)

    for (x, y, w, h) in sorted(keep, key=lambda x: x[0]):
        img = resize_image(inverted_binary_image[y: y+h, x: x+w], (45, 45), 0)

        # Ensure single-channel input
        if img.shape[-1] != 1:
            img = np.expand_dims(img, axis=-1)

        # Expand batch dimension
        second_expand = np.expand_dims(img, axis=0).astype('float32')  # Ensure correct dtype

        # Predict class
        prediction = model.predict(second_expand)
        max_arg = np.argmax(prediction)
        prediction_class = class_names[max_arg]

        # Convert "times" and "div"
        if prediction_class == "times":
            prediction_class = "*"
        elif prediction_class == "div":
            prediction_class = "/"

        eq_list.append(prediction_class)

    eq_string = "".join(eq_list)
    print(eq_string)

    # Convert to proper mathematical symbols
    equation = put_raise_power_symbol(eq_string)
    equation = put_multiply_symbol(equation)

    return equation

