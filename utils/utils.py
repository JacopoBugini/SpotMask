import numpy as np
import cv2
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
import numpy as np

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# -------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------

# Load the trained model
mask_net = keras.models.load_model('models/facemask-correctness/mask_correctness_model.h5')
print("Model Check Mask imported correctly")

detect_net = keras.models.load_model('models/mask-detection/mask_detection_model.h5')
print("Model Detect Mask imported correctly")
print("*********************************************")

suggest_net = keras.models.load_model('models/suggestions-detection/suggestions_model.h5')
print("Model Detect Mask imported correctly")
print("*********************************************")


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def process_frame(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        draw_predict(frame, confidences[i], left, top, left + width, top + height)
    
        colour_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_img_crop = colour_frame[top-30:top+height+30, left-30:left+width+30]

        img_array = prepare_frame(face_img_crop)

        output_mask, colour, mask_result = detect_mask_usage(img_array)

        cv2.rectangle(frame, (left, top), (left+width, top+height), colour, 3)
        cv2.putText(frame, output_mask, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    return final_boxes

def prepare_frame(img, size=[150,150]):

    img_reshaped = cv2.resize(img, (size[0],size[1]))
    img_array = image.img_to_array(img_reshaped)
    img_array = img_array.astype('float32')
    img_array /= 255.0
    img_array = img_array.reshape((1,) + img_array.shape)

    return img_array

def detect_mask_usage(img_array):

    mask_result = detect_net.predict_on_batch(img_array)
    mask_is_proper = mask_net.predict_on_batch(img_array)
    suggestions = suggest_net.predict_on_batch(img_array)
    
    score=np.amax(mask_result[0], axis=0)
    list_scores = list(mask_result[0])
    result_index = list_scores.index(score)
    
    score_2=np.amax(mask_is_proper[0], axis=0)
    list_scores_2 = list(mask_is_proper[0])
    result_index_2 = list_scores_2.index(score_2)

    score_3=np.amax(suggestions[0], axis=0)
    list_scores_3 = list(suggestions[0])
    result_index_3 = list_scores_3.index(score_3)
    print(score_3, result_index_3)

    if result_index == 1:
        output_mask = 'Wear a Mask!' 
        colour = (0,0,255)
    else:      
        if result_index_2 == 1:
            output_mask = 'Good!'
            colour = (0,255,0)
        else:
            output_mask = 'Wear it correctly!'
            colour = (0,152,232)
    
    return output_mask, colour, mask_result
