import numpy as np
import cv2
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
import numpy as np

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
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def process_frame(frame, outs, conf_threshold, nms_threshold, mode):
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
    
        colour_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_img_crop = colour_frame[top-30:top+height+30, left-30:left+width+30]

        img_array = prepare_frame(face_img_crop)

        output_mask, colour, mask_result = detect_mask_usage(img_array, mode)

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

def detect_mask_usage(img_array, mode):

    # predict mask presence: Detect Mask
    mask_result = detect_net.predict_on_batch(img_array)

    # Predict Mask Correctness: Mask Correctness
    mask_is_proper = mask_net.predict_on_batch(img_array)

    # Predict Mask Suggestions: Mask Suggestions
    suggestions = suggest_net.predict_on_batch(img_array)
    
    # Elaborate scores based on prediction values
    # get mask presence results
    score=np.amax(mask_result[0], axis=0)
    list_scores = list(mask_result[0])
    mask_detection_result_index = list_scores.index(score)
    
    # get mask correctness results
    score_2=np.amax(mask_is_proper[0], axis=0)
    list_scores_2 = list(mask_is_proper[0])
    correctness_result_index = list_scores_2.index(score_2)

    # get mask suggestions results
    score_3=np.amax(suggestions[0], axis=0)
    list_scores_3 = list(suggestions[0])
    suggestions_result_index = list_scores_3.index(score_3)

    
    if mask_detection_result_index == 1:
        output_mask = 'Wear a Mask!' 
        colour = (0,0,255)

    else:     

        if mode == 'simple':

            if correctness_result_index == 1:
                output_mask = 'Good!'
                colour = (0,255,0)
            else:
                output_mask = 'Wear it correctly!'
                colour = (0,152,232)
        
        elif mode == 'suggestions':

            if suggestions_result_index == 0:
                output_mask = 'Adjust on Chin!'
                colour = (0,152,232)
            elif suggestions_result_index == 1:
                output_mask = 'Cover your Nose!'
                colour = (0,152,232)
            elif suggestions_result_index == 2:
                output_mask = 'Cover Mouth and Nose!'
                colour = (0,152,232)
            elif suggestions_result_index == 3:
                output_mask = 'Good!'
                colour = (0,255,0)

        else:
            print('Mode not recongized. Please consider giving --mode "suggestions" or --mode "simple"')
    
    return output_mask, colour, mask_result
