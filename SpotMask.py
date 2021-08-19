# *******************************************************************
#
# Author : Jacopo Bugini, 2021
# Email  : jacopo.bugini@gmail.com
# Github : https://github.com/jacopobugini
#
# *******************************************************************

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import cv2
import argparse
from utils.utils import *

# -------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='suggestions')
args = parser.parse_args()


# -------------------------------------------------------------------
# Load Model (YOLO)
# -------------------------------------------------------------------

cfg_dir = './models/face-detection/yolo-v3/cfg/yolov3-face.cfg'
weights_dir ='./models/face-detection/yolo-v3/weights/yolov3-wider_16000.weights'
net = cv2.dnn.readNetFromDarknet(cfg_dir, weights_dir)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model YOLO imported correctly")


# -------------------------------------------------------------------
# SpotMask Main
# -------------------------------------------------------------------

def _main():

    wind_name = 'Facemask Correctness Detection'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    while True:

        has_frame, frame = cap.read()

        # Prepare input for YOLO processing
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], swapRB=False, crop=False)
        net.setInput(blob)

        # get YOLO detection outputs
        outs = net.forward(get_outputs_names(net))

        # Run the different facemask detection models on YOLO outputs
        process_frame(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD, args.mode)                

        cv2.imshow(wind_name, frame)

        # Interruption key method
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> SpotMask Closed Correctly!')
    print('***********************************************************')

if __name__ == '__main__':
    _main()
