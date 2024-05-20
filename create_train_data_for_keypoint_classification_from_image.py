#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # use_brect = True

    # Camera preparation ###############################################################
    # cap = cv.VideoCapture(cap_device)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    #
    # keypoint_classifier = KeyPointClassifier()

    # point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    # with open('model/keypoint_classifier/keypoint_classifier_label.csv',
    #           encoding='utf-8-sig') as f:
    #     keypoint_classifier_labels = csv.reader(f)
    #     keypoint_classifier_labels = [
    #         row[0] for row in keypoint_classifier_labels
    #     ]
    # with open(
    #         'model/point_history_classifier/point_history_classifier_label.csv',
    #         encoding='utf-8-sig') as f:
    #     point_history_classifier_labels = csv.reader(f)
    #     point_history_classifier_labels = [
    #         row[0] for row in point_history_classifier_labels
    #     ]

    # FPS Measurement ########################################################
    # cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    # history_length = 16
    # point_history = deque(maxlen=history_length)

    # # Finger gesture history ################################################
    # finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    data_path = 'data/asl_alphabet_train/asl_alphabet_train_rotated_v3'
    folds = os.listdir(data_path)
    # while True:
    label = -1
    for fold in folds:
        # fps = cvFpsCalc.get()
        if fold not in ['L', 'Z', 'B', 'S', 'V']: continue
        f_path = os.path.join(data_path, fold)
        imgs = os.listdir(f_path)
        index = 0
        label = label + 1
        for img in imgs:
            index = index + 1
            img_path = os.path.join(f_path, img)
            image = cv.imread(img_path)

            number = label
            mode = 1

            # Read image from file
            image = cv.flip(image, 1)  # The asl alphabet is left hand, then I flip to use for right hand
            debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):

                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    logging_csv(number, mode, pre_processed_landmark_list)

                    print("Label %s, Fold %s: %s" %(label, fold, index))


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1:
        csv_path = 'model/keypoint_classifier/keypoint_v4.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

if __name__ == '__main__':
    main()
