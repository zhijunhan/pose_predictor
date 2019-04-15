#!/usr/bin/env python
from __future__ import division

from tfpose_ros.estimator import Human, BodyPart
from tfpose_ros.msg import Persons, Person, BodyPartElm

import cv2
import numpy as np

import tfpose_ros.common as common
from tfpose_ros.common import CocoPairsNetwork, CocoPairs, CocoPart

from tfpose_ros.sort import Sort

POSE_COCO_BODY_PARTS = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar",
    18: "Background",
}

class ActionPredictor(object):

    def __init__(self):
        self.current = []
        self.previous = []
        self.memory = {}
        self.data = {}

        self.output = None

        self.tracker = Sort(20, 3)

        self.move_status = ['', 'Not Crossing', 'Not Crossing', 'Crossing', 'walk close', 'walk away', 'sit down', 'stand up']

    def reg(self, body_parts):
        # All move and actions are based on the joints info difference from the newest frame and oldest frame
        init_x = float(body_parts["Neck"].x + body_parts["RHip"].x + body_parts["LHip"].x) / 3
        init_y = float(body_parts["Neck"].y + body_parts["RHip"].y + body_parts["LHip"].y) / 3
        end_x = float(body_parts["Neck"].x + body_parts["RHip"].x + body_parts["LHip"].x) / 3
        end_y = float(body_parts["Neck"].y + body_parts["RHip"].y + body_parts["LHip"].y) / 3

        # Upper body height change
        init_h1 = float(body_parts["RHip"].y + body_parts["LHip"].y) / 2 - body_parts["Neck"].y
        end_h1 = float(body_parts["RHip"].y + body_parts["LHip"].y) / 2 - body_parts["Neck"].y

        # Upper body height change rate
        try:
            h1 = end_h1 / init_h1
        except:
            h1 = 0.0

        # Thigh height change
        init_h2 = (float(body_parts["RKnee"].y + body_parts["LKnee"].y) - float(body_parts["RHip"].y + body_parts["LHip"].y)) / 2
        end_h2 = (float(body_parts["RKnee"].y + body_parts["LKnee"].y) - float(body_parts["RHip"].y + body_parts["LHip"].y)) / 2

        # Thigh height change rate
        try:
            h2 = end_h2 / init_h2
        except:
            h2 = 0.0

        # Upper body center change
        xc = end_x - init_x
        yc = end_y - init_y

        if abs(xc) < 30. and abs(yc) < 20.:
            ty_1 = float(body_parts["Neck"].y)
            ty_8 = float(body_parts["RHip"].y + body_parts["LHip"].y) / 2
            ty_9 = float(body_parts["RKnee"].y + body_parts["LKnee"].y) / 2
            try:
                t = float(ty_8 - ty_1) / (ty_9 - ty_8)
            except:
                t = 0.0

            if h1 < 1.16 and h1 > 0.84 and h2 < 1.16 and h2 > 0.84:

                if t < 1.73:
                    return self.move_status[1]
                else:
                    return self.move_status[2]
            else:
                if t < 1.7:
                    if h1 >= 1.08:
                        return self.move_status[4]
                    elif h1 < 0.92:
                        return self.move_status[5]
                    else:
                        return self.move_status[0]
                else:
                    return self.move_status[0]
        elif abs(xc) < 30. and abs(yc) >= 30.:
            init_y1 = float(body_parts["Neck"].y)
            init_y8 = float(body_parts["RHip"].y + body_parts["LHip"].y) / 2
            init_y9 = float(body_parts["RKnee"].y + body_parts["LKnee"].y) / 2

            end_y1 = float(body_parts["Neck"].y)
            end_y8 = float(body_parts["RHip"].y + body_parts["LHip"].y) / 2
            end_y9 = float(body_parts["RKnee"].y + body_parts["LKnee"].y) / 2
            try:
                init_yc = float(init_y8 - init_y1) / (init_y9 - init_y8)
            except:
                init_yc = 0.0
            try:
                end_yc = float(end_y8 -end_y1) / (end_y9 -end_y8)
            except:
                end_yc = 0.0

            th_yc = 0.1
            if yc >= 25 and abs(end_yc - init_yc) >= th_yc:
                return self.move_status[6]

            elif yc < -20 and abs(end_yc - int_yc) >= th_yc:
                return self.move_status[7]

            else:
                return self.move_status[0]

        elif abs(xc) > 30. and abs(yc) < 30.:
            return self.move_status[3]

        else:
            return self.move_status[0]

    def proceed(self, image, humans):

        img, joints, bboxes, xcenter, sk = self.get_skeleton(image, humans, imgcopy=False)

        self.output = img

        height = img.shape[0]
        width = img.shape[1]

        if bboxes:
            result = np.array(bboxes)
            det = result[:, 0:5]
            det[:, 0] = det[:, 0] * width
            det[:, 1] = det[:, 1] * height
            det[:, 2] = det[:, 2] * width
            det[:, 3] = det[:, 3] * height
            trackers = self.tracker.update(det)

            self.current = [i[-1] for i in trackers]
            if len(self.previous) > 0:
                for item in self.previous:
                    if item not in self.current and item in self.data:
                        del self.data[item]
                    if item not in self.current and item in self.memory:
                        del self.memory[item]
            self.previous = self.current

            for d in trackers:
                xmin = int(d[0])
                ymin = int(d[1])
                xmax = int(d[2])
                ymax = int(d[3])
                label = int(d[4])

                try:
                    j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                except:
                    j = 0

                if self._joint_filter_(joints[j]):

                    joints[j] = self._joint_complete_(self._joint_complete_(joints[j]))

                    if label not in self.data:
                        self.data[label] = [joints[j]]
                        self.memory[label] = 0
                    else:
                        self.data[label].append(joints[j])

                    if len(self.data[label]) > 10:

                        pred = self.recognize(self.data[label])

                        if pred == 0:
                            pred = self.memory[label]
                        else:
                            self.memory[label] = pred
                        self.data[label].pop(0)

                        location = self.data[label][-1][1]
                        if location[0] <= 30:
                            location = (51, location[1])
                        if location[1] <= 10:
                            location = (location[0], 31)

                        cv2.putText(img, self.move_status[pred], (location[0] - 30, location[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        print "Move status", self.move_status[pred]

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,255,255), 4)

            self.output = img

    @staticmethod
    def _joint_filter_(joint):
        if 1 not in joint:
            return False
        # Check exist of hip
        if 8 not in joint and 11 not in joint:
            return False
        # Check exist of knee
        if 9 not in joint and 12 not in joint:
            return False
        return True

    @staticmethod
    def _joint_complete_(joint):
        if 8 in joint and 11 not in joint:
            joint[11] = joint[8]
        elif 8 not in joint and 11 in joint:
            joint[8] = joint[11]
        if 9 in joint and 12 not in joint:
            joint[12] = joint[9]
        elif 9 not in joint and 12 in joint:
            joint[9] = joint[12]

        return joint

    def recognize(self, joints):
        # All move and actions are based on the joints info difference from the newest frame and oldest frame

        init_x = float(joints[0][1][0] + joints[0][8][0] + joints[0][11][0]) / 3
        init_y = float(joints[0][1][1] + joints[0][8][1] + joints[0][11][1]) / 3
        end_x = float(joints[-1][1][0] + joints[-1][8][0] + joints[-1][11][0]) / 3
        end_y = float(joints[-1][1][1] + joints[-1][8][1] + joints[-1][11][1]) / 3

        # Upper body height change
        init_h1 = float(joints[0][8][1] + joints[0][11][1]) / 2 - joints[0][1][1]
        end_h1 = float(joints[-1][8][1] + joints[-1][11][1]) / 2 - joints[-1][1][1]
        # Upper body height change rate
        try:
            h1 = end_h1 / init_h1
        except:
            h1 = 0.0
        
        # Thigh height change
        init_h2 = (float(joints[0][9][1] + joints[0][12][1]) - float(joints[0][8][1] + joints[0][11][1])) / 2
        end_h2 = (float(joints[-1][9][1] + joints[-1][12][1]) - float(joints[-1][8][1] + joints[-1][11][1])) / 2
        # Thigh height change rate
        try:
            h2 = end_h2 / init_h2
        except:
            h2 = 0.0

        # Upper body center change
        xc = end_x - init_x
        yc = end_y - init_y

        if abs(xc) < 30. and abs(yc) < 20.:
            ty_1 = float(joints[-1][1][1])
            ty_8 = float(joints[-1][8][1] + joints[-1][11][1]) / 2
            ty_9 = float(joints[-1][9][1] + joints[-1][12][1]) / 2
            try:
                t = float(ty_8 - ty_1) / (ty_9 - ty_8)
            except:
                t = 0.0
            if h1 < 1.16 and h1 > 0.84 and h2 < 1.16 and h2 > 0.84:

                if t < 1.73:
                    return 1
                else:
                    return 2
            else:
                if t < 1.7:
                    if h1 >= 1.08:
                        return 4

                    elif h1 < 0.92:
                        return 5
                    else:
                        return 0
                else:
                    return 0
        elif abs(xc) < 30. and abs(yc) >= 30.:
            init_y1 = float(joints[0][1][1])
            init_y8 = float(joints[0][8][1] + joints[0][11][1]) / 2
            init_y9 = float(joints[0][9][1] + joints[0][12][1]) / 2

            end_y1 = float(joints[-1][1][1])
            end_y8 = float(joints[-1][8][1] + joints[-1][11][1]) / 2
            end_y9 = float(joints[-1][9][1] + joints[-1][12][1]) / 2
            try:
                init_yc = float(init_y8 - init_y1) / (init_y9 - init_y8)
            except:
                init_yc = 0.0
            try:
                end_yc = float(end_y8 - end_y1) / (end_y9 - end_y8)
            except:
                end_yc = 0.0
            th_yc = 0.1
            if yc >= 25 and abs(end_yc - init_yc) >= th_yc:
                return 6

            elif yc < -20 and abs(end_yc - init_yc) >= th_yc:
                return 7

            else:
                return 0

        elif abs(xc) > 30. and abs(yc) < 30.:
            return 3

        else:
            return 0

    def get_skeleton(self, npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        sk = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        joints = []
        bboxes = []
        xcenter = []
        for human in humans:
            xs = []
            ys = []
            centers = {}
            # draw point
            for i in range(common.CocoPart.Background.value):

                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]

                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))

                centers[i] = center
                xs.append(center[0])
                ys.append(center[1])
                cv2.circle(sk, center, 3, (0, 0, 255), thickness=3, lineType=8, shift=0)

            # Connect joints that belongs to each person
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv2.line(sk, centers[pair[0]], centers[pair[1]], (0, 255, 0), 2)

            xmin = float(min(xs) / image_w)
            ymin = float(min(ys) / image_h)
            xmax = float(max(xs) / image_w)
            ymax = float(max(ys) / image_h)

            bboxes.append([xmin, ymin, xmax, ymax, 0.9999])
            joints.append(centers)
            if 1 in centers:
                xcenter.append(centers[1][0])

        return npimg, joints, bboxes, xcenter, sk