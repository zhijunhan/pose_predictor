#!/usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tfpose_ros.msg import Persons, Person, BodyPartElm
from tfpose_ros.estimator import Human, BodyPart
from pose_predictor.predictor import ActionPredictor

class PredictionVisulaizer:
    """
    Reference : ros-video-recorder
    https://github.com/ildoonet/ros-video-recorder/blob/master/scripts/recorder.py
    """
    def __init__(self):
        # Ros Parms
        image_topic = rospy.get_param('~camera', '/usb_cam/image_raw')
        pose_topic = rospy.get_param('~pose', '/pose_estimator/pose')
        self.resize_ratio = float(rospy.get_param('~resize_ratio', '-1'))
        # Initialization
        self.bridge = CvBridge()
        self.predictor = ActionPredictor()
        self.frames = []
        self.frame_id = None
        # Pubs and Subs
        self.pub_reg = rospy.Publisher('~output_reg', Image, queue_size=1)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.callback_image, queue_size=1)
        self.pose_sub = rospy.Subscriber(pose_topic, Persons, self.cb_pose, queue_size=1)


    def callback_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr('Converting Image Error. ' + str(e))
            return
        self.frame_id = data.header.frame_id
        self.frames.append((data.header.stamp, cv_image))

    def get_latest(self, at_time, remove_older=True):
        fs = [x for x in self.frames if x[0] <= at_time]
        if len(fs) == 0:
            return None

        f = fs[-1]
        if remove_older:
            self.frames = self.frames[len(fs) - 1:]

        return f[1]

    def cb_pose(self, data_raw):
        # get image with pose time
        t = data_raw.header.stamp
        image = self.get_latest(t, remove_older=True)

        if image is None:
            rospy.logwarn('No received images.')
            return

        h, w = image.shape[:2]

        if self.resize_ratio > 0:
            image = cv2.resize(image, (int(self.resize_ratio*w), int(self.resize_ratio*h)), interpolation=cv2.INTER_LINEAR)

       # ros topic to Person instance
        humans = []
        if data_raw.header.frame_id == self.frame_id:
            for p_idx, person in enumerate(data_raw.persons):
                human = Human([])
                for body_part in person.body_part:
                    part = BodyPart('', body_part.part_id, body_part.x, body_part.y, body_part.confidence)
                    human.body_parts[body_part.part_id] = part
                humans.append(human)
        # Process recognition and publishhumans
        self.predictor.proceed(image, humans)
        self.pub_reg.publish(self.bridge.cv2_to_imgmsg(self.predictor.output, "bgr8"))


if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('PredictionVisulaizer', anonymous=True)    
    pv = PredictionVisulaizer()
    rospy.spin()