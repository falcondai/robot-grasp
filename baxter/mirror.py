import argparse
import os
import sys

import rospy

import baxter_interface

from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from sensor_msgs.msg import JointState, Range, Image

from PIL import Image as PIL_Image
from PIL import ImageDraw as PIL_ImageDraw
import numpy as np

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('controlling_limb', default='right', choices=['left', 'right'])
    # args = parser.parse_args()

    rospy.init_node('robot_mirror_control')

    left = baxter_interface.Limb('left')
    right = baxter_interface.Limb('right')
    grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
    grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
    head = baxter_interface.Head()
    pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
    measurements = {
        'ir_range': 0.,
        'grip_force': 0.,
        'grip_gap': 0.,
    }

    def close_grippers(data):
        if data.state == DigitalIOState.PRESSED:
            grip_right.close()
            grip_left.close()

    def open_grippers(data):
        if data.state == DigitalIOState.PRESSED:
            grip_right.open()
            grip_left.open()

    def change_left_limb(data):
        x = {}
        right_arm_joint_names = ['right_e0', 'right_e1', 'right_s0', 'right_s1', 'right_w0', 'right_w1', 'right_w2']
        for name, position in zip(data.name, data.position):
            if name in right_arm_joint_names:
                x[name.replace('right', 'left')] = position
        if 'left_s0' in x:
            HEAD_LIMB_OFFSET = -0.76
            head.set_pan(x['left_s0'] + HEAD_LIMB_OFFSET)
            x['left_s0'] *= -1
        if len(x.keys()) > 0:
            left.set_joint_positions(x)

    def display_gripper_state(data):
        # print 'position:', data.position, 'force:', data.force
        measurements['grip_force'] = data.force
        measurements['grip_gap'] = data.position
        pass

    def display_ir(data):
        # print 'left range:', data.range
        measurements['ir_range'] = data.range
        pass

    def display_cam(data):
        img = PIL_Image.frombytes('RGBA', (data.width, data.height), data.data)
        img.save('cam.png')
        draw = PIL_ImageDraw.Draw(img)
        draw.multiline_text((100, 100), 'distance: %.4f\ngrip gap: %.2f\ngrip force: %.4f\n' % (measurements['ir_range'], measurements['grip_gap'], measurements['grip_force']))
        data.data = img.tobytes()
        pub.publish(data)


    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    try:
        rs.enable()
        if not grip_left.calibrated:
            grip_left.calibrate()
        if not grip_right.calibrated:
            grip_right.calibrate()
        rospy.Subscriber('/robot/digital_io/right_lower_button/state', DigitalIOState, close_grippers)
        rospy.Subscriber('/robot/digital_io/right_upper_button/state', DigitalIOState, open_grippers)
        rospy.Subscriber('/robot/joint_states', JointState, change_left_limb)
        rospy.Subscriber('/robot/end_effector/left_gripper/state', EndEffectorState, display_gripper_state)
        rospy.Subscriber('/robot/range/left_hand_range/state', Range, display_ir)
        rospy.Subscriber('/cameras/left_hand_camera/image', Image, display_cam)

        rospy.spin()
    except Exception, e:
        rospy.logerr(e.strerror)

    return 0

if __name__ == '__main__':
    sys.exit(main())
