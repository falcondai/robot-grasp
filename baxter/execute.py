import argparse
import os
import sys

import rospy

import baxter_interface
import actionlib

from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from sensor_msgs.msg import JointState, Range, Image
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from PIL import Image as PIL_Image
from PIL import ImageDraw as PIL_ImageDraw
import numpy as np

# baxter's arm links in mm
SHOULDER_LINK = 370.82
ELBOW_LINK = 374.42
SPEED = 100.

def d_arccos(x):
    return -1. / np.sqrt(1 - x**2)

def angle_dot(r, z, v=SPEED, a=SHOULDER_LINK, b=ELBOW_LINK):
    l = np.sqrt(r**2 + z**2)
    q = (a**2 + l**2 - b**2) / (2 * a * l)
    print l, q
    return d_arccos(q) * (l**2 - a**2 + b**2) / (2 * a * l**2) * z / np.sqrt(r**2 + z**2) * v

def solve_joint(r, z, a=SHOULDER_LINK, b=ELBOW_LINK):
    l = np.sqrt(r**2 + z**2)
    theta = -np.arctan2(z, r)
    print theta
    alpha = np.arccos((a**2 + l**2 - b**2) / (2 * a * l))
    beta = alpha + np.arccos((a**2 + l**2 - b**2) / (2 * a * l))
    return -alpha + theta, beta, alpha - theta - beta + np.pi/2

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

    def move_limb(r1, z1):
        def velocity_control_callback(data):
            joints = {}
            for name, position in zip(data.name, data.position):
                joints[name] = position
            print joints
            if 'left_s1' in joints and 'left_e1' in joints:
                alpha = joints['left_s1']
                beta = joints['left_e1']
                z = SHOULDER_LINK * np.sin(-alpha) - ELBOW_LINK * np.sin(beta)
                l = np.sqrt(z**2 + r1**2)
                theta = np.arccos(r1 / l)
                print alpha, beta, l, theta, z
                theta_dot = -1 / (1. + (r1/l)**2) * z / np.sqrt(r1**2 + z**2) * SPEED
                alpha_dot = angle_dot(r1, z)
                beta_dot = alpha_dot + angle_dot(r1, z, a=ELBOW_LINK, b=SHOULDER_LINK)
                left.set_joint_velocities({
                    'left_s1': -alpha_dot + theta_dot,
                    'left_e1': -beta_dot,
                    'left_w1': alpha_dot - theta_dot - beta_dot
                })
        return velocity_control_callback

    def display_gripper_state(data):
        # print 'position:', data.position, 'force:', data.force
        measurements['grip_force'] = data.force
        measurements['grip_gap'] = data.position

    def display_ir(data):
        # print 'left range:', data.range
        measurements['ir_range'] = data.range

    def display_cam(data):
        img = PIL_Image.frombytes('RGBA', (data.width, data.height), data.data)
        draw = PIL_ImageDraw.Draw(img)
        draw.multiline_text((100, 100), 'distance: %.4f\ngrip gap: %.2f\ngrip force: %.4f\n' % (measurements['ir_range'], measurements['grip_gap'], measurements['grip_force']))
        data.data = img.tobytes()
        pub.publish(data)

    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    try:
        rs.enable()
        # grip_left.calibrate()
        # grip_right.calibrate()
        # rospy.Subscriber('/robot/digital_io/right_lower_button/state', DigitalIOState, close_grippers)
        # rospy.Subscriber('/robot/digital_io/right_upper_button/state', DigitalIOState, open_grippers)
        rospy.Subscriber('/robot/end_effector/left_gripper/state', EndEffectorState, display_gripper_state)
        rospy.Subscriber('/robot/range/left_hand_range/state', Range, display_ir)
        rospy.Subscriber('/cameras/left_hand_camera/image', Image, display_cam)
        a0, b0, c0 = solve_joint(400., 0.)
        lowest = np.sqrt((SHOULDER_LINK + ELBOW_LINK) ** 2 - 500 ** 2)
        a1, b1, c1 = solve_joint(400., -lowest)
        print a0, b0, c0
        print a1, b1, c1

        action_client = actionlib.SimpleActionClient(
            '/robot/limb/left/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        server_up = action_client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            return 1
        action_client.cancel_goal()
        # left.move_to_neutral()

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ['left_s1', 'left_e1', 'left_w1', 'left_s0', 'left_e0', 'left_w0', 'left_w2']
        goal.goal_time_tolerance = rospy.Time(1.)

        # specify trajectory
        current_p = JointTrajectoryPoint()
        current_p.positions = [left.joint_angle(joint) for joint in goal.trajectory.joint_names]
        current_p.time_from_start = rospy.Duration(1)
        p0 = JointTrajectoryPoint()
        p0.positions = [a0, b0, c0, 0., 0., 0., 0.]
        p0.time_from_start = rospy.Duration(3.)
        p1 = JointTrajectoryPoint()
        p1.positions = [a1, b1, c1, 0., 0., 0., 0.]
        p1.time_from_start = rospy.Duration(10.)
        goal.trajectory.points.append(current_p)
        goal.trajectory.points.append(p0)
        goal.trajectory.points.append(p1)

        # left.move_to_joint_positions({
        #     'left_s1': a0,
        #     'left_e1': b0,
        #     'left_w1': c0,
        # })
        # rospy.sleep(2.)
        global stopped
        stopped = False
        def stop_when_near(data):
            global stopped
            if data.range < 0.2 and not stopped:
                print 'reached threashold. stopping arm'
                action_client.cancel_goal()
                stopped = True

        rospy.Subscriber('/robot/range/left_hand_range/state', Range, stop_when_near)
        goal.trajectory.header.stamp = rospy.Time.now()
        action_client.send_goal(goal)
        rospy.sleep(2.)
        if not action_client.wait_for_result(rospy.Duration(20.)):
            rospy.logdebug("Canceling goal")
            action_client.cancel_goal()
            if action_client.wait_for_result(rospy.Duration(20.)):
                rospy.logdebug('Preempted')
            else:
                rospy.logdebug("Preempt didn't finish")
        print action_client.get_result()

        # rospy.Subscriber('/robot/joint_states', JointState, move_limb(500., lowest))
        # left.move_to_joint_positions({
        #     'left_s1': a1,
        #     'left_e1': b1,
        #     'left_w1': c1,
        # })
        # rospy.spin()
    except Exception, e:
        print e

    return 0

if __name__ == '__main__':
    sys.exit(main())
