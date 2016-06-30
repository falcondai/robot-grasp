import argparse
import os
import sys

import rospy

import baxter_interface
import actionlib

from baxter_interface import CHECK_VERSION
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from baxter_core_msgs.msg import DigitalIOState
from sensor_msgs.msg import JointState, Range, Image
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from PIL import Image as PIL_Image
from PIL import ImageDraw as PIL_ImageDraw
import numpy as np



class Trajectory(object):
    def __init__(self, limb):
        ns = 'robot/limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.1)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear(limb)

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()
        rospy.sleep(0.1)

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

def get_ik_joints(target_x, target_y, initial_z, target_z, n_steps):
    ns = "ExternalTools/left/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest(seed_mode=SolvePositionIKRequest.SEED_CURRENT)
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    for z in np.arange(initial_z, target_z, (target_z-initial_z)/n_steps):
        pose = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point( x=target_x, y=target_y, z=z, ),
                orientation=Quaternion( x=0., y=1., z=0., w=0., ),
            ),
        )
        ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    return [j for v, j in zip(resp.isValid, resp.joints) if v]


def main():
    rospy.init_node('execute')
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    rs.enable()

    arm = baxter_interface.Limb('left')
    gripper = baxter_interface.Gripper('left')

    gripper.calibrate()

    def open_grippers(msg):
        if msg.state == DigitalIOState.PRESSED:
            gripper.open()
    rospy.Subscriber('/robot/digital_io/left_upper_button/state', DigitalIOState, open_grippers)

    # target position
    target_x = 0.7
    target_y = 0.3
    initial_z = -0.1
    target_z = -0.4

    # build trajectory
    traj = Trajectory('left')
    rospy.on_shutdown(traj.stop)
    current_angles = [arm.joint_angle(joint) for joint in arm.joint_names()]
    traj.add_point(current_angles, 0.0)

    for i, joint in enumerate(get_ik_joints(target_x, target_y, initial_z, target_z, 100)):
        traj.add_point(joint.position, 5. + 0.1 * i)

    global sub
    def near_object(msg):
        # print 'range', msg.range
        if msg.range < 0.17:
            global sub
            sub.unregister()
            print 'near object'
            traj.stop()
            gripper.close()
            rospy.sleep(1.)

            # pick up
            traj2 = Trajectory('left')
            s = arm.joint_angles()
            current_z = arm.endpoint_pose()['position'].z
            ss = [s[x] for x in traj2._goal.trajectory.joint_names]
            traj2.add_point(ss, 0.)
            for j, joint in enumerate(get_ik_joints(target_x, target_y, current_z, initial_z, 10)):
                traj2.add_point(joint.position, 0.2 * j)
            traj2.start()
            # gripper.open()

    sub = rospy.Subscriber('/robot/range/left_hand_range/state', Range, near_object)
    # move arm
    # arm.move_to_joint_positions(initial_pose)
    traj.start()
    rospy.spin()
    print 'executed trajectory'

    return 0


if __name__ == '__main__':
    main()
