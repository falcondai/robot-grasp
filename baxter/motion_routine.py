import sys

import rospy
import actionlib
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import JointTrajectoryPoint
from util import convert_joint_to_dict

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

def get_ik_joints_linear(initial_x, initial_y, initial_z, initial_w2,
target_x, target_y, target_z, target_w2, n_steps):
    ns = "ExternalTools/left/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest(seed_mode=SolvePositionIKRequest.SEED_CURRENT)
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')

    # current_pose = arm.endpoint_pose()
    x0 = initial_x
    y0 = initial_y
    z0 = initial_z

    # linear interpolate between current pose and target pose
    for i in xrange(n_steps):
        t = (i + 1) * 1. / n_steps
        x = (1. - t) * x0 + t * target_x
        y = (1. - t) * y0 + t * target_y
        z = (1. - t) * z0 + t * target_z

        pose = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point( x=x, y=y, z=z, ),
                # endeffector pointing down
                orientation=Quaternion( x=1., y=0., z=0., w=0., ),
            ),
        )
        ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return []

    js = []
    # control w2 separately from other joints
    for i, (v, j) in enumerate(zip(resp.isValid, resp.joints)):
        t = (i + 1) * 1. / n_steps
        if v:
            w2 = (1. - t) * initial_w2 + t * target_w2
            j.position = j.position[:-1] + (w2,)
            js.append(j)
    return js

def get_linear_trajectory_from_current(arm, target_x, target_y, target_z, target_w2, n_steps, duration):
    traj = Trajectory('left')
    x = arm.joint_angles()
    current_pose = arm.endpoint_pose()
    current_position = current_pose['position']
    w2i = x['left_w2']
    dt_step = duration * 1. / n_steps
    for j, joint in enumerate(get_ik_joints_linear(current_position.x, current_position.y, current_position.z, w2i, target_x, target_y, target_z, target_w2, n_steps)):
        traj.add_point(joint.position, dt_step * (j + 1))
    return traj

def execute_linear(arm, target_x, target_y, target_z, target_w2, n_steps=10, duration=4., timeout=10.):
    initial_w2 = arm.joint_angles()['left_w2']
    pos = arm.endpoint_pose()['position']
    traj = get_linear_trajectory_from_current(arm, target_x, target_y, target_z, target_w2, n_steps, duration)
    traj.start()
    traj.wait(timeout)
    return traj

def execute_vertical(arm, target_z, n_steps=10, duration=4., timeout=10.):
    initial_w2 = arm.joint_angles()['left_w2']
    pos = arm.endpoint_pose()['position']
    return execute_linear(arm, pos.x, pos.y, target_z, initial_w2, n_steps, duration, timeout)

def execute_horizontal(arm, target_x, target_y, target_w2, n_steps=10, duration=4., timeout=10.):
    initial_w2 = arm.joint_angles()['left_w2']
    pos = arm.endpoint_pose()['position']
    return execute_linear(arm, target_x, target_y, pos.z, target_w2, n_steps, duration, timeout)

def execute_planar_grasp(arm, gripper, initial_z, target_z, target_w2, n_steps=10, duration=4., timeout=10., sleep=1., lower_to_drop=0.):
    execute_vertical(arm, target_z, n_steps, duration, timeout)
    gripper.close()
    rospy.sleep(sleep)
    execute_vertical(arm, initial_z, n_steps, duration, timeout)
    if lower_to_drop > 0:
        execute_vertical(arm, target_z + lower_to_drop, n_steps, duration, timeout)
    gripper.open()
    rospy.sleep(sleep)

if __name__ == '__main__':
    x0 = 0.81
    y0 = 0.25
    delta = 0.04
    initial_z = 0.1
    bound_z = -0.165

    for _ in xrange(10):
        execute_linear(arm, x0, y0, initial_z, 0.)
        execute_planar_grasp(arm, gripper, initial_z, bound_z, 0., lower_to_drop=0.05)
