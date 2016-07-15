import tensorflow as tf

from ik_execute import *
from model import build_model
from util import *

from sensor_msgs.msg import Image
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from PIL import Image as PIL_Image
from PIL import ImageDraw as PIL_ImageDraw

import cv2
from cv_bridge import CvBridge

import sys

global current_traj

def lift_arm(arm, z0, original_theta, n_steps=10):
    global current_traj
    # if current_traj:
    #     current_traj.stop()

    traj2 = Trajectory('left')
    s = arm.joint_angles()
    current_pose = arm.endpoint_pose()
    current_position = current_pose['position']
    # theta0 = np.arcsin(current_pose['orientation'].x)
    theta0 = original_theta
    ss = [s[x] for x in traj2._goal.trajectory.joint_names]
    traj2.add_point(ss, 0.)
    for j, joint in enumerate(get_ik_joints_linear(arm, current_position.x, current_position.y, current_position.z, theta0, n_steps)):
        traj2.add_point(joint.position, 0.2 * j + 1.)
    traj2.start()
    return traj2

def ik_solve(x, y, z, z_theta):
    ns = "ExternalTools/left/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest(seed_mode=SolvePositionIKRequest.SEED_CURRENT)
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')

    qx = np.sin(z_theta * 0.5)
    qy = np.cos(z_theta * 0.5)

    pose = PoseStamped(
        header=hdr,
        pose=Pose(
            position=Point( x=x, y=y, z=z, ),
            orientation=Quaternion( x=qx, y=qy, z=0., w=0., ),
        ),
    )
    ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    return resp.joints[0]

def get_ik_joints_linear(arm, target_x, target_y, target_z, target_theta, n_steps):
    ns = "ExternalTools/left/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest(seed_mode=SolvePositionIKRequest.SEED_CURRENT)
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')

    current_pose = arm.endpoint_pose()
    x0 = current_pose['position'].x
    y0 = current_pose['position'].y
    z0 = current_pose['position'].z

    try:
        w2f = convert_joint_to_dict(ik_solve(target_x, target_y, target_z, target_theta))['left_w2']
        w2i = arm.joint_angles()['left_w2']
    except:
        return []

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
                orientation=Quaternion( x=1., y=0., z=0., w=0., ),
            ),
        )
        ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    js = []
    for i, (v, j) in enumerate(zip(resp.isValid, resp.joints)):
        t = (i + 1) * 1. / n_steps
        if v:
            w2 = (1. - t) * w2i + t * w2f
            j.position = j.position[:-1] + (w2,)
            js.append(j)
    return js

def planar_grasp(arm, gripper, target_x, target_y, initial_z, target_z, target_theta, grasp_range_threshold=0.12, n_steps=20):
    # build trajectory
    traj = Trajectory('left')
    # rospy.on_shutdown(traj.stop)
    # current_angles = [arm.joint_angle(joint) for joint in arm.joint_names()]
    # traj.add_point(current_angles, 0.0)

    for i, joint in enumerate(get_ik_joints_linear(arm, target_x, target_y, target_z, target_theta, n_steps)):
        traj.add_point(joint.position, 0.2 + 0.3 * i)

    global sub
    global current_traj
    def near_object(msg):
        # print 'range', msg.range
        if msg.range < grasp_range_threshold:
            global sub
            sub.unregister()
            print 'near object'
            traj.stop()
            gripper.close()
            # wait for the gripper to close
            rospy.sleep(1.)

            # lift arm
            traj2 = Trajectory('left')
            s = arm.joint_angles()
            current_z = arm.endpoint_pose()['position'].z
            ss = [s[x] for x in traj2._goal.trajectory.joint_names]
            traj2.add_point(ss, 0.)
            for j, joint in enumerate(get_ik_joints_linear(arm, target_x, target_y, initial_z, target_theta, n_steps)):
                traj2.add_point(joint.position, 0.2 + 0.4 * j)
            traj2.start()
            current_traj = traj2
            traj2.wait(20.)
            rospy.sleep(1.)
            print 'lifted gripper'

    # if sub == None:
    #     sub = rospy.Subscriber('/robot/range/left_hand_range/state', Range, near_object)

    # move arm
    traj.start()
    current_traj = traj
    traj.wait(20.)

    # close gripper
    gripper.close()
    # wait for the gripper to close
    rospy.sleep(1.)

    # lift arm
    traj2 = Trajectory('left')
    s = arm.joint_angles()
    current_z = arm.endpoint_pose()['position'].z
    ss = [s[x] for x in traj2._goal.trajectory.joint_names]
    traj2.add_point(ss, 0.)
    for j, joint in enumerate(get_ik_joints_linear(arm, target_x, target_y, initial_z, target_theta, n_steps)):
        traj2.add_point(joint.position, 0.2 + 0.2 * j)
    traj2.start()
    current_traj = traj2
    traj2.wait(20.)
    rospy.sleep(2.)
    print 'lifted gripper'

    return traj, sub

def main():
    # model
    width = 128
    checkpoint_dir = 'checkpoints-dev-rgb-4-max'
    grasp_class_prediction, logit, grasp_image_ph, keep_prob_ph = build_model(width)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

    with tf.Session() as sess:
        restore_vars(saver, sess, checkpoint_dir)

        rospy.init_node('execute')
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        rs.enable()

        arm = baxter_interface.Limb('left')
        gripper = baxter_interface.Gripper('left')

        # arm.move_to_neutral()
        gripper.calibrate()

        def open_grippers(msg):
            if msg.state == DigitalIOState.PRESSED:
                gripper.open()

        # camera listener
        # bridge = CvBridge()
        global check_grasp
        global found_grasp
        found_grasp = False
        pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
        crop_center_x = 330
        crop_center_y = 160
        grasp_class_threashold = 0.5
        scale = 1.0
        crop_width = width * scale
        crop_box = (crop_center_x - crop_width/2, crop_center_y - crop_width/2, crop_center_x + crop_width/2, crop_center_y + crop_width/2)
        def classify_image(msg):
            img = PIL_Image.frombytes('RGBA', (msg.width, msg.height), msg.data)
            # img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # cv2.imshow('hello', img)
            # msg.data = img.crop((msg.width - width/2, msg.height - width/2, msg.width + width/2, msg.height + width/2)).tobytes()
            # print np.array(img.crop(crop_box)).shape
            crop = np.array(img.crop(crop_box).resize((width, width)))[:,:,:3]
            grasp_pred = grasp_class_prediction.eval(session=sess, feed_dict={
                grasp_image_ph: crop.reshape((1, width, width, 3)),
                keep_prob_ph: 1.,
            })
            draw = PIL_ImageDraw.Draw(img)
            # print grasp_pred[0, 1]
            global force
            draw.text(crop_box[:2], 'prob: %.5f' % grasp_pred[0, 1])
            draw.text((20, 20), 'grasp force: %.5f' % force)
            if grasp_pred[0, 1] > grasp_class_threashold:
                draw.rectangle(crop_box, outline=(0, 255, 0))
            else:
                draw.rectangle(crop_box, outline=(0, 0, 255))
            msg.data = img.tobytes()
            pub.publish(msg)

            global check_grasp
            global found_grasp
            global best_p
            global best_x
            global best_y
            global best_theta

            if check_grasp and grasp_pred[0, 1] > grasp_class_threashold:
                found_grasp = True
                check_grasp = False
                best_p = grasp_pred[0, 1]
                best_x = target_x
                best_y = target_y
                best_theta = target_theta
                # global current_traj
                # current_traj.stop()
                # current_traj.wait(10.)

                # traj, sub = planar_grasp(arm, gripper, target_x, target_y, initial_z, bound_z, target_theta)
                # # current_traj = traj
                # print 'attempting planar grasp'
                # current_traj = traj
                # traj.wait(20.)

        # best grasp
        global best_p
        global best_x
        global best_y
        global best_theta
        best_p = 0.
        best_x = None
        best_y = None
        best_theta = None

        rospy.Subscriber('/robot/digital_io/left_upper_button/state', DigitalIOState, open_grippers)
        rospy.Subscriber('/cameras/left_hand_camera/image', Image, classify_image)

        global force
        def display_gripper_state(msg):
            global force
            force = msg.force

        rospy.Subscriber('/robot/end_effector/left_gripper/state', EndEffectorState, display_gripper_state)

        # positions
        x0 = 0.81
        y0 = 0.25
        delta = 0.04
        initial_z = 0.1
        bound_z = -0.17
        # n_attempts = int(sys.argv[1])

        while True:
            gripper.open()
            found_grasp = False
            global current_traj
            current_traj = None
            traj = None
            global sub
            sub = None
            # for attempt in xrange(n_attempts):
            attempt = 0
            while not found_grasp:
                attempt += 1
                check_grasp = False
                # if found_grasp:
                #     print 'found grasp'
                #     break
                # sample a grasp
                dx = np.random.rand() * (2. * delta) - delta
                dy = np.random.rand() * (2. * delta) - delta
                target_theta = (np.random.rand() * 2. - 1.) * 3.059
                target_x = x0 + dx
                target_y = y0 + dy
                # target_x = x0
                # target_y = y0
                # target_theta = 0.
                joint = ik_solve(target_x, target_y, initial_z, target_theta)

                print attempt
                print dx, dy, target_theta
                print joint.position

                if sub:
                    sub.unregister()

                if len(joint.position) > 0:
                    # arm.move_to_joint_positions(dict(zip(joint.name, joint.position)))
                    traj2 = Trajectory('left')
                    s = arm.joint_angles()
                    current_position = arm.endpoint_pose()['position']
                    ss = [s[x] for x in traj2._goal.trajectory.joint_names]
                    traj2.add_point(ss, 0.)
                    for j, joint in enumerate(get_ik_joints_linear(arm, target_x, target_y, initial_z, target_theta, 10)):
                        traj2.add_point(joint.position, 0.2 * j + 1.)
                    traj2.start()
                    current_traj = traj2
                    traj2.wait(20.)
                    print 'moved to position'
                    rospy.sleep(1.)
                    check_grasp = True
                    rospy.sleep(1.)

            print 'best grasp', best_p, best_x, best_y, best_theta
            if best_x != None:
                print 'executing best grasp'
                traj2 = Trajectory('left')
                for j, joint in enumerate(get_ik_joints_linear(arm, best_x, best_y, initial_z, best_theta, 10)):
                    traj2.add_point(joint.position, 0.2 * j + 1.)
                traj2.start()
                traj2.wait(30.)
                print 'moved to best grasp'
                # traj2.stop()
                print 'attempting planar grasp'
                traj, sub = planar_grasp(arm, gripper, best_x, best_y, initial_z, bound_z, best_theta)
                # current_traj = traj
                current_traj = traj
                traj.wait(20.)

            print 'done'
            # current_traj = lift_arm(arm, initial_z, target_theta)
            # rospy.spin()
            # current_traj.wait(10.)
            print 'executed trajectory'

    return 0


if __name__ == '__main__':
    main()
