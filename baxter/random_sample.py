import rospy
import baxter_interface
import numpy as np

from std_msgs.msg import Header
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from sensor_msgs.msg import Image

from PIL import Image as PIL_Image
from PIL import ImageDraw as PIL_ImageDraw

from motion_routine import *
from model import build_model
from util import *

def main():
    rospy.init_node('execute')
    rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    rs.enable()

    # retrieve images
    global current_image
    def update_image(msg):
        global current_image
        current_image = PIL_Image.frombytes('RGBA', (msg.width, msg.height), msg.data)
        # print msg.width, msg.height, msg.is_bigendian, msg.step, msg.encoding
    rospy.Subscriber('/cameras/left_hand_camera/image', Image, update_image)

    # model
    width = 128
    checkpoint_dir = 'checkpoints-dev-rgb-4-max'
    grasp_class_prediction, logit, grasp_image_ph, keep_prob_ph = build_model(width)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

    arm = baxter_interface.Limb('left')
    arm.move_to_neutral()
    gripper = baxter_interface.Gripper('left')
    gripper.calibrate()

    # grasp crop
    crop_center_x = 330
    crop_center_y = 160
    grasp_class_threashold = 0.5
    scale = 1.0
    crop_width = width * scale
    crop_box = (crop_center_x - crop_width/2, crop_center_y - crop_width/2, crop_center_x + crop_width/2, crop_center_y + crop_width/2)

    # grasp workspace
    x0 = 0.81
    y0 = 0.25
    delta = 0.04
    initial_z = 0.1
    bound_z = -0.165

    grasp_class_threashold = 0.5

    pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
    global force
    def display_gripper_state(msg):
        global force
        force = msg.force

    rospy.Subscriber('/robot/end_effector/left_gripper/state', EndEffectorState, display_gripper_state)

    with tf.Session() as sess:
        restore_vars(saver, sess, checkpoint_dir)

        attemp = 0
        while True:
            # sample a grasp
            dx = np.random.rand() * (2. * delta) - delta
            dy = np.random.rand() * (2. * delta) - delta
            target_theta = (np.random.rand() * 2. - 1.) * 3.059
            target_x = x0 + dx
            target_y = y0 + dy

            # move to the grasp location
            execute_linear(arm, target_x, target_y, initial_z, target_theta)

            # predict grasp
            crop = np.array(current_image.crop(crop_box).resize((width, width)))[:,:,:3]
            grasp_pred = grasp_class_prediction.eval(session=sess, feed_dict={
                grasp_image_ph: crop.reshape((1, width, width, 3)),
                keep_prob_ph: 1.,
            })

            # display image 
            draw = PIL_ImageDraw.Draw(current_image)
            draw.text(crop_box[:2], 'prob: %.5f' % grasp_pred[0, 1])
            draw.text((20, 20), 'grasp force: %.5f' % force)
            if grasp_pred[0, 1] > grasp_class_threashold:
                draw.rectangle(crop_box, outline=(0, 255, 0))
            else:
                draw.rectangle(crop_box, outline=(0, 0, 255))
            msg = Image(
                header=Header(
                    stamp=rospy.Time.now(),
                    frame_id='base',
                ),
                width=640,
                height=400,
                step=640 * 4,
                encoding='bgra8',
                is_bigendian=0,
                data=current_image.tobytes(),
            )
            pub.publish(msg)
            if grasp_pred[0, 1] > grasp_class_threashold:
                execute_planar_grasp(arm, gripper, initial_z, bound_z, target_theta, lower_to_drop=0.05)

            attemp += 1

if __name__ == '__main__':
    main()
