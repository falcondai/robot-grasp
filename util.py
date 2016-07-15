import os
import tensorflow as tf

def convert_joint_to_dict(joint):
    return dict(zip(joint.name, joint.position))

def restore_vars(saver, sess, checkpoint_dir):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.initialize_all_variables())

    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

    path = tf.train.latest_checkpoint(checkpoint_dir)
    if path is None:
        print 'no existing checkpoint found'
        return False
    else:
        print 'restoring from %s' % path
        saver.restore(sess, path)
        return True
