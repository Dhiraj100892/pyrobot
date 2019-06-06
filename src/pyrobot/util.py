# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import PyKDL as kdl
import numpy as np
import rospy
import tf
import tf_conversions
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
import moveit_commander
from moveit_commander import conversions

def get_tf_transform(tf_listener, tgt_frame, src_frame):
    """
    Uses ROS TF to lookup the current transform from tgt_frame to src_frame,
    If the returned transform is applied to data, it will transform data in
    the src_frame into the tgt_frame

    :param tgt_frame: target frame
    :param src_frame: source frame
    :type tgt_frame: string
    :type src_frame: string

    :returns: trans, translation (x,y,z)
    :rtype: tuple (of floats)
    :returns: quat, rotation as a quaternion (x,y,z,w)
    :rtype: tuple (of floats)
    """
    try:
        tf_listener.waitForTransform(tgt_frame, src_frame,
                                     rospy.Time(0),
                                     rospy.Duration(3))
        (trans, quat) = tf_listener.lookupTransform(tgt_frame,
                                                    src_frame,
                                                    rospy.Time(0))
    except (tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException):
        raise RuntimeError('Cannot fetch the transform from'
                           ' {0:s} to {1:s}'.format(tgt_frame, src_frame))
    return trans, quat


def transform_to_ros_pose(trans, quat, frame):
    """
    Converts transformation (translation, quaternion) into ROS PoseStamped

    :param trans: translation (x,y,z)
    :param quat: rotation as a quaternion (x,y,z,w)
    :param frame: reference frame of pose
    :type trans: tuple (of floats)
    :type quat: tuple (of floats)
    :type frame: string

    :returns: pose
    :rtype: ROS geometry_msgs/PoseStamped
    """
    pose = PoseStamped()
    pose.pose = tf_conversions.Pose(trans, quat)
    pose.header.frame_id = frame
    return pose


def quat_to_rot_mat(quat):
    """
    Convert the quaternion into rotation matrix. The quaternion we used
    here is in the form of [x, y, z, w]

    :param quat: quaternion [x, y, z, w] (shape: :math:`[4,]`)
    :type quat: numpy.ndarray

    :return: the rotation matrix (shape: :math:`[3, 3]`)
    :rtype: numpy.ndarray
    """
    return tf.transformations.quaternion_matrix(quat)[:3, :3]


def euler_to_quat(euler):
    """
    Convert the yaw, pitch, roll into quaternion.

    :param euler: the yaw, pitch, roll angles (shape: :math:`[3,]`)
    :type quat: numpy.ndarray

    :return: quaternion [x, y, z, w] (shape: :math:`[4,]`)
    :rtype: numpy.ndarray
    """
    return tf.transformations.quaternion_from_euler(euler[0], euler[1],
                                                    euler[2], axes='rzyx')


def rot_mat_to_quat(rot):
    """
    Convert the rotation matrix into quaternion.

    :param quat: the rotation matrix (shape: :math:`[3, 3]`)
    :type quat: numpy.ndarray

    :return: quaternion [x, y, z, w] (shape: :math:`[4,]`)
    :rtype: numpy.ndarray
    """
    R = np.eye(4)
    R[:3, :3] = rot
    return tf.transformations.quaternion_from_matrix(R)


def kdl_array_to_numpy(kdl_data):
    np_array = np.zeros((kdl_data.rows(), kdl_data.columns()))
    for i in range(kdl_data.rows()):
        for j in range(kdl_data.columns()):
            np_array[i, j] = kdl_data[i, j]
    return np_array


def joints_to_kdl(joint_values):
    """
    Convert the numpy array into KDL data format

    :param joint_values: values for the joints
    :return: kdl data type
    """
    num_jts = joint_values.size
    kdl_array = kdl.JntArray(num_jts)
    for idx in range(num_jts):
        kdl_array[idx] = joint_values[idx]
    return kdl_array


def kdl_frame_to_numpy(frame):
    """
    Convert KDL Frame data into numpy array
    :param frame: data of KDL Frame
    :return: transformation matrix in numpy [4x4]
    """
    p = frame.p
    M = frame.M
    return np.array([[M[0, 0], M[0, 1], M[0, 2], p.x()],
                     [M[1, 0], M[1, 1], M[1, 2], p.y()],
                     [M[2, 0], M[2, 1], M[2, 2], p.z()],
                     [0, 0, 0, 1]])


class MoveitObjectHandler(object):
    '''
    Use this class to create objects that reside in moveit environments
    '''
    def __init__(self):

        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()

    def add_world_object(self, id_name, pose, size):
        '''
        Adds the particular object to the moveit planning scene
        '''
        assert type(size) is tuple, 'size should be tuple'
        assert len(size)==3, 'size should be of length 3'
        pose = conversions.list_to_pose(pose)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = '/base'
        pose_stamped.pose = pose
        self.scene.add_box(id_name, pose_stamped, size)
        rospy.sleep(1.0)
        self.scene.add_box(id_name, pose_stamped, size)

    def remove_world_object(self, id_name):
        '''
        Removes a specified object for the Moveit planning scene
        ''' 
        self.scene.remove_world_object(id_name)
        self.scene.remove_world_object(id_name)

    def attach_arm_object(self, link_name, id_name, pose, size):
        '''
        Attaches the specified object to the robot
        '''
        assert type(size) is tuple, 'size should be tuple'
        assert len(size)==3, 'size should be of length 3'
        pose = conversions.list_to_pose(pose)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = link_name
        pose_stamped.pose = pose
        self.scene.attach_box(link_name, id_name, pose_stamped, size)
        rospy.sleep(1.0)
        self.scene.attach_box(link_name, id_name, pose_stamped, size)

    def detach_arm_object(self, link_name, id_name, remove_from_world=True):
        '''
        Detaches an object earlier attached to the robot
        '''
        self.scene.remove_attached_object(link_name, id_name)
        self.scene.remove_attached_object(link_name, id_name)
        if remove_from_world is True:
            self.remove_world_object(id_name)

    def remove_all_objects(self):
        '''
        Removes all the objects in the current Moveit planning scene
        '''
        ## get add objects
        dict_obj = self.scene.get_objects()
        ## get attach object
        dict_attach_obj = self.scene.get_attached_objects()
        ## remove add objects
        for i in dict_obj.keys():
            self.remove_world_object(i)
        ## remove attached objects
        for i in dict_attach_obj.keys():
            self.detach_arm_object(dict_attach_obj[i].link_name,i)

    def add_table(self, table_yaml=None):
        '''
        This adds a table in the planning scene.
        table_yaml is a yml file describing the pose and size of the table.
        '''
        if table_yaml is not None:
            import yaml
            table_d = yaml.load(open(table_yaml, 'r'))
            self.add_world_object('table', 
                pose=table_d['pose'], 
                size=tuple(table_d['size']))
        else:
            # Default table.
            print('Since table_yaml not supplemented, creating default table.')
            self.add_world_object('table', 
                pose=[0.8,0.0,-0.23,0.,0.,0.,1.],
                size=(1.35,2.0,0.1))

    def add_kinect(self, kinect_yaml=None):
        '''
        This adds a kinect in the planning scene.
        kinect_yaml is a yml file describing the pose and size of the table.
        '''
        if kinect_yaml is not None:
            import yaml
            kinect_d = yaml.load(open(kinect_yaml, 'r'))
            self.add_world_object('kinect', 
                pose=kinect_d['pose'], 
                size=tuple(kinect_d['size']))
        else:
            # Default kinect.
            print('Since kinect_yaml not supplemented, creating default kinect.')
            self.add_world_object('kinect', 
                pose=[0., 0.0,0.75,0.,0.,0.,1.], 
                size=(0.25,0.25,0.3))

    def add_gripper(self, gripper_yaml=None):
        '''
        Attaches gripper to right_gripper link.
        '''
        if gripper_yaml is not None:
            import yaml
            gripper_d = yaml.load(open(gripper_yaml, 'r'))
            self.attach_arm_object('right_gripper',
                'gripper', 
                pose=gripper_d['pose'], 
                size=tuple(gripper_d['size']))
        else:
            # Default gripper.
            print('Since gripper_yaml not supplemented, creating default gripper.')
            self.attach_arm_object('right_gripper',
                'gripper', 
                pose=[0., 0.0, 0.07,0.,0.,0.,1.], 
                size=(0.02,0.1,0.07))

    def remove_table(self):
        '''
        Removes table object from the planning scene
        '''
        self.remove_world_object('table')

    def remove_gripper(self):
        '''
        Removes table object from the planning scene
        '''
        self.detach_object('gripper')
        rospy.sleep(0.2)
        self.detach_object('gripper')
        rospy.sleep(0.2)
        self.remove_world_object('gripper')
        rospy.sleep(0.2)
        self.remove_world_object('gripper')
        

