# agent for collecting pushing and grasping data
from pyrobot import Robot
import cv2
from termcolor import colored
import time
import copy
import numpy as np
from pyrobot.utils.util import MoveitObjectHandler, convert_frames, euler_to_quat
from math import pi
import os
from kbhit import KBHit
from cv_bridge import CvBridge


# TODO: pass the arguments using YAML file, it would be much better
class Agent(object):

    def __init__(self, arm_name, camera_name, z_min, arm_base_frame, camera_img_frame, on_top_height, verbose,
                 crop_region_bottom_right, crop_region_top_left, num_angle, depth_bin_thr, erosion_kernel_size,
                 erosion_iter, empty_thr):
        self.arm = Robot(arm_name)
        self.cam = Robot(camera_name)
        self.z_min = z_min
        self.arm_base_frame = arm_base_frame
        self.camera_img_frame = camera_img_frame
        self.on_top_height = on_top_height
        self.verbose = verbose
        self.object_in_hand_state = 1
        self.crop_region = {'top_left': crop_region_top_left, 'bottom_left': crop_region_bottom_right}
        self.num_angle = num_angle          # if set to <= 0, will be considered continues
        self.depth_bin_thr = depth_bin_thr
        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iter = erosion_iter
        self.empty_thr = empty_thr

    def grasp(self, img_pt, grasp_ang):
        # function that executes grasping on real robot
        """
        :param img_pt: [X,Y] pixel in original camera image
        :return : grasp_ang: float top down angle in which grasp should be executed
        """

        # check if gripper is holding any object ###################
        self.arm.gripper.close()
        if self.arm.gripper.state == self.object_in_hand_state:     # TODO: Check if its right
            inp = input(colored("Looks like there is object in hand , remove objects and  hit enter continue\n",
                                "red"))
            time.sleep(5)
        self.arm.gripper.open()

        force_interaction = False
        vis_interaction = False

        # convert 2D target in image to 3D target in robot
        img_pt_3D, _ = list(self.cam.camera.pix_to_3dpt(img_pt[0], img_pt[1], reduce='mean'))
        robot_pt_3D = convert_frames(img_pt_3D, self.camera_img_frame, self.arm_base_frame)

        # orientation #########################
        # assumption is for 0 angle the gripper long edge will be parallel to long edge of table --> CHECKED
        grasp_orientation = list(euler_to_quat(pi, 0.0, grasp_ang))

        # go on top of the point ##############
        if self.verbose:
            print("going on top of the object")
        self.arm.arm.set_ee_pose(plan=True, position=[robot_pt_3D[0], robot_pt_3D[1], self.on_top_height],
                                 orientation=grasp_orientation)

        # go down in cartesian space ##########
        if self.verbose:
            print("going to grab object")
        self.arm.arm.move_ee_xyz(np.array([0.0, 0.0, self.z_min - self.on_top_height]), plan=True)

        # close gripper #######################
        if self.verbose:
            print("closing the gripper")
        self.arm.gripper.close()
        time.sleep(2)

        # come up #############################
        if self.verbose:
            print("coming up")
        self.arm.arm.move_ee_xyz(np.array([0.0, 0.0, self.on_top_height - self.z_min]), plan=True)

        # check gripper condition #############
        grasped = self.arm.gripper.state == self.object_in_hand_state
        if self.verbose:
            print("reward = {}".format(grasped))

        # if it is grasped , place it at some random location ###########
        if grasped:
            if self.verbose:
                print("going to random location to drop the object")
            # get random location on the same plane
            self.arm.arm.move_ee_xyz(np.array([0.1*(np.random.random()-0.5), 0.1*(np.random.random()-0.5), 0.0]),
                                     plan=True)
            self.arm.gripper.open()
            time.sleep(2)

        # come back to mean position ##########
        if self.verbose:
            print("going to mean position")
            self.arm.arm.go_home()

        return grasped

    def push(self):
        #function that executes pushing on real robot
        pass

    def visualizer(self):
        # visualize the action excuted in real world on the image
        pass

    def collect_data(self, action='grasp', mode='random', store_path='', num_data=1000, meta_file='metadata.txt'):
        """

        :param action:
        :param mode: string ['random', 'object_centric']
        :param store_path:
        :param num_data:
        :param meta_file:
        :return:
        """
        # check if path exist ###################
        if not os.path.isdir(store_path):
            os.makedirs(store_path)

        # check if metadata file exist
        if not os.path.isfile(os.path.join(store_path, meta_file)):
            with open(os.path.join(store_path, meta_file), "wb") as f:
                f.write('0')

        with open(os.path.join(store_path, meta_file), "rb") as f:
            count = int(f.read())

        # Remove all the objects from the table ######
        inp = input(colored("Remove all objects on the table and press enter", "red"))
        time.sleep(2)

        # get the mean depth of the img region which is used for data collection #######
        num_capture = 10
        for i in range(num_capture):
            if i == 0:
                self.mean_depth = self._crop_img(self.cam.camera.get_depth().astype(np.float32))
            else:
                self.mean_depth += self._crop_img(self.cam.camera.get_depth().astype(np.float32))

        self.mean_depth /= num_capture

        self.arm.arm.go_home()
        kbhit = KBHit()
        while count < num_data:
            # give the option to pause process in between to rearrange objects ######
            if kbhit.kbhit():
                c = kbhit.getch()
                if c == 'p':
                    inp = raw_input(
                        colored("Looks like you have paused the process, place objects and press s key and hit enter "
                                "continue\n", "red"))
                    while inp != 's':
                        inp = raw_input(colored(
                                "Looks like you have pressed {} key, press s key and hit enter to continue".format(
                                    inp),"red"))

            # check if there are objects on table #######
            if self._table_empty():
                input(colored("Looks like there is no object on table, place some objects and press enter","red"))

            # sample location based on mode
            if mode == 'random':
                loc = [np.random.randint(self.crop_region['bottom_left'][0], self.crop_region['top_right'][0]),
                       np.random.randint(self.crop_region['bottom_left'][1], self.crop_region['top_right'][1])]
            elif mode == 'object_centric':
                pass
            else:
                raise ValueError("mode can either be 'random' or 'object centric'")

            # sample angle ##############
            if self.num_angle <= 0:
                ang = np.random.random() * 2 * pi
            else:
                ang = np.random.randint(0,self.num_angle) * 2 * pi / float(self.num_angle)

            # perform action based on the action category ######
            if action == 'grasp':
                success = self.grasp(loc, ang)
            elif action == 'push':
                pass
            else:
                raise ValueError("action can either be 'grasp' or 'push'")

            print("\n############")
            print("collected transition num = " + colored('{:04d}'.format(count), 'green'))
            print("############")

            # return should be based on the action we are performing

    def _org_2_crop(self, pt):
        """
        helpful for converting image point from original image to cropped region
        :param pt: [X,Y]
        :return: [cropped_X, cropped_Y]
        """
        return [pt[0]-self.crop_region['top_left'][0], pt[1]-self.crop_region['top_left'][1]]

    def _crop_2_org(self, pt):
        """
        helpful for converting image point from original image to cropped region
        :param pt: [cropped_X, cropped_Y]
        :return: [X, Y]
        """
        return [pt[0] + self.crop_region['top_left'][0], pt[1] + self.crop_region['top_left'][1]]

    def _crop_img(self, img):
        """

        :param img: np.array([width, height, channel])
        :return: cropped img
        """
        return img[self.crop_region['top_left'][0]:self.crop_region['bottom_right'][0],
               self.crop_region['top_left'][1]:self.crop_region['bottom_right'][1]]

    def _table_empty(self):
        """
        :return: True if table is empty .. otherwise False
        """
        diff_erosion = self.get_depth_bin()

        if diff_erosion.mean() > self.empty_thr:
            return False
        else:
            return True

    def _get_depth_bin(self):
        """
        :return: returns the binary uint8{0,255} image out
        """
        diff = self.mean_depth.astype(np.float32) - self.cam.camera.depth.astype(np.float32)
        diff[diff < self.depth_bin_thr ] = 0
        diff[diff != 0] = 255
        diff = diff.astype(np.uint8)

        # erode the image
        kernel = np.ones((self.erosion_kernel_size, self.erosion_kernel_size), np.uint8)
        diff_erosion = cv2.erode(diff, kernel, iterations = self.erosion_iter)

        # dilate
        diff_erosion = cv2.dilate(diff_erosion, kernel, iterations = self.erosion_iter)
        return copy.deepcopy(diff_erosion)

    # for visualizing grasp
    # need to run them in background
    def _vis_grasp(self, img, loc, grasp_angle):
        # input cv_image -->[h,w,c]
        vis_img = img.copy()
        gsize = 100
        grasp_l = gsize/2.5
        grasp_w = gsize/5.0
        points = np.array([[-grasp_l, -grasp_w],
                           [grasp_l, -grasp_w],
                           [grasp_l, grasp_w],
                           [-grasp_l, grasp_w]])
        R = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
                      [np.sin(grasp_angle), np.cos(grasp_angle)]])
        rot_points = np.dot(R, points.transpose()).transpose()
        im_points = rot_points + np.array([loc[1],loc[0]])          # indices swaped to transfer numpy pt to cv pt
        cv2.line(vis_img, tuple(im_points[0].astype(int)), tuple(im_points[1].astype(int)), color=(0,255,0), thickness=5)
        cv2.line(vis_img, tuple(im_points[1].astype(int)), tuple(im_points[2].astype(int)), color=(0,0,255), thickness=5)
        cv2.line(vis_img, tuple(im_points[2].astype(int)), tuple(im_points[3].astype(int)), color=(0,255,0), thickness=5)
        cv2.line(vis_img, tuple(im_points[3].astype(int)), tuple(im_points[0].astype(int)), color=(0,0,255), thickness=5)
        return vis_img

    # for visualizing push
    def _pish_vis(self, img):