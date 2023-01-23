import math
import time
import cv2
import numpy as np
import waypoints_detection as wpd
from datetime import datetime
import os
from robomaster import robot
import queue
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from threading import Thread
import gdown
import common_utils
from moviepy.editor import VideoFileClip


class MyVideoCapture:
    def __init__(self, stream):
        self.cap = cv2.VideoCapture(stream)
        self.q = queue.Queue()
        t = Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        if not (self.cap.isOpened()):
            print("Could not open video device #__ ", self.cap)
            # continue # break
            return
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            time.sleep(0.03)  # 30Hz? #ran

    def read(self):
        return self.q.get()

class DarkRoom:
    def __init__(self):

        self.frame_index = 0
        self.frame_index_only_increasing = 0
        self.simulate_cameras = True
        self.work_with_real_robot = False
        self.main_output_folder = './autonomic_driving_output'
        self.path4simframes = './videos_and_images/images/'
        self.max_eps = 80
        self.cameras_names = ['UOXBGL', 'TGASLM', 'VSYAJL', 'IGXJVS']
        self.cameras_IPs = ['192.168.0.233', '192.168.0.76', '192.168.0.79', '192.168.0.55']
        if not self.work_with_real_robot:
            self.videos_and_images_folder = './videos_and_images'
            self.download_input_videos_from_google_drive(self.videos_and_images_folder)
            common_utils.extract_frames_from_videos(self.videos_and_images_folder)
        self.writePNGs = True
        self.counter = 0
        self.images_output_folder_with_data, self.videos_output_folder_with_data = self.create_images_and_videos_output_folders()
        self.vcaps_list = self.initialize_vcaps()
        self.ninja1 = self.initialize_robot()
        self.num_of_cameras = len(self.vcaps_list)
        self.simulated_frames_indexes = self.create_simulated_frames_indexes(first_frame_index=430, step=5)
        self.base_cameras_frames, self.base_cameras_frames_with_data = self.get_cameras_frames(frame_index=0)
        self.waypoints_calculated = [False] * self.num_of_cameras
        self.waypoints_for_all_cameras = [[]] * self.num_of_cameras
        self.start_waypoint_indexes = np.zeros(self.num_of_cameras, int)
        self.active_camera_index = -1
        self.prev_active_camera_index = -1
        self.cameras_where_robot_finished = [False] * self.num_of_cameras
        self.first_camera_index = None
        self.robot_current_position = None
        self.robot_prev_position = None
        self.robot_current_unit_direction = None
        self.robot_prev_unit_direction = None
        self.image_width = 1920
        self.image_height = 1088
        self.diff_yaw_from_image_angle_in_all_cameras = np.zeros(self.num_of_cameras)
        self.first_time = True
        self.first_time_each_camera = [True] * self.num_of_cameras
        self.current_waypoint_index = None
        self.next_waypoint_index = None
        self.next_waypoint_index_list = []
        self.all_cameras_frames = None
        self.all_cameras_frames_with_frame_index = None
        self.calc_waypoints_again_per_camera = [False] * self.num_of_cameras
        self.aruco_detected_per_camera = [False] * self.num_of_cameras
        self.active_cameras_each_frame = []
        self.is_bridge = False
        self.robot_contour = None
        self.is_robot_found_by_aruco = False
        self.min_num_of_frames = None
        self.prev_pitch = None
        self.prev_yaw = None
        self.robot_yaw = None
        self.robot_pitch = None
        self.robot_roll = None
        self.robot_start_pitch = None
        self.first_time_getting_robot_pitch = True
        self.continue_straight_text = ""
        self.robot_first_position = None
        self.robot_first_unit_direction = None
        self.straight_distance_to_move_in_pixels = 600
        self.num_of_pixels_in_one_meter = 500
        self.robot_3d_current_position = None
        self.robot_3d_prev_position = None
        self.moving_in_pitch = False
        self.first_time_reading_pitch = True
        self.waiting_for_completed = False
        self.first_time_moving = True
        self.create_gif_video = False
        #self.init_pitch_control()

    def download_input_videos_from_google_drive(self, videos_and_images_folder):
        videos_sub_folder_name = 'videos'
        videos_sub_folder_full_path = videos_and_images_folder + '/' + videos_sub_folder_name

        is_folder_exist = os.path.exists(videos_sub_folder_full_path)
        if is_folder_exist:
            return
        else:
            os.makedirs(videos_sub_folder_full_path)

        google_drive_prefix_url = 'https://drive.google.com/uc?id='

        url1_id_UOXBGL = '1-0STdItDTaGQbmMfrljidEZdaDhsqfw1'
        url2_id_TGASLM = '1cEb91AWmzwiNN4q55ghYcTUIdQM3Jo4s'
        url3_id_VSYAJL = '1S9wYkk_6vm17fsbaj6keftYsqtTuodmi'
        url4_id_IGXJVS = '170dVWpcfFjLaBjQKRvXT0Njx-NdgCqeE'

        url_ids = [url1_id_UOXBGL, url2_id_TGASLM, url3_id_VSYAJL, url4_id_IGXJVS]

        num_of_videos = len(self.cameras_names)

        for video_index in range(0, num_of_videos):
            video_name = self.cameras_names[video_index] + '.avi'
            print()
            print(f'Downloading video number {video_index + 1} out of {num_of_videos}. video name: {video_name}')
            url_id = url_ids[video_index]
            url = google_drive_prefix_url + url_id
            output = videos_sub_folder_full_path + '/' + video_name
            gdown.download(url, output, quiet=False)
        print('Finished downloading all videos')

    def init_pitch_control(self):
        t = Thread(target=self.special_move_at_incline)
        print('started thread')
        t.daemon = True
        t.start()
    def special_move_at_incline(self):
        state = True
        while state:
            if self.first_time_reading_pitch:
                if self.robot_pitch is not None:
                    start_pitch = self.robot_pitch
                    self.first_time_reading_pitch = False
            pitch = self.robot_pitch
            time.sleep(0.05)
            if pitch is not None and start_pitch is not None and not self.waiting_for_completed:
                print(f'start_pitch = {start_pitch}')
                delta_pitch = pitch - start_pitch
                #print('delta_pitch = ' + str(delta_pitch))
                #if delta_pitch > 5:
                while delta_pitch > 5:
                    print(f'delta_pitch = {delta_pitch}')
                    self.moving_in_pitch = True
                    #print('moving up @ ' + str(pitch))
                    #self.ninja1.chassis.drive_speed(1, 0, 0)
                    pitch = self.robot_pitch
                    delta_pitch = pitch - start_pitch
                    time.sleep(0.1)
                self.moving_in_pitch = False
                self.ninja1.chassis.drive_speed(0, 0, 0)
                # elif delta_pitch < -5:
                while delta_pitch < -5:
                    self.moving_in_pitch = True
                    print('moving down @ ' + str(pitch))
                    pitch = self.robot_pitch
                    delta_pitch = pitch - start_pitch
                    time.sleep(0.1)
                self.moving_in_pitch = False
                self.ninja1.chassis.drive_speed(0, 0, 0)

    def sub_info_handler_att(self, attitude_info):
        self.robot_yaw, self.robot_pitch, self.robot_roll = attitude_info
        # print('self.robot_pitch:' + str(self.robot_pitch))
        if self.first_time_getting_robot_pitch:
            self.robot_start_pitch = self.robot_pitch
            self.first_time_getting_robot_pitch = False

    def sub_info_handler_pos(self, position_info):
        pos_x, pos_y, pos_z = position_info
        self.robot_3d_current_position = np.array((pos_x, pos_y, pos_z))

    def init_ip_cam(self, cameras_names, cameras_IPs):
        vcap = [MyVideoCapture("rtsp://admin:{id}@{ip}:554/H.264".format(id=njnj_id, ip=njnj_IP)) \
                for njnj_id, njnj_IP in zip(cameras_names, cameras_IPs)]
        return vcap

    def initialize_vcaps(self):
        if self.simulate_cameras:
            vCaps = [0, 0, 0, 0]
        else:
            vCaps = self.init_ip_cam(self.cameras_names, self.cameras_IPs)
        return vCaps

    def initialize_robot(self):
        if self.work_with_real_robot:
            ninja1 = robot.Robot()
            ninja1.initialize(conn_type="ap")
            ninja1.chassis.sub_attitude(freq=10, callback=self.sub_info_handler_att)
            ninja1.chassis.sub_position(freq=10, callback=self.sub_info_handler_pos, cs=0)
        else:
            ninja1 = []
        return ninja1

    def get_min_num_of_frames_for_simulation(self):
        cameras_names = os.listdir(self.path4simframes)
        num_of_images_per_camera = []
        for camera_name in cameras_names:
            subfolder_full_path = os.path.join(self.path4simframes, camera_name)
            images_files = sorted(glob.glob(subfolder_full_path + '/*.jpg'))
            num_of_images_for_current_camera = len(images_files)
            num_of_images_per_camera.append(num_of_images_for_current_camera)
        num_of_images = min(num_of_images_per_camera)
        return num_of_images

    def create_simulated_frames_indexes(self, first_frame_index=430, step=20):
        simulated_frames_indexes = []
        if self.simulate_cameras:
            self.min_num_of_frames = self.get_min_num_of_frames_for_simulation()
            simulated_frames_indexes = np.arange(first_frame_index, self.min_num_of_frames, step)
        return simulated_frames_indexes



    def create_images_and_videos_output_folders(self):
        now = datetime.now()
        now = now.strftime("date_%m_%d_%Y__time_%H_%M_%S")
        images_output_folder_with_data = self.main_output_folder + '/' + now + '/images'
        videos_output_folder_with_data = self.main_output_folder + '/' + now + '/videos'
        self.create_folder_if_not_exist(images_output_folder_with_data)
        self.create_folder_if_not_exist(videos_output_folder_with_data)
        return images_output_folder_with_data, videos_output_folder_with_data


    def create_folder_if_not_exist(self, folder_full_path):
        isExist = os.path.exists(folder_full_path)
        if not isExist:
            os.makedirs(folder_full_path)

    def get_cameras_frames(self, frame_index):
        num_of_cameras = len(self.cameras_names)
        cameras_frames = [None] * num_of_cameras
        cameras_frames_with_data = [None] * num_of_cameras
        for camera_index in range(0, num_of_cameras):
            camera_name = self.cameras_names[camera_index]
            if self.simulate_cameras:
                frame, frame_with_data = self.get_simulated_frame(camera_name, frame_index, self.simulated_frames_indexes)
                cameras_frames_with_data[camera_index] = frame_with_data
            else:
                vcap_single = self.vcaps_list[camera_index]
                time.sleep(0.1)
                frame = vcap_single.read()
                frame_with_data = frame.copy()
                frame_with_data = wpd.write_text_on_frame(frame_with_data, f'{frame_index}', (50, 50))
                cameras_frames_with_data[camera_index] = frame_with_data
            cameras_frames[camera_index] = frame
        return cameras_frames, cameras_frames_with_data

    def get_simulated_frame(self, njnj_id, frame_index, simulated_frames_indexes):
        if njnj_id == 'UOXBGL':
            frame_vid_ind = simulated_frames_indexes[frame_index]
        elif njnj_id == 'TGASLM':
            frame_vid_ind = simulated_frames_indexes[frame_index]
        elif njnj_id == 'VSYAJL':
            frame_vid_ind = simulated_frames_indexes[frame_index]
        elif njnj_id == 'IGXJVS':
            frame_vid_ind = simulated_frames_indexes[frame_index]

        frame_vid_ind += 1
        temp_name = f"{frame_vid_ind:05}" + '.jpg'
        image_full_path = self.path4simframes + njnj_id + '/' + temp_name
        frame = cv2.imread(image_full_path)

        frame_with_frame_index = frame.copy()
        frame_with_frame_index = wpd.write_text_on_frame(frame_with_frame_index,
                                                         f'{simulated_frames_indexes[frame_index]}, {self.frame_index_only_increasing}',
                                                         (50, 50))

        return frame, frame_with_frame_index

    def get_eps(self, next_WP, current_robo_pos):
        dist = np.linalg.norm(next_WP - current_robo_pos)
        return dist

    def get_azimuth(self, vector_1, vector_2):
        if np.linalg.norm(vector_1) == 0:
            unit_vector_1 = vector_1
        else:
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        if np.linalg.norm(vector_2) == 0:
            unit_vector_2 = vector_2
        else:
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        azi = np.arccos(dot_product) * 180 / np.pi  # *rad2deg # should be positive number 0-180deg
        # if azi<0:
        #     print("azi received negative number! ? singleWP , move_vec are : ", single_WP, Move_vec)

        # https://www.geeksforgeeks.org/how-to-compute-the-cross-product-of-two-given-vectors-using-numpy/
        cross = np.cross(vector_1, vector_2)
        if cross < 0:
            # required rotete right from vec1 to vec2 (from current pos to desired WP)
            pass
        else:
            # required rotete left from vec1 to vec2 (from current pos to desired WP)
            azi = -azi

        # todo ? - azi so far is only by point vector from prev to current.
        #  consider and add/trim aruco vector relative to move_vec. negative cross is pos addition..

        return azi

    # def handle_robot_rotation(self, vector_1, vector_2):
    #     azimuth = self.get_azi(vector_1, vector_2)
    #     return azimuth

    def move_robot(self, azimuth, dist_in_pixels):
        if self.work_with_real_robot:
            #yaw = self.robot_yaw
            #pitch = self.robot_pitch
            #roll = self.robot_roll
            #start_pitch = self.robot_start_pitch
            #delta_pitch = pitch - start_pitch
            # if delta_pitch > 5:
            #     print('on ramp')
            # else:
            #     print('strait')
            #
            # print(f'yaw = {yaw}')
            # print(f'pitch = {pitch}')
            # print(f'roll = {roll}')
            # print()

            speed_x = 0.3
            speed_azimuth = 20
            num_of_meters = dist_in_pixels / self.num_of_pixels_in_one_meter
            if self.robot_pitch is not None and self.prev_pitch is not None and self.robot_pitch - self.prev_pitch > 5:
                print(f'self.prev_pitch = {self.prev_pitch}')
                print(f'self.robot_pitch = {self.robot_pitch}')
                self.ninja1.chassis.drive_speed(1, 0, 0)
            else:
                print('regular move')
                # if azimuth < 0:
                #     speed_azimuth_with_sign = speed_azimuth
                # else:
                #     speed_azimuth_with_sign = -speed_azimuth
                # sleep_time_rotation = math.fabs(azimuth / speed_azimuth)
                # self.ninja1.chassis.drive_speed(0, 0, speed_azimuth_with_sign)
                # time.sleep(sleep_time_rotation)
                # if not self.moving_in_pitch:
                #     self.ninja1.chassis.drive_speed(0, 0, 0)
                self.waiting_for_completed = True
                self.ninja1.chassis.move(x=0, y=0, z=azimuth, z_speed=100).wait_for_completed()
                self.waiting_for_completed = False


                self.waiting_for_completed = True
                if self.current_waypoint_index == (len(self.waypoints_for_all_cameras[self.active_camera_index]) - 1) and self.active_camera_index == 0:
                    self.ninja1.chassis.move(x=0, y=0, z=5, z_speed=100).wait_for_completed()
                    print('driving faaaaaast!')
                    self.ninja1.chassis.drive_speed(1, 0, 0, timeout=2)
                    time.sleep(2)
                elif self.current_waypoint_index == 0 and self.active_camera_index == 3:
                        print('driving faaaaaast!')
                        self.ninja1.chassis.drive_speed(1, 0, 0, timeout=1)
                        time.sleep(1)
                else:
                    self.ninja1.chassis.move(x=num_of_meters, y=0, z=0, xy_speed=1).wait_for_completed()
                self.waiting_for_completed = False

    def create_rgb_image_with_data(self, rgb_image, single_WP, body_pix, prev_best_robo_pos, waypoints, Az, robot_data, eps):
        rgb_image_with_data = rgb_image.copy()
        if len(robot_data) > 0:
            front_aruco_mean_location = robot_data['robot_mean_aruco_locations']['front_aruco']
            back_aruco_mean_location = robot_data['robot_mean_aruco_locations']['back_aruco']
        else:
            front_aruco_mean_location = None
            back_aruco_mean_location = None

        flipped_waypoints = []
        for point in waypoints:
            flipped_waypoints.append(np.array([point[1], point[0]]))
        rgb_image_with_data = wpd.plot_path_on_image(rgb_image_with_data, flipped_waypoints)

        # if front_aruco_mean_location is not None:
        #     front_aruco_text = f"front_aruco: ({int(front_aruco_mean_location[0])},{int(front_aruco_mean_location[1])})"
        #     cv2.putText(rgb_image_with_data,
        #                 front_aruco_text,
        #                 (front_aruco_mean_location[0], front_aruco_mean_location[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7,
        #                 (0, 0, 255), 2)
        #
        # if back_aruco_mean_location is not None:
        #     back_aruco_text = f"back_aruco: ({int(back_aruco_mean_location[0])},{int(back_aruco_mean_location[1])})"
        #     cv2.putText(rgb_image_with_data,
        #                 back_aruco_text,
        #                 (back_aruco_mean_location[0], back_aruco_mean_location[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7,
        #                 (0, 255, 0), 2)
        #
        # if back_aruco_mean_location is not None and front_aruco_mean_location is not None:
        #     cv2.arrowedLine(rgb_image_with_data, back_aruco_mean_location, front_aruco_mean_location, (255, 0, 255), 5)
        cv2.arrowedLine(rgb_image_with_data, body_pix, single_WP, (255, 0, 0), 5)
        cv2.circle(img=rgb_image_with_data, center=[int(body_pix[0]), int(body_pix[1])], radius=10, color=(0, 0, 255),
                   thickness=-1)
        cv2.circle(img=rgb_image_with_data, center=[int(body_pix[0]), int(body_pix[1])], radius=int(self.max_eps),
                   color=(0, 255, 0), thickness=2)
        # cv2.circle(rgb_image_with_data, [int(prev_best_robo_pos[0]), int(prev_best_robo_pos[1])],
        #            5, (255, 0, 0), -1)  # Draws a red dot in the center of the yellow circle
        # cv2.circle(rgb_image_with_data, single_WP, 5, (0, 255, 0),
        #            -1)  # Draws a red dot in the center of the yellow circle
        # cv2.arrowedLine(rgb_image_with_data,
        #                 [int(prev_best_robo_pos[0]), int(prev_best_robo_pos[1])],
        #                 [int(body_pix[0]), int(body_pix[1])], (0, 0, 255), 5)
        # cv2.arrowedLine(rgb_image_with_data,
        #                 [int(prev_best_robo_pos[0]), int(prev_best_robo_pos[1])],
        #                 single_WP, (0, 0, 255), 5)
        #
        cv2.putText(rgb_image_with_data,
                    f"({int(body_pix[0])},{int(body_pix[1])})",
                    (body_pix[0], body_pix[1]), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 0, 255), 2)
        #
        # cv2.putText(rgb_image_with_data, f" eps: {round(eps, 2)}", (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(rgb_image_with_data, f" max eps: {round(max_eps, 2)}", (30, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(rgb_image_with_data, f" Az: {round(Az, 2)}", (30, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # for pp in waypoints:
        #     cv2.circle(rgb_image_with_data, pp, 5, (0, 255, 0), -1)
        resized_rgb_image_with_data = self.resize_image(rgb_image_with_data, 60)
        return resized_rgb_image_with_data

    def resize_image(self, rgb_image, scale_percent):
        width = int(rgb_image.shape[1] * scale_percent / 100)
        height = int(rgb_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        interpolation_method = cv2.INTER_AREA
        resized_rgb_image = cv2.resize(rgb_image, dim, interpolation=interpolation_method)
        return resized_rgb_image

    def rotate_vector(self, unit_vector, angle_degrees):
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])
        rotated_unit_direction = rotation_matrix.dot(unit_vector)
        return rotated_unit_direction

    def check_if_point_inside_polygon(self, waypoint, polygon_points):
        point = Point(waypoint[0], waypoint[1])
        polygon = Polygon(polygon_points)
        is_waypoint_inside_polygon = polygon.contains(point)
        return is_waypoint_inside_polygon

    def get_polygon_in_robot_direction(self, angle_degrees=20, vector_size1=450, vector_size2=250, vector_size3=50):
        if self.robot_current_unit_direction is None:
            return None
        unit_vec1 = self.rotate_vector(unit_vector=self.robot_current_unit_direction, angle_degrees=angle_degrees)
        unit_vec2 = self.rotate_vector(unit_vector=self.robot_current_unit_direction, angle_degrees=-angle_degrees)
        position_behind_robot = (self.robot_current_position - vector_size3 * self.robot_current_unit_direction).astype(int)
        unit_perpendicular_direction1 = np.array((-self.robot_current_unit_direction[1], self.robot_current_unit_direction[0]))
        #unit_perpendicular_direction2 = np.array((self.robot_current_unit_direction[1], -self.robot_current_unit_direction[0]))
        point1 = (self.robot_current_position + vector_size1 * unit_vec1).astype(int)
        point2 = (self.robot_current_position + vector_size1 * unit_vec2).astype(int)
        point3 = (position_behind_robot - vector_size2 * unit_perpendicular_direction1).astype(int)
        point4 = (position_behind_robot + vector_size2 * unit_perpendicular_direction1).astype(int)
        polygon_points = [(point1[0], point1[1]), (point2[0], point2[1]), (point3[0], point3[1]), (point4[0], point4[1])]
        return polygon_points

    def plot_polygon_on_image(self, rgb_image, polygon_points):
        rgb_image_with_polygon = rgb_image.copy()
        num_of_points = len(polygon_points)
        for index in range(0, num_of_points):
            next_index = index + 1
            if next_index == num_of_points:
                next_index = 0
            current_point = polygon_points[index]
            next_point = polygon_points[next_index]
            rgb_image_with_polygon = wpd.draw_circle_on_image(rgb_image_with_polygon, center=current_point, color=(0, 0, 255), radius=10, thickness=-1)
            cv2.putText(rgb_image_with_polygon, "p{:1.0f}=({:3.0f}, {:3.0f})".format(index + 1, current_point[0], current_point[1]),
                        (current_point[0] + 30, current_point[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.0,
                        color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(rgb_image_with_polygon, current_point, next_point, color=(0, 255, 0), thickness=4)
        return rgb_image_with_polygon



    def find_closest_waypoint_to_robot(self, waypoints, waypoint_index):
        num_of_waypoints = len(waypoints)
        max_delta = 5
        min_index = max(waypoint_index - max_delta, 0)
        max_index = min(waypoint_index + max_delta, num_of_waypoints - 1)
        distances = dict()
        polygon_points = self.get_polygon_in_robot_direction(angle_degrees=20, vector_size1=700, vector_size2=350,
                                                             vector_size3=150)
        # rgb_image = self.all_cameras_frames_with_frame_index[self.active_camera_index]
        # rgb_image = self.plot_polygon_on_image(rgb_image, polygon_points)
        dist = None
        for i in range(min_index, max_index + 1):
            current_waypoint = waypoints[i]
            is_waypoint_inside_polygon = self.check_if_point_inside_polygon(current_waypoint, polygon_points)
            if is_waypoint_inside_polygon:
                prev_dist = dist
                if dist is not None and dist > prev_dist:
                    return i
                dist = np.linalg.norm(current_waypoint - self.robot_current_position)
                distances[i] = dist
        if len(distances) == 0:
            return None
        else:
            new_waypoint_index = min(distances, key=distances.get)
        return new_waypoint_index

    def find_next_waypoint_in_one_line(self, waypoint_index, waypoints, max_angle_delta):
        num_of_waypoints = len(waypoints)
        next_waypoint_index = waypoint_index + 1
        if next_waypoint_index == num_of_waypoints:
            return None
        next_of_next_waypoint_index = waypoint_index + 2

        current_waypoint = waypoints[waypoint_index]
        next_waypoint = waypoints[next_waypoint_index]

        vec1 = next_waypoint - current_waypoint
        size_vec1 = np.linalg.norm(vec1)
        if math.isclose(size_vec1, 0):
            return next_waypoint_index #should never reach here
        unit_vec1 = vec1 / size_vec1
        while next_of_next_waypoint_index <= num_of_waypoints - 1:
            next_waypoint = waypoints[next_waypoint_index]
            next_of_next_waypoint = waypoints[next_of_next_waypoint_index]
            vec2 = next_of_next_waypoint - next_waypoint
            size_vec2 = np.linalg.norm(vec2)
            if math.isclose(size_vec2, 0):
                return next_of_next_waypoint_index  # should never reach here
            unit_vec2 = vec2 / size_vec2
            cos_angle = np.dot(unit_vec1, unit_vec2)
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            abs_angle_degrees = math.fabs(angle_degrees)
            if abs_angle_degrees > max_angle_delta:
                return next_waypoint_index
            else:
                next_waypoint_index += 1
                next_of_next_waypoint_index += 1
        return next_waypoint_index



    def find_next_waypoint(self, waypoints, waypoint_index, max_azimuth):
        num_of_waypoints = len(waypoints)
        next_waypoint_index = self.find_next_waypoint_in_one_line(waypoint_index, waypoints, max_angle_delta=10)
        if next_waypoint_index is None:
            return None
        next_waypoint = waypoints[next_waypoint_index]
        eps = self.get_eps(next_waypoint, self.robot_current_position)
        direction_from_robot_to_next_waypoint = next_waypoint - self.robot_current_position
        azimuth = self.get_azimuth(self.robot_current_unit_direction, direction_from_robot_to_next_waypoint)
        #while eps < self.max_eps:
        while eps < self.max_eps and azimuth < max_azimuth:
            next_waypoint_index += 1
            if next_waypoint_index == num_of_waypoints:
                return None
            next_waypoint = waypoints[next_waypoint_index]
            direction_from_robot_to_next_waypoint = next_waypoint - self.robot_current_position
            azimuth = self.get_azimuth(self.robot_current_unit_direction, direction_from_robot_to_next_waypoint)
            eps = self.get_eps(next_waypoint, self.robot_current_position)

        min_tries = 5
        if len(self.next_waypoint_index_list) >= min_tries:
            for i in range(0, min_tries):
                index = len(self.next_waypoint_index_list) - 1 - i
                if self.next_waypoint_index_list[index] != next_waypoint_index:
                    return next_waypoint_index
        return next_waypoint_index

    def move_robot_to_next_waypoint_along_active_camera(self):
        waypoints = self.waypoints_for_all_cameras[self.active_camera_index]
        num_of_waypoints = len(waypoints)
        self.current_waypoint_index = self.start_waypoint_indexes[self.active_camera_index]
        self.current_waypoint_index = self.find_closest_waypoint_to_robot(waypoints, self.current_waypoint_index)
        if self.current_waypoint_index == num_of_waypoints - 1:
            self.waypoints_calculated[self.active_camera_index] = True
            #just continue striaght
            self.continue_straight_text = "continue straight"
            self.plot_image()
            self.move_robot(azimuth=0, dist_in_pixels=self.straight_distance_to_move_in_pixels)
            self.calc_waypoints_again_per_camera[self.active_camera_index] = True
            return
        if self.current_waypoint_index is None:
            #just continue striaght
            self.continue_straight_text = "continue straight"
            self.plot_image()
            self.move_robot(azimuth=0, dist_in_pixels=self.straight_distance_to_move_in_pixels)
            self.calc_waypoints_again_per_camera[self.active_camera_index] = True
            return
        self.start_waypoint_indexes[self.active_camera_index] = self.current_waypoint_index
        self.next_waypoint_index = self.find_next_waypoint(waypoints, self.current_waypoint_index, max_azimuth=45)
        self.next_waypoint_index_list.append(self.next_waypoint_index)





        if self.next_waypoint_index is None:
            return
        next_waypoint = waypoints[self.next_waypoint_index]
        eps = self.get_eps(next_waypoint, self.robot_current_position)
        direction_from_robot_to_next_waypoint = next_waypoint - self.robot_current_position
        azimuth = self.get_azimuth(self.robot_current_unit_direction, direction_from_robot_to_next_waypoint)
        self.move_robot(azimuth=azimuth, dist_in_pixels=eps)
        return

    def get_robot_data_for_all_cameras_by_aruco(self, all_cameras_frames):
        num_of_cameras = len(all_cameras_frames)
        robot_positions_in_all_cameras_by_aruco = [None] * num_of_cameras
        robot_unit_directions_in_all_cameras_by_aruco = [None] * num_of_cameras
        is_robot_found_by_aruco = False
        for camera_index, frame in enumerate(all_cameras_frames):
            robot_data = wpd.get_robot_info(frame)
            if len(robot_data) > 0:
                is_robot_found_by_aruco = True
                robot_position_by_aruco = wpd.get_robot_current_position_by_aruco(robot_data)
                robot_unit_direction_by_aruco = wpd.get_robot_unit_direction_by_aruco(robot_data)
                robot_positions_in_all_cameras_by_aruco[camera_index] = robot_position_by_aruco
                robot_unit_directions_in_all_cameras_by_aruco[camera_index] = robot_unit_direction_by_aruco
        return is_robot_found_by_aruco, robot_positions_in_all_cameras_by_aruco, robot_unit_directions_in_all_cameras_by_aruco

    def get_waypoints_for_specific_camera(self, camera_index, all_cameras_frames, robot_position, robot_unit_direction):
        rgb_image = all_cameras_frames[camera_index]
        base_rgb_image = self.base_cameras_frames[camera_index]
        base_robot_data = wpd.get_robot_info(base_rgb_image)
        if len(base_robot_data) == 0:
            if robot_position is not None:
                ddd = rgb_image.copy()
                cv2.circle(img=ddd, center=[int(robot_position[0]), int(robot_position[1])], radius=5,
                           color=(0, 0, 255), thickness=-1)
            waypoints_for_current_camera = wpd.get_waypoints(base_rgb_image, robot_position, robot_unit_direction,
                                                             wpd.ImageDateType.IMAGE_WITHOUT_ROBOT)
        else:
            waypoints_for_current_camera = wpd.get_waypoints(rgb_image, robot_position, robot_unit_direction,
                                                             wpd.ImageDateType.IMAGE_WITH_ROBOT)
        if len(waypoints_for_current_camera) > 0:
            self.waypoints_calculated[camera_index] = True
        flipped_waypoints_for_current_camera = []
        for point in waypoints_for_current_camera:
            flipped_waypoints_for_current_camera.append(np.array([point[1], point[0]]))
        # flipped_waypoints_for_current_camera1 = wpd.minimize_waypoints_in_one_line(flipped_waypoints_for_current_camera,
        #                                                                          max_angle_delta=5)
        return flipped_waypoints_for_current_camera

    def get_waypoints_for_all_cameras(self, all_cameras_frames, robot_positions_in_all_cameras, robot_unit_directions_in_all_cameras):
        is_success = True
        num_of_cameras = len(all_cameras_frames)
        #all_cameras_frames[0] = cv2.line(all_cameras_frames[0], (226, 302), (361, 613), color=(255, 255, 255), thickness=4)
        #all_cameras_frames[0] = cv2.line(all_cameras_frames[0])
        for camera_index in range(0, num_of_cameras):
            if self.waypoints_calculated[camera_index] == True and self.calc_waypoints_again_per_camera[camera_index] == False:
                continue
            robot_position = robot_positions_in_all_cameras[camera_index]
            robot_unit_direction = robot_unit_directions_in_all_cameras[camera_index]

            if self.robot_first_position is not None and robot_position is not None:
                dist_from_first_position = np.linalg.norm(robot_position - self.robot_first_position)
                if dist_from_first_position < 50 and camera_index == self.first_camera_index:
                    waypoints_for_specific_camera = self.get_waypoints_for_specific_camera(camera_index, all_cameras_frames, self.robot_first_position, self.robot_first_unit_direction)
                else:
                    waypoints_for_specific_camera = self.get_waypoints_for_specific_camera(camera_index,
                                                                                           all_cameras_frames,
                                                                                           robot_position,
                                                                                           robot_unit_direction)
            else:
                waypoints_for_specific_camera = self.get_waypoints_for_specific_camera(camera_index, all_cameras_frames, robot_position, robot_unit_direction)
            if len(waypoints_for_specific_camera) == 0 and robot_positions_in_all_cameras[camera_index] is not None:
                is_success = False
                return is_success
            new_frame = all_cameras_frames[camera_index]
            rgb_new = wpd.plot_path_on_image(new_frame, waypoints_for_specific_camera)
            self.waypoints_for_all_cameras[camera_index] = waypoints_for_specific_camera
            self.start_waypoint_indexes[camera_index] = 0
            self.calc_waypoints_again_per_camera[camera_index] = False
        return is_success

    def get_active_camera_index(self,
                                robot_positions_in_all_cameras_by_aruco,
                                waypoints_for_all_cameras,
                                start_waypoint_indexes,
                                current_active_camera,
                                cameras_where_robot_finished):
        num_of_cameras = len(waypoints_for_all_cameras)
        waypoints_lengths = np.zeros(num_of_cameras, int)
        for camera_index, waypoints in enumerate(waypoints_for_all_cameras):
            if robot_positions_in_all_cameras_by_aruco[camera_index] is None:
                waypoints_lengths[camera_index] = 0
                continue
            # if cameras_where_robot_finished[camera_index] == True:
            #     waypoints_lengths[camera_index] = 0
            #     continue
            waypoint_index = start_waypoint_indexes[camera_index]
            num_of_waypoints_in_full_path = len(waypoints)
            num_of_waypoints_left_for_the_robot = num_of_waypoints_in_full_path - waypoint_index
            waypoints_lengths[camera_index] = num_of_waypoints_left_for_the_robot
        if np.sum(waypoints_lengths) == 0:
            #The robot might be seen in more than one camera. So you can randomly pick a camera.
            for first_index_not_none in range(0, len(robot_positions_in_all_cameras_by_aruco)):
                if robot_positions_in_all_cameras_by_aruco[first_index_not_none] is not None:
                    return first_index_not_none
        new_active_camera = np.argmax(waypoints_lengths)
        # if current_active_camera != new_active_camera and current_active_camera >= 0:
        #     cameras_where_robot_finished[current_active_camera] = True
        return new_active_camera


    def get_unit_direction_from_angle(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        x = math.cos(angle_radians)
        y = math.sin(angle_radians)
        unit_direction = np.array((x, y))
        return unit_direction



    def get_yaw_angle_from_robot(self, ninja1): #TODO - write this function with Ran Dviri
        if self.active_camera_index == 0:   #2
            return -41.57
        elif self.active_camera_index == 1: #3
            return 150
        elif self.active_camera_index == 2: #1
            return -122.96
        else:                               #4
            return 63.51

    def get_diff_yaw_from_image_angle(self, angle_degrees_from_image, yaw_angle_degrees_from_robot):
        diff_angle_degrees = angle_degrees_from_image - yaw_angle_degrees_from_robot
        return diff_angle_degrees

    def convert_image_unit_direction_to_yaw_angle(self, unit_direction):
        angle_degrees = wpd.get_angle_from_unit_direction(unit_direction)
        diff_yaw_from_image_angle = self.diff_yaw_from_image_angle_in_all_cameras[self.active_camera_index]
        yaw_angle_degrees = angle_degrees - diff_yaw_from_image_angle
        if yaw_angle_degrees < -180:
            yaw_angle_degrees += 360
        return yaw_angle_degrees

    def calc_unit_direction_when_aruco_is_not_detected(self, prev_unit_direction, yaw_degrees_prev_frame, yaw_degrees_current_frame):
        if yaw_degrees_current_frame is None or yaw_degrees_prev_frame is None:
            return None
        diff_angle_degrees = yaw_degrees_current_frame - yaw_degrees_prev_frame
        diff_angle_radians = np.radians(diff_angle_degrees)
        rotation_matrix = np.array([
            [np.cos(diff_angle_radians), -np.sin(diff_angle_radians)],
            [np.sin(diff_angle_radians), np.cos(diff_angle_radians)]
        ])
        current_unit_direction = rotation_matrix.dot(prev_unit_direction)
        return current_unit_direction

    def calc_robot_position_when_aruco_is_not_detected(self, prev_robot_position, robot_3d_current_position, robot_3d_prev_position):
        robot_current_position_xy_plane = robot_3d_current_position[0:2]
        robot_3d_prev_position_xy_plane = robot_3d_prev_position[0:2]
        dist = np.linalg.norm(robot_current_position_xy_plane - robot_3d_prev_position_xy_plane)
        dist_pixels = (self.num_of_pixels_in_one_meter * dist).astype(int)
        current_robot_position = (prev_robot_position + self.robot_current_unit_direction * dist_pixels).astype(int)
        return current_robot_position


    def update_aruco_detected_per_camera(self):
        num_of_frames_to_check = 3
        num_of_frames_so_far = len(self.active_cameras_each_frame)
        if num_of_frames_so_far <= num_of_frames_to_check:
            return
        for camera_index in range(self.num_of_cameras):
            if camera_index == self.active_camera_index:
                continue
            else:
                update_aruco_for_camera = True
                for i in range(num_of_frames_to_check):
                    if self.active_cameras_each_frame[-(i + 1)] == camera_index:
                        update_aruco_for_camera = False
                        break
                if update_aruco_for_camera:
                    self.aruco_detected_per_camera[camera_index] = False
        return

    def get_camera_indexes_with_robot(self, robot_positions_in_all_cameras_by_aruco):
        camera_indexes_with_robot = []
        for camera_index in range(self.num_of_cameras):
            if robot_positions_in_all_cameras_by_aruco[camera_index] is not None:
                camera_indexes_with_robot.append(camera_index)
        return camera_indexes_with_robot

    def check_if_waypoints_should_be_calculated_again(self, robot_positions_in_all_cameras_by_aruco):
        camera_indexes_with_robot = self.get_camera_indexes_with_robot(robot_positions_in_all_cameras_by_aruco)
        for camera_index in camera_indexes_with_robot:
            current_waypoint_index = self.start_waypoint_indexes[camera_index]
            waypoints = self.waypoints_for_all_cameras[camera_index]
            if len(waypoints) > 0:
                current_waypoint_index = self.find_closest_waypoint_to_robot(waypoints, current_waypoint_index)
                if current_waypoint_index is None:
                    self.calc_waypoints_again_per_camera[camera_index] = True
        return

    def check_if_position_is_inside_frame(self, robot_current_position):
        x_position = robot_current_position[0]
        y_position = robot_current_position[1]
        if x_position < 0 or x_position >= self.image_width or y_position < 0 or y_position >= self.image_height:
            is_inside_frame = False
        else:
            is_inside_frame = True
        return is_inside_frame



    def check_if_in_middle_of_frame(self, robot_prev_position, x_dist_from_boundaries=150, y_dist_from_boundaries=150):
        is_in_middle = False
        x_position = robot_prev_position[0]
        y_position = robot_prev_position[1]
        dist_from_right_end = self.image_width - x_position
        dist_from_bottom_end = self.image_height - y_position

        if x_position > x_dist_from_boundaries and dist_from_right_end > x_dist_from_boundaries and \
            y_position > y_dist_from_boundaries and dist_from_bottom_end > y_dist_from_boundaries:
            is_in_middle = True
        return is_in_middle


    def create_image_with_data(self):
        rgb_image = self.all_cameras_frames_with_frame_index[self.active_camera_index]
        if self.robot_current_position is None:
            return rgb_image
        waypoints_for_active_camera = self.waypoints_for_all_cameras[self.active_camera_index]
        
        if self.is_robot_found_by_aruco:
            rgb_image = wpd.draw_circle_on_image(rgb_image,
                                                 center=self.robot_current_position, color=(0, 255, 0), radius=15,
                                                 thickness=-1)
        else:
            rgb_image = wpd.draw_circle_on_image(rgb_image,
                                                 center=self.robot_current_position, color=(0, 0, 255), radius=15,
                                                 thickness=-1)

        polygon_points = self.get_polygon_in_robot_direction(angle_degrees=20, vector_size1=700, vector_size2=350,
                                                             vector_size3=150)
        rgb_image = self.plot_polygon_on_image(rgb_image, polygon_points)
        if self.robot_contour is not None:
            rect = cv2.minAreaRect(self.robot_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            angle_degrees = rect[2]
            if box is not None:
                cv2.drawContours(rgb_image, [box], 0, (0, 0, 255), 2)
                cv2.drawContours(rgb_image, [self.robot_contour], 0, (255, 255, 255), 2)
                # mean_box = np.mean(box, axis=0).astype(int)
                # cv2.putText(rgb_image, "angle={:3.1f}".format(angle_degrees),
                #             (mean_box[0] + 30, mean_box[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.3,
                #             color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        rgb_image = wpd.plot_path_on_image(rgb_image, path=waypoints_for_active_camera)
        if self.current_waypoint_index is not None:
            if self.current_waypoint_index <= len(waypoints_for_active_camera) - 1:
                current_waypoint = waypoints_for_active_camera[self.current_waypoint_index]
                rgb_image = wpd.draw_circle_on_image(rgb_image, center=current_waypoint, color=(0, 255, 255), radius=20,
                                                     thickness=-1)
        if self.next_waypoint_index is not None:
            if self.next_waypoint_index <= len(waypoints_for_active_camera) - 1:
                next_waypoint = waypoints_for_active_camera[self.next_waypoint_index]
                rgb_image = wpd.draw_circle_on_image(rgb_image, center=next_waypoint, color=(100, 0, 0), radius=20,
                                                     thickness=-1)
        rgb_image = wpd.draw_circle_on_image(rgb_image, center=self.robot_current_position, color=(0, 255, 255),
                                             radius=int(self.max_eps), thickness=2)
        second_point = (self.robot_current_position + 200 * self.robot_current_unit_direction).astype(int)
        cv2.arrowedLine(rgb_image, self.robot_current_position, second_point, (255, 0, 255), 5)
        font_for_direction = cv2.FONT_HERSHEY_COMPLEX
        font_scale_for_direction = 0.8
        text_color_for_direction = (255, 0, 255)
        text_thickness_for_direction = 2
        cv2.putText(rgb_image, "({:3.2f}, {:3.2f})".format(self.robot_current_unit_direction[0],
                                                           self.robot_current_unit_direction[1]),
                    (second_point[0], second_point[1] + 30), font_for_direction, font_scale_for_direction,
                    text_color_for_direction, text_thickness_for_direction, cv2.LINE_AA)

        if self.is_bridge:
            bridge_text = 'BRIDGE'
            bridge_font_type = cv2.FONT_HERSHEY_COMPLEX
            bridge_font_scale = 1.5
            bridge_font_color = (0, 255, 0)
            bridge_font_thickness = 3
        else:
            bridge_text = 'NO BRIDGE'
            bridge_font_type = cv2.FONT_HERSHEY_COMPLEX
            bridge_font_scale = 0.5
            bridge_font_color = (0, 0, 255)
            bridge_font_thickness = 1
        cv2.putText(rgb_image, bridge_text,
                    (50, 150), bridge_font_type, bridge_font_scale,
                    bridge_font_color, bridge_font_thickness, cv2.LINE_AA)

        font_for_point = cv2.FONT_HERSHEY_COMPLEX
        font_scale_for_point = 0.8
        text_color_for_point = (255, 255, 255)
        text_thickness_for_point = 2
        cv2.putText(rgb_image, f'({self.robot_current_position[0]}, {self.robot_current_position[1]})',
                    (self.robot_current_position[0], self.robot_current_position[1] + 30), font_for_point,
                    font_scale_for_point,
                    text_color_for_point, text_thickness_for_point,
                    cv2.LINE_AA)

        if self.is_robot_found_by_aruco:
            aruco_text = 'aruco recognized'
            aruco_font_type = cv2.FONT_HERSHEY_COMPLEX
            aruco_font_scale = 1.5
            aruco_font_color = (0, 255, 100)
            aruco_font_thickness = 3
        else:
            aruco_text = 'aruco NOT recognized'
            aruco_font_type = cv2.FONT_HERSHEY_COMPLEX
            aruco_font_scale = 0.5
            aruco_font_color = (0, 0, 255)
            aruco_font_thickness = 1
        cv2.putText(rgb_image, aruco_text,
                    (50, 250), aruco_font_type, aruco_font_scale,
                    aruco_font_color, aruco_font_thickness, cv2.LINE_AA)


        continue_straight_font_type = cv2.FONT_HERSHEY_COMPLEX
        continue_straight_font_scale = 1.5
        continue_straight_font_color = (255, 255, 0)
        continue_straight_font_thickness = 3
        cv2.putText(rgb_image, self.continue_straight_text,
                    (50, 350), continue_straight_font_type, continue_straight_font_scale,
                    continue_straight_font_color, continue_straight_font_thickness, cv2.LINE_AA)

        return rgb_image

    def plot_image(self):
        rgb_image_with_data = self.create_image_with_data()
        file_full_path_with_data = "{}/{:05d}.jpg".format(self.images_output_folder_with_data, self.frame_index_only_increasing)
        cv2.imwrite(file_full_path_with_data, rgb_image_with_data)
        resized_rgb_image = wpd.resize_image(rgb_image_with_data, 60)
        cv2.imshow('resized_rgb_image', resized_rgb_image)
        cv2.waitKey(1)

    def create_videos(self):
        frame_rate = 20
        video_name = 'robot_autonomic_driving'
        video_path = self.videos_output_folder_with_data + '/' + video_name + '.avi'
        print(f'Creating video {video_path}')
        common_utils.create_video(self.images_output_folder_with_data, 'jpg', video_path, frame_rate)
        print(f'Finised creating video {video_path}')

        if self.create_gif_video:
            gif_video_path = self.videos_output_folder_with_data + '/' + video_name + '.gif'
            print(f'Creating gif video {gif_video_path}')
            videoClip = VideoFileClip(video_path)
            videoClip.write_gif(gif_video_path)
            print(f'Finished creating gif video {gif_video_path}')

    def increase_frame_index(self):
        self.frame_index += 1
        self.frame_index_only_increasing += 1
        if self.simulate_cameras:
            if self.frame_index == len(self.simulated_frames_indexes):
                self.frame_index = 0

    def move_robot_through_all_cameras(self):
        for i in range(0, 1000):
            self.continue_straight_text = ""
            self.all_cameras_frames, self.all_cameras_frames_with_frame_index = self.get_cameras_frames(self.frame_index)
            self.is_robot_found_by_aruco, robot_positions_in_all_cameras_by_aruco, robot_unit_directions_in_all_cameras_by_aruco = self.get_robot_data_for_all_cameras_by_aruco(self.all_cameras_frames)

            #since we assume that the aruco MUST be detected at the first frame:
            self.robot_contour = None



            if self.is_robot_found_by_aruco == True:
                self.is_bridge = False
                self.check_if_waypoints_should_be_calculated_again(robot_positions_in_all_cameras_by_aruco)
                is_success = self.get_waypoints_for_all_cameras(self.all_cameras_frames, robot_positions_in_all_cameras_by_aruco, robot_unit_directions_in_all_cameras_by_aruco)
                if not is_success:
                    # something is wrong. maybe the robot is on the white markers and therefore the wayopints can't be calculated
                    self.continue_straight_text = "continue straight"
                    self.plot_image()
                    self.move_robot(azimuth=0, dist_in_pixels=self.straight_distance_to_move_in_pixels) #just continue straint
                    self.increase_frame_index()
                    continue
                self.prev_active_camera_index = self.active_camera_index
                self.active_camera_index = self.get_active_camera_index(robot_positions_in_all_cameras_by_aruco, self.waypoints_for_all_cameras, self.start_waypoint_indexes, self.active_camera_index, self.cameras_where_robot_finished)
                self.active_cameras_each_frame.append(self.active_camera_index)
                self.aruco_detected_per_camera[self.active_camera_index] = True
                self.update_aruco_detected_per_camera()
                self.robot_prev_position = self.robot_current_position
                self.robot_current_position = robot_positions_in_all_cameras_by_aruco[self.active_camera_index]
                self.robot_prev_unit_direction = self.robot_current_unit_direction
                self.robot_current_unit_direction = robot_unit_directions_in_all_cameras_by_aruco[self.active_camera_index]


                #angle_degrees = wpd.get_angle_from_unit_direction(self.robot_current_unit_direction)
                #unit_direction1 = self.get_unit_direction_from_angle(angle_degrees)

                if self.first_time_each_camera[self.active_camera_index]:
                    self.first_time_each_camera[self.active_camera_index] = False
                    angle_degrees_from_image = wpd.get_angle_from_unit_direction(self.robot_current_unit_direction)
                    yaw_angle_degrees_from_robot = self.get_yaw_angle_from_robot(self.ninja1)
                    diff_yaw_from_image_angle = self.get_diff_yaw_from_image_angle(angle_degrees_from_image,
                                                                                   yaw_angle_degrees_from_robot)
                    self.diff_yaw_from_image_angle_in_all_cameras[self.active_camera_index] = diff_yaw_from_image_angle
                else:
                    yaw_angle_degrees_from_robot = self.convert_image_unit_direction_to_yaw_angle(
                        self.robot_current_unit_direction)



                if self.first_time:
                    self.first_time = False
                    self.first_camera_index = self.active_camera_index
                    base_frame_first_camera = self.base_cameras_frames[self.first_camera_index]
                    self.robot_first_position = self.robot_current_position
                    self.robot_first_unit_direction = self.robot_current_unit_direction
            else:
                self.prev_active_camera_index = self.active_camera_index
                self.robot_prev_position = self.robot_current_position
                self.robot_prev_unit_direction = self.robot_current_unit_direction
                if self.work_with_real_robot:
                    self.robot_current_unit_direction = self.calc_unit_direction_when_aruco_is_not_detected(
                        prev_unit_direction=self.robot_prev_unit_direction, yaw_degrees_prev_frame=self.prev_yaw,
                        yaw_degrees_current_frame=self.robot_yaw)
                    if self.robot_current_unit_direction is None:
                        self.continue_straight_text = "continue straight"
                        self.plot_image()
                        self.move_robot(azimuth=0, dist_in_pixels=self.straight_distance_to_move_in_pixels)  # just continue straint
                        self.increase_frame_index()
                        continue
                    self.robot_current_position = self.calc_robot_position_when_aruco_is_not_detected(self.robot_prev_position,
                                                                                              self.robot_3d_current_position,
                                                                                              self.robot_3d_prev_position)
                    is_in_middle_of_frame = self.check_if_in_middle_of_frame(self.robot_current_position,
                                                                             x_dist_from_boundaries=150,
                                                                             y_dist_from_boundaries=150)
                    is_position_inside_frame = self.check_if_position_is_inside_frame(self.robot_current_position)
                    if is_in_middle_of_frame:
                        # we are probably under a bridge
                        self.is_bridge = True
                    else:
                        self.is_bridge = False
                    if is_position_inside_frame == False:
                        # we switched to a different camera
                        self.continue_straight_text = "continue straight"
                        self.plot_image()
                        self.move_robot(azimuth=0, dist_in_pixels=self.straight_distance_to_move_in_pixels)  # just continue straint
                        self.increase_frame_index()
                        continue
                else:
                    self.robot_current_position, _, self.robot_contour, active_camera_index = wpd.get_robot_position_from_all_cameras_by_frames_diff(
                        self.base_cameras_frames, self.all_cameras_frames, self.first_camera_index,
                        self.robot_first_position, self.cameras_where_robot_finished)
                    if active_camera_index is not None:
                        self.active_camera_index = active_camera_index
                self.active_cameras_each_frame.append(self.active_camera_index)
                if self.robot_current_position is None:
                    #robot is not seen in any of the cameras
                    self.continue_straight_text = "continue straight"
                    self.plot_image()
                    self.move_robot(azimuth=0, dist_in_pixels=self.straight_distance_to_move_in_pixels) #just continue straint
                    self.increase_frame_index()
                    continue
                if self.waypoints_calculated[self.active_camera_index] == False or self.calc_waypoints_again_per_camera[self.active_camera_index]:
                    waypoints_for_specific_camera = self.get_waypoints_for_specific_camera(self.active_camera_index, self.all_cameras_frames, self.robot_current_position, self.robot_current_unit_direction)
                    self.waypoints_for_all_cameras[self.active_camera_index] = waypoints_for_specific_camera
                    self.start_waypoint_indexes[self.active_camera_index] = 0
                    self.calc_waypoints_again_per_camera[self.active_camera_index] = False



            self.prev_pitch = self.robot_pitch
            self.prev_yaw = self.robot_yaw
            self.robot_3d_prev_position = self.robot_3d_current_position
            self.move_robot_to_next_waypoint_along_active_camera()

            self.plot_image()

            self.increase_frame_index()



