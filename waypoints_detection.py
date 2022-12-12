import cv2
import numpy as np
import skimage.graph
import imutils
from skimage.morphology import medial_axis, skeletonize, thin
from enum import Enum
from skimage.graph import mcp
from skimage.metrics import structural_similarity as compare_ssim
import math

dict_type = cv2.aruco.DICT_4X4_250
aruco_dict = cv2.aruco.Dictionary_get(dict_type)
front_aruco_id = 0
back_aruco_id = 21
front_aruco_key = 'front_aruco'
back_aruco_key = 'back_aruco'
robot_possible_ids = {front_aruco_key: front_aruco_id, back_aruco_key: back_aruco_id}
aruco_parameters = cv2.aruco.DetectorParameters_create()
front_back_movement_vec_priorities = np.array([[back_aruco_id, front_aruco_id]])


class ImageDateType(Enum):
    IMAGE_WITH_ROBOT = 0
    IMAGE_WITHOUT_ROBOT = 1


def write_text_on_frame(rgb_image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 255)
    thickness = 2
    cv2.putText(rgb_image, f'{text}', position, font, fontScale, color, thickness, cv2.LINE_AA)
    return rgb_image


def get_aruco_data(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=aruco_parameters)
    frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    return (corners, ids, frame_markers)


def get_robot_position_and_direction(rgb_image):
    robot_data = get_robot_info(rgb_image)
    robot_position = get_robot_current_location(robot_data)
    robot_unit_direction = get_robot_unit_direction(robot_data)
    return robot_position, robot_unit_direction

def get_robot_info(rgb_image):
    (corners, ids, frame_markers) = get_aruco_data(rgb_image)
    robot_unit_directions_for_both_arucos = dict()
    robot_data = dict()
    # robot_data['robot_mean_aruco_locations'] = None
    # robot_data['ids'] = None
    # robot_data['robot_unit_directions_for_both_arucos'] = None
    # robot_data['frame_markers'] = None
    # robot_data['robot_unit_direction'] = None
    # robot_data['robot_pixel_location'] = None
    # robot_data['direction_from_back_aruco_to_front_aruco'] = None
    # robot_data['unit_direction_from_back_aruco_to_front_aruco'] = None
    # robot_data['robot_mean_of_front_and_back_mean_locations'] = None

    if ids is None:
        return robot_data
    robot_mean_aruco_locations = dict()
    for counter, possible_id_key in enumerate(robot_possible_ids):
        robot_mean_aruco_locations[possible_id_key] = None
        robot_unit_directions_for_both_arucos[possible_id_key] = None
        possible_id_value = robot_possible_ids[possible_id_key]
        if possible_id_value in ids:
            id_index = np.where(ids == possible_id_value)[0][0]
            robot_corners = corners[id_index]
            unit_direction = get_unit_direction_from_aruco(robot_corners, ids, rgb_image)
            robot_unit_directions_for_both_arucos[possible_id_key] = unit_direction
            current_robot_location_np = np.mean(robot_corners, 1)[0]
            current_robot_location_np_int = current_robot_location_np.astype(int)
            robot_mean_aruco_locations[possible_id_key] = current_robot_location_np_int

    robot_data['robot_mean_aruco_locations'] = robot_mean_aruco_locations
    robot_data['ids'] = ids
    robot_data['robot_unit_directions_for_both_arucos'] = robot_unit_directions_for_both_arucos
    robot_data['frame_markers'] = frame_markers
    robot_unit_direction = get_robot_unit_direction_by_aruco(robot_data)
    robot_pixel_location = get_robot_current_position_by_aruco(robot_data)
    robot_data['robot_unit_direction'] = robot_unit_direction
    robot_data['robot_pixel_location'] = robot_pixel_location
    front_aruco_mean_location = robot_mean_aruco_locations['front_aruco']
    back_aruco_mean_location = robot_mean_aruco_locations['back_aruco']
    if front_aruco_mean_location is None or back_aruco_mean_location is None:
        robot_data['direction_from_back_aruco_to_front_aruco'] = robot_unit_direction
        robot_data['unit_direction_from_back_aruco_to_front_aruco'] = robot_unit_direction
        robot_data['robot_mean_of_front_and_back_mean_locations'] = robot_pixel_location
    else:
        robot_locations = np.array([front_aruco_mean_location, back_aruco_mean_location])
        robot_mean_of_front_and_back_mean_locations = np.mean(robot_locations, axis=0).astype(int)
        direction_from_back_aruco_to_front_aruco = front_aruco_mean_location - back_aruco_mean_location
        unit_direction_from_back_aruco_to_front_aruco = direction_from_back_aruco_to_front_aruco/np.linalg.norm(direction_from_back_aruco_to_front_aruco)
        robot_data['direction_from_back_aruco_to_front_aruco'] = direction_from_back_aruco_to_front_aruco
        robot_data['unit_direction_from_back_aruco_to_front_aruco'] = unit_direction_from_back_aruco_to_front_aruco
        robot_data['robot_mean_of_front_and_back_mean_locations'] = robot_mean_of_front_and_back_mean_locations
    return robot_data


def shortest_path_from_start_to_end(start_point, end_point, binary_image):
    very_bin_number = 10000
    costs = np.where(binary_image == 0, 1, very_bin_number)
    path, cost = skimage.graph.route_through_array(
        costs,
        start=start_point,
        end=end_point,
        fully_connected=False,
        geometric=False)
    return path, cost


def shortest_path(start_point, binary_image):
    binary_image_with_data_3_channels = expand_1_channel_image_to_3_channels_image(binary_image)
    binary_image_with_data_3_channels = draw_circle_on_image(rgb_image=binary_image_with_data_3_channels, center=(start_point[1], start_point[0]), color=(0, 255, 0), radius=10, thickness=-1)
    very_bin_number = 1000000
    costs = np.where(binary_image == 0, 1, very_bin_number)
    m = mcp.MCP(costs, fully_connected=True)
    cost_mcp, path_mcp = m.find_costs([start_point])
    cost_mcp[cost_mcp >= very_bin_number] = -10
    end_point = np.unravel_index(np.argmax(cost_mcp, axis=None), cost_mcp.shape)
    m.traceback(end_point)
    path, cost = skimage.graph.route_through_array(
        costs,
        start=start_point,
        end=end_point,
        fully_connected=True,
        geometric=False)
    return path, cost


def resize_image(rgb_image, scale_percent):
    width = int(rgb_image.shape[1] * scale_percent / 100)
    height = int(rgb_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    interpolation_method = cv2.INTER_AREA
    resized_rgb_image = cv2.resize(rgb_image, dim, interpolation=interpolation_method)
    return resized_rgb_image

def get_azi(vector_1, vector_2):
    if np.linalg.norm(vector_1)==0:
        unit_vector_1 = vector_1
    else:
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    if np.linalg.norm(vector_2)==0:
        unit_vector_2 = vector_2
    else:
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    azi = np.arccos(dot_product) * 180/np.pi  # *rad2deg # should be positive number 0-180deg

    # https://www.geeksforgeeks.org/how-to-compute-the-cross-product-of-two-given-vectors-using-numpy/
    cross = np.cross(vector_1, vector_2)
    if cross < 0 :
        #required rotete right from vec1 to vec2 (from current pos to desired WP)
        pass
    else:
        #required rotete left from vec1 to vec2 (from current pos to desired WP)
        azi = -azi

    #todo ? - azi so far is only by point vector from prev to current.
    #  consider and add/trim aruco vector relative to move_vec. negative cross is pos addition..

    return -azi ## , np.linalg.norm(vector_1), np.linalg.norm(vector_2)


def get_unit_direction_from_aruco(corners, ids, rgb_image):
    unit_vector = None
    if [front_aruco_id] in ids or [back_aruco_id] in ids:
        point1 = np.array(corners[0][0]).astype(int)
        point2 = np.array(corners[0][1]).astype(int)
        point3 = np.array(corners[0][2]).astype(int)
        point4 = np.array(corners[0][3]).astype(int)
        unit_vector = get_unit_direction_by_points(point2, point3)

        # rgb_image = draw_circle_on_image(rgb_image, point1, circle_color=(0, 255, 0), radius=3)
        # rgb_image = write_text_on_frame(rgb_image, 'p1', point1)
        # rgb_image = draw_circle_on_image(rgb_image, point2, circle_color=(0, 255, 0), radius=3)
        # rgb_image = write_text_on_frame(rgb_image, 'p2', point2)
        # rgb_image = draw_circle_on_image(rgb_image, point3, circle_color=(0, 255, 0), radius=3)
        # rgb_image = write_text_on_frame(rgb_image, 'p3', point3)
        # rgb_image = draw_circle_on_image(rgb_image, point4, circle_color=(0, 255, 0), radius=3)
        # rgb_image = write_text_on_frame(rgb_image, 'p4', point4)
        # rgb_image = plot_robot_direction_on_frame(rgb_image, point3, unit_vector, arrow_color=(0, 0, 255))

    return unit_vector


def get_unit_direction_by_points(back_point, front_point):
    direction = front_point - back_point
    direction_size = np.linalg.norm(direction)
    if direction_size == 0:
        print('division by zero')
        david = 5
    unit_direction = direction / direction_size
    return unit_direction


# def plot_points_on_image(rgb_image, points):
#     int_points = points.astype(int)
#     rgb_image_with_data = rgb_image.copy()
#     circle_radius = 8
#     circle_color = (0, 255, 0)
#     circle_thickness = -1
#     num_of_points = int_points.shape[1]
#     for i in range(0, num_of_points):
#         point = int_points[:, i]
#         cv2.circle(rgb_image_with_data, point, circle_radius, circle_color, circle_thickness)
#     return rgb_image_with_data


def  plot_path_on_image(rgb_image, path):
    rgb_image_with_path = rgb_image.copy()
    radius = 10
    circle_color = (200, 100, 100)
    thickness = -1
    path_length = len(path)

    font_for_index = cv2.FONT_HERSHEY_COMPLEX
    font_scale_for_index = 1.2
    text_color_for_index = (0, 0, 255)
    text_thickness_for_index = 2

    font_for_point = cv2.FONT_HERSHEY_COMPLEX
    font_scale_for_point = 0.8
    text_color_for_point = (0, 255, 255)
    text_thickness_for_point = 2

    for i in range(0, path_length):
        point1 = path[i]
        x1 = int(point1[0])
        y1 = int(point1[1])
        cv2.circle(rgb_image_with_path, (x1, y1), radius, circle_color, thickness)
        cv2.putText(rgb_image_with_path, f'{i + 1}', (x1, y1), font_for_index, font_scale_for_index, text_color_for_index, text_thickness_for_index, cv2.LINE_AA)
        cv2.putText(rgb_image_with_path, f'({x1}, {y1})', (x1, y1+30), font_for_point, font_scale_for_point, text_color_for_point, text_thickness_for_point,
                    cv2.LINE_AA)
        if i < path_length - 1:
            point2 = path[i + 1]
            x2 = int(point2[0])
            y2 = int(point2[1])
            arrow_color = (0, 255, 0)
            arrow_thickness = 3
            cv2.arrowedLine(rgb_image_with_path, (x1, y1), (x2, y2), arrow_color, arrow_thickness)
            # cv2.line(rgb_image_with_path, (y1, x1), (y2, x2), line_color, thickness=line_thickness)
    return rgb_image_with_path


def expand_1_channel_image_to_3_channels_image(image_1_channel):
    rows = image_1_channel.shape[0]
    cols = image_1_channel.shape[1]
    image_3_channels = np.zeros((rows, cols, 3), 'uint8')
    image_3_channels[:, :, 0] = image_1_channel
    image_3_channels[:, :, 1] = image_1_channel
    image_3_channels[:, :, 2] = image_1_channel
    return image_3_channels


def get_robot_unit_direction_by_current_and_previous_locations(current_robot_location,
                                                               robot_location_in_previous_frame):
    if current_robot_location is None or robot_location_in_previous_frame is None:
        return None
    robot_direction = current_robot_location - robot_location_in_previous_frame
    robot_direction_size = np.linalg.norm(robot_direction)
    if np.isclose(robot_direction_size, 0):
        return None
    unit_direction = robot_direction / robot_direction_size
    return unit_direction

def get_pependicular_vector(unit_vector):
    unit_perpendicular_vector = [unit_vector[1], -unit_vector[0]]
    return unit_perpendicular_vector


def plot_mcp_points(rgb_image, current_waypoint, next_waypoint):
    binary_image = get_binary_image(rgb_image)
    binary_image_3_channels = expand_1_channel_image_to_3_channels_image(binary_image)
    start_point = [current_waypoint[1], current_waypoint[0]]
    end_point = [next_waypoint[1], next_waypoint[0]]
    path, cost = shortest_path_from_start_to_end(start_point, end_point, binary_image)
    simplified_path = simplify_coords_vw(path, 800.0)
    rgb_image_with_path = plot_path_on_image(rgb_image, binary_image_3_channels, simplified_path)
    binary_image_3_channels_with_path = plot_path_on_image(binary_image_3_channels, simplified_path)
    resized_binary_image_3_channels_with_path = resize_image(rgb_image=binary_image_3_channels_with_path,
                                                             scale_percent=80)
    cv2.imshow('resized_binary_image_3_channels_with_path', resized_binary_image_3_channels_with_path)
    cv2.waitKey(0)


def get_next_waypoint(rgb_image, current_waypoint, unit_vector):
    vector_size = 200
    direction = vector_size * unit_vector
    #binary_image = get_binary_image(rgb_image)
    #binary_image_3_channels = expand_1_channel_image_to_3_channels_image(binary_image)
    #binary_image_3_channels_with_line = binary_image_3_channels.copy()
    next_waypoint = current_waypoint + direction
    next_waypoint = next_waypoint.astype(int)
    return next_waypoint

    # line_color = (0, 255, 0)
    # line_thickness = 3
    # cv2.line(binary_image_3_channels_with_line, current_waypoint, next_waypoint, line_color, thickness=line_thickness)
    # cv2.imshow('binary_image_3_channels_with_line', binary_image_3_channels_with_line)
    # cv2.waitKey(0)


def complete_partial_lines_in_image(binary_image):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(binary_image, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    binary_image_with_complete_lines_3_channels = expand_1_channel_image_to_3_channels_image(binary_image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(binary_image_with_complete_lines_3_channels, (x1, y1), (x2, y2), color=(255, 0, 255),
                         thickness=3)

    return binary_image_with_complete_lines_3_channels


def calc_edge_detection(gray_image_no_noise):
    low_threshold = 0
    high_threshold = 90
    edges_image = cv2.Canny(gray_image_no_noise, low_threshold, high_threshold)
    values = np.unique(edges_image)
    # print(values)
    # cv2.imshow('edges_image', edges_image)
    # cv2.waitKey(0)
    return edges_image


def remove_noise(gray_image):
    # kernel_size = 5
    # kernel = np.ones((kernel_size, kernel_size), np.uint8) / (kernel_size * kernel_size)
    # kernel_center = kernel_size // 2
    # #kernel[kernel_center, kernel_center] = 1
    #
    # binary_image_no_noise = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)

    kernel_size = 5
    gray_image_no_noise = cv2.GaussianBlur(gray_image,
                                           (kernel_size, kernel_size),
                                           0)
    return gray_image_no_noise


def get_gray_image_no_noise_3_channels(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    gray_image_no_noise = remove_noise(gray_image)
    gray_image_no_noise_3_channels = expand_1_channel_image_to_3_channels_image(gray_image_no_noise)
    return gray_image_no_noise_3_channels


def get_all_contours_boxes(contours):
    boxes = list()
    local_max_screen_width = 900
    local_min_screen_width = 500
    local_max_screen_height = 500
    local_min_screen_height = 200
    for single_contour in contours:
        x, y, w, h = cv2.boundingRect(single_contour)
        # if local_max_screen_width > w > local_min_screen_width and \
        #         local_max_screen_height > h > local_min_screen_height:
        x1 = x
        x2 = x1 + w
        y1 = y
        y2 = y1 + h
        box = (x1, y1, x2, y2)
        boxes.append(box)
    return boxes


def draw_boxes_and_write_their_sizes_on_image(rgb_image, boxes, box_color, box_thickness):
    # x, y, w, h = cv2.boundingRect(single_contour)
    # if self.max_screen_width > w > self.min_screen_width and \
    #         self.max_screen_height > h > self.min_screen_height:
    #     x1 = x
    #     x2 = x1 + w
    #     y1 = y
    #     y2 = y1 + h
    #     boxes.append((x1, y1, x2, y2))

    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        width = x2 - x1
        height = y2 - y1
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.3
        text_color = (0, 0, 255)
        text_thickness = 1

        local_max_screen_width = 300
        local_min_screen_width = 100

        local_max_screen_height = 450
        local_min_screen_height = 100

        # local_max_square_size = 185
        # local_min_square_size = 100

        # if self.max_screen_width > width > self.min_screen_width and \
        #         self.max_screen_height > height > self.min_screen_height:
        # if local_max_square_size > width > local_min_square_size and \
        #         local_max_square_size > height > local_min_square_size:
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), box_color, box_thickness)
        cv2.putText(rgb_image, f'({width},{height})', (x1, y1), font, font_scale, text_color, text_thickness,
                    cv2.LINE_AA)
    return rgb_image


def get_binary_image(rgb_image):
    gray_image_no_noise_3_channels = get_gray_image_no_noise_3_channels(rgb_image)
    gray_image_no_noise = gray_image_no_noise_3_channels[:, :, 0]
    binary_image_edge_detection_black_background = calc_edge_detection(gray_image_no_noise)
    binary_image_adaptive_thresh_white_background = cv2.adaptiveThreshold(gray_image_no_noise, 255,
                                                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                          cv2.THRESH_BINARY, 21, 2)
    binary_image_adaptive_thresh_black_background = cv2.bitwise_not(binary_image_adaptive_thresh_white_background)
    binary_image = binary_image_adaptive_thresh_black_background
    binary_image = binary_image_edge_detection_black_background
    # kernel_size = 3
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # erode_binary_image = cv2.erode(binary_image, kernel)

    dilate_kernel_size = 5
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilate_binary_image = cv2.dilate(binary_image, dilate_kernel)

    binary_image_with_complete_lines_3_channels = complete_partial_lines_in_image(dilate_binary_image)
    binary_image_with_complete_lines = binary_image_with_complete_lines_3_channels[:, :, 0]

    binary_image_white_background = cv2.bitwise_not(binary_image_with_complete_lines)

    # findContoursResults = cv2.findContours(binary_image_adaptive_thresh_black_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(findContoursResults)
    # all_boxes = get_all_contours_boxes(contours)
    # binary_image_3_channels = expand_1_channel_image_to_3_channels_image(binary_image)
    # rgb_image_with_all_boxes = rgb_image.copy()
    # rgb_image_with_all_boxes = draw_boxes_and_write_their_sizes_on_image(rgb_image=rgb_image_with_all_boxes,
    #                                                                           boxes=all_boxes,
    #                                                                           box_color=(0, 255, 0),
    #                                                                           box_thickness=2)
    # cv2.imshow('rgb_image_with_all_boxes', rgb_image_with_all_boxes)
    # cv2.waitKey(0)

    # cv2.imshow('binary_image_with_complete_lines', binary_image_with_complete_lines)
    # cv2.waitKey(0)
    return binary_image_white_background


# def get_front_and_back_locations(robot_locations):
#     not_nan_indexes = ~(np.isnan(robot_locations).any(axis=0))
#     valid_ids = robot_possible_ids[not_nan_indexes]
#     current_back_id = -1
#     current_front_id = -1
#     vec_option = front_back_movement_vec_priorities[0]
#     if vec_option[0] in valid_ids and vec_option[1] in valid_ids:
#         current_back_id = vec_option[0]
#         current_front_id = vec_option[1]
#     else:
#         return None, None
#     if current_back_id >= 0 and current_front_id >= 0:
#         front_index = np.where(robot_possible_ids == current_front_id)[0][0]
#         back_index = np.where(robot_possible_ids == current_back_id)[0][0]
#         front_location = robot_locations[:, front_index].astype(int)
#         back_location = robot_locations[:, back_index].astype(int)
#     else:
#         return None, None
#     return front_location, back_location


def draw_circle_on_image(rgb_image, center, color=(0, 255, 0), radius=10, thickness=-1):
    rgb_image_with_data = rgb_image.copy()
    x = int(center[1])
    y = int(center[0])
    cv2.circle(rgb_image_with_data, (y, x), radius, color, thickness)
    return rgb_image_with_data


def find_closest_point_on_line(point, line):
    start_point = line['start_point']
    end_point = line['end_point']

    a = start_point
    b = end_point
    c = point

    line_direction = b - a
    v = c - a

    t = np.dot(v, line_direction) / np.dot(line_direction, line_direction)
    if t < 0 or t > 1:
        closest_point = None
    else:
        closest_point = a + t * line_direction
    return closest_point


def find_closest_point(single_contour, point):
    squeezed_single_contour = np.squeeze(single_contour)
    num_of_contour_points = squeezed_single_contour.shape[0]
    line = dict()
    list_closest_points_and_dists = []
    min_dist = np.inf
    closest_point_on_contour = None
    for contour_point_index in range(0, num_of_contour_points - 1):
        start_point = squeezed_single_contour[contour_point_index]
        end_point = squeezed_single_contour[contour_point_index + 1]
        line['start_point'] = start_point
        line['end_point'] = end_point
        closest_point_on_line = find_closest_point_on_line(point, line)
        if closest_point_on_line is None:
            dist_from_point_to_line = np.inf
        else:
            dist_from_point_to_line = np.linalg.norm(point - closest_point_on_line)
            if dist_from_point_to_line < min_dist:
                min_dist = dist_from_point_to_line
                closest_point_on_contour = closest_point_on_line
    return closest_point_on_contour, min_dist


def draw_point_on_image(rgb_image, np_point, circle_radius, circle_color):
    tuple_point = (int(np_point[0]), int(np_point[1]))
    circle_thickness = -1
    cv2.circle(rgb_image, tuple_point, circle_radius, circle_color, circle_thickness)
    return rgb_image


def find_contour_encapsulates_point(point, contours):
    for single_contour in contours:
        dist = cv2.pointPolygonTest(single_contour, point, False)
        if dist > 0:
            return single_contour
    return None


def get_approx_contour(single_contour):
    peri = cv2.arcLength(single_contour, True)
    single_approx_contour = cv2.approxPolyDP(single_contour, 0.001 * peri, True)
    return single_approx_contour


def get_binary_image_with_filled_single_contour(rgb_image, single_contour):
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]
    binary_image_with_filled_single_contour = np.zeros((height, width), np.uint8)
    cv2.fillPoly(binary_image_with_filled_single_contour, pts=[single_contour], color=(255, 255, 255))
    return binary_image_with_filled_single_contour


def convert_true_false_image_to_uint8(true_false_image):
    height = true_false_image.shape[0]
    width = true_false_image.shape[1]
    binary_image = np.zeros((height, width), np.uint8)
    binary_image[true_false_image == True] = 255
    return binary_image


def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:, :, 0] - target[0]) ** 2 + (nonzero[:, :, 1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    nearest_index_2d = nonzero[nearest_index]
    point_index = (nearest_index_2d[0][1], nearest_index_2d[0][0])
    return point_index


def put_black_pixels_in_image_bounderies(binary_image, boundary_width):
    height = binary_image.shape[0]
    width = binary_image.shape[1]
    binary_image[0:boundary_width, :] = 0
    binary_image[height - boundary_width - 1:height, :] = 0
    binary_image[:, 0:boundary_width] = 0
    binary_image[:, width - boundary_width - 1:width]
    return binary_image


def reduce_path_lenth(path, max_dist_between_consecutive_points):
    filtered_path = [path[0]]
    point1 = np.array(path[0])
    num_of_points = len(path)
    david = 5
    for i in range(1, num_of_points):
        point2 = np.array(path[i])
        dist = np.linalg.norm(point2 - point1)
        if dist >= max_dist_between_consecutive_points:
            filtered_path.append(point2)
            point1 = point2
    return filtered_path

def add_lines_in_robot_direction(binary_image_with_filled_single_contour, point, unit_driving_direction, scale_factor_perpendicular, scale_factor_negative_driving_direction, scale_factor_driving_direction):
    unit_direction_perpendicular_to_driving_direction1 = np.array((-unit_driving_direction[1], unit_driving_direction[0]))
    unit_direction_perpendicular_to_driving_direction2 = np.array((unit_driving_direction[1], -unit_driving_direction[0]))


    point1 = (point + scale_factor_perpendicular * unit_direction_perpendicular_to_driving_direction1).astype(int)
    point2 = (point + scale_factor_perpendicular * unit_direction_perpendicular_to_driving_direction2).astype(int)


    start_point_line_1 = (point1 - scale_factor_negative_driving_direction * unit_driving_direction).astype(int)
    end_point_line_1 = (point1 + scale_factor_driving_direction * unit_driving_direction).astype(int)

    start_point_line_2 = (point2 - scale_factor_negative_driving_direction * unit_driving_direction).astype(int)
    end_point_line_2 = (point2 + scale_factor_driving_direction * unit_driving_direction).astype(int)

    binary_image_with_filled_single_contour_and_black_lines_in_driving_direction = binary_image_with_filled_single_contour.copy()
    cv2.line(binary_image_with_filled_single_contour_and_black_lines_in_driving_direction, start_point_line_1, end_point_line_1, color=0, thickness=8)
    cv2.line(binary_image_with_filled_single_contour_and_black_lines_in_driving_direction, start_point_line_2, end_point_line_2, color=0, thickness=8)
    return binary_image_with_filled_single_contour_and_black_lines_in_driving_direction


def add_lines_in_robot_direction_for_debugging(binary_image_with_filled_single_contour, point, unit_driving_direction, scale_factor_perpendicular, scale_factor_negative_driving_direction, scale_factor_driving_direction):
    binary_image_with_data_3_channels = expand_1_channel_image_to_3_channels_image(binary_image_with_filled_single_contour)
    binary_image_with_data_3_channels = draw_circle_on_image(rgb_image=binary_image_with_data_3_channels, center=point, color=(0, 255, 0), radius=5, thickness=-1)
    binary_image_with_data_3_channels = plot_direction_on_frame(rgb_image=binary_image_with_data_3_channels, point=point, unit_direction=unit_driving_direction,
                                                         arrow_color=(255, 0, 0), scale_factor=50, arrow_thickness=2)
    unit_direction_perpendicular_to_driving_direction1 = np.array((-unit_driving_direction[1], unit_driving_direction[0]))
    unit_direction_perpendicular_to_driving_direction2 = np.array((unit_driving_direction[1], -unit_driving_direction[0]))


    point1 = (point + scale_factor_perpendicular * unit_direction_perpendicular_to_driving_direction1).astype(int)
    point2 = (point + scale_factor_perpendicular * unit_direction_perpendicular_to_driving_direction2).astype(int)
    binary_image_with_data_3_channels = draw_circle_on_image(rgb_image=binary_image_with_data_3_channels,
                                                             center=point1, color=(0, 0, 255), radius=5, thickness=-1)
    binary_image_with_data_3_channels = draw_circle_on_image(rgb_image=binary_image_with_data_3_channels,
                                                             center=point2, color=(0, 0, 255), radius=5, thickness=-1)

    binary_image_with_data_3_channels = plot_direction_on_frame(rgb_image=binary_image_with_data_3_channels, point=point1, unit_direction=unit_driving_direction,
                                                         arrow_color=(255, 0, 255), scale_factor=50, arrow_thickness=2)

    binary_image_with_data_3_channels = plot_direction_on_frame(rgb_image=binary_image_with_data_3_channels, point=point2, unit_direction=unit_driving_direction,
                                                         arrow_color=(255, 0, 255), scale_factor=50, arrow_thickness=2)

    start_point_line_1 = (point1 - scale_factor_negative_driving_direction * unit_driving_direction).astype(int)
    end_point_line_1 = (point1 + scale_factor_driving_direction * unit_driving_direction).astype(int)

    start_point_line_2 = (point2 - scale_factor_negative_driving_direction * unit_driving_direction).astype(int)
    end_point_line_2 = (point2 + scale_factor_driving_direction * unit_driving_direction).astype(int)

    binary_image_with_data_3_channels = cv2.line(binary_image_with_data_3_channels, start_point_line_1, end_point_line_1, color=(0, 255, 0), thickness=4)
    binary_image_with_data_3_channels = cv2.line(binary_image_with_data_3_channels, start_point_line_2, end_point_line_2, color=(0, 255, 0), thickness=4)
    return binary_image_with_data_3_channels


def remove_path_going_back(path, biggest_allowed_angle):
    num_of_points = len(path)
    if num_of_points <= 2:
        return path
    first_point = path[0]
    second_point = path[1]
    filtered_path = [first_point, second_point]
    for i in range(0, num_of_points - 2):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        p3 = np.array(path[i + 2])
        vec1 = np.array(p2 - p1)
        vec2 = np.array(p3 - p2)
        unit_vec1 = vec1 / np.linalg.norm(vec1)
        unit_vec2 = vec2 / np.linalg.norm(vec2)
        dot_product = np.dot(unit_vec1, unit_vec2)
        if dot_product > 1:
            dot_product = 1
        if dot_product < -1:
            dot_product = -1
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)
        if angle_degrees > biggest_allowed_angle:
            return filtered_path
        else:
            filtered_path.append(p3)
    return filtered_path

def get_angle_from_unit_direction(unit_direction):
    x = unit_direction[0]
    y = unit_direction[1]
    angle_radians = math.atan2(y, x)
    angle_degrees = math.degrees(angle_radians)

    #check
    x1 = math.cos(angle_radians)
    y1 = math.sin(angle_radians)

    diff_x = math.fabs(x - x1)
    diff_y = math.fabs(y - y1)
    eps = 0.00001
    if diff_x > eps or diff_y > eps:
        print('error angle')
        return None
    return angle_degrees

def get_angles_to_axis_x(waypoints):
    num_of_waypoints = len(waypoints)
    if num_of_waypoints == 0:
        return None
    angles_to_x_axis = np.zeros(num_of_waypoints - 1)
    for waypoint_index in range(1, num_of_waypoints):
        current_waypoint = waypoints[waypoint_index]
        prev_waypoint = waypoints[waypoint_index - 1]
        vec = current_waypoint - prev_waypoint
        vec[1] = -vec[1] #since the y in image is from top to bottom
        size_vec = np.linalg.norm(vec)
        if math.isclose(size_vec, 0):
            angles_to_x_axis[waypoint_index - 1] = 0
            continue
        unit_vec = vec / size_vec
        angle_degrees = get_angle_from_unit_direction(unit_vec)
        angles_to_x_axis[waypoint_index - 1] = angle_degrees
    return angles_to_x_axis

def minimize_waypoints_in_one_line(waypoints, max_angle_delta=5):
    num_of_waypoints = len(waypoints)
    if num_of_waypoints <= 2:
        return waypoints
    angles_to_x_axis = get_angles_to_axis_x(waypoints)
    filtered_waypoints = [waypoints[0], waypoints[1]]
    for waypoint_index in range(2, num_of_waypoints):
        prev_angle_to_x = angles_to_x_axis[waypoint_index - 2]
        current_angle_to_x = angles_to_x_axis[waypoint_index - 1]
        diff_angles = current_angle_to_x - prev_angle_to_x
        abs_diff_angles = math.fabs(diff_angles)
        if abs_diff_angles > max_angle_delta:
            filtered_waypoints.append(waypoints[waypoint_index])
    return filtered_waypoints



def get_polygon_in_robot_direction(self, angle_degrees=20, vector_size1=450, vector_size2=250, vector_size3=50):
    unit_vec1 = self.rotate_vector(unit_vector=self.robot_current_unit_direction, angle_degrees=angle_degrees)
    unit_vec2 = self.rotate_vector(unit_vector=self.robot_current_unit_direction, angle_degrees=-angle_degrees)
    position_behind_robot = (self.robot_current_position - vector_size3 * unit_vec1).astype(int)
    unit_perpendicular_direction1 = np.array((-self.robot_current_unit_direction[1], self.robot_current_unit_direction[0]))
    #unit_perpendicular_direction2 = np.array((self.robot_current_unit_direction[1], -self.robot_current_unit_direction[0]))
    point1 = (self.robot_current_position + vector_size1 * unit_vec1).astype(int)
    point2 = (self.robot_current_position + vector_size1 * unit_vec2).astype(int)
    point3 = (position_behind_robot - vector_size2 * unit_perpendicular_direction1).astype(int)
    point4 = (position_behind_robot + vector_size2 * unit_perpendicular_direction1).astype(int)
    polygon_points = [(point1[0], point1[1]), (point2[0], point2[1]), (point3[0], point3[1]), (point4[0], point4[1])]
    return polygon_points

def add_lines_to_make_sure_the_robot_drives_in_the_right_direction(binary_image, robot_position, unit_direction, vector_size1, vector_size2, vector_size3):
    position_behind_robot = (robot_position - vector_size3 * unit_direction).astype(int)
    unit_perpendicular_direction = np.array((-unit_direction[1], unit_direction[0]))
    point1_behind_robot = (position_behind_robot - vector_size2 * unit_perpendicular_direction).astype(int)
    point2_behind_robot = (position_behind_robot + vector_size2 * unit_perpendicular_direction).astype(int)
    point3 = (point1_behind_robot + vector_size1 * unit_direction).astype(int)
    point4 = (point2_behind_robot + vector_size1 * unit_direction).astype(int)
    binary_image_with_robot_boundaries = binary_image.copy()
    cv2.line(binary_image_with_robot_boundaries, point1_behind_robot, point2_behind_robot, color=0, thickness=8)
    cv2.line(binary_image_with_robot_boundaries, point1_behind_robot, point3, color=0, thickness=8)
    cv2.line(binary_image_with_robot_boundaries, point2_behind_robot, point4, color=0, thickness=8)
    return binary_image_with_robot_boundaries



def get_path(robot_position, point, unit_direction, rgb_image, max_dist_between_consecutive_points=140):
    scale_factor_perpendicular = 30
    scale_factor_negative_driving_direction = 100
    scale_factor_driving_direction = 50

    binary_image = get_binary_image(rgb_image)
    if robot_position is not None:
        binary_image_with_lines_around_robot = add_lines_to_make_sure_the_robot_drives_in_the_right_direction(binary_image, robot_position, unit_direction,
                                                                       vector_size1=300, vector_size2=150, vector_size3=50)
        binary_image = binary_image_with_lines_around_robot
    findContoursResults = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(findContoursResults)

    contour_encapsulates_point = find_contour_encapsulates_point(point, contours)
    if contour_encapsulates_point is None:
        return [], None
    #approx_contour_encapsulates_point = get_approx_contour(contour_encapsulates_point)
    # rgb_image = draw_circle_on_image(rgb_image, point)
    binary_image_with_filled_single_contour = get_binary_image_with_filled_single_contour(rgb_image,
                                                                                          contour_encapsulates_point)
    binary_image_with_filled_single_contour_with_black_lines_around_unit_direction = add_lines_in_robot_direction(binary_image_with_filled_single_contour, point, unit_direction, scale_factor_perpendicular, scale_factor_negative_driving_direction, scale_factor_driving_direction)
    binary_image_with_filled_single_contour_with_black_lines_around_unit_direction_3_channels = add_lines_in_robot_direction_for_debugging(
        binary_image_with_filled_single_contour, point, unit_direction, scale_factor_perpendicular,
        scale_factor_negative_driving_direction, scale_factor_driving_direction)
    # skeleton, distances = medial_axis(binary_image_with_filled_single_contour, return_distance=True)
    binary_image_with_filled_single_contour_with_black_lines_around_unit_direction_0_1 = binary_image_with_filled_single_contour_with_black_lines_around_unit_direction.copy()
    binary_image_with_filled_single_contour_with_black_lines_around_unit_direction_0_1[binary_image_with_filled_single_contour_with_black_lines_around_unit_direction == 255] = 1
    skeleton_contour = skeletonize(binary_image_with_filled_single_contour_with_black_lines_around_unit_direction_0_1)
    #skeleton_contour = thin(binary_image_with_filled_single_contour)
    binary_skeleton_contour_black_background = convert_true_false_image_to_uint8(skeleton_contour)
    # binary_skeleton_contour_black_background_black_boundaries = put_black_pixels_in_image_bounderies(binary_skeleton_contour_black_background, boundary_width=80)
    binary_skeleton_contour_white_background = cv2.bitwise_not(binary_skeleton_contour_black_background)
    nearest_point = find_nearest_white(binary_skeleton_contour_black_background, point)
    path, cost = shortest_path(nearest_point, binary_skeleton_contour_white_background)

    # simplified_path = simplify_coords_vw(path, 200.0)
    simplified_path = reduce_path_lenth(path=path, max_dist_between_consecutive_points=max_dist_between_consecutive_points)
    simplified_path_without_going_back = remove_path_going_back(path=simplified_path, biggest_allowed_angle=120)
    #simplified_path_without_lasts = simplified_path[:-1]

    # rgb_image_with_path = plot_path_on_image(rgb_image=rgb_image, path=path)
    # rgb_image_with_simplified_path = plot_path_on_image(rgb_image=rgb_image, path=simplified_path)
    flipped_simplified_path_without_going_back = []
    for single_waypoint in simplified_path_without_going_back:
        flipped_simplified_path_without_going_back.append(np.array([single_waypoint[1], single_waypoint[0]]))
    rgb_image_with_simplified_path = plot_path_on_image(rgb_image=rgb_image, path=flipped_simplified_path_without_going_back)
    return simplified_path_without_going_back, binary_skeleton_contour_black_background


# def check_if_there_is_bridge(rgb_image, path):
#     if len(path) <= 10:
#         return True
#     else:
#         return False


def get_first_point_after_bridge(point, rgb_image, path, bridge_width):
    num_of_waypoints = len(path)
    if num_of_waypoints > 4:
        before_last_point = path[-4]
        last_point = path[-3]
    else:
        before_last_point = path[-3]
        last_point = path[-2]
    np_before_last_point = np.array(before_last_point)
    np_last_point = np.array(last_point)
    np_flipped_before_last_point = np.flip(np_before_last_point)
    np_flipped_last_point = np.flip(np_last_point)
    direction = np_flipped_last_point - np_flipped_before_last_point
    direction_size = np.linalg.norm(direction)
    unit_direction = direction / direction_size
    next_point_in_path = np_flipped_last_point + bridge_width * unit_direction
    next_point_in_path = next_point_in_path.astype(int)

    # rgb_image = plot_path_on_image(rgb_image, path)
    # rgb_image = draw_circle_on_image(rgb_image, np_flipped_last_point, circle_color=(255, 0, 255), radius=10)
    # rgb_image = draw_circle_on_image(rgb_image, next_point_in_path, circle_color=(255, 0, 255), radius=20)
    # cv2.imshow('rgb_image', rgb_image)
    # cv2.waitKey(0)

    return next_point_in_path


def get_path_after_bridge(point, robot_unit_direction, rgb_image, path, bridge_width, max_dist_between_consecutive_points):
    first_point_after_bridge = get_first_point_after_bridge(point, rgb_image, path, bridge_width)
    point = ((int)(first_point_after_bridge[0]), (int)(first_point_after_bridge[1]))
    path_after_bridge, binary_skeleton_contour_black_background = get_path(None, point, robot_unit_direction, rgb_image, max_dist_between_consecutive_points)
    return path_after_bridge


def get_robot_unit_direction_by_aruco(robot_data):
    if len(robot_data) == 0:
        return None
    robot_unit_directions_for_both_arucos = robot_data['robot_unit_directions_for_both_arucos']
    if robot_unit_directions_for_both_arucos[front_aruco_key] is None:
        robot_single_unit_direction = robot_unit_directions_for_both_arucos[back_aruco_key]
    else:
        robot_single_unit_direction = robot_unit_directions_for_both_arucos[front_aruco_key]
    return robot_single_unit_direction

def get_robot_current_position_by_aruco(robot_data):
    if len(robot_data) == 0:
        return None
    robot_mean_aruco_locations = robot_data['robot_mean_aruco_locations']
    if robot_mean_aruco_locations[front_aruco_key] is None:
        robot_location = robot_mean_aruco_locations[back_aruco_key]
    else:
        robot_location = robot_mean_aruco_locations[front_aruco_key]
    return robot_location

def plot_direction_on_frame(rgb_image, point, unit_direction, arrow_color, scale_factor = 50, arrow_thickness = 2):
    scaled_robot_direction = (scale_factor * unit_direction).astype(int)
    first_point = point
    second_point = first_point + scaled_robot_direction
    frame_with_robot_direction = cv2.arrowedLine(rgb_image, first_point, second_point, arrow_color, arrow_thickness)
    return frame_with_robot_direction


def plot_robot_direction_on_frame(frame, robot_data, arrow_color):
    unit_robot_direction = get_robot_unit_direction(robot_data)
    current_robot_location = get_robot_current_location(robot_data)
    frame_with_robot_direction = plot_direction_on_frame(frame, current_robot_location, unit_robot_direction, arrow_color, scale_factor=50, arrow_thickness=2)
    return frame_with_robot_direction

def plot_waypoints_on_image(rgb_image, binary_skeleton_contour_black_background, waypoints, robot_data):
    frame_index = 0
    camera_index = 1
    binary_skeleton_contour_black_background_3_channels = expand_1_channel_image_to_3_channels_image(
        binary_skeleton_contour_black_background)
    rgb_image_with_path = plot_path_on_image(rgb_image,waypoints)
    binary_image_3_channels_with_path = plot_path_on_image(binary_skeleton_contour_black_background_3_channels, waypoints)
    rgb_image_with_path_and_robot_direction = plot_robot_direction_on_frame(
        rgb_image_with_path, robot_data, arrow_color=(0, 0, 255))
    rgb_image_with_path_and_robot_direction = write_text_on_frame(
        rgb_image_with_path_and_robot_direction, f'frame {frame_index + 1}', (20, 50))

    rgb_image_with_path_and_robot_direction = write_text_on_frame(
        rgb_image_with_path_and_robot_direction, f'camera {camera_index}', (20, 100))

    resized_rgb_image_with_path_and_robot_direction = resize_image(
        rgb_image=rgb_image_with_path_and_robot_direction, scale_percent=60)
    cv2.imshow(f'camera {camera_index}', resized_rgb_image_with_path_and_robot_direction)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def get_image_with_robot_data(rgb_image_with_robot, robot_data):
    if robot_data is None:
        return rgb_image_with_robot
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.4
    text_color = (0, 255, 0)
    text_thickness = 1

    single_contour = robot_data['single_contour']
    w = robot_data['width']
    h = robot_data['height']
    x_middle = robot_data['x_middle']
    y_middle = robot_data['y_middle']
    angle_degrees = robot_data['angle_degrees']
    robot_box = robot_data['robot_box']

    middle_point = np.mean(robot_box, axis=0).astype(int)


    rgb_image_with_robot_and_data = rgb_image_with_robot.copy()
    cv2.drawContours(rgb_image_with_robot_and_data, [robot_box], 0, (0, 0, 255), 2)
    cv2.putText(rgb_image_with_robot_and_data, f'({w},{h})', (x_middle, y_middle), font, font_scale, text_color,
                text_thickness, cv2.LINE_AA)

    return rgb_image_with_robot_and_data

def get_robot_position_from_all_cameras_by_frames_diff(base_cameras_frames, all_cameras_frames, first_camera_index, start_location, cameras_where_robot_finished):
    num_of_cameras = len(base_cameras_frames)
    areas_robot_boxes_all_cameras = np.zeros(num_of_cameras, int)
    robot_data_for_all_cameras = [[]] * num_of_cameras
    for camera_index in range(0, num_of_cameras):
        if cameras_where_robot_finished[camera_index] == True:
            areas_robot_boxes_all_cameras[camera_index] = 0
            continue
        current_base_frame = base_cameras_frames[camera_index]
        current_frame = all_cameras_frames[camera_index]
        is_frame_of_first_camera = first_camera_index == camera_index
        robot_data_by_frames_diff_for_current_camera = detect_robot_location_by_diff_images(current_base_frame, current_frame,
                                                                        is_frame_of_first_camera, start_location)
        if robot_data_by_frames_diff_for_current_camera is None:
            area_robot_box_current_camera = 0
        else:
            robot_data_by_frames_diff_for_current_camera = robot_data_by_frames_diff_for_current_camera
            robot_data_for_all_cameras[camera_index] = robot_data_by_frames_diff_for_current_camera
            width = robot_data_by_frames_diff_for_current_camera['width']
            height = robot_data_by_frames_diff_for_current_camera['height']
            area_robot_box_current_camera = width * height
        areas_robot_boxes_all_cameras[camera_index] = area_robot_box_current_camera
    active_camera_index = np.argmax(areas_robot_boxes_all_cameras)
    robot_data_by_frames_diff = robot_data_for_all_cameras[active_camera_index]
    if len(robot_data_by_frames_diff) == 0:
        return None, None, None, None
    robot_contour = robot_data_by_frames_diff['single_contour']



    # robot_data_by_frames_diff['width'] = w
    # robot_data_by_frames_diff['height'] = h
    # robot_data_by_frames_diff['single_contour'] = single_contour
    # robot_data_by_frames_diff['robot_box'] = robot_box_candidate
    # robot_data_by_frames_diff['robot_area'] = robot_area

    x_middle = robot_data_by_frames_diff['x_middle']
    y_middle = robot_data_by_frames_diff['y_middle']

    angle_degrees = robot_data_by_frames_diff['angle_degrees']
    angle_radians = math.radians(angle_degrees)
    robot_unit_direction = np.array((math.cos(angle_radians), math.sin(angle_radians)))

    robot_box_by_frames_diff = robot_data_by_frames_diff['robot_box']
    robot_position = np.array((x_middle, y_middle))
    frame_with_robot = all_cameras_frames[active_camera_index]
    #ddd = get_image_with_robot_box(frame_with_robot, robot_box_by_frames_diff)
    # if box is not None:
    #     cv2.drawContours(frame_with_robot, [box], 0, (0, 0, 255), 2)
    # cv2.imshow('frame_with_robot', frame_with_robot)
    # cv2.waitKey(1)
    return robot_position, robot_unit_direction, robot_contour, active_camera_index



def get_center_of_box(box):
    if box is None:
        return None
    x = box['x']
    y = box['y']
    width = box['width']
    height = box['height']
    x_middle_box = int(x + 0.5 * width)
    y_middle_box = int(y + 0.5 * height)
    center_position = np.array((x_middle_box, y_middle_box))
    return center_position

def get_robot_position(rgb_image_base, rgb_image_with_robot, first_camera_index, active_camera_index, start_location, robot_data):
    if len(robot_data) == 0:
        is_frame_of_first_camera = first_camera_index == active_camera_index
        robot_box_by_frames_diff = detect_robot_location_by_diff_images(rgb_image_base, rgb_image_with_robot,
                                                                            is_frame_of_first_camera, start_location)
        if robot_box_by_frames_diff is None:
            david = 5
        robot_position = get_center_of_box(robot_box_by_frames_diff)
    else:
        robot_position = robot_data['robot_mean_of_front_and_back_mean_locations']
    return robot_position

def detect_robot_location_by_diff_images(rgb_image_base, rgb_image_with_robot, is_frame_of_first_camera, start_location):
    robot_box = None
    robot_min_width = 50
    robot_max_width = 300

    robot_min_height = 50
    robot_max_height = 300

    gray_image_base = cv2.cvtColor(rgb_image_base, cv2.COLOR_BGR2GRAY)
    gray_image_with_robot = cv2.cvtColor(rgb_image_with_robot, cv2.COLOR_BGR2GRAY)



    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(gray_image_base, gray_image_with_robot, full=True)
    diff_image = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))

    diff_image1_float = gray_image_with_robot.astype(float) - gray_image_base.astype(float)
    abs_diff_image1_float = np.abs(diff_image1_float)
    diff_image1 = abs_diff_image1_float.astype(np.uint8)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    binary_image = cv2.threshold(diff_image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    erode_kernel_size = 9
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    erode_binary_image = cv2.erode(binary_image, erode_kernel)

    dilate_kernel_size = 9
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilate_binary_image = cv2.dilate(erode_binary_image, dilate_kernel)


    contours = cv2.findContours(dilate_binary_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.4
    text_color = (0, 255, 0)
    text_thickness = 1
    rgb_image_with_robot_and_data = rgb_image_with_robot.copy()
    robot_boxes_data_candidates = []
    for single_contour in contours:
        #(x, y, w, h) = cv2.boundingRect(single_contour)
        rect = cv2.minAreaRect(single_contour)
        robot_box_candidate = cv2.boxPoints(rect)
        robot_box_candidate = np.int0(robot_box_candidate)
        (x_middle, y_middle) = rect[0]
        (w, h) = rect[1]
        x_middle = int(x_middle)
        y_middle = int(y_middle)
        w = int(w)
        h = int(h)
        angle_degrees = rect[2]

        cv2.drawContours(rgb_image_with_robot_and_data, [robot_box_candidate], 0, (0, 0, 255), 2)
        cv2.putText(rgb_image_with_robot_and_data, f'({w},{h})', (x_middle, y_middle), font, font_scale, text_color, text_thickness,
                    cv2.LINE_AA)

        #cv2.rectangle(gray_image_with_robot, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if w <= robot_max_width and w >=robot_min_width and h <= robot_max_height and h >= robot_min_height:
            if is_frame_of_first_camera == True:
                x_dist_from_start_location = abs(start_location[0] - x_middle)
                y_dist_from_start_location = abs(start_location[1] - y_middle)
                if x_dist_from_start_location < robot_max_width and y_dist_from_start_location < robot_max_height:
                    # robot_box_candidate = {'x': x, 'y': y, 'width': w, 'height': h}
                    # ddd = get_image_with_robot_box(rgb_image_with_robot, robot_box_candidate)
                    # cv2.imshow('ddd', ddd)
                    # cv2.waitKey(0)
                    continue
            robot_area = cv2.contourArea(single_contour)
            single_robot_data_candidate = dict()
            single_robot_data_candidate['width'] = w
            single_robot_data_candidate['height'] = h
            single_robot_data_candidate['x_middle'] = x_middle
            single_robot_data_candidate['y_middle'] = y_middle
            single_robot_data_candidate['angle_degrees'] = angle_degrees
            single_robot_data_candidate['single_contour'] = single_contour
            single_robot_data_candidate['robot_box'] = robot_box_candidate
            single_robot_data_candidate['robot_area'] = robot_area
            robot_boxes_data_candidates.append(single_robot_data_candidate)
    robot_candidates_areas = np.array(
        [single_robot_data_candidate['robot_area'] for single_robot_data_candidate in robot_boxes_data_candidates])
    if len(robot_candidates_areas) == 0:
        return None
    index_max = np.argmax(robot_candidates_areas)
    robot_data = robot_boxes_data_candidates[index_max]
    robot_box = robot_data['robot_box']

    ddd = get_image_with_robot_data(rgb_image_with_robot, robot_data)
    # cv2.imshow('ddd', ddd)
    # cv2.waitKey(1)

    # cv2.imshow('david', rgb_image_with_robot_and_data)
    # cv2.waitKey(0)
    # cv2.drawContours(rgb_image_with_robot, [robot_data['single_contour']], 0, (0, 0, 255), 2)
    # cv2.imshow('rgb_image_with_robot', rgb_image_with_robot)
    # cv2.waitKey(0)
    return robot_data

def get_waypoints(rgb_image, robot_position, robot_unit_direction, image_data_type):
    if robot_position is None or robot_unit_direction is None:
        return []
    if image_data_type == ImageDateType.IMAGE_WITH_ROBOT:
        next_waypoint = get_next_waypoint(rgb_image, robot_position, robot_unit_direction)
    else:
        next_waypoint = robot_position
    rgb_image_with_next_point = draw_circle_on_image(rgb_image, next_waypoint, color=(0, 255, 0), radius=10, thickness=-1)
    point = ((int)(next_waypoint[0]), (int)(next_waypoint[1]))
    # path, binary_skeleton_contour_black_background = get_path(point, robot_unit_direction, rgb_image, max_dist_between_consecutive_points=50)
    # rgb_image_with_path = plot_path_on_image(rgb_image=rgb_image, path=path)
    # is_bridge = check_if_there_is_bridge(rgb_image, path)
    # if is_bridge:
    #     path_after_bridge = get_path_after_bridge(point, robot_unit_direction, rgb_image, path, bridge_width=600, max_dist_between_consecutive_points=140)
    #     # path, binary_skeleton_contour_black_background = get_path(point, robot_unit_direction, rgb_image,
    #     #                                                           max_dist_between_consecutive_points=140)
    #     path = path + path_after_bridge
    #     rgb_image_with_path_including_bridge = plot_path_on_image(rgb_image=rgb_image, path=path)
    # else:
    path, binary_skeleton_contour_black_background = get_path(robot_position, point, robot_unit_direction, rgb_image,
                                                                  max_dist_between_consecutive_points=140)
    rgb_image_with_path = plot_path_on_image(rgb_image=rgb_image, path=path)
    return path
