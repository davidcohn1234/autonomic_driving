import os
import cv2

def extract_frames_from_videos(videos_and_images_folder):
    video_folder_path = videos_and_images_folder + '/videos'
    images_folder_path = videos_and_images_folder + '/images'
    video_names = os.listdir(video_folder_path)

    isFolderExist = os.path.exists(images_folder_path)
    if isFolderExist:
        return
    else:
        os.mkdir(images_folder_path)

    video_count = 0
    num_of_videos = len(video_names)
    for video_name_with_extension in video_names:
        video_count += 1
        video_name_without_extension = os.path.splitext(video_name_with_extension)[0]
        current_video_full_path = video_folder_path + '/' + video_name_with_extension
        vidcap = cv2.VideoCapture(current_video_full_path)
        current_video_num_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vidcap.read()
        frame_count = 0
        images_sub_folder = images_folder_path + '/' + video_name_without_extension
        isSubFolderExist = os.path.exists(images_sub_folder)
        if isSubFolderExist == False:
            os.mkdir(images_sub_folder)
        while success:
            frame_count += 1
            frame_name = "%05d.jpg" % frame_count
            frame_full_path = images_sub_folder + '/' + frame_name
            cv2.imwrite(frame_full_path, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print(
                f'video {video_count} out of {num_of_videos} videos. frame {frame_count} out of {current_video_num_of_frames} frames')
        print()