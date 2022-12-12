import gdown
import os
import common_utils
from dark_room import DarkRoom


def download_input_videos_from_google_drive(videos_and_images_folder):
    videos_sub_folder_name = 'videos'
    videos_sub_folder_full_path = videos_and_images_folder + '/' + videos_sub_folder_name

    is_folder_exist = os.path.exists(videos_sub_folder_full_path)
    if is_folder_exist:
        return
    else:
        os.makedirs(videos_sub_folder_full_path)

    google_drive_prefix_url = 'https://drive.google.com/uc?id='

    url1_id = '170dVWpcfFjLaBjQKRvXT0Njx-NdgCqeE'
    url2_id = '1cEb91AWmzwiNN4q55ghYcTUIdQM3Jo4s'
    url3_id = '1-0STdItDTaGQbmMfrljidEZdaDhsqfw1'
    url4_id = '1S9wYkk_6vm17fsbaj6keftYsqtTuodmi'

    url_ids = [url1_id, url2_id, url3_id, url4_id]
    videos_names = ['IGXJVS.avi', 'TGASLM.avi', 'UOXBGL.avi', 'VSYAJL.avi']

    num_of_videos = len(videos_names)

    for video_index in range(0, num_of_videos):
        video_name = videos_names[video_index]
        print()
        print(f'Downloading video number {video_index + 1} out of {num_of_videos}. video name: {video_name}')
        url_id = url_ids[video_index]
        url = google_drive_prefix_url + url_id
        output = videos_sub_folder_full_path + '/' + video_name
        gdown.download(url, output, quiet=False)
    print('Finished downloading all videos')





def main():
    videos_and_images_folder = './videos_and_images'
    download_input_videos_from_google_drive(videos_and_images_folder)
    common_utils.extract_frames_from_videos(videos_and_images_folder)
    dr = DarkRoom()
    dr.move_robot_through_all_cameras()


if __name__ == "__main__":
    main()
