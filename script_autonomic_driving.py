from dark_room import DarkRoom

def main():
    dr = DarkRoom()
    dr.move_robot_through_all_cameras()
    dr.create_videos()

if __name__ == "__main__":
    main()
