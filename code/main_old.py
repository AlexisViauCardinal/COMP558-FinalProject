import argparse
from os.path import isdir, isfile
from tracker import Tracker


def handle_photo(dir_path : str, out_video_path : str):

    
    pass

def handle_video(vid_path : str, out_video_path : str):
    pass

def main():
    parser = argparse.ArgumentParser(description="Process photo and video paths.")
    
    # Define the arguments
    parser.add_argument('--photo_dir', type=str, help="Directory containing photos.")
    parser.add_argument('--video_path', type=str, help="Path to the video file.")
    parser.add_argument('--out_video_path', type=str, help="Path to the output video file.")
    
    # Parse the arguments
    args = parser.parse_args()

    if (not (args.photo_dir or args.video_path)) or (args.photo_dir and args.video_path):
        raise ValueError("Specify one of --photo_dir or --video_path")
    
    if not args.out_video_path:
        raise ValueError("Specify --out_video_path")

    # Check the validity of the arguments
    if args.photo_dir:
        if not isdir(args.photo_dir):
            raise ValueError("Invalid photo directory specified")
        print(f"Processing photos from directory: {args.photo_dir}")
        handle_photo(args.photo_dir, args.out_video_path)
    
    if args.video_path:
        if not isfile(args.video_path):
            raise ValueError("Invalid video path specified")
        print(f"Processing video from file: {args.video_path}")
        handle_video(args.video_path, args.out_video_path)

if __name__ == '__main__':
    main()
