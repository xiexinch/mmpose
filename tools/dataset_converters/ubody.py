# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from multiprocessing import Pool


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def convert(video_path: str):
    video_name = video_path.split('/')[-1]
    image_path = video_path.replace(video_name, video_name.split('.')[0])
    image_path = image_path.replace('/videos/', '/images/')
    os.makedirs(image_path, exist_ok=True)
    print(
        f'ffmpeg -i {video_path} -f image2 -r 30 -b:v 5626k {image_path}/%06d.png'  # noqa
    )
    raise '123'
    os.system(
        f'ffmpeg -i {video_path} -f image2 -r 30 -b:v 5626k {image_path}/%06d.png'  # noqa
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video-folder', type=str, default='data/ubodyv1/videos')
    args = parser.parse_args()
    video_paths = findAllFile(args.video_folder)
    pool = Pool(processes=1)
    pool.map(convert, video_paths)
    pool.close()
    pool.join()
