# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import numpy as np
import mmengine
from multiprocessing import Pool

from pycocotools.coco import COCO


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
    os.system(
        f'ffmpeg -i {video_path} -f image2 -r 30 -b:v 5626k {image_path}/%06d.png'  # noqa
    )
    
def split_dataset(annotation_path: str, split_path: str):
    folders = os.listdir(annotation_path)
    splits = np.load(split_path)
    train_annos = dict()
    val_annos = dict()
    train_imgs = []
    val_imgs = []
    t_id = 0
    v_id = 0
    for scene in folders:
        data = COCO(os.path.join(annotation_path, scene, 'keypoint_annotation.json'))
        for aid in data.anns.keys():
            ann = data.anns[aid]
            img = data.loadImgs(ann['image_id'])[0]
            
            if img['file_name'].startswith('/'):
                file_name = img['file_name'][1:]   # [1:] means delete '/'
            else:
                file_name = img['file_name']
            video_name = file_name.split('/')[-2]
            if 'Trim' in video_name:
                video_name = video_name.split('_Trim')[0]
            
            if video_name in splits:
                val_imgs.append(img)
                val_annos[v_id] = ann
                v_id += 1
            else:
                train_imgs.append(img)
                train_annos[t_id] = ann
                t_id += 1
    train_data = dict(images=train_imgs, annotations=train_annos)
    val_data = dict(images=val_imgs, annotations=val_annos)
    
    mmengine.dump(train_data, os.path.join(annotation_path, 'train_annotation.json'))
    mmengine.dump(val_data, os.path.join(annotation_path, 'val_annotation.json'))
    
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', type=str, default='data/ubodyv1')
    args = parser.parse_args()
    video_root = f'{args.data_root}/videos'
    split_path = f'{args.data_root}/splits/intra_scene_test_list.npy'
    annotation_path = f'{args.data_root}/annotations'

    video_paths = findAllFile(video_root)
    pool = Pool(processes=1)
    pool.map(convert, video_paths)
    pool.close()
    pool.join()

    split_dataset(annotation_path, split_path)
