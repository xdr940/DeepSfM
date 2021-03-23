
from path import Path
from utils.official import readlines,writelines
import os
from datasets.kitti_dataset_v2 import relpath_split
def main():
    data_base = Path('/home/roit/datasets/kitti')
    path = Path('/home/roit/datasets/splits/kitti/eigen_std/test.txt')
    out_file = Path('/home/roit/datasets/splits/kitti/eigen_std/wo.txt')
    lines = readlines(path)
    note_in =[]
    for item in lines:
        date, scene, camera, frame = relpath_split(item)
        reframe_forward = str(int(frame) -1)
        reframe_passward = str(int(frame) +1)

        path = os.path.join(
            date,
            scene,
            camera,
            'data',
            "{:010d}".format(int(reframe_forward))
        )
        image_path1 = data_base / path + '.png'
        path = os.path.join(
            date,
            scene,
            camera,
            'data',
            "{:010d}".format(int(reframe_passward))
        )
        image_path2 = data_base / path + '.png'
        if not Path.exists(image_path1) or not Path.exists(image_path2):

            note_in.append(item)
            print(item)


    for item in note_in:
        lines.remove(item)
    print(len(lines))
    print('total {}'.format(len(note_in)))
    writelines(out_file,lines)






if __name__ == '__main__':
    main()