import os
import os.path as osp
from .bases import BaseImageDataset
import re
import pdb
import random


class WMVEID863(BaseImageDataset):

    def __init__(self, root, verbose=True):
        self.root = root
        self.train_dir = os.path.join(root, 'train')
        self.gallery_dir = os.path.join(root, 'test')
        self.query_dir = os.path.join(root, 'query')

        self.check_dir_exist()

        # self.train, self.num_train_pids, self.num_train_cams = self.get_imagedata_info(self.train_dir, relabel=True)
        # self.gallery, self.num_gallery_pids, self.num_gallery_cams = self.get_imagedata_info(self.gallery_dir, relabel=False)
        # self.query, self.num_query_pids, self.num_query_cams = self.get_imagedata_info(self.query_dir, relabel=False, ratio=1)

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def get_data(self, folder, relabel=False, ratio=1):
        vids = os.listdir(folder)
        # pdb.set_trace()

        if ratio != 1:
            print('randomly sample ', ratio, 'ids for ttt')
            vids = random.sample(vids, int(len(vids) * ratio))
        labels = [int(vid) for vid in vids]

        if relabel:
            label_map = dict()
            for i, lab in enumerate(labels):
                label_map[lab] = i
        cam_set = set()
        img_info = []
        for vid in vids:
            id_vimgs = os.listdir(os.path.join(folder, vid, "vis"))
            id_nimgs = os.listdir(os.path.join(folder, vid, "ni"))
            # print(vid)
            id_timgs = os.listdir(os.path.join(folder, vid, "th"))
            for i, img in enumerate(id_vimgs):
                vpath = os.path.join(folder, vid, "vis", id_vimgs[i])
                npath = os.path.join(folder, vid, "ni", id_nimgs[i])
                tpath = os.path.join(folder, vid, "th", id_timgs[i])
                label = label_map[int(vid)] if relabel else int(vid)

                night = re.search('n+\d', img).group(0)[1]
                cam = re.search('v+\d', img).group(0)[1]
                cam = int(cam)
                night = int(night)
                cam_set.add(cam)
                trackid = -1
                img_info.append(((vpath, npath, tpath), label, cam, trackid))

        return img_info, len(vids), len(cam_set)

    def check_dir_exist(self):
        if not os.path.exists(self.root):
            raise Exception('Error path: {}'.format(self.root))
        if not os.path.exists(self.train_dir):
            raise Exception('Error path:{}'.format(self.train_dir))
        if not os.path.exists(self.gallery_dir):
            raise Exception('Error path:{}'.format(self.gallery_dir))
        if not os.path.exists(self.query_dir):
            raise Exception('Error path:{}'.format(self.query_dir))

    def _process_dir(self, dir_path, relabel=False):
        vid_container = set()
        for vid in os.listdir(dir_path):
            vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for vid in os.listdir(dir_path):
            vid_path = osp.join(dir_path, vid)
            r_data = os.listdir(osp.join(vid_path, 'vis'))
            vid = int(vid)
            if relabel:
                vid = vid2label[vid]
            for img in r_data:
                r_img_path = osp.join(vid_path, 'vis', img)
                n_img_path = osp.join(vid_path, 'ni', img)
                t_img_path = osp.join(vid_path, 'th', img)
                camid = int(img[1])
                sceneid = int(img[7:10])  # scene id
                assert 0 <= camid <= 7
                dataset.append(((r_img_path, n_img_path, t_img_path), vid, camid, sceneid))
        return dataset
