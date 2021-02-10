import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import pathlib
import re


def iou_all_at_once(box, boxes):  # [xmin, ymin, xmax, ymax]
    eps = 1e-5
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    #intersの計算
    inter_boxes = np.zeros_like(boxes)
    np.clip(boxes[:, 0::2], box[0], box[2], out=inter_boxes[:, 0::2])
    np.clip(boxes[:, 1::2], box[1], box[3], out=inter_boxes[:, 1::2])
    inters = (inter_boxes[:, 2] - inter_boxes[:, 0]) * (inter_boxes[:, 3] - inter_boxes[:, 1])
    #IoUの計算
    unions = boxes_areas + box_area - inters
    IoUs = inters / (unions + eps)
    return IoUs


class RefDataProduction:
    def __init__(self, anno_df, save_folder=None, non_ref_save_folder=None):
        self.eps = 1e-5
        self.ADD_RATE = 0.008
        self.IMG_SIZE = 448
        self.anno_df = anno_df
        self.save_folder = save_folder
        self.non_ref_save_folder = non_ref_save_folder
        
        
    def make_non_ref_data(self, png_path):
        img = cv2.imread(png_path)
        H,W,C = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[:, :, np.newaxis] * np.ones((H, W, C))
        self.additional_range = int(np.sqrt(self.ADD_RATE * H * W))
        png_file_name = pathlib.Path(png_path).name
        non_ref_boxes, ref_boxes_num = self._get_non_ref_boxes(png_file_name, img.shape)
        print('len: non_ref: ', len(non_ref_boxes))
#         print('len ref: ', ref_boxes_num)
        if len(non_ref_boxes) != 0:
            partial_img_list = self._get_non_ref_data(img, non_ref_boxes)
            partial_imgs_selected, _ = self._select_non_ref_data(partial_img_list, ref_boxes_num)#ref_boxes_num
#             partial_imgs_selected = self._exclude_empty_data(partial_img_list)
            for j, partial_img in enumerate(partial_imgs_selected):
                cv2.imwrite(self.non_ref_save_folder + png_file_name[:-4] + '_no' + str(j) + png_file_name[-4:], partial_img)
        
     
    
    
    def main(self, png_path):
        img = cv2.imread(png_path)
        H,W,C = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[:, :, np.newaxis] * np.ones((H, W, C))
        self.additional_range = int(np.sqrt(self.ADD_RATE * H * W))
        png_file_name = pathlib.Path(png_path).name
        ref_boxes = self._get_ref_boxes(png_file_name)

        if len(ref_boxes) != 0:
            for j, bbox in enumerate(ref_boxes):
                partial_img = self._patch(img, bbox)

                if partial_img is None:
                    print('noneです。img_path: ', png_path)
        #             print(partial_img.shape)
        #             print(save_folder + png_file_name[:-4] + '_no' + str(j) + png_file_name[-4:])
        #             print('partial_Img shape: ', partial_img.shape)
                partial_img = cv2.resize(partial_img, (self.IMG_SIZE, self.IMG_SIZE))
                cv2.imwrite(self.save_folder + png_file_name[:-4] + '_no' + str(j) + png_file_name[-4:], partial_img)
                
    
    def _get_non_ref_data(self, img, non_ref_boxes):
        partial_img_list = []
        for bbox in non_ref_boxes:
            partial_img = self._patch(img, bbox, non_ref=True)
#             print('partial_img shape: ', partial_img.shape)

            if partial_img is None:
                print('noneです。')
#             print(partial_img.shape)
    #             print(save_folder + png_file_name[:-4] + '_no' + str(j) + png_file_name[-4:])
#             print('partial_Img shape: ', partial_img.shape)
            partial_img = cv2.resize(partial_img, (self.IMG_SIZE, self.IMG_SIZE))
            partial_img_list.append(partial_img)
        return partial_img_list
        
        
        
        
        
        
    def _select_non_ref_data(self, partial_img_list, ref_data_num):
        partial_imgs = np.stack(partial_img_list, axis=0)
        N, H, W, C = partial_imgs.shape
        partial_imgs_INV = 255 - partial_imgs
        partial_imgs_areas = np.sum(partial_imgs_INV.reshape(N, H*W*C), axis=1)
        idxes = np.argsort(-partial_imgs_areas)
        selected_idxes = idxes[:ref_data_num]
        partial_imgs = partial_imgs[selected_idxes, :, :, :]
        return partial_imgs, selected_idxes
    
    
    def _exclude_empty_data(self, partial_img_list):
        partial_imgs = np.stack(partial_img_list, axis=0)
        N, H, W, C = partial_imgs.shape
        partial_imgs_INV = 255 - partial_imgs
        partial_imgs_areas = np.sum(partial_imgs_INV.reshape(N, H*W*C), axis=1)
        idxes = np.where(partial_imgs_areas != 0)[0]
        partial_imgs = partial_imgs[idxes]
        return partial_imgs
        
        
        
        
    def _get_ref_boxes(self, png_file_name):
        mask0 = self.anno_df.loc[:, 'ファイル名'] == png_file_name
    #     print(np.unique(np.array(df.loc[:, '枠'])))
        mask1 = self.anno_df.loc[:, '枠'] == '矩形'
        mask = mask0 * mask1
        mask = mask.astype(bool)
#         print('mask: ', sum(mask))
        ref_boxes = self.anno_df.loc[mask, ['座標X','座標Y', '座標X.1','座標Y.1']]
        ref_boxes_arr = np.array(ref_boxes)
        x_length = np.abs(ref_boxes_arr[:, 2] - ref_boxes_arr[:, 0])
        y_length = np.abs(ref_boxes_arr[:, 3] - ref_boxes_arr[:, 1])
        tateyoko_mask = x_length > y_length
    #     print(len(tateyoko_mask))
        tateyoko_ratios = np.zeros(len(tateyoko_mask))
        x_bunbo =  y_length / (x_length + self.eps)
        y_bunbo = x_length / (y_length + self.eps)
    #     print(len(tateyoko_ratios))
        tateyoko_ratios[tateyoko_mask] =x_bunbo[tateyoko_mask]
        tateyoko_ratios[~tateyoko_mask] = y_bunbo[~tateyoko_mask]
        tateyoko_ratio_mask = tateyoko_ratios > 0.5
        ref_boxes_arr = ref_boxes_arr[tateyoko_ratio_mask, :]
        return ref_boxes_arr

    
    

    def _get_non_ref_boxes(self, png_file_name,  img_shape):
        non_ref_boxes = []
        ref_boxes_arr = self._get_ref_boxes(png_file_name)
        if len(ref_boxes_arr) <=1:
            return [], 0
        ref_boxes_arr_cpy = np.copy(ref_boxes_arr)
        ref_boxes_arr_cpy[:, 0] = np.amin(ref_boxes_arr[:, 0::2], axis=1)
        ref_boxes_arr_cpy[:, 1] = np.amin(ref_boxes_arr[:, 1::2], axis=1)
        ref_boxes_arr_cpy[:, 2] = np.amax(ref_boxes_arr[:, 0::2], axis=1)
        ref_boxes_arr_cpy[:, 3] = np.amax(ref_boxes_arr[:, 1::2], axis=1)
        ref_boxes_arr = ref_boxes_arr_cpy
        ref_boxes_num = len(ref_boxes_arr)
        ref_boxes_length_means = np.mean(ref_boxes_arr[:, 2:] - ref_boxes_arr[:, :2], axis=0)
        for i in range(int(ref_boxes_num * 1.2)):

            while True:
                x_start, y_start = np.random.rand(2) * (np.array(img_shape[:2])[::-1] - 1)
                x_start, y_start = int(img_shape[1] * 0.1 + x_start), int(img_shape[0] * 0.1 + y_start)
                
                is_change_box_size = np.random.rand() < 0.2
                
                if is_change_box_size:
                    length_error = np.random.normal(loc=1.5, scale=1, size=1)
                    x_length_error, y_length_error = length_error *  (1 + np.random.normal(loc=0.0, scale=0.08, size=2))
                else:
                    x_length_error, y_length_error = np.random.normal(loc=0.0, scale=0.08, size=2)
                    
                x_length, y_length =  ref_boxes_length_means[0] *(1+ x_length_error), ref_boxes_length_means[1] * (1+ y_length_error)
                x_end = x_start +x_length
                y_end = y_start + y_length
                if x_end > img_shape[1]-1 or y_end > img_shape[0]-1:
                    continue
                non_ref_box = np.array([x_start, y_start, x_end, y_end])
                
                IoUs = iou_all_at_once(non_ref_box, ref_boxes_arr)
                if np.sum(IoUs) == 0:
                    non_ref_boxes.append(non_ref_box)
                    break
        
        non_ref_boxes = np.vstack(non_ref_boxes)
        return non_ref_boxes, ref_boxes_num
    
                
        
        
                                
        
        
        
    
    
    
    
    
    def _patch(self, img, bbox, non_ref=False):
        bbox = list(map(int, bbox))
        ref_x_min, ref_y_min, ref_x_max, ref_y_max = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])

#             print('bbox: ', bbox)
        x_min, x_max = max(int(ref_x_min - self.additional_range), 0) ,  min(int(ref_x_max + self.additional_range), img.shape[1]-1)

        if x_min == 0:
#                 print('x_minがゼロなのでx_maxを広げます。x_max:{}'.format(x_max))
            x_max += abs(ref_x_min - self.additional_range)
#                 print('after x_max:', x_max)
        elif x_max ==  img.shape[1]-1:
#                 print('x_maxがふちなのでx_minを広げます。x_min:{}'.format(x_min))
            x_min -= ref_x_max + self.additional_range - (img.shape[1]-1)

        y_min, y_max = max(int(ref_y_min - self.additional_range), 0),  min(int(ref_y_max + self.additional_range), img.shape[0]-1)

        if y_min == 0:
#                 print('y_minがゼロなのでy_maxを広げます。y_max:{}'.format(y_max))
            y_max += abs(ref_y_min - self.additional_range)
#                 print('after y_max:', y_max)
        elif y_max ==  img.shape[0]-1:
#                 print('y_maxがふちなのでy_minを広げます。y_min:{}'.format(y_min))
            y_min -= ref_y_max + self.additional_range - (img.shape[0]-1)

#             print('x_length:{}, y_length:{}'.format(x_max - x_min, y_max - y_min))

        ref_x_min, ref_x_max = ref_x_min - x_min, ref_x_max - x_min
        ref_y_min, ref_y_max = ref_y_min  -  y_min, ref_y_max - y_min

        if not non_ref:
            ref_x_min, ref_y_min, ref_x_max, ref_y_max = self._add_box_error(np.array([ref_x_min, ref_y_min, ref_x_max, ref_y_max]),
                                                                       0.008, 0.08, img.shape)
        partial_img = img[y_min : y_max, x_min : x_max].copy()
        partial_img = cv2.rectangle(partial_img, (ref_x_min, ref_y_min), (ref_x_max, ref_y_max), (0, 0, 255), 2)
        return partial_img
        
        
        
        
    def _add_box_error(self, box, position_scale, length_scale, img_shape):
        x_position_error, y_position_error = np.random.normal(loc=0.0, scale=position_scale, size=2)
        x_length_error, y_length_error = np.random.normal(loc=0.0, scale=length_scale, size=2)
        box[2:] -= box[:2]
        box[0] += box[0]*x_position_error
        box[1]+=box[1]*y_position_error
        box[2]+=box[2]*x_length_error
        box[3]+=box[3]*y_length_error
        box[2:] +=box[:2]
        box[0::2] = np.maximum(np.minimum(box[0::2], img_shape[1]-1), 0)
        box[1::2] = np.maximum(np.minimum(box[1::2], img_shape[0]-1), 0)
        return box