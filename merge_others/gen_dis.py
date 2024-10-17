import sys
# sys.path.append('Users/helenpeng/Desktop/project1/')
# print(sys.path)
from x_distortion import add_distortion
import json
import cv2
import mmcv
import os


def add_dis_md(json_path_in, im_path_in, im_path_out):
    with open(json_path_in) as f:
        file = json.load(f)
    for i in file:
        img = cv2.cvtColor(mmcv.imread(os.path.join(im_path_in, i["image_A"].split('/')[-1].split('_')[0]+'.png')), cv2.COLOR_BGR2RGB)
        if i["distortion_names"] is not None:
            for idx in range(len(i['distortion_names'])):
                img=add_distortion(img=img, severity=i['severities'][idx], distortion_name=i['distortion_names'][idx])
        mmcv.imwrite(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), os.path.join(im_path_out, i["image_A"].split('/')[-1]))


def add_dis_sd(json_path_in, im_path_in, im_path_out):
    with open(json_path_in) as f:
        file = json.load(f)
    for i in file:
        img = cv2.cvtColor(mmcv.imread(os.path.join(im_path_in, i["image_A"].split('/')[-1].split('_')[0]+'.png')), cv2.COLOR_BGR2RGB)
        if i["distortion_name"] is not None:
            img=add_distortion(img=img, severity=i['severity'], distortion_name=i['distortion_name'])
        mmcv.imwrite(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), os.path.join(im_path_out, i["image_A"].split('/')[-1].split('_')[0]+'.png'))
            

if __name__=='__main__':
    # add_dis_sd(json_path_in=r'/root/project1/DataDepictQA_ours/KADIS700K/A_sd_detail/train_A_sd_detail_20k.json', 
    #            im_path_in=r'/root/autodl-tmp/kadis700k/ref_imgs',
    #            im_path_out=r'/root/autodl-tmp/out')
    add_dis_md(json_path_in=r'/root/project1/DataDepictQA_ours/KADIS700K/A_md_detail/train_A_md_detail_8k.json', 
               im_path_in=r'/root/autodl-tmp/kadis700k/ref_imgs',
               im_path_out=r'/root/autodl-tmp/out')
    