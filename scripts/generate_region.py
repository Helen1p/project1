from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
# utils_image是加混合distortion的
import utils_image
# iqa_distortions是加单种distortion的，Re-IQA
from iqa_distortions import *
import argparse
import json
import os
import torch
import pycocotools.mask as maskUtils
import random
from typing import Dict, Optional, Union
from functools import reduce
import operator

# np.random.seed(666)

distortion = {
    1: 'noise',
    2: 'blur',
    3: 'jpeg',
}


def show_anns(anns, image):
    if len(anns) == 0:
        return
    print(len(anns))
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    dpi = plt.rcParams['figure.dpi']
    height, width = image.shape[:2]
    plt.figure(figsize=(width/dpi, height/dpi))
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.6]])
        img[m] = color_mask
    ax.imshow(img)
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img = np.asarray(img)
    buf.close()
    plt.close()
    return img


def region_regress(anns, region_thresh, img_area, area_initial=0.01):
    while len(anns['annotations']) > region_thresh:
        anns['annotations'] = [m for m in anns['annotations'] if m['area'] > img_area * (area_initial**2)]
        # 0.05太大了
        area_initial += 0.003
    print(f'The area thresh is {area_initial**2:2f}.')
    return anns


def iqa_transformations(choice, im):

        level = random.randint(0,2)

        dis_dict={'1':'gaussian blur', '2':'lens blur', '3':'color diffuse', '4':'color shift', '5':'color saturation decrease', '6':'JPEG_compression', \
                  '7':'white noise', '8':'impulse noise', '9':'multiplicative noise', '10':'over exposure', '11':'low light', '12':['mean increase', 'mean decrease'], \
                  '13':'bicubic interpolation resize', '14':'sharpen', '15':['contrast increase', 'contrast decrease'], '16':'color block', '17':'pixelate', '18':'jitter', '19':'motion blur', \
                    '20':'lanczos interpolation resize', '21':'bilinear interpolation resize'}

        # 去掉color相关distortion里面levels出现的负值level
        if choice == 1:
            # gaussian blur
            im = imblurgauss(im, level)

        elif choice == 2 :
            # lens blur
            im = imblurlens(im,level)
        
        elif choice == 3 :
            # color diffuse
            im = imcolordiffuse(im,level)

        elif choice == 4 :
            # color shift
            im = imcolorshift(im,level)

        elif choice == 5 :
            # color saturate decrease in HSV color space
            im = imcolorsaturate(im,level)

        elif choice == 6 :
            # jpeg compression
            im = imcompressjpeg(im,level)

        elif choice == 7 :
            # white noise in RGB space
            im = imnoisegauss(im,level)

        elif choice == 8 :
            # impulse noise
            im = imnoiseimpulse(im,level)

        elif choice == 9 :
            # multiplicative noise
            im = imnoisemultiplicative(im,level)

        elif choice == 10 :
            # brighten
            im = imbrighten(im,level)

        elif choice == 11 :
            # darken
            im = imdarken(im, level)

        elif choice == 12 :
            # mean shift
            i = random.randint(0,1)
            im = immeanshift(im,level, i)

        elif choice == 13 :
            # bicubic interpolation resize
            im = imresizedist_bicubic(im,level)

        elif choice == 14 :
            # sharpen
            im = imsharpenHi(im,level)

        elif choice == 15 :
            # contrast change
            i = random.randint(0,1)
            im = imcontrastc(im,level, i)

        elif choice == 16 :
            # color block
            im = imcolorblock(im,level)

        elif choice == 17 :
            # pixelate
            im = impixelate(im,level)

        elif choice == 18 :
            # jitter
            im = imjitter(im,level)

        elif choice == 19:
            # motion blur
            im = imblurmotion(im,level)
        
        elif choice == 20:
            # lanczos interpolation resize
            im = imresizedist_lanczos(im,level)
        
        elif choice == 21:
            # bilinear interpolation resize
            im = imresizedist_bilinear(im,level)

        else :
            pass
        
        if choice == 12 or choice == 15:
            return im, dis_dict[str(choice)][i], level
        else:
            return im, dis_dict[str(choice)], level


def add_single_region_distortion(anns, image: np.ndarray):
    # d_image = image.copy().astype(np.float32) / 255.
    d_image = image.copy()
    # bbox_list=[]
    # distortion_list=[]
    if anns['annotations'] is not None:
        for i in range(len(anns['annotations'])):
            ann = anns['annotations'][i]
            # ann['segmentation']['counts']=torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            d_region = d_image[y: y+h, x: x+w]
            
            d_region_ori = d_region.copy()
            d_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()[y: y+h, x: x+w]
            # d_mask = ann['segmentation']['counts'][y: y+h, x: x+w]
            
            # d_region: np.array -> Image
            # d_region = Image.fromarray(np.uint8(d_region))

            # d_region = Image.fromarray((d_region * 255).astype(np.uint8))
            d_region = Image.fromarray(d_region)
            distortion_type_list =[]
            distortion_level_list = []
            num_distortion = random.randint(1,3)
            # distortion不重复
            choice_all = random.sample(range(1, 22), num_distortion)
            for i in choice_all:
                # choice = random.randint(1,19)
                d_region, distortion_type, level = iqa_transformations(i, d_region)
                distortion_type_list.append(distortion_type)
                distortion_level_list.append(level)

            # d_region: Image -> np.array
            # d_region = np.array(d_region)/225.
            d_region = np.array(d_region)

            ann.update({'distortion':distortion_type_list})
            ann.update({'distortion_level':distortion_level_list})
            # discription = anns['annotations'][i]['class_name'] + ' with ' + distortion_type
            # ann.update({'discription': discription})

            d_region_ori[d_mask == True] = d_region[d_mask == True]
            d_image[y: y+h, x: x+w] = d_region_ori
            
            # distortion_list.append(ann['distortion'])

    # return np.clip((d_image * 255.).round(), 0, 255).astype(np.uint8)
    return np.clip(d_image, 0, 255).astype(np.uint8)


# 包括去除mask过小的功能
def delete_overlap_anns(anns, p=0.8):
    # 如果丢弃重复的，那么不同部分之间存在边缘微小重复，会导致丢弃过多
    # 如果丢弃存在包含关系的，那么存在不同部分之间非完全包含，仅大部分包含的情况
    if len(anns['annotations']) == 0:
        return
    anns_1 = anns['annotations']
    # 从大到小
    sorted_anns = sorted(anns_1, key=(lambda x: x['area']), reverse=True)
    i=0
    while i < len(sorted_anns):
        ann2int = maskUtils.decode(sorted_anns[i]['segmentation'])
        j=len(sorted_anns)-1
        while j>i:
            # torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            x = ann2int+maskUtils.decode(sorted_anns[j]['segmentation']).astype(int)
            overlap=np.sum(x==2)
            thresh=np.sum(maskUtils.decode(sorted_anns[j]['segmentation']).astype(int))*p
            # print('overlap',overlap,'thresh',thresh)
            if overlap>thresh:
            # 丢弃具有包含关系的
            # if -1 not in x:
                del sorted_anns[j]
            j-=1
        i+=1

    anns['annotations'] = [x for x in sorted_anns if 3500 < x['area']]
    # anns['annotations'] = sorted_anns
    return anns


def add_region_distortion(anns, image: np.ndarray):
    d_image = image.copy().astype(np.float32) / 255.
    if anns['annotations'] is not None:
        for i in range(len(anns['annotations'])):
            ann = anns['annotations'][i]
            # ann['segmentation']['counts']=torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            d_region = d_image[y: y+h, x: x+w]
            
            d_region_ori = d_region.copy()
            d_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()[y: y+h, x: x+w]
            # d_mask = ann['segmentation']['counts'][y: y+h, x: x+w]
            
            order = [1 if np.random.rand()< 0.8 else 0 for i in range(4)]
            d_region, distortion_type = utils_image.task(d_region, order)
            ann.update({'distortion':distortion_type})

            d_region_ori[d_mask == True] = d_region[d_mask == True]
            d_image[y: y+h, x: x+w] = d_region_ori
            
            # distortion_list.append(ann['distortion'])

    return np.clip((d_image * 255.).round(), 0, 255).astype(np.uint8)
        
        
def mask_filter(anns) -> Union[Optional[dict], Optional[Dict[str, str]]]:
    '''
    it/that/something 丢掉，flower plant
    多个semantic label相同的丢掉，
    多个mask加起来面积过小 丢掉
    如果是多个mask，6个，并且有3个大的的，可以留下 aalto-theatre-228663.png, andalusia-106714.png, horses-918757_semantic
    > 4的就可以重新分一下了
    '''
    label_dict={}
    reason=['specific words', 'repetition', 'overall too small']
    words1=['flower', 'plant']
    words2=['it', 'that', 'something']
    # 注意顺序不要变
    if len(anns['annotations']) >= 7 :
        return None

    for i in anns['annotations']:
        if any(i['class_name'] in item for item in words1):
            return None
        if any(i['class_name'] in item for item in words2):
            del i
            continue
        if i['class_name'] not in label_dict.keys():
            label_dict[i['class_name']] = 1
        else:
            label_dict[i['class_name']] += 1
    if any(map(lambda x : x >1, label_dict.values())):
        return None

    if len(anns['annotations']) <= 4 :
        return anns

    size = reduce(operator.add, [x['area'] for x in anns['annotations']])
    if size < 65000:
        return None

    anns['annotations']=[i for i in anns['annotations'] if i['area']> 30000]
    return anns


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_image', type=str, default='../DF2K/train/000002.png', required=False)
    parser.add_argument('--input_image', type=str, default='./autodl-tmp/DIV2K_train_HR/', required=False)
    parser.add_argument('--output_image', type=str, default='./autodl-tmp/DIV2K_train_HR_output/', required=False)
    parser.add_argument('--sam_weight', type=str, default='./EIQA/sam_vit_h_4b8939.pth', required=False)

    args = parser.parse_args()


    sam = sam_model_registry["default"](checkpoint=args.sam_weight)
    sam.to('cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)

    # dict_write={}
    image_all = os.listdir(args.input_image)
    for i in image_all:
        print('************{} is processing***********'.format(i))
        input_image = os.path.join(args.input_image, i)
        # print(input_image)

        image = Image.open(input_image).convert('RGB')
        image = np.array(image)

        h, w, _ = image.shape

        # mask_generator.min_mask_region_area = h * w * 0.09  ## 0.5**2

        masks = mask_generator.generate(image)

        masks = region_regress(masks, 15, img_area=h*w)

        masks=delete_overlap_anns(masks)

        # 看看效果
        # image = show_anns(masks, image)
        
        image, bbox_list, distortion_list = add_region_distortion(masks, image)
        # dict_write[i]={'bbox': bbox_list, 'distortion': distortion_list}


        output_path = args.output_image+i
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        new_data = {i: {'bbox': bbox_list, 'distortion': distortion_list}}

        with open("data1.json", "r", encoding="utf-8") as f:
            file = f.read()
            if len(file) > 0:
                old_data = json.loads(file)
            else:
                old_data = {}
            old_data.update(new_data)
        with open("data1.json", "w", encoding="utf-8") as f:
            json.dump(old_data, f)

        print('************{} is done***********'.format(i))


# 细节的东西一般都没有distortion
if __name__ == '__main__':
    main()
    


