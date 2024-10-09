import json
import numpy as np
import cv2
import pycocotools.mask as maskUtils
import os

def tojson(path_in, path_out):
    with open(path_in, 'r') as f:
        output = json.load(f)
    height, width = output['imageHeight'], output['imageWidth']
    caption = output['image_text']
    name = output['imagePath'].split('.')[0]
    # mask 合并
    json_file={'annotations':[], 'caption':caption}
    label_mask={}
    for i in output['shapes']:
        # a boy on the left;blur:1;noise:2
        poly = i['points']
        mask=poly2mask(poly,height,width)

        if i['label'] not in label_mask.keys():
            label_mask[i['label']]=mask
        else:
            label_mask[i['label']]+=mask

    for k, mask in label_mask.items():
        every_json_file={'distortion': [], 'distortion_level': []}
        class_name, distortion = k.split(';')[0],k.split(';')[1:]
        dis_ = [(x.split(':')[0], x.split(':')[1]) for x in distortion]

        for d in dis_:
            every_json_file['distortion'].append(d[0])
            every_json_file['distortion_level'].append(eval(d[1]))

        # mask大小是整图的
        all=np.argwhere(mask != 0)
        x_1=np.min(all[:, 0]).item()
        y_1=np.min(all[:, 1]).item()
        x_2=np.max(all[:, 0]).item()
        y_2=np.max(all[:, 1]).item()
        bbox=[y_1,x_1,y_2,x_2]
        rle=singleMask2rle(mask)
        every_json_file['segmentation']=rle
        every_json_file['area']=len(all)
        every_json_file['bbox']=bbox
        every_json_file['class_name']=class_name
        json_file['annotations'].append(every_json_file)

    with open(os.path.join(path_out, name+'.json'), 'w') as ff:
        json.dump(json_file, ff, indent=4)


def poly2mask(points, height, width):
    mask = np.zeros((height, width), dtype=np.int32)
    obj = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, obj, 1)
    return mask


def singleMask2rle(mask):
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def show_bbox(img_path, bbox):
    img = cv2.imread(img_path)
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]],[bbox[0], bbox[3]]], dtype=np.int32)
    cv2.polylines(img, [points], True, (255, 0, 0), 3)

    cv2.imshow("window", img)
    cv2.waitKey(0)


def walk_json(path_in, path_out):
    all_json_file = filter(lambda x: x.endswith('.json'),os.listdir(path_in))
    for i in all_json_file:
        tojson(os.path.join(path_in,i), path_out)

if __name__ == '__main__':
    # path_in = r'/Users/helenpeng/Desktop/img_in/'
    # path_out = r'/Users/helenpeng/Desktop/img_out/'
    # walk_json(path_in, path_out)
    tojson('/Users/helenpeng/Desktop/img_in/7345.json', '/Users/helenpeng/Desktop/project1/')
