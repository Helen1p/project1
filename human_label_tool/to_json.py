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
    # 同一个mask分离成2块的就当做两个mask
    json_file={'annotations':[], 'caption':caption}
    label_list=[]
    for i in output['shapes']:
        every_json_file={}
        poly = i['points']
        # mask大小是整图的
        mask=poly2mask(poly,height,width)
        all=np.argwhere(mask != 0)
        x_1=np.min(all[:, 0]).item()
        y_1=np.min(all[:, 1]).item()
        x_2=np.max(all[:, 0]).item()
        y_2=np.max(all[:, 1]).item()
        bbox=[y_1,x_1,y_2,x_2]
        rle=singleMask2rle(mask)
        class_name, distortion = i['label'].split(';')[0],i['label'].split(';')[1:]
        every_json_file['segmentation']=rle
        every_json_file['area']=len(all)
        every_json_file['bbox']=bbox
        every_json_file['class_name']=class_name
        every_json_file['distortion']=distortion
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
    path_in = r'/Users/helenpeng/Desktop/img_in/'
    path_out = r'/Users/helenpeng/Desktop/img_out/'
    walk_json(path_in, path_out)