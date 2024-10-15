'''
select koniq from q instruct
score range: 
[0,1): 168 
[1,2): 940
[2,3): 2169
[3,4): 1524
[4,5): 379
1000张: 0.193
'''
import os
import json
import random
import sys
from shutil import copy


def select(path, each_num=[63, 231, 429, 264, 13]):
    select_list=[]
    with open(path) as f:
        file=json.load(f)
        img_0_1=list(filter(lambda x: x['gt_score']>=0 and x['gt_score']<1, file))
        output_img_0_1=random.sample(img_0_1, each_num[0])
        img_1_2=list(filter(lambda x: x['gt_score']>=1 and x['gt_score']<2, file))
        output_img_1_2=random.sample(img_1_2, each_num[1])
        img_2_3=list(filter(lambda x: x['gt_score']>=2 and x['gt_score']<3, file))
        output_img_2_3=random.sample(img_2_3, each_num[2])
        img_3_4=list(filter(lambda x: x['gt_score']>=3 and x['gt_score']<4, file))
        output_img_3_4=random.sample(img_3_4, each_num[3])
        img_4_5=list(filter(lambda x: x['gt_score']>=4 and x['gt_score']<5, file))
        output_img_4_5=random.sample(img_4_5, each_num[4])
    select_list = [*output_img_0_1, *output_img_1_2, *output_img_2_3, *output_img_3_4, *output_img_4_5]

    with open(r'human_label_tool/koniq_select.json','w') as ff:
        json.dump(select_list, ff, indent=4)


# 将图片转移到每个打分者的文件夹里面
def move_img(img_path, json_path):
    path_=r'/data/pxg1/data/q-instruct-images/'
    with open(json_path) as f:
        file=json.load(f)
    random.shuffle(file)
    for i in range(1, 5):
        for n in file[(i-1)*250:i*250]:
            if not os.path.exists(os.path.join(img_path, str(i))):
                os.mkdir(os.path.join(img_path, str(i)))
            copy(os.path.join(path_,n['img_path'].split('/')[1]), os.path.join(img_path, str(i)))


if __name__=='__main__':
    select(r'human_label_tool/KonIQ_Q_score.json')
    # move_img(r'/data/pxg1/data/1k',r'/data/pxg1/data/1k/koniq_select.json')
