import mmcv
import numpy as np
from PIL import Image, ImageFilter
import skimage.morphology
from scipy import ndimage
import random
from skimage import color,filters,io
from sklearn.preprocessing import normalize
import io
from scipy.interpolate import UnivariateSpline
import PIL
from scipy import interpolate
import skimage
import cv2
import os
from x_distortion import add_distortion, multi_distortions_dict, distortions_dict


img = cv2.cvtColor(mmcv.imread('/root/autodl-tmp/example/kadis_HR/koala-185736.png'), cv2.COLOR_BGR2RGB)


for i in multi_distortions_dict.keys():
    img1=add_distortion(img=img, severity=3, distortion_name=distortions_dict[i][0])
    for n in multi_distortions_dict[i]:
        path='/root/autodl-tmp/order/'+i+'_'+n
        os.mkdir(path)
        mmcv.imwrite(cv2.cvtColor(add_distortion(img=img1, severity=3, distortion_name=distortions_dict[n][0]), cv2.COLOR_RGB2BGR), os.path.join(path, 'QQ'+i+'_'+n + '.png'))

        img2=add_distortion(img=img, severity=3, distortion_name=distortions_dict[n][0])
        mmcv.imwrite(cv2.cvtColor(add_distortion(img=img2, severity=3, distortion_name=distortions_dict[i][0]), cv2.COLOR_RGB2BGR), os.path.join(path, n+'_'+i + '.png'))



# path1=r'/root/autodl-tmp/order2'
# for i in multi_distortions_dict.keys():

#     img1=add_distortion(img=img, severity=3, distortion_name=distortions_dict[i][0])
#     for n in multi_distortions_dict[i]:
#         filename=i+'_'+n
#         mmcv.imwrite(cv2.cvtColor(add_distortion(img=img1, severity=3, distortion_name=distortions_dict[n][0]), cv2.COLOR_RGB2BGR), os.path.join(path2, filename + '.png'))

