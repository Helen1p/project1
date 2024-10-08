import os
import torch
import argparse
import sys
sys.path.append('/root/project1')
sys.path.append('/data1/pxg/project1')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pipeline import semantic_annotation_pipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.distributed as dist
import torch.multiprocessing as mp
# os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '12355'
os.environ['MASTER_PORT'] = '6006'

# 改/root/miniconda3/envs/ssa/lib/python3.8/site-packages/transformers/models/oneformer/image_processing_oneformer.py
# 的prepare_metadata，防止从hugging face上读json


def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/example/kadis_HR/', help='HR image root')
    parser.add_argument('--out_dir_json', default='/root/autodl-tmp/example/json/', help='semantic annotations root')
    parser.add_argument('--out_dir_semantic', default='/root/autodl-tmp/example/semantic/', help='semantic img root')
    parser.add_argument('--out_dir_dis', default='/root/autodl-tmp/example/kadis_output/', help='output img root')
    
    parser.add_argument('--save_img', default=True, action='store_true', help='whether to save annotated images')
    # parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
    parser.add_argument('--sam', default=True, action='store_true', help='use SAM but not given annotation json, default is False')
    parser.add_argument('--ckpt_path', default='/root/project1/ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--light_mode', default=False, action='store_true', help='use light mode')
    args = parser.parse_args()
    return args


def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    if args.light_mode:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(rank)
    else:
        # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(rank)
        clip_processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/pretrained/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("/root/autodl-tmp/pretrained/clip-vit-large-patch14").to(rank)

    if args.light_mode:
        oneformer_ade20k_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny").to(rank)
    else:
        # oneformer_ade20k_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        # oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to(rank)
        oneformer_ade20k_processor = OneFormerProcessor.from_pretrained("/root/autodl-tmp/pretrained/oneformer_ade20k_swin_large")
        oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained("/root/autodl-tmp/pretrained/oneformer_ade20k_swin_large").to(rank)

    # oneformer_coco_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    # oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(rank)
    oneformer_coco_processor = OneFormerProcessor.from_pretrained("/root/autodl-tmp/pretrained/oneformer_coco_swin_large")
    oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained("/root/autodl-tmp/pretrained/oneformer_coco_swin_large").to(rank)

    # blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    # blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(rank)

    if args.light_mode:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(rank)
    else:
        # blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        # blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(rank)
        blip_processor = BlipProcessor.from_pretrained("/root/autodl-tmp/pretrained/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("/root/autodl-tmp/pretrained/blip-image-captioning-large").to(rank)

    if args.light_mode:
        clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd16")
        clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd16").to(rank)
        clipseg_processor.image_processor.do_resize = False
    else:
        # clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        # clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(rank)
        clipseg_processor = AutoProcessor.from_pretrained("/root/autodl-tmp/pretrained/clipseg-rd64-refined")
        clipseg_model = CLIPSegForImageSegmentation.from_pretrained("/root/autodl-tmp/pretrained/clipseg-rd64-refined").to(rank)
        clipseg_processor.image_processor.do_resize = False
    if args.sam:
        # from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(rank)
        if args.light_mode:
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=16,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,  # 1 by default
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires open-cv to run post-processing
                output_mode='coco_rle',
            )
        else:
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                # points_per_side=32,
                points_per_side=16, #或8
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,  # 1 by default
                crop_n_points_downscale_factor=2,
                min_mask_region_area=2000,  # Requires open-cv to run post-processing
                output_mode='coco_rle',
            )

        # print('*******************')
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        filenames = [fn_.replace('.' + fn_.split('.')[-1], '') for fn_ in os.listdir(args.data_dir) if '.'+fn_.split('.')[-1] in image_extensions]
    else:
        mask_generator = None
        filenames = [fn_[:-5] for fn_ in os.listdir(args.data_dir) if '.json' in fn_]  # if sam is not used, the filenames are the same as the json files
    if rank==0:
        print('Total number of files: ', len(filenames))
    local_filenames = filenames[(len(filenames) // args.world_size + 1) * rank : (len(filenames) // args.world_size + 1) * (rank + 1)]
    # 半路挂了
    # have_done = os.listdir('/root/autodl-tmp/semantic/')
    # have_done = [x.split('.')[0].split('_')[0] for x in have_done]
    # have_all = os.listdir('/root/autodl-tmp/DIV2K_train_HR')
    # have_all = [x.split('.')[0] for x in have_all]
    # local_filenames = list(set(have_all)-set(have_done))
    # # local_filenames = [x.split('.')[0].split('_')[0] for x in local_filenames]
    # print('Total number of files need to be done: ', len(local_filenames))

    for file_name in local_filenames:
        with torch.no_grad():
            semantic_annotation_pipeline(file_name, args.data_dir, args.out_dir_json, args.out_dir_semantic, args.out_dir_dis, rank, save_img=args.save_img,
                                        clip_processor=clip_processor, clip_model=clip_model,
                                        oneformer_ade20k_processor=oneformer_ade20k_processor, oneformer_ade20k_model=oneformer_ade20k_model,
                                        oneformer_coco_processor=oneformer_coco_processor, oneformer_coco_model=oneformer_coco_model,
                                        blip_processor=blip_processor, blip_model=blip_model,
                                        clipseg_processor=clipseg_processor, clipseg_model=clipseg_model, mask_generator=mask_generator)
        # torch.cuda.empty_cache()
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir_json):
        os.mkdir(args.out_dir_json)
    if not os.path.exists(args.out_dir_semantic):
        os.mkdir(args.out_dir_semantic)
    if not os.path.exists(args.out_dir_dis):
        os.mkdir(args.out_dir_dis)
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)