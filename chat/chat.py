import numpy as np
import os
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
import json

weird_str=["```json\n", "\n```", '\n      ', '\n    ', '\n  ', '\n']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.7))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.3)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


class chat_internvl_artificial(torch.nn.Module):
    def __init__(self, model_path, img_prefix, semantic_prefix, json_prefix, output_json_prefix):
        super().__init__()
        self.img_prefix = img_prefix
        self.semantic_prefix = semantic_prefix
        self.json_prefix = json_prefix
        self.output_json_prefix = output_json_prefix
        split_model_path=model_path.split('/')[-1]
        device_map = split_model(split_model_path)
        self.model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def forward(self, img):
        img_path = self.img_prefix+str(img)
        json_path = self.json_prefix+img.split('.')[0]+'_info.json'
        with open(json_path) as f:
            s=json.load(f)
        
        pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        '''
        对输入的bbox处理能力弱，
        对semantic label错误的情况无法改正（很多seg结果的本来也很奇怪），
        distortion description无法incorporate具体的semantic
        '''

        prompt_gen = []
        prompt_gen.append('There are {} main visual elements in this image'.format(len(s['annotations'])))
        for idx, i in enumerate(s['annotations']):
            dis_list = []
            for x, y in zip(i['distortion_type'], i['distortion_level']):
                sstr = x+'(level '+ str(y+1)+')'
                dis_list.append(sstr)
            output_dis = (', ').join(dis_list)
            sstr1 = 'element {}: bounding box: [{},{}],[{},{}]; semantic label reference: {}; distortion type and level: {}' \
        .format(str(idx+1), i['bbox'][0],i['bbox'][1],i['bbox'][0]+i['bbox'][2],i['bbox'][1]+i['bbox'][3],i['class_name'],output_dis)
            prompt_gen.append(sstr1)
        question_im = ('. ').join(prompt_gen)
        
#         basic_prompt = 'provide the most accurate semantic label for each visual element in each bounding box. The semantic label should be specific, for example, instead of output the label as "pen", you might specify it to "crayon". \
# Make sure the bounding box corresponds to the right visual element, especially the bounding boxes have big overlaps. \
# These are the rules you have to follow: \
# 1. The coordinate origin (0,0) of bounding box is at the top-left corner of the image. The given coordinates is the top-left and bottom-right corners, respectively. \
# 2. Each bounding box should only contain one intact visual elements, it can either be a foreground object(e.g., cat, bird) or just the background(e.g., sky, floor). \
# 3. Either an object or the background is entirely within the boundaries of the bounding box, and avoid recognizing targets that just partially within the boundaries in the scenes with dense objects. \
# 4. the provided semantic label reference are probably wrong. You have to output accurate semantic labels by your own. \
# 5. If the semantic labels in your output are identical, please add distinguishing details to differentiate them. For example, instead of labeling both as "bird," you might specify "blue bird" for one of them. \
# 6. The distortion levels are categorized into three tiers: Level 1 is the lowest, while Level 3 is the highest. '

# 3. The target visual element is entirely within the boundaries of the bounding box, and avoid recognizing wrong targets that just partially within the boundaries in the scenes with dense objects. \
         # If the bounding box contains multiple visual elements, identify the taget visual element whose entire area is contained within the box. 
# avoid identifying wrong targets whose edges are beyond the boundaries especially when the bounding box contains multiple visual elements. \

        basic_prompt = 'provide the most accurate semantic label for the target visual element in each bounding box. The semantic label should be specific, for example, instead of output the label as "pen", you might specify it to "crayon". \
These are the rules you have to follow: \
1. The coordinate origin (0,0) of bounding box is at the top-left corner of the image. The given coordinates of the bounding box is the top-left and bottom-right corners, respectively. \
2. There is only one target visual element in each bounding box, it can either be a foreground object(e.g., cat, bird) or just the background(e.g., sky, floor). \
3. If the bounding box contains multiple objects, identify the taget visual element whose entire area is contained within the box.  \
4. The provided semantic label references are probably wrong. You have to output accurate semantic labels by your own. \
5. If the semantic labels in your output are identical, please add distinguishing details to differentiate them. For example, instead of labeling both as "bird," you might specify "blue bird" for one of them. \
6. The distortion levels are categorized into three tiers: Level 1 is the lowest, while Level 3 is the highest. '

        
        # 此次不需要收集answer，不给format
        # question = basic_prompt + question_im
        question = "<image>\n" + question_im + basic_prompt
        
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        print(f'User: {question}\nAssistant: {response}')

        # 只留风景照，植物的也不要，其他的更不要了

        question_base = 'You have generate descriptions for the image. Take use of the accurate semantic lable you have generated in last question. \
Firstly, give a mixed description of 3 acpects for each visual element in each bounding box. \
The mixed description is compriseed of 3 acpects: 1.Basic Information: the type, color, and any notable features of the target; 2.Position and Orientation of the target; \
3.the visual effects and texture damage of each distortion, if there are more than one distortions, state the the visual effects and texture damage of each distortion one by one. \
The description should not contain the number of level, bounding box, use vivid words to replace them. '

        question_format = 'The output must be a raw json format, I will give you an example and do not imitate the sentence structure in the example, make it diverse. \
If the Basic Information is: "There is a building with a white exterior. It has a window and a door, both of which are made of glass and wood, respectively.", \
and Position and Orientation information is: "The building occupies the left side of the image, with the window and door positioned centrally.", \
and the Distortion Effects: 1.Gaussian Blur(Level 1): "The edges of the building appear slightly blurred, reducing the sharpness of the structure.", 2.Mean Shift(Level 2): "The colors of the building are slightly shifted, especially the roof, making the white exterior appear less bright and more muted."}, \
Then, the output should be like: \
{"element description": {"element 1": {"semantic label": "building", "bounding box": [[12, 34], [56, 78]], \
"mixed description": "A building with a white exterior is positioned on the left side, featuring a glass window and a wooden door centrally placed. The edges are slightly blurred due to a Gaussian effect, giving a soft, hazy look. Additionally, a mean shift effect has muted the colors, \
making the white exterior and roof appear less bright and more subdued."}}}'
        
        question = question_base + question_format
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        caption1_json_path = os.path.join(self.output_json_prefix, 'caption1', img.split('.')[0]+'.json')
        for i in range(len(weird_str)):
            response=response.replace(weird_str[i],'')
        with open(caption1_json_path, 'w') as f:
            json.dump(json.loads(response), f, indent=4)


        question_base = 'refer to the mixed description for each visual element the and give a global description about this whole image, you should mention every element and the whole image structure, \
especially for the impact of the distortions and quality evaluation. \
The description should not contain the number of level, bounding box, use vivid words to replace them. '

        question_format = 'The output must be a raw json format, this is a format example and do not imitate the sentence structure in the example, make it diverse. \
{"global description": "The image showcases a scene with a building dominating the background, with three distinct elements in the foreground: a fire hydrant, a wooden fence, and a plant. The building, occupying most of the image, is rendered with a soft blur and shifted colors, \
giving it a slightly hazy and surreal appearance. This creates a backdrop that feels out of focus and less defined. In the foreground, the fire hydrant stands out with its colors subtly diffused, making it less vibrant and somewhat muted. Nearby, the wooden fence appears smeared due to motion blur, \
with its colors slightly intensified and softened by additional blurring. This combination results in a fence that lacks clear definition and sharpness. Finally, the plant is depicted under dim lighting, making it appear darker and less prominent. Its colors are intensely vivid, \
but the details are compromised due to a resizing effect that has softened its edges and textures."}'
        
        question = question_base + question_format
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        caption2_json_path = os.path.join(self.output_json_prefix, 'caption2', img.split('.')[0]+'.json')
        for i in range(len(weird_str)):
            response=response.replace(weird_str[i],'')
        with open(caption2_json_path, 'w') as f:
            json.dump(json.loads(response), f, indent=4)


        question = 'Give a description about the spatial relations of each visual element of each bounding box in this image.'.format(str(len(s['annotations'])))
        question_format = 'The answer must be a json format. This is an example: {"spatial relations of all elements": {"element 1": {"bounding box": [[12, 34], [56, 78]], "spatial relations": "The cabinet is positioned in the background, behind the keyboard and synthesizer. It is placed against the wall and is partially visible due to the angle of the image."}'
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        spatial_json_path = os.path.join(self.output_json_prefix, 'spatial', img.split('.')[0]+'.json')
        for i in range(len(weird_str)):
            response=response.replace(weird_str[i],'')
        with open(spatial_json_path, 'w') as f:
            json.dump(json.loads(response), f, indent=4)

        
        question_base1 = 'use the spatial relations of {} main visual element in the previous question to generate {} referring questions for {} objects. \
The question must include spatial relations of the visual elements. \
The answer must be the distortion of the visual elements. Modify the level to diverse adjectives. For example, modify "jpeg compression(level 1)" to "moderate jpeg compression".'\
        .format(str(len(s['annotations'])),str(len(s['annotations'])),str(len(s['annotations'])))
        question_format1 = 'The output must be a json format, follow this example: {"referring": {"element 1": {"question": "What is the distortion of the book in the lower-right corner?", "answer": "Minor jpeg compression, severe motion blur."}}}'
        question = question_base1 + question_format1
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        referring_json_path = os.path.join(self.output_json_prefix, 'referring', img.split('.')[0]+'.json')
        for i in range(len(weird_str)):
            response=response.replace(weird_str[i],'')
        with open(referring_json_path, 'w') as f:
            json.dump(json.loads(response), f, indent=4)


class chat_internvl_humanlabel(torch.nn.Module):
    def __init__(self, model_path, img_prefix, json_prefix, output_json_prefix):
        super().__init__()
        self.img_prefix = img_prefix
        self.json_prefix = json_prefix
        self.output_json_prefix = output_json_prefix
        split_model_path=model_path.split('/')[-1]
        device_map = split_model(split_model_path)
        self.model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def forward(self, img):
        img_path = self.img_prefix+str(img)
        json_path = self.json_prefix+img.split('.')[0]+'.json'
        with open(json_path) as f:
            s=json.load(f)
        
        pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        prompt_gen = []
        prompt_gen.append('There are {} main visual elements in this image'.format(len(s['annotations'])))
        for idx, i in enumerate(s['annotations']):
            dis_list = []
            for x, y in zip(i['distortion_type'], i['distortion_level']):
                sstr = x+'(level '+ str(y+1)+')'
                dis_list.append(sstr)
            output_dis = (', ').join(dis_list)
            sstr1 = 'element {}: bounding box: [{},{}],[{},{}]; semantic label: {}; distortion type and level: {}; brief description: {} ' \
        .format(str(idx+1), i['bbox'][0],i['bbox'][1],i['bbox'][0]+i['bbox'][2],i['bbox'][1]+i['bbox'][3],i['class_name'],output_dis,i['brief_description'])
            prompt_gen.append(sstr1)
        question_im = ('. ').join(prompt_gen)
        
        basic_prompt = 'The provided semantic label is accurate and specific, additionally some also have the detailed attributes or the spatial description of the element. \
There is also a brief discription about the detailed information of each element, which hlp you to understand the distortion and texture damage of each element. \
These are the rules you have to follow: \
1. The coordinate origin (0,0) of bounding box is at the top-left corner of the image. The given coordinates of the bounding box is the top-left and bottom-right corners, respectively. \
2. There is only one target visual element in each bounding box, it can either be a foreground object(e.g., cat, bird) or or a couple of foreground objects in the same type (e.g., 3 cats) or just the background(e.g., sky, floor). \
3. The distortion levels are categorized into three tiers: Level 1 is the lowest, while Level 3 is the highest. '
        # 3. If the bounding box contains multiple objects, identify the taget visual element whose entire area is contained within the box.  \
        
        
        # question = "<image>\n" + question_im + basic_prompt
        
        # response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')

        
        question_base = 'You have generate descriptions for the image. Take use of the accurate semantic lable and the brief discription carefully. \
Firstly, give a mixed description of 3 acpects for each visual element in each bounding box. \
The mixed description is compriseed of 3 acpects: 1.Basic Information: the type, color, and any notable features of the target; 2.Position and Orientation of the target; \
3.the visual effects and texture damage of each distortion, if there are more than one distortions, state the the visual effects and texture damage of each distortion one by one. \
The description should not contain the number of level, bounding box, use vivid words to replace them. '

        question_format = 'The output must be a raw json format, I will give you an example and do not imitate the sentence structure in the example, make it diverse. \
If the Basic Information is: "There is a building with a white exterior. It has a window and a door, both of which are made of glass and wood, respectively.", \
and Position and Orientation information is: "The building occupies the left side of the image, with the window and door positioned centrally.", \
and the Distortion Effects: 1.Gaussian Blur(Level 1): "The edges of the building appear slightly blurred, reducing the sharpness of the structure.", 2.Mean Shift(Level 2): "The colors of the building are slightly shifted, especially the roof, making the white exterior appear less bright and more muted."}, \
Then, the output should be like: \
{"element description": {"element 1": {"semantic label": "building", "bounding box": [[12, 34], [56, 78]], \
"mixed description": "A building with a white exterior is positioned on the left side, featuring a glass window and a wooden door centrally placed. The edges are slightly blurred due to a Gaussian effect, giving a soft, hazy look. Additionally, a mean shift effect has muted the colors, \
making the white exterior and roof appear less bright and more subdued."}}}'
        
        question = "<image>\n" + question_im + basic_prompt+question_base + question_format
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        caption1_json_path = os.path.join(self.output_json_prefix, 'caption1', img.split('.')[0]+'.json')
        for i in range(len(weird_str)):
            response=response.replace(weird_str[i],'')
        with open(caption1_json_path, 'w') as f:
            json.dump(json.loads(response), f, indent=4)


        question_base = 'refer to the mixed description for each visual element the and give a global description about this whole image, you should mention every element and the whole image structure, \
especially for the impact of the distortions and quality evaluation. \
The description should not contain the number of level, bounding box, use vivid words to replace them. '

        question_format = 'The output must be a raw json format, this is a format example and do not imitate the sentence structure in the example, make it diverse. \
{"global description": "The image showcases a scene with a building dominating the background, with three distinct elements in the foreground: a fire hydrant, a wooden fence, and a plant. The building, occupying most of the image, is rendered with a soft blur and shifted colors, \
giving it a slightly hazy and surreal appearance. This creates a backdrop that feels out of focus and less defined. In the foreground, the fire hydrant stands out with its colors subtly diffused, making it less vibrant and somewhat muted. Nearby, the wooden fence appears smeared due to motion blur, \
with its colors slightly intensified and softened by additional blurring. This combination results in a fence that lacks clear definition and sharpness. Finally, the plant is depicted under dim lighting, making it appear darker and less prominent. Its colors are intensely vivid, \
but the details are compromised due to a resizing effect that has softened its edges and textures."}'
        
        question = question_base + question_format
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        caption2_json_path = os.path.join(self.output_json_prefix, 'caption2', img.split('.')[0]+'.json')
        for i in range(len(weird_str)):
            response=response.replace(weird_str[i],'')
        with open(caption2_json_path, 'w') as f:
            json.dump(json.loads(response), f, indent=4)


        # question = 'Give a description about the spatial relations of each visual element of each bounding box in this image.'.format(str(len(s['annotations'])))
        # question_format = 'The answer must be a json format. This is an example: {"spatial relations of all elements": {"element 1": {"bounding box": [[12, 34], [56, 78]], "spatial relations": "The cabinet is positioned in the background, behind the keyboard and synthesizer. It is placed against the wall and is partially visible due to the angle of the image."}'
        # response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        # spatial_json_path = os.path.join(self.output_json_prefix, 'spatial', img.split('.')[0]+'.json')
        # for i in range(len(weird_str)):
        #     response=response.replace(weird_str[i],'')
        # with open(spatial_json_path, 'w') as f:
        #     json.dump(json.loads(response), f, indent=4)

        
        question_base1 = 'use the spatial relations of {} main visual element in the previous question to generate {} referring questions for {} objects. \
The question must include spatial relations of the visual elements. \
The answer must be the distortion of the visual elements. Modify the level to diverse adjectives. For example, modify "jpeg compression(level 1)" to "moderate jpeg compression".'\
        .format(str(len(s['annotations'])),str(len(s['annotations'])),str(len(s['annotations'])))
        question_format1 = 'The output must be a json format, follow this example: {"referring": {"element 1": {"question": "What is the distortion of the book in the lower-right corner?", "answer": "Minor jpeg compression, severe motion blur."}}}'
        question = question_base1 + question_format1
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        referring_json_path = os.path.join(self.output_json_prefix, 'referring', img.split('.')[0]+'.json')
        for i in range(len(weird_str)):
            response=response.replace(weird_str[i],'')
        with open(referring_json_path, 'w') as f:
            json.dump(json.loads(response), f, indent=4)


if __name__ == '__main__':
    # chat=chat_internvl_artificial(model_path='/root/autodl-tmp/pretrained/OpenGVLab/InternVL2-Llama3-76B', 
    #                img_prefix='/root/autodl-tmp/example/kadis_output/', 
    #                semantic_prefix='/root/autodl-tmp/example/semantic/', 
    #                json_prefix='/root/autodl-tmp/example/json/', 
    #                output_json_prefix='/root/autodl-tmp/example/chat/'
    #                )
    chat=chat_internvl_humanlabel(model_path='/hy-tmp/OpenGVLab/InternVL2-Llama3-76B', 
                   img_prefix='/hy-tmp/example/kadis_output/', 
                   json_prefix='/hy-tmp/example/json/', 
                   output_json_prefix='/hy-tmp/example/chat/'
                   )
    chat('07824.jpg')

    # for i in os.listdir('/root/autodl-tmp/example/kadis_output/'):
    #     if i.endswith('png'):
    #         a = chat(i)
    #         torch.cuda.empty_cache()

