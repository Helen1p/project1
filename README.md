## Steps for Processing

### Step 1: semantic segmentation and add distortions

```bash
cd /data1/pxg/project1/scripts
CUDA_VISIBLE_DEVICES=0 python main_ssa_engine.py
```

### step 2: generate grounding Q&A
```
CUDA_VISIBLE_DEVICES=0 python generate_grounding.py
```

### step 3: generate long caption and referring Q&A
4 * A100-80G / 8 * L20-48G
```
CUDA_VISIBLE_DEVICES='1,2,3,4' python chat.py
```



**to modify human labels:**
```
python to_json.py
``` 

#### TODO：
1.把软件里几个橘子的情况合成一个

2.重新修改to_json.py 加更多标签的转换

3.severity要不要改

4.grounding question pool要修改，以及要不要加上answer pool