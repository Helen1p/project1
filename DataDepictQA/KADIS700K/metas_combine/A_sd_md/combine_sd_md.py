import json
import random

random.seed(131)


if __name__ == "__main__":
    json_sd = "../../A_sd_brief/metas/train_A_sd_mix_98k_exclude.json"
    json_md = "../../A_md_brief/metas/train_A_md_mix_75k_exclude.json"

    with open(json_sd) as fr:
        metas_sd = json.load(fr)
    metas_sd_dict = {}
    for meta_sd in metas_sd:
        metas_sd_dict[meta_sd["id"]] = meta_sd

    with open(json_md) as fr:
        metas_md = json.load(fr)
    metas_md_dict = {}
    for meta_md in metas_md:
        metas_md_dict[meta_md["id"]] = meta_md

    ids_sd = set([_["id"] for _ in metas_sd])
    ids_md = set([_["id"] for _ in metas_md])
    print("num ids_sd", len(ids_sd))
    print("num ids_md", len(ids_md))
    ids_all = ids_sd.union(ids_md)
    num_all = len(ids_all)
    print("num ids_all", num_all)
    num_sd_only = len(ids_sd - ids_md)
    num_md_only = len(ids_md - ids_sd)
    num_inter = len(ids_sd.intersection(ids_md))
    print("num_sd_only", num_sd_only)
    print("num_md_only", num_md_only)
    print("num_inter", num_inter)

    props_sd = [0.25, 0.50, 0.75]
    save_paths = [
        "train_A_mix_sd32k_md75k.json", 
        "train_A_mix_sd54k_md54k.json", 
        "train_A_mix_sd80k_md27k.json", 
    ]
    for save_path, prop_sd in zip(save_paths, props_sd):
        num_sd = int(num_all * prop_sd)
        num_sd_inter = max(num_sd - num_sd_only, 0)
        prop_sd_inter = num_sd_inter / num_inter

        num_sd = 0
        num_md = 0
        metas = []
        for item_id in ids_all:
            if item_id in ids_sd and item_id not in ids_md:
                num_sd += 1
                meta_sd = metas_sd_dict[item_id]
                metas.append(meta_sd)
            elif item_id not in ids_sd and item_id in ids_md:
                num_md += 1
                meta_md = metas_md_dict[item_id]
                metas.append(meta_md)
            else: 
                if random.random() < prop_sd_inter:
                    num_sd += 1
                    meta_sd = metas_sd_dict[item_id]
                    metas.append(meta_sd)
                else:
                    num_md += 1
                    meta_md = metas_md_dict[item_id]
                    metas.append(meta_md)

        print("=" * 100)
        print(save_path)
        print(f"prop: {prop_sd}, sd: {num_sd}, md: {num_md}")
        with open(save_path, "w") as fw:
            fw.write(json.dumps(metas, indent=4))
