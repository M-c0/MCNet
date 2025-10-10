import os
import re
import cv2
import numpy as np

# ———— 配置 ————
# 四个源文件夹（按顺序分别对应网格左上、右上、左下、右下）
src_dirs = [
    # "../gradcam_vis/MSVR310/RGB/Baseline_73",
    # "../gradcam_vis/MSVR310/RGB/PAT_73",
    # "../gradcam_vis/MSVR310/RGB/PAT+common_l_g_73",
    # "../gradcam_vis/MSVR310/RGB/PAT+common_l_g+ME_73"

    "../gradcam_vis/MSVR310/emb_v/baseline_2",
    "../gradcam_vis/MSVR310/emb_v/all_2",

]
# 输出文件夹
# out_dir = "../gradcam_vis/MSVR310/RGB/merged"
out_dir = "../gradcam_vis/MSVR310/emb_v/merged_2"
os.makedirs(out_dir, exist_ok=True)

# 自然数排序函数：提取文件名中的数字
def natural_sort_key(fname):
    # 比如 "12.jpg" → 12；若有多个数字，则取第一个
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else fname

# 按自然数排序
# ———— 读取文件名列表并排序 ————
# 假设每个文件夹内文件名一一对应，且数量都为 191 张
file_lists = [sorted(os.listdir(d), key=natural_sort_key) for d in src_dirs]
n = len(file_lists[0])  # 191

# ———— 确定统一的目标尺寸（可选） ————
# 这里取第一个文件夹的第一张图尺寸作为基准
first_img = cv2.imread(os.path.join(src_dirs[0], file_lists[0][0]))
h, w = first_img.shape[:2]

margin_h = 20  # 水平留白宽度
margin_v = 20  # 垂直留白高度

# 白色留白块
blank_h = np.full((h, margin_h, 3), 255, dtype=np.uint8)
# 顶部行宽度 = 两张图加一个水平留白
top_width = w * 2 + margin_h
blank_v = np.full((margin_v, top_width, 3), 255, dtype=np.uint8)

# ———— 开始循环拼接 ————
for i in range(n):
    # 1. 依次读取四张图，并 resize 到相同大小
    imgs = []
    for folder, flist in zip(src_dirs, file_lists):
        img_path = os.path.join(folder, flist[i])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{img_path}")
        # 如果不是同样大小，就做 resize
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        imgs.append(img)

    # 拼四张图
    # # 拼接左上 + 右上，并在中间加水平留白
    # top = cv2.hconcat([imgs[0], blank_h, imgs[1]])
    # # 拼接左下 + 右下，并在中间加水平留白
    # bottom = cv2.hconcat([imgs[2], blank_h, imgs[3]])
    # # 最后在 top 和 bottom 之间插入垂直留白
    # grid = cv2.vconcat([top, blank_v, bottom])

    # 拼两张图
    grid = cv2.hconcat([imgs[0], imgs[1]])

    # 5. 保存
    out_path = os.path.join(out_dir, f"merged_{i:03d}.jpg")
    cv2.imwrite(out_path, grid)
    print(f"Saved {out_path}")

print("全部拼接完成！")
