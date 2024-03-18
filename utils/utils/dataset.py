import os,re, cv2
import logging
import random
import logging
import jsonlines
from io import BytesIO
from PIL import Image
import torch
from utils.box_ops import box_xyxy_to_cxcywh
from torch.utils.data import Dataset
from sat.helpers import print_rank0
import uuid
import json

def find_all_files(path, suffix=".jpg"):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'find {len(target_files)} files...')
    return target_files

# class ItemDataset(Dataset):
#     def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
#         super().__init__()
#         self.data = self.load_data(data_dirs)
#         self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
#     def process_img(self, img):
#         img_dict = {'vision': self.image_processor(img)}
#         if self.cross_image_processor:
#             img_dict.update({'cross': self.cross_image_processor(img)})
#         return img_dict
    
#     def process_text(self, answer, prompt):
#         return self.text_processor(answer, prompt)
    
#     def load_data(self, data_dir):
#         all_files = find_all_files(data_dir, suffix=".jpg")
#         print_rank0(f"find {len(all_files)} samples in all...")
#         return all_files
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         data = self.data[index]
#         # img
#         try:
#             img = Image.open(data).convert('RGB')
#         except Exception as e:
#             print_rank0(e, level=logging.WARNING)
#             return {}
#         img_dict = self.process_img(img)
#         # text
#         label = data.split('/')[-1].split('.')[0]
#         uni_key = label
#         text_dict = self.process_text(label, "CAPTCHA:")
#         if text_dict is None:
#             print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
#             return {}
#         # other attr
#         ret = {**img_dict, **text_dict, "question_id": uni_key}
#         return ret

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
        for item in data:
            item['imagePath'] = "/workspace/CogAgent/data/" + item['imagePath'][1:]
        return data

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None,grounding_image_processor=None, **kwargs):
        super().__init__()
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
        self.grounding_image_processor = grounding_image_processor
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        if self.grounding_image_processor:
            gnd_images, ratios = self.grounding_image_processor.process(img)
            img_dict.update({'gndimg': gnd_images})
            img_dict.update({'ratios': ratios})
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def postprocess_bbox(self, bboxes_raw, ratios):
        h, w = self.grounding_image_processor.image_size
        boxes_gt = []
        for box in bboxes_raw:
            if len(box) == 0:
                boxes_gt.append([])
                continue

            if isinstance(box, list):
                box = torch.tensor(box)

            ratio_width, ratio_height = ratios
            scaled_boxes = box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

            scaled_boxes = box_xyxy_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(scaled_boxes.device)
            boxes_gt.append(scaled_boxes)
        return boxes_gt
    
    def load_data(self, data_dir):
        all_data = read_json(data_dir)
        print_rank0(f"find {len(all_data)} samples in all...")
        print(f'all_data: {all_data[0]}')
        return all_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(data['imagePath']).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        
        # text
        label = data['Answer']
        prompt = data['Question']
        
        bbox_gt = re.findall(r"\[.*?\]", label)
        assert len(bbox_gt)==1
        assert len(re.findall(r"\[.*?\]", prompt))==0
        right_part = label.split(bbox_gt[0])[1].strip()
        left_part = label.split(bbox_gt[0])[0].strip()
        label = left_part + " " + right_part
        
        # Add VG token
        label = label + '[VG]'
        
        bbox_gt = bbox_gt[0].replace("[", "").replace("]", "")
        if "," in bbox_gt:
            bbox_gt = bbox_gt.split(",")
        else:
            bbox_gt = bbox_gt.split(" ")
        img = cv2.imread(data["imagePath"])
        bbox_gt = [float(x) for x in bbox_gt]
        bbox_gt = [int(bbox_gt[0]*img.shape[1]), int(bbox_gt[1]*img.shape[0]), int(bbox_gt[2]*img.shape[1]), int(bbox_gt[3]*img.shape[0])]
        bboxes_gpt = [bbox_gt]
        bboxes_gpt = self.postprocess_bbox([bboxes_gpt], img_dict['ratios'])

        uni_key = str(uuid.uuid4())
        text_dict = self.process_text(label, prompt)
        
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}
        
        text_dict.update({"bboxes_gt_list": [bboxes_gpt]})
        
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret