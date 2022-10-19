"""
Download the weights in ./checkpoints beforehand for fast inference
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth
"""

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from models.blip import blip_decoder
import argparse
import os

class Predictor :
    def __init__(self):
        self.device = "cuda:0"
        self.models = blip_decoder(pretrained='checkpoints/model_base_caption.pth', image_size=384, vit='base')

    def predict(self, path):
        model = self.models
        model.eval()
        model = model.to(self.device)

        caption_list = []
        image_list = []
        img_names = os.listdir(path)
        for img_name in tqdm(img_names) :
            img = f'{path}/{img_name}'
            im, im_b = load_image(img, image_size=384, device=self.device)
                
            # similar coding like pokemon 
            bytes_dict = {}
            bytes_dict['bytes'] = im_b
            bytes_dict['path'] = None
            image_list.append(bytes_dict)            
            with torch.no_grad():
                caption = model.generate(im, sample=False, num_beams=3, max_length=20, min_length=5)
                caption_list.append(caption[0])
            
        return image_list, caption_list



def load_image(image, image_size, device):
    with open(image,'rb') as f:
        image_binary = f.read()
        f.close()

    raw_image = Image.open(str(image)).convert('RGB')
    

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image, image_binary





if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--image", default = './image/test.jpg') #test용
    parser.add_argument('-p',"--path",type=str, default='./image')
    parser.add_argument('-o',"--outdir",type=str,default='')

    args = parser.parse_args()
    path = args.path
    outdir = args.outdir


    # 모델 Load
    print("모델을 로드합니다...")
    model = Predictor()


    # 모델 Inference 및 caption, image_bytes 생성
    print('캡션을 생성합니다...')
    image_list, caption_list = model.predict(path)

    # DataFrame 생성
    caption_df = pd.DataFrame({
            'image' : image_list,
            'text' : caption_list
        })
    
    # outdir가 default 일때
    if outdir=='' :
        outdir = path.split('/')[-1]

    # save
    print('캡션을 저장합니다...')
    # caption_df.to_csv(f'./output/{outdir}.csv') #csv
    caption_df.to_parquet(f'./output/{outdir}.parquet', engine='pyarrow', index=False) #parquet
    print("완료")
