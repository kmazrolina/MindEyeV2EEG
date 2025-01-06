'''
This script generates image captions for test images of thingseeg dataset (all_captions.pt).
It also reads and saves all images into tensor (all_images.pt)

Needed for evaluation purposes.
'''

import argparse
import numpy as np
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=os.getcwd(),
        help="Path to where data is stored",
    )
    parser.add_argument(
        "--evals_path", type=str, default="../evals",
        help="Path to where captions and images will be stored",
    )

    args = parser.parse_args()

    test_images = np.load(f'{args.data_path}/GetData/test_imgpaths.npy').tolist()
    all_captions = []
    all_images = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    model.to(device)
    
    for img_path in test_images:
        img = Image.open(img_path)
        if all_images is None:
            all_images = img
        else:
            all_images = torch.vstack((all_images, img))
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_captions = np.hstack((all_captions, generated_caption))

    torch.save(all_captions,f"{args.evals_path}/all_captions.pt")
    torch.save(all_images,f"{args.evals_path}/all_captions.pt")

