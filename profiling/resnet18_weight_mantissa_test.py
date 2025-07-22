import sys
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

# create result file
fw = open("./ResNet18.txt", 'a') 
fw.write("Test Begin:\n")
fw.write("="*50 + "\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")
fw.write(f"Using Device: {device}\n")

# Input transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

'''
# Note that when using Efficient Net, transform need to change:
weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
transform = weights.transforms()
'''

# Load Model
model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model = model.to(device)
model_app = models.resnet18()
model_app.load_state_dict(torch.load('./weight/resnet18_truncated_weights.pth'))
model_app = model_app.to(device)
model.eval()
model_app.eval()
print("ResNet18 model load complete")
fw.write("ResNet18 model load complete\n")

# dataset class
class ImageNetDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def load_test_data(data_dir, label_file):
    with open(label_file, 'r') as f:
        gt_labels = [int(line.strip()) for line in f.readlines()]
    
    image_files = []
    for file in sorted(os.listdir(data_dir)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(data_dir, file))
    
    assert len(image_files) == len(gt_labels), f"Number of pictures ({len(image_files)}) do not match the label ({len(gt_labels)})"
    print(f"find {len(image_files)} testing pictures")
    fw.write(f"find {len(image_files)} testing pictures\n")
    
    return image_files, gt_labels

# Main test
def test_model(image_dir, label_file):
    # load test data
    image_files, gt_labels = load_test_data(image_dir, label_file)
    num_images = len(image_files)
    
    # build dataset using dataset class
    dataset = ImageNetDataset(image_files, gt_labels, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1000,  # batch
        num_workers=min(8, mp.cpu_count()),  # multiprocess
        pin_memory=True,
        prefetch_factor=4,  # prefetch
        persistent_workers=True
    )
    
    # initial
    correct_top1 = 0
    correct_top5 = 0
    correct_top1_app = 0
    correct_top5_app = 0
    processed = 0
    
    # use in batch
    batch_top1 = 0
    batch_top5 = 0
    batch_top1_app = 0
    batch_top5_app = 0
    batch_processed = 0
    
    start_time = time.time()
    
    for batch_idx, (image_batch, label_batch) in enumerate(dataloader):

        image_batch = image_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)
        
        batch_size = label_batch.size(0) 
        
        # model inference
        with torch.no_grad():
            outputs = model(image_batch)
            outputs_app = model_app(image_batch)
        
        # Top-1 for accurate model
        _, preds = torch.max(outputs, 1)
        correct_top1 += (preds == label_batch).sum().item()
        batch_top1 += (preds == label_batch).sum().item()

        # Top-5 for accurate model
        _, top5_preds = torch.topk(outputs, 5, dim=1)
        correct_top5 += (top5_preds == label_batch.view(-1, 1)).any(dim=1).sum().item()
        batch_top5 += (top5_preds == label_batch.view(-1, 1)).any(dim=1).sum().item()

        # Top-1 for approximate model
        _, preds_app = torch.max(outputs_app, 1)
        correct_top1_app += (preds_app == label_batch).sum().item()
        batch_top1_app += (preds_app == label_batch).sum().item()

        # Top-5 for approximate model
        _, top5_preds_app = torch.topk(outputs_app, 5, dim=1)
        correct_top5_app += (top5_preds_app == label_batch.view(-1, 1)).any(dim=1).sum().item()
        batch_top5_app += (top5_preds_app == label_batch.view(-1, 1)).any(dim=1).sum().item()
        
        processed += batch_size
        batch_processed += batch_size
        
        # each 10 batch print result
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            elapsed = time.time() - start_time
            speed = processed / elapsed

            print(f"Processed: {processed}/{num_images} [{batch_idx+1}/{len(dataloader)} batch] | "
                  f"Top-1 of Accurate Model: {batch_top1/batch_processed:.5%} | "
                  f"Top-5 of Accurate Model: {batch_top5/batch_processed:.5%} | "
                  f"Top-1 of Approximate Model: {batch_top1_app/batch_processed:.5%} | "
                  f"Top-5 of Approximate Model: {batch_top5_app/batch_processed:.5%} | "
                  f"Spped: {speed:.1f} pictures/second")
            
            fw.write(f"Processed: {processed}/{num_images} [{batch_idx+1}/{len(dataloader)} batch] | "
                  f"Top-1 of Accurate Model: {batch_top1/batch_processed:.5%} | "
                  f"Top-5 of Accurate Model: {batch_top5/batch_processed:.5%} | "
                  f"Top-1 of Approximate Model: {batch_top1_app/batch_processed:.5%} | "
                  f"Top-5 of Approximate Model: {batch_top5_app/batch_processed:.5%} | "
                  f"Spped: {speed:.1f} pictures/second \n")
            
            batch_top1 = 0
            batch_top5 = 0
            batch_top1_app = 0
            batch_top5_app = 0
            batch_processed = 0
    
    # final accuracy
    top1_accuracy = correct_top1 / num_images
    top5_accuracy = correct_top5 / num_images
    top1_accuracy_app = correct_top1_app / num_images
    top5_accuracy_app = correct_top5_app / num_images

    # print final result
    print("\n" + "="*70)
    print(f"AVG Top-1 of Accurate Model: {top1_accuracy:.5%}")
    print(f"AVG Top-5 of Accurate Model: {top5_accuracy:.5%}")
    print(f"AVG Top-1 of Approximate Model: {top1_accuracy_app:.5%}")
    print(f"AVG Top-5 of Approximate Model: {top5_accuracy_app:.5%}")
    print("="*70)
    
    fw.write("\n" + "="*70 + "\n")
    fw.write(f"AVG Top-1 Accurate Model: {top1_accuracy:.5%}\n")
    fw.write(f"AVG Top-5 Accurate Model: {top5_accuracy:.5%}\n")
    fw.write(f"AVG Top-1 of Approximate Model: {top1_accuracy_app:.5%}\n")
    fw.write(f"AVG Top-5 of Approximate Model: {top5_accuracy_app:.5%}\n")
    fw.write("="*70 + "\n")

    return top1_accuracy, top5_accuracy, top1_accuracy_app, top5_accuracy_app

if __name__ == "__main__":
    # data dir
    image_dir = "../Val_dataset/imagenet/"
    label_file = "../Val_dataset/imagenet/val.txt"

    # file path check
    if not os.path.exists(image_dir):
        print(f"Error: image_dir '{image_dir}' do not exist")
        fw.write(f"Error: image_dir '{image_dir}' do not exist\n")
        exit(1)
    
    if not os.path.exists(label_file):
        print(f"Error: label_dir '{label_file}' do not exist")
        fw.write(f"Error: label_dir '{label_file}' do not exist\n")
        exit(1)
    
    # run test
    print(f"Staring test ResNet18 on '{image_dir}' ...")
    fw.write(f"Staring test ResNet18 on '{image_dir}' ...\n")
    start_time = time.time()
    
    top1_acc, top5_acc, top1_app, top5_app = test_model(image_dir, label_file)
    
    duration = time.time() - start_time
    speed = len(os.listdir(image_dir)) / duration if os.path.exists(image_dir) else 0
    
    print(f"Test Finish! Total time: {duration:.2f} second | AVG Speed: {speed:.1f} figs/s")
    fw.write(f"Test Finish! Total time: {duration:.2f} second | AVG Speed: {speed:.1f} figs/s\n")
    fw.close()
