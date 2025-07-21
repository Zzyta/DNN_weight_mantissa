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
fw = open("./ResNet18-layer-diff.txt", 'a') 
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
        return img, label, self.image_files[idx]

def load_test_data(data_dir, label_file):
    with open(label_file, 'r') as f:
        gt_labels = [int(line.strip()) -1 for line in f.readlines()]
    
    image_files = []
    for file in sorted(os.listdir(data_dir)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(data_dir, file))
    
    assert len(image_files) == len(gt_labels), f"Number of pictures ({len(image_files)}) do not match the label ({len(gt_labels)})"
    print(f"find {len(image_files)} testing pictures")
    fw.write(f"find {len(image_files)} testing pictures\n")
    
    return image_files, gt_labels

# Hook every layer's output
def register_activation_hooks(model, model_name):

    activations = OrderedDict()
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    handles = []
    for name, module in model.named_modules():
        if name:
            handle = module.register_forward_hook(hook_fn(name))
            handles.append(handle)
    
    return activations, handles

# calculate difference of acc and app model
def compute_output_differences(acts1, acts2, image_path):
    """Calculate the differences in the outputs of each layer of the two models"""
    differences = OrderedDict()
    
    # Compare the same layer
    assert set(acts1.keys()) == set(acts2.keys()), "Error:Model structure is not the same"
    
    for layer_name in acts1.keys():
        output1 = acts1[layer_name]
        output2 = acts2[layer_name]
        
        if output1.shape != output2.shape:
            print(f"Error: {layer_name} output shape is not the same: {output1.shape} vs {output2.shape}")
            continue
        
        sum1 = output1.sum()
        sum2 = output2.sum()
        abs_diff = torch.abs(output1 - output2)
        mae = torch.mean(abs_diff).item()  # Mean absolute error
        mse = torch.mean(torch.square(output1 - output2)).item()  # Mean square error
        max_diff = torch.max(abs_diff).item()  # Max error
        norm_diff = torch.mean(abs_diff / abs(output1.max())).item() # Normalized error
        sum_diff = abs(sum1-sum2)/abs(sum1) # error for the sum of the layer
        cos_sim = nn.functional.cosine_similarity(output1.flatten(), output2.flatten(), dim=0).item() + 1
        
        differences[layer_name] = {
            'MAE': mae,
            'MSE': mse,
            'MaxDiff': max_diff,
            'NormDiff': norm_diff,
            'SumDiff': sum_diff,
            'Cos_Sim': cos_sim,
            'Shape': output1.shape
        }
    
    return differences

def test_layer_comparison(image_dir, label_file):
    # load test data
    image_files, gt_labels = load_test_data(image_dir, label_file)
    num_images = len(image_files)
    
    # create dateset
    dataset = ImageNetDataset(image_files, gt_labels, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1, # need to run the picture one by one!!!
        num_workers=min(4, mp.cpu_count()),
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True
    )
    
    # hook
    activations_model, handles_model = register_activation_hooks(model, "model")
    activations_app, handles_app = register_activation_hooks(model_app, "model_app")
    
    # initial
    correct_top1 = 0
    correct_top5 = 0
    correct_top1_app = 0
    correct_top5_app = 0
    processed = 0
    
    layer_diff_stats = defaultdict(lambda: {
        'MAE_sum': 0.0,
        'MSE_sum': 0.0,
        'MaxDiff_sum': 0.0,
        'MaxDiff_max': 0.0,
        'NormDiff_sum': 0.0,
        'SumDiff_sum': 0.0,
        'Cos_Sim_sum': 0.0,
        'Cos_Sim_min': 2.0,
        'count': 0
    })
    
    start_time = time.time()
    
    # run the test each picture!!!
    for batch_idx, (image_batch, label_batch, image_paths) in enumerate(dataloader):

        activations_model.clear()
        activations_app.clear()
        
        
        image_batch = image_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)
        
        # model inference
        with torch.no_grad():
            outputs = model(image_batch)
            outputs_app = model_app(image_batch)
        
        # Top-1
        _, preds = torch.max(outputs, 1)
        correct_top1 += (preds == label_batch).sum().item()
        _, preds_app = torch.max(outputs_app, 1)
        correct_top1_app += (preds_app == label_batch).sum().item()

        # Top-5
        _, top5_preds = torch.topk(outputs, 5, dim=1)
        correct_top5 += (top5_preds == label_batch.view(-1, 1)).any(dim=1).sum().item()
        _, top5_preds_app = torch.topk(outputs_app, 5, dim=1)
        correct_top5_app += (top5_preds_app == label_batch.view(-1, 1)).any(dim=1).sum().item()
        
        differences = compute_output_differences(activations_model, activations_app, image_paths[0])
        for layer_name, diff_metrics in differences.items():
            stats = layer_diff_stats[layer_name]
            stats['MAE_sum'] += diff_metrics['MAE']
            stats['MSE_sum'] += diff_metrics['MSE']
            stats['MaxDiff_sum'] += diff_metrics['MaxDiff']
            stats['NormDiff_sum'] += diff_metrics['NormDiff']
            stats['SumDiff_sum'] += diff_metrics['SumDiff']
            stats['Cos_Sim_sum'] += diff_metrics['Cos_Sim']
            stats['Cos_Sim_min'] = min(stats['Cos_Sim_min'], diff_metrics['Cos_Sim'])
            stats['MaxDiff_max'] = max(stats['MaxDiff_max'], diff_metrics['MaxDiff'])
            stats['count'] += 1
        
        processed += 1
        if processed % 100 == 0 or processed == num_images:
            elapsed = time.time() - start_time
            speed = processed / elapsed
            
            print(f"Processed: {processed}/{num_images} | "
                  f"Top-1 of accurate model: {correct_top1/processed:.5%} | "
                  f"Top-1 of approximate model: {correct_top1_app/processed:.5%} | "
                  f"Speed: {speed:.1f} figs/s")
            
            fw.write(f"Processed: {processed}/{num_images} | "
                  f"Top-1 of accurate model: {correct_top1/processed:.5%} | "
                  f"Top-1 of approximate model: {correct_top1_app/processed:.5%} | "
                  f"Speed: {speed:.1f} figs/s\n")
    
    # remove hook
    for handle in handles_model + handles_app:
        handle.remove()
    
    # final accuracy
    top1_accuracy = correct_top1 / num_images
    top5_accuracy = correct_top5 / num_images
    top1_accuracy_app = correct_top1_app / num_images
    top5_accuracy_app = correct_top5_app / num_images

    # print result
    print("\n" + "="*70)
    print(f"AVG Top-1 of Accurate Model: {top1_accuracy:.5%}")
    print(f"AVG Top-5 of Accurate Model: {top5_accuracy:.5%}")
    print(f"AVG Top-1 of Approximate Model: {top1_accuracy_app:.5%}")
    print(f"AVG Top-5 of Approximate Model: {top5_accuracy_app:.5%}")
    print("="*70)
    
    fw.write("\n" + "="*70 + "\n")
    fw.write(f"AVG Top-1 of Accurate Model: {top1_accuracy:.5%}\n")
    fw.write(f"AVG Top-5 of Accurate Model: {top5_accuracy:.5%}\n")
    fw.write(f"AVG Top-1 of Approximate Model: {top1_accuracy_app:.5%}\n")
    fw.write(f"AVG Top-5 of Approximate Model: {top5_accuracy_app:.5%}\n")
    fw.write("="*70 + "\n")
    
    # print difference
    print("\nOutput difference statistics for each layer of the model:")
    fw.write("\nOutput difference statistics for each layer of the model:\n")
    print("=" * 300)
    fw.write("=" * 300)
    fw.write('\n')
    print(f"{'Layer name':<25} | {'AVG MAE':<25} | {'AVG MSE':<25} | {'AVG Max':<25} | {'AVG Norm':<25} | {'AVG Sum':<25} | {'AVG Cos':<25} | {'min Cos':<25} | {'Max MaxDiff':<25} | {'Test counts':<20}")
    fw.write(f"{'Layer name':<25} | {'AVG MAE':<25} | {'AVG MSE':<25} | {'AVG Max':<25} | {'AVG Norm':<25} | {'AVG Sum':<25} | {'AVG Cos':<25} | {'min Cos':<25} | {'Max MaxDiff':<25} | {'Test counts':<20}\n")
    
    for layer_name, stats in layer_diff_stats.items():
        if stats['count'] > 0:
            avg_mae = stats['MAE_sum'] / stats['count']
            avg_mse = stats['MSE_sum'] / stats['count']
            avg_maxdiff = stats['MaxDiff_sum'] / stats['count']
            avg_normdiff = stats['NormDiff_sum'] / stats['count']
            avg_sumdiff = stats['SumDiff_sum'] / stats['count']
            avg_cossim = stats['Cos_Sim_sum'] / stats['count']
            min_cossim = stats['Cos_Sim_min']
            max_diff = stats['MaxDiff_max']

            if (avg_cossim > 2):
                avg_cossim=2.0
            
            print(f"{layer_name:<25} | {avg_mae:<25.20f} | {avg_mse:<25.20f} | {avg_maxdiff:<25.20f} | {avg_normdiff:<25.20f} | {avg_sumdiff:<25.20f} | {avg_cossim:<25.20f} | {min_cossim:<25.20f} | {max_diff:<25.20f} | {stats['count']:<10}")
            fw.write(f"{layer_name:<25} | {avg_mae:<25.20f} | {avg_mse:<25.20f} | {avg_maxdiff:<25.20f} | {avg_normdiff:<25.20f} | {avg_sumdiff:<25.20f} | {avg_cossim:<25.20f} | {min_cossim:<25.20f} | {max_diff:<25.20f} | {stats['count']:<10}\n")
    
    return top1_accuracy, top5_accuracy, top1_accuracy_app, top5_accuracy_app

if __name__ == "__main__":

    image_dir = ""
    label_file = ""

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
    print(f"Staring test ResNet18 on '{image_dir}' including difference of layers...")
    fw.write(f"Staring test ResNet18 on '{image_dir}' including difference of layers...\n")
    start_time = time.time()
    
    top1_acc, top5_acc, top1_app, top5_app = test_layer_comparison(image_dir, label_file)
    
    duration = time.time() - start_time
    speed = len([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) / duration
    
    print(f"Test Finish! Total time: {duration:.2f} second | AVG Speed: {speed:.1f} figs/s")
    fw.write(f"Test Finish! Total time: {duration:.2f} second | AVG Speed: {speed:.1f} figs/s\n")
    fw.close()
