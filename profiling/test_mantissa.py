import torch
from torchvision import transforms, models
import numpy as np

# Load pre-trained model and the modified mantissa model
model_app = models.resnet18()
model_app.load_state_dict(torch.load('./weight/resnet18_truncated_weights.pth'))
model_acc = models.resnet18(pretrained=True)

def weights_to_binary(weights, max_bits=320):
    byte_data = weights.cpu().detach().numpy().tobytes()
    binary_str = ''.join(format(byte, '08b') for byte in byte_data)
    return binary_str, len(binary_str)

def format_binary_preview(binary_str, max_bits=320, groups_per_line=4):
    truncated = binary_str[:max_bits]
    groups = []
    
    for i in range(0, len(truncated), 8):
        group = truncated[i:i+8]
        if len(group) < 8: 
            group += ' ' * (8 - len(group))
        groups.append(group)
    
    lines = []
    for i in range(0, len(groups), groups_per_line):
        line_groups = groups[i:i+groups_per_line]
        lines.append(line_groups)
    
    return lines

def highlight_diff_group(group1, group2):
    """highlight different bits"""
    if group1 == group2:
        return group1, group2
    
    highlighted1 = []
    highlighted2 = []
    for b1, b2 in zip(group1, group2):
        if b1 == b2 or b1 == ' ' or b2 == ' ':
            highlighted1.append(b1)
            highlighted2.append(b2)
        else:
            highlighted1.append("\033[91m" + b1 + "\033[0m")
            highlighted2.append("\033[91m" + b2 + "\033[0m")
    
    return ''.join(highlighted1), ''.join(highlighted2)

def print_layer_comparison(layer_name, param_acc, param_app):
    print(f"\n{'-'*80}")
    print(f"LAYER: {layer_name}")
    print(f"Shape: {tuple(param_acc.shape)} | Data type: {param_acc.dtype}")
    print(f"{'-'*80}")
    
    # get binary
    binary_acc, total_acc = weights_to_binary(param_acc)
    binary_app, total_app = weights_to_binary(param_app)
    
    # get view
    lines_acc = format_binary_preview(binary_acc, groups_per_line=4)
    lines_app = format_binary_preview(binary_app, groups_per_line=4)
    
    print(f"{'PRETRAINED (model_acc)':<38}    {'CUSTOM WEIGHTS (model_app)':<38}")
    print(f"{'Total bits: '+str(total_acc):<38}    {'Total bits: '+str(total_app):<38}")
    print('-'*80)
    
    max_lines = max(len(lines_acc), len(lines_app))
    
    for i in range(max_lines):

        acc_groups = lines_acc[i] if i < len(lines_acc) else ["", "", "", ""]
        app_groups = lines_app[i] if i < len(lines_app) else ["", "", "", ""]
        
        acc_line_parts = []
        app_line_parts = []
        
        for j in range(4):
            acc_group = acc_groups[j] if j < len(acc_groups) else "        "
            app_group = app_groups[j] if j < len(app_groups) else "        "
            
            hl_acc, hl_app = highlight_diff_group(acc_group, app_group)
            acc_line_parts.append(hl_acc)
            app_line_parts.append(hl_app)
        
        acc_line = ' '.join(acc_line_parts)
        app_line = ' '.join(app_line_parts)
        
        print(f"{acc_line:<38}    {app_line:<38}")
    
    # only show the first 10 values
    if total_acc > 320 or total_app > 320:
        shown_bits = min(320, total_acc, total_app)
        print(f"\nNOTE: Showing first {shown_bits} bits of {max(total_acc, total_app)} total bits")
    print('='*80)

# make sure two models are the same structure
params_acc = list(model_acc.named_parameters())
params_app = list(model_app.named_parameters())

# print all layers
print("\n" + "="*100)
print("MODEL WEIGHT COMPARISON: PRETRAINED vs CUSTOM WEIGHTS")
print("="*100)

for (name_acc, param_acc), (name_app, param_app) in zip(params_acc, params_app):
    if name_acc != name_app:
        print(f"Warning: Layer name mismatch! {name_acc} vs {name_app}")
        continue
        
    if param_acc.requires_grad and param_app.requires_grad:
        print_layer_comparison(name_acc, param_acc.data, param_app.data)
