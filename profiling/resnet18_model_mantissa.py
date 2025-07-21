import sys
import os
import torch
import torchvision
import torchvision.models as models
from torchsummary import summary
import pickle
import numpy as np

fw = open("./ResNet18-layer-diff.txt", 'a')
fw.write("\n")
fw.write("\n")
fw.write("\n")
fw.write("\n")
fw.write("\n")
fw.write("\n")

bits = 13
fw.write(f'bits = {bits}')
fw.write('\n')

def truncate_mantissa(tensor, bits):
    """
    Clear the specified number of bits of the mantissum of the floating-point tensor to zero
    :param tensor: The input floating-point tensor
    :param bits: The number of last digits to be reset to zero (default is 10 digits)
    :return: Modified tensor
    """
    # Make sure to handle only floating-point tensors
    if not tensor.is_floating_point():
        return tensor
    
    cpu_tensor = tensor.detach().cpu()
    float_array = cpu_tensor.numpy()
    
    original_dtype = float_array.dtype
    int_array = float_array.view(np.int32)
    
    # Create a tail number mask (0x007FFFFF is the mask for the tail number part)
    mantissa_mask = 0x007FFFFF  # 23bits
    other_mask = 0b11111111100000000000000000000000
    clear_mask = ~((1 << bits) - 1) 
    final_mask = mantissa_mask & clear_mask | other_mask  # mask

    int_array &= final_mask
    
    # Convert back to a floating-point number and return a PyTorch tensor
    modified_array = int_array.view(original_dtype)
    return torch.from_numpy(modified_array).to(tensor.device)

def modify_weights():
    """
    Modify the model's weight and set the mantissa to zero
    :return: The modified model
    """
    # Load the Pytorch pre-trained model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.data.dim() > 1:  # Only handle the weight matrix and ignore the bias
                param.data = truncate_mantissa(param.data, bits)
    
    return model

def verify_modification(model):
    """
    Verify whether the weight modification was successful
    param model: The modified VGG16 model
    """
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            float_tensor = param.data.cpu().numpy().flatten()
            int_view = float_tensor.view(np.int32)
            
            mantissas = int_view & 0x007FFFFF
            
            for i, mantissa in enumerate(mantissas[:5]):
                last_10_bits = mantissa & 0x3FF
                if last_10_bits != 0:
                    print(f"Fail! Layer: {name} value#{i} mantissa last:{bits}位: {bin(last_10_bits)}")
                    fw.write(f"Fail! Layer: {name} value#{i} mantissa last:{bits}位: {bin(last_10_bits)}\n")
                    return
            
    print(f"Success! All weight's last mantissa {bits} bits are set to zero")
    fw.write(f"Success! All weight's last mantissa {bits} bits are set to zero")


# main
if __name__ == "__main__":

    modified_model = modify_weights()
    
    verify_modification(modified_model)
    
    torch.save(modified_model.state_dict(), "./weight/resnet18_truncated_weights.pth")
    print("The modified model weights have been saved to: resnet18_truncated_weights.pth")
    fw.write("The modified model weights have been saved to: resnet18_truncated_weights.pth\n")

    input_tensor = torch.randn(1, 3, 224, 224)
    output = modified_model(input_tensor)
    print(f"Test model output shape: {output.shape}")
    fw.write(f"Test model output shape: {output.shape}\n")
    fw.close()