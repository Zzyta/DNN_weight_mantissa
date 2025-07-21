import sys
import os
import torch
import torchvision
import torchvision.models as models
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import stats

fw = open("./weight-distribution.txt", 'a')

def analyze_weight_distribution(model, 
                                bins=100):

    results = {
        'global': {'weights': [],
            'bins': bins,
            'interval_counts': None,
            'interval_edges': None},
        'layers': OrderedDict()
    }
    
    # 1. gather weight of all layers
    all_weights = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.data.numel() > 0:
            weights = param.data.cpu().numpy().flatten()
            all_weights.append(weights)

            min_val = weights.min()
            max_val = weights.max()
            counts, edges = np.histogram(weights, bins=bins, range=(min_val, max_val))
            
            # Statistical indicators for each layer
            layer_stats = {
                'values': len(weights),
                'min': min_val,
                'max': max_val,
                'mean': weights.mean(),
                'median': np.median(weights),
                'std': weights.std(),
                'interval_counts': counts.tolist(),
                'interval_edges': edges.tolist(),
                'skewness': stats.skew(weights),
                'kurtosis': stats.kurtosis(weights),
                'percentiles': {
                    '1%': np.percentile(weights, 1),
                    '5%': np.percentile(weights, 5),
                    '25%': np.percentile(weights, 25),
                    '50%': np.percentile(weights, 50),
                    '75%': np.percentile(weights, 75),
                    '95%': np.percentile(weights, 95),
                    '99%': np.percentile(weights, 99)
                },
                'shape': param.data.shape
            }
            results['layers'][name] = layer_stats
            
            # print layer-wise result
            print(f"Layer: {name:<30} | Shape: {str(layer_stats['shape']):<25} | "
                  f"Min: {layer_stats['min']:>9.4f} | Max: {layer_stats['max']:>9.4f} | "
                  f"Mean: {layer_stats['mean']:>9.4f} | Std: {layer_stats['std']:>9.4f} | "
                  f"Skew: {layer_stats['skewness']:>7.2f} | {layer_stats['values']} weights")
            print(f'counts = {layer_stats['interval_counts']}')
            print(f'edges = {layer_stats['interval_edges']}\n')
            fw.write(f"Layer: {name:<30} | Shape: {str(layer_stats['shape']):<25} | "
                  f"Min: {layer_stats['min']:>9.4f} | Max: {layer_stats['max']:>9.4f} | "
                  f"Mean: {layer_stats['mean']:>9.4f} | Std: {layer_stats['std']:>9.4f} | "
                  f"Skew: {layer_stats['skewness']:>7.2f} | {layer_stats['values']} weights\n")
            fw.write(f'counts = {layer_stats['interval_counts']}\n')
            fw.write(f'edges = {layer_stats['interval_edges']}\n\n')

    # 2. Calculate the statistical indicators of the entire model
    if all_weights:
        all_weights = np.concatenate(all_weights)

        min_val = all_weights.min()
        max_val = all_weights.max()
        
        global_counts, global_edges = np.histogram(all_weights, bins=bins, range=(min_val, max_val))

        results['global']['values'] = len(all_weights)
        results['global']['min'] = min_val
        results['global']['max'] = max_val
        results['global']['mean'] = all_weights.mean()
        results['global']['median'] = np.median(all_weights)
        results['global']['std'] = all_weights.std()
        results['global']['skewness'] = stats.skew(all_weights)
        results['global']['kurtosis'] = stats.kurtosis(all_weights)
        results['global']['interval_counts'] = global_counts.tolist()
        results['global']['interval_edges'] = global_edges.tolist()
    
    return results

# main
if __name__ == "__main__":
    for name in ['alexnet','googlenet','efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7','efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l','mobilenet_v2','mobilenet_v3_large','vgg16','resnet18','resnet34','resnet50','resnet101','resnet152','shufflenet_v2_x0_5','shufflenet_v2_x1_0','shufflenet_v2_x1_5','shufflenet_v2_x2_0','vit_b_16','vit_b_32','vit_l_16','vit_l_32']:
        print('\n \n \n')
        print('='*50 + f'\nAnalyzing {name.upper()}\n')
        fw.write('\n \n \n')
        fw.write('='*50 + f'\nAnalyzing {name.upper()}\n\n')
        model = getattr(models, name)(pretrained=True)
        
        weight_stats = analyze_weight_distribution(
            model,
            bins=100
        )
        print(model)
        summary(model, (3, 224, 224))
        fw.write(str(model))
        fw.write('\n')
        fw.write(str(summary(model, (3, 224, 224))))
        fw.write('\n')
        print("\nLayer Statistics:")
        fw.write("\nLayer Statistics:\n")
        for layer_name in weight_stats['layers']:
            print(f"Layer: {layer_name:<30} | Shape: {str(weight_stats['layers'][layer_name]['shape']):<25} | "
                    f"Min: {weight_stats['layers'][layer_name]['min']:>9.4f} | Max: {weight_stats['layers'][layer_name]['max']:>9.4f} | "
                    f"Mean: {weight_stats['layers'][layer_name]['mean']:>9.4f} | Std: {weight_stats['layers'][layer_name]['std']:>9.4f} | "
                    f"Skew: {weight_stats['layers'][layer_name]['skewness']:>7.2f} | {weight_stats['layers'][layer_name]['values']} weights")
            fw.write(f"Layer: {layer_name:<30} | Shape: {str(weight_stats['layers'][layer_name]['shape']):<25} | "
                    f"Min: {weight_stats['layers'][layer_name]['min']:>9.4f} | Max: {weight_stats['layers'][layer_name]['max']:>9.4f} | "
                    f"Mean: {weight_stats['layers'][layer_name]['mean']:>9.4f} | Std: {weight_stats['layers'][layer_name]['std']:>9.4f} | "
                    f"Skew: {weight_stats['layers'][layer_name]['skewness']:>7.2f} | {weight_stats['layers'][layer_name]['values']} weights\n")
        

        print("\nGlobal Statistics:")
        print(f"- Value range: [{weight_stats['global']['min']:.6f}, "
            f"{weight_stats['global']['max']:.6f}]")
        print(f"- Num Weight: {weight_stats['global']['values']}")
        print(f"- Median: {weight_stats['global']['median']:.6f}")
        print(f"- Skewness: {weight_stats['global']['skewness']:.4f} "
            "(>0 means right-skewed)")
        print(f"- Kurtosis: {weight_stats['global']['kurtosis']:.4f} "
            "(>3 means heavy-tailed)")
        print(f'counts = {weight_stats['global']['interval_counts']}')
        print(f'edges = {weight_stats['global']['interval_edges']}')
        print('\n')
        fw.write("\nGlobal Statistics:\n")
        fw.write(f"- Value range: [{weight_stats['global']['min']:.6f}, "
            f"{weight_stats['global']['max']:.6f}]\n")
        fw.write(f"- Num Weight: {weight_stats['global']['values']}\n")
        fw.write(f"- Median: {weight_stats['global']['median']:.6f}\n")
        fw.write(f"- Skewness: {weight_stats['global']['skewness']:.4f} "
            "(>0 means right-skewed)\n")
        fw.write(f"- Kurtosis: {weight_stats['global']['kurtosis']:.4f} "
            "(>3 means heavy-tailed)\n")
        fw.write(f'counts = {weight_stats['global']['interval_counts']}\n')
        fw.write(f'edges = {weight_stats['global']['interval_edges']}\n')
        fw.write('\n')
        edges = weight_stats['global']['interval_edges']
        counts = weight_stats['global']['interval_counts']
        max_count = weight_stats['global']['values']
        
        for i in range(len(edges)-1):
            density = counts[i] / max_count #  Draw simple results
            bar = '█' * int(50 * density) 
            print(f"[{edges[i]:>9.6f}, {edges[i+1]:>9.6f}]: {counts[i]:>20,} weights | {bar}")
            fw.write(f"[{edges[i]:>9.6f}, {edges[i+1]:>9.6f}]: {counts[i]:>20,} weights | {bar}\n")

    fw.close()