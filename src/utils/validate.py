"""
Módulo de validação para o modelo DC-UNet.
"""
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import sys

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import THRESHOLD, LOSS_FUNCTION
from src.utils.utils import save_images, get_metrics, calculate_precision_recall_f1
from src.utils.loss import structure_loss


def validate(model, loader, save_test_images=False, print_metrics=False, 
             use_only_original_images=False, apply_tta=False, device='cuda',
             img_path=None, val_img_dir=None, val_mask_dir=None):
    """
    Valida o modelo no conjunto de validação.
    
    Args:
        model: Modelo a ser validado
        loader: DataLoader para validação
        save_test_images: Se True, salva imagens de resultado
        print_metrics: Se True, imprime métricas por imagem
        use_only_original_images: Se True, usa apenas imagens originais
        apply_tta: Se True, aplica Test Time Augmentation
        device: Device para execução ('cuda' ou 'cpu')
        img_path: Caminho para salvar imagens (opcional)
        val_img_dir: Diretório de imagens de validação (opcional)
        val_mask_dir: Diretório de máscaras de validação (opcional)
    
    Returns:
        Tupla com métricas médias e dicionário de métricas por imagem
    """
    model.eval()
    
    loader.index = 0
    loader_size = loader.size
    
    metrics = {
        'name': [], 'area': [], 'loss': [], 'precision': [], 
        'recall': [], 'f1 score': [], 'iou': [], 'dice score': []
    }
    mean_loss = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1_score = 0
    mean_IoU = 0
    mean_dice = 0
    mean_auc = 0
    n = 0
    
    with torch.no_grad():
        for i in range(loader_size):
            image, target, name = loader.load_data()
            
            aux = name.replace('.png', '')[-1]
            if (aux == 'R' or aux == 'L') or not (use_only_original_images or apply_tta):
                n += 1
                
                image = image.to(device)
                target_array = np.asarray(target, np.float32)
                target_array /= (target_array.max() + 1e-8)
                
                prediction = model(image)
                prediction = F.interpolate(
                    prediction, size=target_array.shape, 
                    mode='bilinear', align_corners=False
                )
                
                # Calcular loss
                if LOSS_FUNCTION == 'Focal Loss':
                    from torchvision.ops import sigmoid_focal_loss
                    loss = sigmoid_focal_loss(
                        prediction.cpu(), 
                        torch.from_numpy(target_array).unsqueeze(0).unsqueeze(0),
                        reduction='mean', gamma=1
                    ).item()
                else:  # IoU
                    loss = structure_loss(
                        prediction.cpu(),
                        torch.from_numpy(target_array).unsqueeze(0).unsqueeze(0)
                    ).item()
                
                prediction = prediction.sigmoid().data.cpu().numpy().squeeze()
                prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
                
                # Aplicar threshold
                prediction_binary = (prediction >= THRESHOLD).astype(np.float32)
                
                # Aplicar TTA se solicitado
                if apply_tta:
                    from src.utils.TTA import TTA
                    prediction_binary = TTA(model, image, target_array, prediction_binary, threshold=THRESHOLD)
                
                # Calcular métricas
                p, r, f1, IoU, dice, auc = get_metrics(prediction_binary, target_array)
                
                # Caso especial: se target é todo zero
                if np.all(target_array == 0):
                    pred_flat = np.reshape(prediction_binary, (-1))
                    target_flat = np.reshape(target_array, (-1))
                    p, r, f1 = calculate_precision_recall_f1(pred_flat, target_flat, positive_class=0)
                
                if save_test_images:
                    save_images(
                        prediction_binary, name, p, r, f1, IoU, dice,
                        img_path=img_path, val_img_dir=val_img_dir, val_mask_dir=val_mask_dir
                    )
                
                mean_loss += loss
                mean_precision += p
                mean_recall += r
                mean_f1_score += f1
                mean_IoU += IoU
                mean_dice += dice
                mean_auc += auc
                
                metrics['name'].append(name)
                metrics['area'].append(target_array.sum())
                metrics['loss'].append(loss)
                metrics['precision'].append(p)
                metrics['recall'].append(r)
                metrics['f1 score'].append(f1)
                metrics['iou'].append(IoU)
                metrics['dice score'].append(dice)
                
                if print_metrics:
                    print(name)
                    print(f"validation/precision = {p:.4f}")
                    print(f"validation/recall = {r:.4f}")
                    print(f"validation/f1_score = {f1:.4f}")
                    print(f"validation/IoU_score = {IoU:.4f}")
                    print(f"validation/dice_score = {dice:.4f}")
    
    if n > 0:
        mean_loss = mean_loss / n
        mean_precision = mean_precision / n
        mean_recall = mean_recall / n
        mean_f1_score = mean_f1_score / n
        mean_IoU = mean_IoU / n
        mean_dice = mean_dice / n
        mean_auc = mean_auc / n
    else:
        print("Warning: No images were processed!")
    
    return mean_loss, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, mean_auc, metrics
