"""
Script de teste do modelo DC-UNet.
Avalia o modelo em um conjunto de teste e gera métricas detalhadas.
"""
import torch
import argparse
from pathlib import Path
import sys
import pandas as pd

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    VAL_IMG_DIR, VAL_MASK_DIR, TRAIN_SIZE, IN_CHANNELS, 
    DEVICE, CHECKPOINT_PATH, THRESHOLD
)
from src.models import DC_Unet
from src.data import Ateroma_test_dataset
from src.utils.validate import validate
from src.utils.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Testar modelo DC-UNet')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Caminho para o checkpoint do modelo')
    parser.add_argument('--val_img_dir', type=str, default=str(VAL_IMG_DIR),
                        help='Diretório com imagens de validação')
    parser.add_argument('--val_mask_dir', type=str, default=str(VAL_MASK_DIR),
                        help='Diretório com máscaras de validação')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device para execução (cuda ou cpu)')
    parser.add_argument('--apply_tta', action='store_true',
                        help='Aplicar Test Time Augmentation')
    parser.add_argument('--save_images', action='store_true',
                        help='Salvar imagens de resultado')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Diretório para salvar resultados (opcional)')
    
    args = parser.parse_args()
    
    if args.checkpoint is None:
        raise ValueError("É necessário fornecer um checkpoint via --checkpoint ou configurar CHECKPOINT_PATH no config.py")
    
    # Carregar modelo
    print(f"Carregando modelo de {args.checkpoint}")
    model = DC_Unet(in_channels=IN_CHANNELS).to(args.device)
    load_checkpoint(torch.load(args.checkpoint, map_location=torch.device(args.device)), model)
    
    # Preparar data loader
    loader = Ateroma_test_dataset(args.val_img_dir, args.val_mask_dir, TRAIN_SIZE)
    
    # Validar
    print("Iniciando validação...")
    mean_loss, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, mean_auc, metrics = validate(
        model, loader,
        save_test_images=args.save_images,
        apply_tta=args.apply_tta,
        print_metrics=False,
        use_only_original_images=False,
        device=args.device,
        val_img_dir=args.val_img_dir,
        val_mask_dir=args.val_mask_dir
    )
    
    # Criar DataFrame com métricas
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df['name'] = metrics_df['name'].str.replace(".png", "")
    
    # Imprimir métricas médias
    print("\n" + "="*50)
    print("MÉTRICAS MÉDIAS")
    print("="*50)
    print(f"validation/loss = {mean_loss:.4f}")
    print(f"validation/precision = {mean_precision:.4f}")
    print(f"validation/recall = {mean_recall:.4f}")
    print(f"validation/f1_score = {mean_f1_score:.4f}")
    print(f"validation/IoU_score = {mean_IoU:.4f}")
    print(f"validation/dice_score = {mean_dice:.4f}")
    print(f"validation/auc = {mean_auc:.4f}")
    
    # Métricas das imagens originais (terminadas em R ou L)
    print("\n" + "="*50)
    print("MÉTRICAS DAS IMAGENS ORIGINAIS (terminadas em R ou L)")
    print("="*50)
    metrics_df_orig = metrics_df.copy()
    orig_names = metrics_df_orig[metrics_df_orig['name'].str.endswith(('R', 'L'))]
    if len(orig_names) > 0:
        media_orig = orig_names.mean(numeric_only=True)
        print(media_orig)
    else:
        print("Nenhuma imagem original encontrada")
    
    # Métricas das melhores (uma por caso, mantendo a melhor IoU)
    print("\n" + "="*50)
    print("MÉTRICAS DAS MELHORES (uma por caso)")
    print("="*50)
    metrics_df_best = metrics_df.copy()
    # Remover sufixo se não for R ou L
    metrics_df_best['name_base'] = metrics_df_best['name'].apply(
        lambda x: x[:-1] if x[-1] not in ['R', 'L'] else x
    )
    # Manter apenas a melhor por caso (maior IoU)
    metrics_df_best = metrics_df_best.sort_values('iou', ascending=False).drop_duplicates(subset='name_base')
    if len(metrics_df_best) > 0:
        media_best = metrics_df_best.mean(numeric_only=True)
        print(media_best)
    else:
        print("Nenhuma métrica encontrada")
    
    # Salvar resultados se solicitado
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path / 'metrics_detailed.csv', index=False)
        print(f"\nMétricas detalhadas salvas em {output_path / 'metrics_detailed.csv'}")


if __name__ == '__main__':
    main()
