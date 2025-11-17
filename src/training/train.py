"""
Script de treinamento do modelo DC-UNet para segmentação de ateroma.
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import sys

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm
from config.config import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
    IN_CHANNELS, TRAIN_SIZE, AUGMENTATION, NUM_EPOCHS, LEARNING_RATE,
    BATCH_SIZE, NUM_WORKERS, DEVICE, CUDA_DEVICE_ID, RUNS_DIR,
    create_run_dir_with_description
)
from src.models import DC_Unet
from src.data import get_loader, Ateroma_test_dataset
from src.utils import structure_loss, adjust_lr, save_checkpoint
from src.utils.validate import validate


def load_checkpoint(checkpoint_path, model, device):
    """Carrega um checkpoint do modelo."""
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def train(model, loader, optimizer, device):
    """Loop de treinamento."""
    model.train()
    mean_loss = 0
    loop = tqdm(loader)
    len_loader = len(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        optimizer.zero_grad()
        data = Variable(data).to(device)
        targets = Variable(targets).to(device)
        predictions = model(data)
        
        loss = structure_loss(predictions, targets)
        loss_value = loss.item()
        mean_loss += loss_value
        loop.set_postfix(loss=loss_value)
        loss.backward()
        optimizer.step()
    
    mean_loss = mean_loss / len_loader
    return mean_loss


if __name__ == '__main__':
    # Criar diretório para esta run
    run_directory = create_run_dir_with_description(RUNS_DIR)
    checkpoint_path = run_directory / "checkpoint.pth.tar"
    writer = SummaryWriter(log_dir=str(run_directory / "plot"))
    
    # Configurar device
    if 'cuda' in DEVICE:
        torch.cuda.set_device(CUDA_DEVICE_ID)
    
    # Construir modelo
    model = DC_Unet(in_channels=IN_CHANNELS).to(DEVICE)
    
    # Otimizador
    params = model.parameters()
    optimizer = torch.optim.Adam(params, LEARNING_RATE)
    
    # Data loaders
    train_loader = get_loader(
        str(TRAIN_IMG_DIR), str(TRAIN_MASK_DIR),
        batchsize=BATCH_SIZE,
        trainsize=TRAIN_SIZE,
        num_workers=NUM_WORKERS,
        augmentation=AUGMENTATION
    )
    
    val_loader = Ateroma_test_dataset(
        str(VAL_IMG_DIR), str(VAL_MASK_DIR), TRAIN_SIZE
    )
    
    total_step = len(train_loader)
    
    # Loop de treinamento
    for epoch in range(1, NUM_EPOCHS + 1):
        adjust_lr(optimizer, LEARNING_RATE, epoch, 0.1, 100)
        mean_loss = train(model, train_loader, optimizer, DEVICE)
        
        # Salvar checkpoint
        checkpoint_dict = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint_dict, filename=str(checkpoint_path))
        
        # Validação
        mean_loss_val, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, _, _ = validate(
            model, val_loader, save_test_images=False, device=DEVICE
        )
        
        # Log no tensorboard
        writer.add_scalar('training/loss', mean_loss, epoch)
        writer.add_scalar('validation/loss', mean_loss_val, epoch)
        writer.add_scalar('validation/precision', mean_precision, epoch)
        writer.add_scalar('validation/recall', mean_recall, epoch)
        writer.add_scalar('validation/f1_score', mean_f1_score, epoch)
        writer.add_scalar('validation/IoU', mean_IoU, epoch)
        writer.add_scalar('validation/score', mean_dice, epoch)
    
    writer.close()
    print(f"Treinamento concluído! Checkpoint salvo em: {checkpoint_path}")
