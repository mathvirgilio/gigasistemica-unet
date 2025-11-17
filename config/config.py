"""
Configurações centralizadas para o projeto DC-UNet de segmentação de ateroma.
Todas as configurações podem ser sobrescritas via variáveis de ambiente.
"""
import torch
import os
from pathlib import Path
import datetime


# ==================== PATHS ====================
# Use variáveis de ambiente ou ajuste os paths abaixo
BASE_DIR = Path(__file__).parent.parent

# Dataset paths - configure via variáveis de ambiente ou ajuste aqui
DATASET_DIR = os.getenv('DATASET_DIR', 'data/dataset/')
DATASET_DIR = Path(DATASET_DIR)

# Estrutura esperada do dataset:
# dataset/
#   images/
#     train/
#     val/
#   masks/
#     train/
#     val/
TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
TRAIN_MASK_DIR = DATASET_DIR / "masks" / "train"
VAL_IMG_DIR = DATASET_DIR / "images" / "val"
VAL_MASK_DIR = DATASET_DIR / "masks" / "val"

# Diretório para salvar runs e checkpoints
RUNS_DIR = Path(os.getenv('RUNS_DIR', BASE_DIR / 'runs'))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Path para checkpoint pré-treinado (opcional)
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', None)


# ==================== MODEL HYPERPARAMETERS ====================
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '300'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-4'))
MIN_LEARNING_RATE = float(os.getenv('MIN_LEARNING_RATE', str(LEARNING_RATE)))
GAMMA = float(os.getenv('GAMMA', '1'))

# Arquitetura
TRAIN_SIZE = int(os.getenv('TRAIN_SIZE', '512'))
IN_CHANNELS = int(os.getenv('IN_CHANNELS', '1'))
OUT_CHANNELS = 1

# Loss function: 'IoU' ou 'Focal Loss'
LOSS_FUNCTION = os.getenv('LOSS_FUNCTION', 'IoU')

# Threshold para binarização
THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))


# ==================== TRAINING SETTINGS ====================
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '0'))
PIN_MEMORY = os.getenv('PIN_MEMORY', 'True').lower() == 'true'
USE_DATA_PARALLEL = os.getenv('USE_DATA_PARALLEL', 'False').lower() == 'true'
AUGMENTATION = os.getenv('AUGMENTATION', 'True').lower() == 'true'
APPLY_TTA = os.getenv('APPLY_TTA', 'False').lower() == 'true'

# Device
DEVICE_STR = os.getenv('DEVICE', None)
if DEVICE_STR:
    DEVICE = DEVICE_STR
elif torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

# GPU device ID (se usando CUDA)
CUDA_DEVICE_ID = int(os.getenv('CUDA_DEVICE_ID', '0'))


# ==================== OPTIMIZER ====================
OPTIMIZER = os.getenv('OPTIMIZER', 'Adam')  # 'Adam' ou 'SGD'
SGD_WEIGHT_DECAY = float(os.getenv('SGD_WEIGHT_DECAY', '1e-4'))
SGD_MOMENTUM = float(os.getenv('SGD_MOMENTUM', '0.9'))


# ==================== VALIDATION SETTINGS ====================
SAVE_TEST_IMAGES = os.getenv('SAVE_TEST_IMAGES', 'False').lower() == 'true'
PRINT_METRICS = os.getenv('PRINT_METRICS', 'False').lower() == 'true'
USE_ONLY_ORIGINAL_IMAGES = os.getenv('USE_ONLY_ORIGINAL_IMAGES', 'False').lower() == 'true'


# ==================== RUN MANAGEMENT ====================
NOTES = os.getenv('NOTES', 'DC-UNet Training Run')
LOAD_RUN = os.getenv('LOAD_RUN', 'False').lower() == 'true'
TRAINING = os.getenv('TRAINING', 'True').lower() == 'true'


def create_run_dir_with_description(runs_directory=None):
    """
    Cria um diretório para a run atual com timestamp e salva as configurações.
    
    Args:
        runs_directory: Diretório base para runs. Se None, usa RUNS_DIR.
    
    Returns:
        Path do diretório criado
    """
    if runs_directory is None:
        runs_directory = RUNS_DIR
    else:
        runs_directory = Path(runs_directory)
    
    run_name = str(datetime.datetime.now()).replace(':', '-').replace(' ', '_')
    run_dir = runs_directory / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar subdiretórios
    imgs_saved_dir = run_dir / "saved_images"
    (imgs_saved_dir / "val").mkdir(parents=True, exist_ok=True)
    (imgs_saved_dir / "test").mkdir(parents=True, exist_ok=True)
    
    # Salvar parâmetros do modelo
    params_file = run_dir / "model_parameters.txt"
    with open(params_file, "w") as f:
        f.write(f"Dataset: {DATASET_DIR}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Number of epochs: {NUM_EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE} -> {MIN_LEARNING_RATE}\n")
        f.write(f"Loss: {LOSS_FUNCTION} (Gamma = {GAMMA})\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Train size: {TRAIN_SIZE}\n")
        f.write(f"In channels: {IN_CHANNELS}\n")
        f.write(f"Augmentation: {AUGMENTATION}\n")
        f.write(f"Notes: {NOTES}\n")
    
    return run_dir


# Criar diretório de run se necessário
RUN_DIR = None
IMG_PATH = None
NAME_CHECKPOINT = None

# Inicializar apenas se estiver em modo de treinamento
if TRAINING:
    RUN_DIR = create_run_dir_with_description()
    IMG_PATH = RUN_DIR / "saved_images"
    NAME_CHECKPOINT = RUN_DIR / "checkpoint.pth.tar"


# ==================== DATASET DICT ====================
DATASET_DICT = {
    "train_images": str(TRAIN_IMG_DIR),
    "train_masks": str(TRAIN_MASK_DIR),
    "val_images": str(VAL_IMG_DIR),
    "val_masks": str(VAL_MASK_DIR),
}

