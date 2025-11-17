# DC-UNet para Segmentação de Ateroma

Este projeto implementa uma arquitetura DC-UNet (Dense Convolutional UNet) para segmentação de ateroma em imagens médicas. O modelo utiliza uma arquitetura baseada em U-Net com blocos Dense Convolutional para melhorar a captura de características em diferentes escalas.

## 📁 Estrutura do Projeto

```
gigasistemica-unet/
├── src/
│   ├── models/           # Arquiteturas do modelo
│   │   └── DC_UNet.py    # Implementação do DC-UNet
│   ├── data/             # Data loaders e datasets
│   │   ├── dataloader.py # Data loaders principais
│   │   └── ateroma_dataloader.py  # Data loader específico para ateroma
│   ├── training/         # Scripts de treinamento
│   │   └── train.py      # Script principal de treinamento
│   └── utils/            # Utilitários
│       ├── loss.py       # Funções de loss
│       ├── utils.py      # Funções auxiliares
│       ├── validate.py   # Função de validação
│       └── TTA.py        # Test Time Augmentation
├── config/               # Configurações
│   └── config.py         # Arquivo de configuração centralizado
├── scripts/              # Scripts de execução
│   └── test.py           # Script de teste/validação
├── requirements.txt      # Dependências do projeto
└── README.md            # Este arquivo
```

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd gigasistemica-unet
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuração

Todas as configurações estão centralizadas no arquivo `config/config.py`. Você pode:

1. **Editar diretamente o arquivo `config/config.py`** para ajustar paths e hiperparâmetros

2. **Usar variáveis de ambiente** (recomendado para diferentes ambientes):
```bash
export DATASET_DIR="/caminho/para/dataset"
export RUNS_DIR="/caminho/para/runs"
export DEVICE="cuda:0"
export BATCH_SIZE=4
export NUM_EPOCHS=300
export LEARNING_RATE=1e-4
```

### Estrutura Esperada do Dataset

O dataset deve seguir a seguinte estrutura:
```
dataset/
├── images/
│   ├── train/     # Imagens de treinamento (.jpg ou .png)
│   └── val/       # Imagens de validação (.jpg ou .png)
└── masks/
    ├── train/     # Máscaras de treinamento (.png)
    └── val/       # Máscaras de validação (.png ou .tif)
```

## 📊 Uso

### Treinamento

Para treinar o modelo, execute:

```bash
python src/training/train.py
```

Ou com variáveis de ambiente customizadas:
```bash
DATASET_DIR="/caminho/dataset" DEVICE="cuda:0" python src/training/train.py
```

O script irá:
- Criar um diretório de run com timestamp em `runs/`
- Salvar checkpoints periodicamente
- Registrar métricas no TensorBoard
- Executar validação a cada época

### Teste/Validação

Para testar um modelo treinado:

```bash
python scripts/test.py --checkpoint /caminho/para/checkpoint.pth.tar --save_images
```

Opções disponíveis:
- `--checkpoint`: Caminho para o checkpoint do modelo (obrigatório)
- `--val_img_dir`: Diretório com imagens de validação (opcional, usa config)
- `--val_mask_dir`: Diretório com máscaras de validação (opcional, usa config)
- `--device`: Device para execução (`cuda` ou `cpu`)
- `--apply_tta`: Aplicar Test Time Augmentation
- `--save_images`: Salvar imagens de resultado
- `--output_dir`: Diretório para salvar métricas detalhadas (CSV)

Exemplo completo:
```bash
python scripts/test.py \
    --checkpoint runs/2024-11-17_15-30-00/checkpoint.pth.tar \
    --save_images \
    --apply_tta \
    --output_dir results/
```

## 🏗️ Arquitetura

O DC-UNet é baseado na arquitetura U-Net com as seguintes características:

- **Encoder**: Blocos DCBlock (Dense Convolutional) com ResPath para preservar informações
- **Decoder**: Upsampling com skip connections
- **Loss Function**: Suporta IoU Loss e Focal Loss
- **Input**: Imagens em escala de cinza (1 canal) ou RGB (3 canais)
- **Output**: Máscara binária de segmentação

### Hiperparâmetros Principais

- **TRAIN_SIZE**: Tamanho das imagens de entrada (padrão: 512x512)
- **IN_CHANNELS**: Número de canais de entrada (padrão: 1 para grayscale)
- **BATCH_SIZE**: Tamanho do batch (padrão: 4)
- **LEARNING_RATE**: Taxa de aprendizado (padrão: 1e-4)
- **NUM_EPOCHS**: Número de épocas (padrão: 300)
- **LOSS_FUNCTION**: Função de loss ('IoU' ou 'Focal Loss')

## 📈 Métricas

O modelo calcula as seguintes métricas:

- **Precision**: Precisão da segmentação
- **Recall**: Recall da segmentação
- **F1 Score**: Média harmônica de precision e recall
- **IoU (Intersection over Union)**: Sobreposição entre predição e ground truth
- **Dice Score**: Coeficiente de Dice
- **AUC**: Área sob a curva ROC

## 🔧 Funcionalidades

### Test Time Augmentation (TTA)

O projeto suporta TTA para melhorar a robustez das predições. Quando ativado, o modelo faz predições em múltiplas versões aumentadas da imagem e combina os resultados.

### Data Augmentation

Durante o treinamento, as seguintes transformações podem ser aplicadas:
- Rotação aleatória (até 90 graus)
- Flip horizontal e vertical
- Ajustes de brilho e contraste
- Transformações elásticas

## 📝 Notas

- O projeto foi desenvolvido para segmentação de ateroma, mas pode ser adaptado para outras tarefas de segmentação semântica
- Os paths hardcoded foram removidos e centralizados em `config/config.py`
- O código foi organizado em módulos para facilitar manutenção e extensão

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob licença [especificar licença].

## 👥 Autores

- [Seu Nome] - Desenvolvimento inicial

## 🙏 Agradecimentos

- Baseado na arquitetura U-Net original
- Utiliza componentes do PyTorch e bibliotecas open-source
