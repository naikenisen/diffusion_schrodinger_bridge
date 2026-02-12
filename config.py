# ----- Dataset -----
DATASET = "hes_cd30"
DATASET_TRANSFER = "hes_cd30"
IMAGE_SIZE = 256
CHANNELS = 3
DATA_DIR = "./"                          # dataset_v2/ doit être dans ce dossier
TRANSFER = True                          # mode transfer : HES -> CD30
LOAD = False

# ----- Modèle (UNET) -----
MODEL = "UNET"
NUM_CHANNELS = 96
NUM_RES_BLOCKS = 2
NUM_HEADS = 4
NUM_HEADS_UPSAMPLE = -1
ATTENTION_RESOLUTIONS = "16"
DROPOUT = 0.0
USE_CHECKPOINT = True
USE_SCALE_SHIFT_NORM = True

import torch
# ----- Device -----
DEVICE = "cuda"
DATAPARALLEL = True
NUM_WORKERS = 8  # Augmenté pour accélérer le dataloader
PIN_MEMORY = True

# ----- Mixed Precision (fp16) -----
USE_FP16 = True  # Active le mode fp16/mixed precision pour accélérer l'entraînement sur V100

# ----- Entraînement -----
BATCH_SIZE = 64
LR = 1e-4
NUM_ITER = 10000
N_IPF = 15
N_IPF_INIT = 1
CACHE_NPAR = 32
NUM_CACHE_BATCHES = 10
CACHE_REFRESH_STRIDE = 300
USE_PREV_NET = True
MEAN_MATCH = True

# ----- EMA -----
EMA = True
EMA_RATE = 0.999

# ----- Gradient clipping -----
GRAD_CLIPPING = True
GRAD_CLIP = 1.0

# ----- Schedule de diffusion -----
NUM_STEPS = 20
GAMMA_MAX = 0.1
GAMMA_MIN = 1e-5
GAMMA_SPACE = "linspace"                 # "linspace" ou "geomspace"
WEIGHT_DISTRIB = True
WEIGHT_DISTRIB_ALPHA = 100
FAST_SAMPLING = True

# ----- Gaussian final (non utilisé en mode transfer, mais requis) -----
FINAL_ADAPTIVE = False
ADAPTIVE_MEAN = False
MEAN_FINAL = "torch.zeros([3, 256, 256])"
VAR_FINAL = "torch.ones([3, 256, 256])"

# ----- Logging -----
LOGGER = "CSV"
CSV_LOG_DIR = "./"
LOG_STRIDE = 10
GIF_STRIDE = 5000
PLOT_NPAR = 16
PLOT_LEVEL = 1

# ----- Checkpoint -----
CHECKPOINT_RUN = False
CHECKPOINT_IT = 1
CHECKPOINT_PASS = "b"
SAMPLE_CHECKPOINT_F = ""
SAMPLE_CHECKPOINT_B = ""
