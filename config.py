# ----- Dataset -----
DATASET = "hes_cd30"
DATASET_TRANSFER = "hes_cd30"
IMAGE_SIZE = 256
CHANNELS = 3
RANDOM_FLIP = True
DATA_DIR = "./"                          # dataset_v2/ doit être dans ce dossier
TRANSFER = True                          # mode transfer : HES -> CD30
LOAD = False

# ----- Modèle (UNET) -----
MODEL = "UNET"
NUM_CHANNELS = 32
NUM_RES_BLOCKS = 2
NUM_HEADS = 4
NUM_HEADS_UPSAMPLE = -1
ATTENTION_RESOLUTIONS = "16"
DROPOUT = 0.0
USE_CHECKPOINT = False
USE_SCALE_SHIFT_NORM = True

# ----- Device -----
DEVICE = "cuda"                          # "cpu" si pas de GPU
DATAPARALLEL = True
NUM_WORKERS = 1
PIN_MEMORY = True

# ----- Entraînement -----
BATCH_SIZE = 4
LR = 1e-4
NUM_ITER = 50000
N_IPF = 20
N_IPF_INIT = 1
CACHE_NPAR = 4
NUM_CACHE_BATCHES = 1
CACHE_REFRESH_STRIDE = 100
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
