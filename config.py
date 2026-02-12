# --- Architecture & Data ---
IMAGE_SIZE = 256
NUM_CHANNELS = 128          # Augmenté (vs 64) pour plus de détails
NUM_RES_BLOCKS = 2
ATTENTION_RESOLUTIONS = "16" # Ajout de 32 pour la cohérence globale
DROPOUT = 0.0               # OK, ou 0.1 si vous avez peu de données (<1000 paires)

# --- Training ---
BATCH_SIZE = 48             # Compromis sûr avec 128 channels. Tentez 64 si ça passe.
LR = 1e-4
NUM_ITER = 5000             # Suffisant par IPF step, permet de cycler plus vite
N_IPF = 15                  # Très bien
GRAD_CLIP = 1.0

# --- Cache & Langevin (Schrödinger Bridge) ---
# C'est ici que la VRAM aide le plus :
CACHE_NPAR = 64             # Génération plus rapide (parallélisme accru)
NUM_CACHE_BATCHES = 40      # 40 * 64 = 2560 images en cache (beaucoup plus stable)
CACHE_REFRESH_STRIDE = 400  # On rafraîchit moins souvent car le cache est plus gros

# --- Diffusion Physics ---
NUM_STEPS = 20              # Bon compromis qualité/vitesse
GAMMA_MAX = 0.1             # OK pour I2SB
GAMMA_MIN = 1e-5
GAMMA_SPACE = "linspace"
EMA_RATE = 0.999