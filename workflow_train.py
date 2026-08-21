import os
import numpy as np
import torch
from training import train_deep_url

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

NPY_PATH = "images/IN.npy"
KERNEL_SIZE = 3
NUM_LAYERS = 5
EPOCHS = 2
LR = 0.1
LAMBDA_TV = 0.1

MODEL_PATH = f'model/deep_url_{KERNEL_SIZE}_{NUM_LAYERS}_{EPOCHS}_{LR}_{LAMBDA_TV}.pth'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_PATH, exist_ok=True)

# ============================================================
# 2. CARREGA IMAGEM BLURRED
# ============================================================

blurred_image = np.load(NPY_PATH)

y = blurred_image.astype("float32")

y = (y - y.min()) / (
    y.max() - y.min() + 1e-8
)

y = torch.from_numpy(y)
y = y.reshape(50, 352, 1400)

# ------------------------------------------------------------
# O modelo espera:
#
# [batch, channel, height, width]
#
# Caso o .npy tenha:
#
# [H, W]
#
# transformamos em:
#
# [1, 1, H, W]
# ------------------------------------------------------------

if y.ndim == 2:

    y = y.unsqueeze(0).unsqueeze(0)


# Caso já esteja no formato [B, H, W]
elif y.ndim == 3:

    y = y.unsqueeze(1)


else:
    raise ValueError(
        f"Formato não suportado: {y.shape}"
    )
    
print(y.shape)

# ============================================================
# 3. TREINA MODELO
# ============================================================


model, x_hat, H_hat = train_deep_url(
    y,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS,
    epochs=EPOCHS,
    lr=LR,
    lambda_tv=LAMBDA_TV
)

print("Imagem recuperada:", x_hat.shape)
print("Kernel estimado:", H_hat.shape)