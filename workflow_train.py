import os
import numpy as np
import torch
from training import train_deep_url


# ============================================================
# 6. CONFIGURAÇÕES
# ============================================================

NPY_PATH = "/home/data/IN.npy"
KERNEL_SIZE = 15
NUM_LAYERS = 5
EPOCHS = 5000
LR = 0.1
LAMBDA_TV = 0.1
MODEL_DIR = '/home/src/model'
MODEL_PATH = f'/home/src/model/deep_url_{KERNEL_SIZE}_{NUM_LAYERS}_{EPOCHS}_{LR}_{LAMBDA_TV}.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 7. CARREGA IMAGEM BLURRED
# ============================================================

y = np.load(NPY_PATH).astype("float64")
y = y.mean() / (3 * y.std()) # aplica normalização zScore com n_std = 3
y = torch.from_numpy(y)
y = y.reshape(50, 352, 1400)

if y.ndim == 3 and y.shape[0] > 3:
    y = y.unsqueeze(1)
elif y.ndim == 2:
    y = y.unsqueeze(0).unsqueeze(0)
else:
    raise ValueError(f"Formato não suportado: {y.shape}")
print(f'Shape do dado blurred: {y.shape}')


# ============================================================
# 8. TREINA MODELO
# ============================================================


x_hat, H_hat, model = train_deep_url(
    y,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS,
    epochs=EPOCHS,
    lr=LR,
    lambda_tv=LAMBDA_TV,
    model_path=MODEL_PATH
)

print("Imagem recuperada:", x_hat.shape)
print("Kernel estimado:", H_hat.shape)
print("Modelo salvo:", MODEL_PATH)