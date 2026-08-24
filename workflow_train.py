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
EPOCHS = 500
LR = 0.1
LAMBDA_TV = 0.1
MODEL_DIR = '/home/src/model'
RESULTS_DIR = f'/home/src/results/deep_url_{KERNEL_SIZE}_{NUM_LAYERS}_{EPOCHS}_{LR}_{LAMBDA_TV}'
MODEL_PATH = f'/home/src/model/deep_url_{KERNEL_SIZE}_{NUM_LAYERS}_{EPOCHS}_{LR}_{LAMBDA_TV}.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 7. CARREGA IMAGEM BLURRED
# ============================================================

y = np.load(NPY_PATH).astype("float32")
y = (y - y.mean()) / (3 * y.std()) # aplica normalização zScore com n_std = 3
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
    device=DEVICE,
    model_path=MODEL_PATH
)

x0_np = x_hat.cpu().numpy
H0_np = H_hat.cpu().numpy 

np.save(f'{RESULTS_DIR}/x0_np.npy', x0_np)
np.save(f'{RESULTS_DIR}/H0_np.npy', H0_np)

print("Imagem recuperada:", x_hat.shape)
print("Kernel estimado:", H_hat.shape)
print("Modelo salvo:", MODEL_PATH)