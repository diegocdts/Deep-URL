import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from deep_url import DeepURL

# ============================================================
# 9. CONFIGURAÇÕES
# ============================================================

NPY_PATH = "/home/data/IMG.npy"
KERNEL_SIZE = 15
NUM_LAYERS = 5
EPOCHS = 500
LR = 0.1
LAMBDA_TV = 0.1
MODEL_PATH = f'/home/src/model/deep_url_{KERNEL_SIZE}_{NUM_LAYERS}_{EPOCHS}_{LR}_{LAMBDA_TV}.pth'
RESULTS_DIR = f'/home/src/results/deep_url_{KERNEL_SIZE}_{NUM_LAYERS}_{EPOCHS}_{LR}_{LAMBDA_TV}'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_X_PATH = f'{RESULTS_DIR}/x_hat.npy'
OUTPUT_H_PATH = f'{RESULTS_DIR}/H_hat.npy'
OUTPUT_FIG_PATH = f'{RESULTS_DIR}/output.png'


# ============================================================
# 10. CARREGAR OS DADOS
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
# 11. CARREGAR O CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

print("\nInformações do modelo:")
print("Kernel size:", checkpoint["kernel_size"])
print("Número de layers:", checkpoint["num_layers"])


# ============================================================
# 12. RECRIAR A ARQUITETURA
# ============================================================

model = DeepURL(num_layers=NUM_LAYERS, 
                kernel_size=KERNEL_SIZE, 
                image_H=y.shape[-2], 
                image_W=y.shape[-1]).to(DEVICE)


# ============================================================
# 13. CARREGAR OS PESOS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 14. TESTE / INFERÊNCIA
# ============================================================

with torch.no_grad():

    x_hat, H_hat = model(y)


# ============================================================
# 15. CONVERTER RESULTADOS PARA NUMPY
# ============================================================

x_hat = x_hat.squeeze().cpu().numpy()
H_hat = H_hat.squeeze().cpu().numpy()


print("\nResultados:")
print("x_hat:", x_hat.shape)
print("H_hat:", H_hat.shape)


# ============================================================
# 16. SALVAR RESULTADOS
# ============================================================

np.save(
    OUTPUT_X_PATH,
    x_hat
)

np.save(
    OUTPUT_H_PATH,
    H_hat
)

print("\nResultados salvos:")
print(OUTPUT_X_PATH)
print(OUTPUT_H_PATH)


# ============================================================
# 17. VISUALIZAÇÃO
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

plt.imshow(
    y.squeeze().cpu().numpy(),
    cmap="seismic"
)

plt.title("Imagem de entrada (y)")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    x_hat,
    cmap="seismic"
)

plt.title("Imagem recuperada (x_hat)")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    H_hat,
    cmap="seismic"
)

plt.title("Kernel estimado (H_hat)")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_FIG_PATH, bbox_inches='tight')