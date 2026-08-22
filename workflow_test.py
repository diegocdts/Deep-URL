import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from deep_url import DeepURL

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

NPY_PATH = "/home/data/IMG.npy"
MODEL_PATH = "deep_url.pth"

OUTPUT_X_PATH = "x_hat.npy"
OUTPUT_H_PATH = "H_hat.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. CARREGAR OS DADOS
# ============================================================

y = np.load(NPY_PATH)

print("Shape original:", y.shape)
print("Dtype:", y.dtype)


# ============================================================
# 3. NORMALIZAÇÃO
# ============================================================

y_min = y.min()
y_max = y.max()

y = (y - y_min) / (y_max - y_min + 1e-8)

y = y.astype(np.float32)


# ============================================================
# 4. CONVERTER PARA TENSOR
# ============================================================

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


y = y.to(DEVICE)

print("Shape usado pelo modelo:", y.shape)


# ============================================================
# 5. CARREGAR O CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

print("\nInformações do modelo:")
print("Kernel size:", checkpoint["kernel_size"])
print("Número de layers:", checkpoint["num_layers"])


# ============================================================
# 6. RECRIAR A ARQUITETURA
# ============================================================

model = DeepURL(
    kernel_size=checkpoint["kernel_size"],
    num_layers=checkpoint["num_layers"]
).to(DEVICE)


# ============================================================
# 7. CARREGAR OS PESOS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 8. TESTE / INFERÊNCIA
# ============================================================

with torch.no_grad():

    x_hat, H_hat = model(y)


# ============================================================
# 9. CONVERTER RESULTADOS PARA NUMPY
# ============================================================

x_hat = x_hat.squeeze().cpu().numpy()
H_hat = H_hat.squeeze().cpu().numpy()


print("\nResultados:")
print("x_hat:", x_hat.shape)
print("H_hat:", H_hat.shape)


# ============================================================
# 10. SALVAR RESULTADOS
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
# 11. VISUALIZAÇÃO
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

plt.imshow(
    y.squeeze().cpu().numpy(),
    cmap="gray"
)

plt.title("Imagem de entrada (y)")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    x_hat,
    cmap="gray"
)

plt.title("Imagem recuperada (x_hat)")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    H_hat,
    cmap="gray"
)

plt.title("Kernel estimado (H_hat)")
plt.axis("off")


plt.tight_layout()
plt.show()