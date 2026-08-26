import os
import numpy as np
import torch
from training import train_deep_url
from visualization import plot_images, create_patches, reorganize_patches

torch.manual_seed(42)

# ============================================================
# 6. CONFIGURAÇÕES
# ============================================================

Y_PATH = "/home/data/IN.npy"
X_PATH = "/home/data/RFLT.npy"
SUP_AUTOSUP = 'supervised' if X_PATH is not None else 'self-supervised'
KERNEL_SIZE = 3
NUM_LAYERS = 5
EPOCHS = 5000
LR = 0.1
LAMBDA_TV = 0.1
RESULTS_DIR = f'/home/src/results/deep-url_{KERNEL_SIZE}_{NUM_LAYERS}_{EPOCHS}_{LR}_{LAMBDA_TV}_{SUP_AUTOSUP}'
MODEL_PATH = f'{RESULTS_DIR}/model.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f'KERNEL_SIZE: {KERNEL_SIZE} - NUM_LAYERS: {NUM_LAYERS} - EPOCHS: {EPOCHS} - LR: {LR} - LAMBDA_TV: {LAMBDA_TV}')

N_PATCHES = (1,1)
OG_HEIGHT = 352
OG_WIDTH = 1400
HEIGHT = OG_HEIGHT // N_PATCHES[0]
WIDTH = OG_WIDTH // N_PATCHES[1]

# ============================================================
# 7. CARREGA IMAGEM BLURRED E GROUND TRUTH (SE HOUVER)
# ============================================================

def minmax_normalize(data):
    data_min, data_max = data.min(), data.max()
    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm

def load_data(data_path, batch, height, width):
    data = np.load(data_path).astype("float32")
    data = create_patches(data, n_patches=N_PATCHES, height=HEIGHT, width=WIDTH, og_height=OG_HEIGHT, og_width=OG_WIDTH)
    data = torch.from_numpy(data)
    data = minmax_normalize(data)
    data = data.reshape(batch, height, width)

    if data.ndim == 3:
        data = data.unsqueeze(1)
    elif data.ndim == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Formato não suportado: {data.shape}")
    return data

y = load_data(data_path=Y_PATH, batch=N_PATCHES[0]*N_PATCHES[1], height=HEIGHT, width=WIDTH)
print(f'Imagem blurred: {y.shape}', f'min: {y.min()} - max: {y.max()}')

if X_PATH is not None:
    x = load_data(data_path=X_PATH, batch=N_PATCHES[0]*N_PATCHES[1], height=HEIGHT, width=WIDTH)
    print(f'Imagem limpa:   {x.shape}', f'min: {x.min()} - max: {x.max()}')
    print('TREINAMENTO SUPERVISIONADO')
else:
    x = None
    print('TREINAMENTO AUTO-SUPERVISIONADO')


# ============================================================
# 8. TREINA MODELO
# ============================================================


x_hat, H_hat, model = train_deep_url(
    y, x,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS,
    epochs=EPOCHS,
    lr=LR,
    lambda_tv=LAMBDA_TV,
    device=DEVICE,
    model_path=MODEL_PATH,
    results_dir=RESULTS_DIR
)

y_np = y.detach().cpu().numpy()
x_np = x.detach().cpu().numpy()
x_hat_np = x_hat.detach().cpu().numpy()
H_hat_np = H_hat.detach().cpu().numpy()

np.save(f'{RESULTS_DIR}/x_hat_np.npy', x_hat_np)
np.save(f'{RESULTS_DIR}/H_hat_np.npy', H_hat_np)

print("Imagem recuperada:", x_hat.shape)
print("Kernel estimado:", H_hat.shape)
print("Modelo salvo:", MODEL_PATH)

y_np = reorganize_patches(data=y_np, n_patches=N_PATCHES, height=HEIGHT, width=WIDTH, og_height=OG_HEIGHT, og_width=OG_WIDTH)
x_hat_np = reorganize_patches(data=x_hat_np, n_patches=N_PATCHES, height=HEIGHT, width=WIDTH, og_height=OG_HEIGHT, og_width=OG_WIDTH)
x_np = reorganize_patches(data=x_np, n_patches=N_PATCHES, height=HEIGHT, width=WIDTH, og_height=OG_HEIGHT, og_width=OG_WIDTH)

plot_images(blur_image=y_np, x_hat=x_hat_np, ground_truth=x_np, cmap='seismic', results_dir=RESULTS_DIR, name='final')
plot_images(blur_image=y_np, x_hat=x_hat_np, ground_truth=x_np, cmap='gray', results_dir=RESULTS_DIR, name='final')