import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional.image import structural_similarity_index_measure


class DeepURLLayer(nn.Module):
    """
    Uma camada do Deep-URL.

    Cada camada representa uma iteração do algoritmo
    Richardson-Lucy, mas com parâmetros aprendíveis.

    Entradas:
        y       : imagem borrada
        x       : estimativa atual da imagem limpa
        H       : estimativa atual do kernel

    Saídas:
        x_next  : nova estimativa da imagem
        H_next  : nova estimativa do kernel
    """

    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

        # Pesos aprendíveis associados à atualização de x
        self.W_x = nn.Parameter(
            torch.randn(1, 1, kernel_size, kernel_size) * 0.01
        )

        # Pesos aprendíveis associados à atualização de H
        self.W_H = nn.Parameter(
            torch.randn(1, 1, kernel_size, kernel_size) * 0.01
        )

        # Parâmetros escalares aprendíveis
        self.alpha_x = nn.Parameter(torch.tensor(1.0))
        self.alpha_H = nn.Parameter(torch.tensor(1.0))

    def forward(self, y, x, H):

        eps = 1e-8

        # ---------------------------------------------------------
        # 1. Atualização do kernel H
        # ---------------------------------------------------------

        # Convolução da imagem atual com o kernel atual
        y_hat = F.conv2d(
            x,
            H,
            padding=self.kernel_size // 2
        )

        # Residual entre observação e reconstrução
        residual = y / (y_hat + eps)

        # Operação parametrizada
        correction_H = F.conv2d(
            residual,
            self.W_H,
            padding=self.kernel_size // 2
        )

        # Atualização do kernel
        H_next = H * (
            1.0 + self.alpha_H * correction_H
        )

        # ---------------------------------------------------------
        # 2. Garantir H >= 0
        # ---------------------------------------------------------

        H_next = F.relu(H_next)

        # Normalizar kernel para que sua soma seja 1
        H_next = H_next / (
            H_next.sum(dim=(-1, -2), keepdim=True) + eps
        )

        # ---------------------------------------------------------
        # 3. Atualização da imagem x
        # ---------------------------------------------------------

        y_hat = F.conv2d(
            x,
            H_next,
            padding=self.kernel_size // 2
        )

        residual = y / (y_hat + eps)

        correction_x = F.conv2d(
            residual,
            self.W_x,
            padding=self.kernel_size // 2
        )

        x_next = x * (
            1.0 + self.alpha_x * correction_x
        )

        # ---------------------------------------------------------
        # 4. Garantir x >= 0
        # ---------------------------------------------------------

        x_next = F.relu(x_next)

        # Sigmoid para manter os valores em [0, 1]
        x_next = torch.sigmoid(x_next)

        return x_next, H_next


class DeepURL(nn.Module):
    """
    Deep Unfolded Richardson-Lucy.

    L layers = L unfolded Richardson-Lucy iterations.
    """

    def __init__(self, kernel_size=15, num_layers=5):
        super().__init__()

        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            DeepURLLayer(kernel_size)
            for _ in range(num_layers)
        ])

    def forward(self, y, x0=None, H0=None):

        batch_size, _, height, width = y.shape

        # ---------------------------------------------------------
        # Inicialização
        # ---------------------------------------------------------

        if x0 is None:
            x = torch.rand_like(y)

        else:
            x = x0.clone()

        if H0 is None:

            H = torch.rand(
                batch_size,
                1,
                self.kernel_size,
                self.kernel_size,
                device=y.device
            )

            H = H / H.sum(
                dim=(-1, -2),
                keepdim=True
            )

        else:
            H = H0.clone()

        # ---------------------------------------------------------
        # Deep unfolding
        # ---------------------------------------------------------

        for layer in self.layers:

            x, H = layer(
                y=y,
                x=x,
                H=H
            )

        return x, H


def total_variation(x):

    tv_h = torch.mean(
        torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    )

    tv_w = torch.mean(
        torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    )

    return tv_h + tv_w


def deep_url_loss(y, x, H, lambda_tv=0.1):

    # Reconstrução da imagem observada
    kernel_size = H.shape[-1]

    y_hat = F.conv2d(
        x,
        H,
        padding=kernel_size // 2
    )

    # -SSIM
    ssim_loss = -structural_similarity_index_measure(
        y_hat,
        y,
        data_range=1.0
    )

    # Regularização TV
    tv_loss = total_variation(x)

    loss = ssim_loss + lambda_tv * tv_loss

    return loss, y_hat, ssim_loss, tv_loss