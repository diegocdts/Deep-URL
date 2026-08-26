"""
Deep-URL: Deep Unfolded Richardson-Lucy Network
================================================
Reimplementação em PyTorch da proposta de:
Agarwal, Khobahi, Bose, Soltanalian, Schonfeld,
"Deep-URL: A Model-Aware Approach to Blind Deconvolution Based on
Deep Unfolded Richardson-Lucy Network", arXiv:2002.01053, 2020.
 
IDEIA CENTRAL DO ARTIGO
------------------------
O problema de "blind deconvolution" consiste em recuperar, a partir de uma
única imagem borrada y, tanto a imagem nítida x quanto o kernel de
borramento (PSF) H, sabendo que:
 
    y = H (*) x + n                                            Eq. (1)
 
O artigo parte do algoritmo clássico de Richardson-Lucy (RL), que resolve
 
    min_{x,H} ||y - H (*) x||^2 + lambda * TV(x)                Eq. (3)
 
por meio de atualizações multiplicativas (Eq. 4a-4b), e faz um "deep
unfolding": cada iteração do RL vira uma camada de uma rede neural, na
qual os termos fixos do algoritmo clássico (a estimativa atual de H e de
x usada nos denominadores) são substituídos por PESOS TREINÁVEIS W_H^k e
W_x^k (Eq. 5a-5b). Isso torna o método "model-aware": a arquitetura da
rede é derivada diretamente do algoritmo de otimização, e não escolhida
empiricamente -- por isso é interpretável.
 
O treinamento é AUTO-SUPERVISIONADO (zero-shot): a única informação usada
é a própria imagem borrada y. A função de perda é a SSIM negativa entre y
e a reconstrução x^L (*) H^L, mais um termo de regularização TV sobre a
imagem recuperada (Eq. 6), otimizada seguindo o Algoritmo 1 do artigo.
 
SIMPLIFICAÇÕES ASSUMIDAS NESTA IMPLEMENTAÇÃO (documentadas para
transparência, já que o artigo não fornece código-fonte):
  1. Imagens em escala de cinza (1 canal), como no experimento com MNIST.
  2. Kernel de tamanho ímpar (k), para permitir padding 'same' simétrico.
  3. Seguindo a Seção 4 ("batch-wise optimization ... using the same
     kernel"), o kernel H é ÚNICO E COMPARTILHADO para todo o lote (mesmo
     kernel borra todas as imagens do batch), enquanto a imagem x é
     estimada individualmente por amostra. A atualização do kernel
     (Eq. 5a) é, portanto, calculada por amostra e depois agregada
     (média) no lote -- ver `kernel_update_correlation`.
  4. lambda (peso do TV) e a taxa de aprendizado seguem os valores do
     artigo (0.1 e 0.1, respectivamente), com decaimento em 40% e 60%
     das épocas, otimizador RMSprop.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
 
# ---------------------------------------------------------------------------
# 1. OPERAÇÕES BÁSICAS: convolução, flip e a "correlação restrita ao kernel"
# ---------------------------------------------------------------------------

def flip2d(t: torch.Tensor) -> torch.Tensor:
    """Retorna a versão espacialmente invertida (rotação de 180 graus) de um
    tensor (·)†, conforme definido logo após a Eq. (4b) do artigo."""
    return torch.flip(t, dims=[-2, -1])


def conv_same(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolução/correlação 'same': aplica um kernel PEQUENO e
    COMPARTILHADO (1,1,k,k) sobre um lote de imagens GRANDES (B,1,H,W),
    preservando o tamanho espacial da imagem.
 
    Usada para os termos x (*) H (ex.: Eq. 5a denominador, Eq. 5b
    denominador, e a reconstrução final x^L (*) H^L).
    Assume canal único e kernel de lado ímpar (ver docstring do módulo).
    """
    k = kernel.shape[-1]
    assert k % 2 == 1, "Esta implementação assume kernel de tamanho ímpar."
    pad = k // 2
    return F.conv2d(image, kernel, padding=pad)


def kernel_update_correlation(ratio: torch.Tensor, x_k: torch.Tensor,
                               kernel_size: int) -> torch.Tensor:
    """Calcula o termo `(y / ReLU(x^k (*) W_H^k)) (*) x^{k}dagger` da
    Eq. (5a), cujo resultado deve ter o TAMANHO DO KERNEL (k x k), e não o
    tamanho da imagem.
 
    Isso é o análogo, em deep-unfolding, da etapa clássica de correlação
    do algoritmo RL para atualizar o PSF: correlaciona-se o resíduo
    (`ratio`, tamanho da imagem) com a estimativa atual da imagem (`x_k`,
    também tamanho da imagem), mas mantendo-se apenas os deslocamentos
    (lags) compatíveis com o suporte espacial do kernel.
 
    Implementação: fazemos padding de `ratio` com (k-1) pixels em cada
    borda e em seguida uma correlação cruzada 'valid' com `x_k` (usado
    como filtro). O resultado tem tamanho (2k-1, 2k-1); extraímos a janela
    central k x k. Como cada amostra do lote tem sua própria imagem x_k,
    mas o kernel H é ÚNICO para o lote (ver docstring do módulo), a
    correlação é feita POR AMOSTRA (via convolução agrupada, groups=B) e
    depois agregada pela média no lote.
    """
    B, C, H, W = ratio.shape
    assert C == 1, "Esta implementação assume imagens em escala de cinza."
    pad = kernel_size - 1
 
    ratio_padded = F.pad(ratio, [pad, pad, pad, pad])          # (B,1,H+2p,W+2p)
    x_flipped = flip2d(x_k)                                    # x^{k}†
 
    # Truque de convolução "por amostra": cada elemento do lote usa seu
    # próprio x_flipped como filtro. Reorganizamos o batch como grupos.
    inp = ratio_padded.reshape(1, B * C, H + 2 * pad, W + 2 * pad)
    weight = x_flipped.reshape(B * C, 1, H, W)
    out = F.conv2d(inp, weight, groups=B * C)                  # (1,B*C,2k-1,2k-1)
    out = out.reshape(B, C, 2 * kernel_size - 1, 2 * kernel_size - 1)
 
    # Recorte da janela central k x k (deslocamentos compatíveis com o
    # suporte do kernel), validado numericamente contra correlação
    # completa antes de escrever este código.
    start = (2 * kernel_size - 1 - kernel_size) // 2
    cropped = out[:, :, start:start + kernel_size, start:start + kernel_size]
 
    kernel_update = cropped.mean(dim=0, keepdim=True)          # agrega o lote -> (1,1,k,k)
    return kernel_update
 
 
# ---------------------------------------------------------------------------
# 2. FUNÇÃO DE PERDA: SSIM negativa + regularização TV, Eq. (6)
# ---------------------------------------------------------------------------
 
def ssim4deep_url(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return ssim(img1.detach().cpu().numpy().squeeze(),img2.detach().cpu().numpy().squeeze(),data_range=data_range)
 
 
def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Regularização de variação total (TV) anisotrópica: TV(x) na Eq. (3)/(6)."""
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw
 
 
def deep_url_loss(y: torch.Tensor, x_L: torch.Tensor, H_L: torch.Tensor,
                   lambda_tv: float = 0.1, x: torch.Tensor = None, is_supervised: bool = False):
    """L(x^L (*) H^L, y) + lambda * TV(x^L), Eq. (6).
    L(.) é a SSIM negativa entre a imagem borrada real y e a reconstrução
    y_hat = x^L (*) H^L."""
    y_hat = conv_same(x_L, H_L)
    _ssim = ssim4deep_url(x_L, x) if is_supervised and x is not None else ssim4deep_url(y_hat, y)
    x_tv = tv_loss(x_L)
    loss = (1.0 - _ssim) + lambda_tv * x_tv
    return loss, _ssim, x_tv, y_hat


# ---------------------------------------------------------------------------
# 3. UMA CAMADA DO DEEP-URL = UMA ITERAÇÃO DO RICHARDSON-LUCY, Eq. (5a)-(5b)
# ---------------------------------------------------------------------------
 
class DeepURLLayer(nn.Module):
    """Uma camada k do Deep-URL. Contém os parâmetros treináveis
    Upsilon^k = {W_x^k, W_H^k} (um por camada, pesos NÃO compartilhados
    entre camadas, como em redes de deep unfolding padrão)."""
 
    def __init__(self, kernel_size: int, image_H: int, image_W: int, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        # W_H^k: mesmo formato do kernel H (1,1,k,k)
        self.W_H = nn.Parameter(torch.rand(1, 1, kernel_size, kernel_size))
        # W_x^k: mesmo formato da imagem x (1,1,image_H,image_W)
        self.W_x = nn.Parameter(torch.rand(1, 1, image_H, image_W))     

    def forward(self, y: torch.Tensor, x_k: torch.Tensor, H_k: torch.Tensor):
        # ---------------- Atualização do kernel, Eq. (5a) ----------------
        #   H^{k+1} = sigma( ReLU( (y / ReLU(x^k (*) W_H^k)) (*) x^{k}† ) . W_H^k )
        blur_est_H = F.relu(conv_same(x_k, self.W_H))
        ratio_H = y / (blur_est_H + self.eps)
        corr_H = kernel_update_correlation(ratio_H, x_k, self.kernel_size)
        H_next = torch.sigmoid(F.relu(corr_H) * self.W_H)
 
        # ---------------- Atualização da imagem, Eq. (5b) -----------------
        #   x^{k+1} = sigma( ReLU( (y / ReLU(W_x^k (*) H^{k+1})) (*) H^{k+1}† ) . W_x^k )
        blur_est_x = F.relu(conv_same(self.W_x, H_next))
        ratio_x = y / (blur_est_x + self.eps)
        corr_x = conv_same(ratio_x, flip2d(H_next))
        x_next = torch.sigmoid(F.relu(corr_x) * self.W_x)
 
        return x_next, H_next


# ---------------------------------------------------------------------------
# 4. REDE COMPLETA: empilha L camadas (Fig. 1 / Algoritmo 1, laço interno)
# ---------------------------------------------------------------------------
 
class DeepURL(nn.Module):
    def __init__(self, num_layers: int, kernel_size: int, image_H: int, image_W: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DeepURLLayer(kernel_size, image_H, image_W) for _ in range(num_layers)]
        )
 
    def forward(self, y: torch.Tensor, x0: torch.Tensor, H0: torch.Tensor):
        x, H = x0, H0
        for layer in self.layers:
            x, H = layer(y, x, H)
        return x, H  