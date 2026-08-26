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
from deep_url import DeepURL, deep_url_loss
from visualization import plot_lower_loss


# ---------------------------------------------------------------------------
# 5. TREINAMENTO AUTO-SUPERVISIONADO -- Algoritmo 1 do artigo
# ---------------------------------------------------------------------------
 
def train_deep_url(y: torch.Tensor, x: torch.Tensor = None, kernel_size: int = 15, num_layers: int = 5,
                    epochs: int = 500, lr: float = 0.1, lambda_tv: float = 0.1,
                    decay_factor: float = 0.1, device: str = "cpu",
                    verbose: bool = True, model_path: str = None, results_dir = None, is_supervised: bool = False):
    """
    Reproduz o Algoritmo 1 (DEEP-URL) do artigo.
 
    Entradas
    --------
    y : (B,1,H,W) imagem(ns) borrada(s) -- única informação de supervisão.
    num_layers : número de camadas (= número de iterações do RL "desenrolado").
    epochs : número de épocas de treinamento.
 
    Saída
    -----
    x_star, H_star : imagem nítida e kernel estimados (linha 10 do Alg. 1).
    model          : rede treinada (pesos podem ser reutilizados para
                     deborrar outras imagens com o MESMO kernel, como
                     observado no artigo, Seção 3, último parágrafo).
    """
    y = y.to(device)
    if x is not None:
        x = x.to(device)
    B, C, H_img, W_img = y.shape
    assert C == 1, "Esta implementação assume imagens em escala de cinza."
    assert model_path is not None, "O path para o modelo precisa ser definido"
    assert results_dir is not None, "O diretório de resultados precisa ser definido"
 
    model = DeepURL(num_layers, kernel_size, H_img, W_img).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    milestones = [int(0.4 * epochs), int(0.6 * epochs)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=decay_factor
    )
 
    # Linha "Initialize": H^0 ~ U(0,1), x^0 ~ U(0,1)
    H0 = torch.rand(1, 1, kernel_size, kernel_size, device=device)
    x0 = torch.rand(B, 1, H_img, W_img, device=device)

    lower_loss = None
 
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_L, H_L = model(y, x0, H0)                                     # laço interno (linhas 2-5)
        loss, _ssim, x_tv, _ = deep_url_loss(y=y, x=x, x_L=x_L, H_L=H_L, lambda_tv=lambda_tv, is_supervised=is_supervised)    # linha 6: gradiente de Eq. (6)
        loss.backward()
        optimizer.step()                                                # linha 7: atualiza Upsilon
        scheduler.step()
 
        # Linha 8: realimenta o estado para a próxima época
        x0 = x_L.detach()
        H0 = H_L.detach()
 
        if verbose:
            print(f"  época {epoch:4d}/{epochs} - perda: {loss.item():.4f} - _ssim: {_ssim:.4f} - x_tv: {x_tv:.4f}")

        if lower_loss is None or loss < lower_loss:
            lower_loss = loss
            plot_lower_loss(blur_image=y.detach().cpu().numpy(), x_hat=x0.cpu().numpy(), ground_truth=x.detach().cpu().numpy(), results_dir=results_dir, epoch=epoch)          

    
    # Salvar modelo
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kernel_size": kernel_size,
            "num_layers": num_layers,
        },
        model_path
    )
 
    return x0, H0, model  # linha 10: x*, H*