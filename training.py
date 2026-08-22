import torch
from deep_url import DeepURL, deep_url_loss

def train_deep_url(
    y,
    kernel_size=15,
    num_layers=5,
    epochs=5000,
    lr=0.1,
    lambda_tv=0.1,
    model_path='model/deep_url.pth',
    device="cuda"
):

    device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )

    y = y.to(device)

    model = DeepURL(
        kernel_size=kernel_size,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=lr
    )

    # ---------------------------------------------------------
    # Learning-rate scheduler
    #
    # O paper reduz o LR em 40% e 60% das epochs.
    # ---------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(0.4 * epochs),
            int(0.6 * epochs)
        ],
        gamma=0.1
    )

    for epoch in range(epochs):

        optimizer.zero_grad()

        # Forward
        x_hat, H_hat = model(y)

        # Loss
        loss, y_hat, ssim_loss, tv_loss = deep_url_loss(
            y,
            x_hat,
            H_hat,
            lambda_tv=lambda_tv
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        scheduler.step()

        if epoch % 10 == 0:

            print(
                f"Epoch [{epoch:4d}/{epochs}] "
                f"Loss: {loss.item():.6f} "
                f"SSIM loss: {ssim_loss.item():.6f} "
                f"TV: {tv_loss.item():.6f}"
            )

    # Resultado final
    with torch.no_grad():
        x_hat, H_hat = model(y)

    # Salvar modelo
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kernel_size": kernel_size,
            "num_layers": num_layers,
        },
        model_path
    )

    return model, x_hat, H_hat