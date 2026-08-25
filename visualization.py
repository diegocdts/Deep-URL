import matplotlib.pyplot as plt

def plot_images(blur_image, x_hat, cmap, results_dir, name, ground_truth=None, epoch=None):
    fig = plt.figure(figsize=(15, 5))

    epoch = f'\nÉpoca: {epoch}' if epoch is not None else ''

    if ground_truth is not None:
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.imshow(blur_image[0, 0], cmap=cmap, aspect='auto')
    ax1.set_title('Imagem de entrada (y)')
    ax1.axis('off')

    ax2.imshow(x_hat[0, 0], cmap=cmap, aspect='auto')
    ax2.set_title(f'Imagem recuperada (x_hat){epoch}')
    ax2.axis('off')

    if ground_truth is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(ground_truth[0,0], cmap=cmap, aspect='auto')
        ax3.set_title('Ground Truth (x)')
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig(
        f'{results_dir}/input_output_{cmap}_{name}.png',
        bbox_inches='tight'
    )
    plt.close()


def plot_lower_loss(blur_image, x_hat, results_dir, ground_truth=None, epoch=None):
    plot_images(blur_image=blur_image, x_hat=x_hat, ground_truth=ground_truth, cmap='gray', results_dir=results_dir, name='lower_loss', epoch=epoch)
    plot_images(blur_image=blur_image, x_hat=x_hat, ground_truth=ground_truth, cmap='seismic', results_dir=results_dir, name='lower_loss', epoch=epoch)