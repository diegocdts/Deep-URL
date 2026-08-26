import matplotlib.pyplot as plt

def plot_images(blur_image, x_hat, cmap, results_dir, name, ground_truth=None, epoch=None):
    fig = plt.figure(figsize=(15, 5))

    blur_image = blur_image[0,0] if blur_image.ndim == 4 else blur_image
    x_hat = x_hat[0,0] if x_hat.ndim == 4 else x_hat

    epoch = f'\nÉpoca: {epoch}' if epoch is not None else ''

    if ground_truth is not None:
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        ground_truth = ground_truth[0,0] if ground_truth.ndim == 4 else ground_truth
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.imshow(blur_image, cmap=cmap, aspect='auto')
    ax1.set_title('Imagem de entrada (y)')
    ax1.axis('off')

    ax2.imshow(x_hat, cmap=cmap, aspect='auto')
    ax2.set_title(f'Imagem recuperada (x_hat){epoch}')
    ax2.axis('off')

    if ground_truth is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(ground_truth, cmap=cmap, aspect='auto')
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

def create_patches(data, n_patches, height, width, og_height, og_width):
    image = data[0,:og_height,:og_width]
    patches = image.reshape(n_patches[0], height, n_patches[1], width)
    patches = patches.transpose(0, 2, 1, 3)
    patches = patches.reshape(-1, height, width)
    return patches

def reorganize_patches(data, n_patches, height, width, og_height, og_width):
    data = data.reshape(n_patches[0], n_patches[1], height, width)
    img = data.transpose(0, 2, 1, 3).reshape(og_height, og_width)
    return img
