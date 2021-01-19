import math

import matplotlib.pyplot as plt


def plot_average_per_target(ds_per_target):

    vmin, vmax = 0, 255
    x_axis, y_axis = math.ceil(math.sqrt(len(ds_per_target))), math.floor(
        math.sqrt(len(ds_per_target)))
    fig = plt.figure()
    plt.title("Average Image per Target")
    fig.tight_layout()
    spec = fig.add_gridspec(nrows=x_axis, ncols=y_axis)

    x_ax, y_ax = 0, -1
    for i in range(len(ds_per_target)):
        ds = ds_per_target[i]

        avg_data = ds.data.float().mean(dim=0)

        x_ax = i % x_axis
        y_ax = (y_ax + 1 if i % x_axis == 0 else y_ax)

        ax = fig.add_subplot(spec[x_ax, y_ax])

        im = ax.imshow(avg_data, cmap='Reds', vmin=vmin, vmax=vmax)

        ax.set_title("Digit {}".format(i))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
