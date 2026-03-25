import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from icecream import ic


def show_interactively(
    data: np.ndarray,
    colormap,
    vmin0: float,
    vmax0: float,
    extent: tuple[float, float, float, float],
    origin: str,
):

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.12, bottom=0.25)

    im = ax.imshow(
        data, cmap=colormap, vmin=vmin0, vmax=vmax0, extent=extent, origin=origin
    )
    fig.colorbar(im, ax=ax)

    # slider axes
    ax_vmin = plt.axes((0.12, 0.14, 0.76, 0.03))
    ax_vmax = plt.axes((0.12, 0.08, 0.76, 0.03))

    # slider ranges chosen from data min/max
    data_min, data_max = float(np.nanmin(data)), float(np.nanmax(data))
    if data_min > 0:
        data_min = 0
    data_max = (data_max - data_min) * 1.1 + data_min
    slider_vmin = Slider(ax_vmin, "vmin", data_min, data_max, valinit=vmin0)
    slider_vmax = Slider(ax_vmax, "vmax", data_min, data_max, valinit=vmax0)

    def update(val):
        vmin = slider_vmin.val
        vmax = slider_vmax.val
        # enforce vmin <= vmax
        if vmin > vmax:
            # adjust the other slider to keep ordering
            if val is slider_vmin.val:
                slider_vmax.set_val(vmin)
                vmax = vmin
            else:
                slider_vmin.set_val(vmax)
                vmin = vmax
        im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)

    # reset button
    reset_ax = plt.axes((0.12, 0.020, 0.08, 0.04))
    reset_btn = Button(reset_ax, "Reset", hovercolor="0.975")

    save_ax = plt.axes((0.71, 0.020, 0.08, 0.04))
    save_btn = Button(save_ax, "save", hovercolor="0.975")

    cancel_ax = plt.axes((0.80, 0.020, 0.08, 0.04))
    cancel_btn = Button(cancel_ax, "cancel", hovercolor="0.975")

    def reset(event):
        slider_vmin.reset()
        slider_vmax.reset()

    reset_btn.on_clicked(reset)

    results = dict(
        canceled=True
    )  # Using a dict as a dirty work around to let it be captured byt the close function

    def save_and_close(event):
        results["canceled"] = False
        plt.close(fig)

    def close(event):
        results["canceled"] = True
        plt.close(fig)

    cancel_btn.on_clicked(close)
    save_btn.on_clicked(save_and_close)

    plt.show()

    return slider_vmin.val, slider_vmax.val, not results["canceled"]
