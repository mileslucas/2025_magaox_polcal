import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numpy as np
import proplot as pro
import pandas as pd

import fit_model_A
import paths
import plot_utils


class ReactiveFitGUI:
    def __init__(self, param_ranges, table):
        """
        param_ranges: dict {name: (min, max)}
        model_func: function f(x, **params) → model y
        xdata, ydata: data to plot
        """
        self.param_ranges = param_ranges

        table = table.sort_values(["hwp", "imr"])
        self.imr_groups = table.groupby("imr")
        self.hwp_angs = table["hwp"].unique()
        self.imr_angs = table["imr"].unique()
        self.test_hwps = np.linspace(self.hwp_angs.min(), self.hwp_angs.max(), 100)

        cmap = pro.Colormap("inferno", left=0.2, right=0.8, discrete=True)
        self.colors = cmap(np.linspace(0, 1, len(self.imr_angs)))

        # internal param state (initialized mid-range)
        self.params = {k: 0.5 * (v[0] + v[1]) for k, v in param_ranges.items()}

        self._build_gui()

    def _build_gui(self):
        fig, ax = pro.subplots(left="0.25", bottom="0.5")

        # Plot original data
        ax = self.plot_table(ax)

        # Plot initial model line
        model_curves = self.get_model_curve(self.params)
        self.model_lines = []
        for i, curve in enumerate(model_curves):
            self.model_lines.append(ax.plot(self.test_hwps, curve, c=self.colors[i])[0])

        # Store sliders + textboxes
        self.sliders = {}
        self.textboxes = {}

        slider_height = 0.03
        text_width = 0.05
        bottom = 0.25

        for i, (name, (vmin, vmax)) in enumerate(self.param_ranges.items()):
            ypos = bottom - i * 0.05

            # slider axis
            ax_slider = plt.axes([0.25, ypos, 0.65, slider_height])
            slider = Slider(ax_slider, name, vmin, vmax, valinit=self.params[name])
            self.sliders[name] = slider

            # textbox axis
            ax_text = plt.axes([0.92, ypos, text_width, slider_height])
            text = TextBox(ax_text, "", initial=f"{self.params[name]:.3g}")
            self.textboxes[name] = text

            # callbacks
            slider.on_changed(self._make_slider_callback(name))
            text.on_submit(self._make_textbox_callback(name))

        plt.show()

    def _update_plot(self):
        # Recompute model
        model_curves = self.get_model_curve(self.params)
        for i, curve in enumerate(model_curves):
            self.model_lines[i].set_ydata(curve)
            self.model_lines[i].figure.canvas.draw_idle()

        # Redraw

    def _make_slider_callback(self, name):
        def callback(val):
            self.params[name] = val
            self.textboxes[name].set_val(f"{val:.3g}")  # update textbox
            self._update_plot()

        return callback

    def _make_textbox_callback(self, name):
        def callback(text):
            try:
                v = float(text)
            except ValueError:
                return
            # clamp to range
            lo, hi = self.param_ranges[name]
            v = max(lo, min(hi, v))
            self.params[name] = v
            self.sliders[name].set_val(v)
            self._update_plot()

        return callback

    def get_model_curve(self, params_dict):
        X = list(params_dict.values())
        curves = []
        for imr in self.imr_angs:
            inner_curve = []
            for hwp in self.test_hwps:
                inner_curve.append(
                    fit_model_A.model(X, hwp, imr, np.array([1, 1, 0, 0]))
                )
            curves.append(inner_curve)
        return np.array(curves)

    def plot_table(self, ax):
        for idx, (imr_ang, group) in enumerate(self.imr_groups):
            ax.scatter(
                group["hwp"],
                group["single_norm"],
                # bardata=group["single_norm_err"],
                c=self.colors[idx],
                marker="o",
                label=f"{imr_ang:.02f}°",
                zorder=1000,
            )

        ax.axhline(0, lw=1, c="0.2", zorder=100)
        ax.legend(ncols=3, loc="bottom", title="IMR angle")

        ax.format(
            xlabel="HWP angle (°)",
            ylabel="Normalized single difference",
            ylim=(-1, 1),
            xlocator=22.5,
        )
        return ax


if __name__ == "__main__":
    plot_utils.setup_plots()
    pro.switch_backend("QtAgg")
    table = pd.read_csv(paths.data / "20251126_magaox_lp0_single_diffs.csv")
    subtable = table.query("filter == 'i'")

    ReactiveFitGUI(
        param_ranges={
            "hwp_offset": [-20, 20],
            "hwp_retardance": [0, 360],
            "imr_diatt": [0, 1],
            "imr_offset": [-90, 90],
            "imr_retardance": [0, 360],
            "pbs_through_1": [0, 1],
            "pbs_through_2": [0, 1],
        },
        table=subtable,
    )
