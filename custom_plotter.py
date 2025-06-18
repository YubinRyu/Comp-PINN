import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

class _Plotter:
    def __call__(self, *args):
        raise NotImplementedError

    def _add_figures(self, group, name, results_dir, writer, step, *args):
        "Try to make plots and write them to tensorboard summary"

        # catch exceptions on (possibly user-defined) __call__
        try:
            fs = self(*args)
        except Exception as e:
            print(f"error: {self}.__call__ raised an exception:", str(e))
        else:
            for f, tag in fs:
                f.savefig(
                    results_dir + name + "_" + tag + ".png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                writer.add_figure(group + "/" + name + "/" + tag, f, step, close=True)
            plt.close("all")

    def _interpolate_2D(self, size, invar, *outvars):
        "Interpolate 2D outvar solutions onto a regular mesh"

        assert len(invar) == 2

        # define regular mesh to interpolate onto
        xs = [invar[k][:, 0] for k in invar]
        extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size),
            np.linspace(extent[2], extent[3], size),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (xs[0], xs[1]), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp

    def custom_interpolate_2D(self, size, radius, height, invar, *outvars):
        "Interpolate 2D outvar solutions onto a regular mesh"

        assert len(invar) == 2

        min_r = - radius[0]
        max_r = radius[0]

        min_h = - height[0] / 2
        max_h = height[0] / 2

        # define regular mesh to interpolate onto
        xs = [invar[k][:, 0] for k in invar]
        extent = (min_r, max_r, min_h, max_h)
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size),
            np.linspace(extent[2], extent[3], size),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (xs[0], xs[1]), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp

class ValidatorPlotter(_Plotter):
    "Default plotter class for validator"

    def __call__(self, invar, true_outvar, pred_outvar):
        "Default function for plotting validator data"

        r = invar["r"][0]
        h = invar["h"][0]

        radius = 0.05 + 0.02 * r
        height = 0.045 * h

        invar = {"x": invar["x"], "z": invar["z"]}

        # interpolate 2D data onto grid
        extent, true_outvar, pred_outvar = self._interpolate_2D(
                1000, invar, true_outvar, pred_outvar
            )

        # make plots
        fs = []
        for k in pred_outvar:
            factor = 1

            if k == "p":
                factor = 520

            f = plt.figure(figsize=(15/2.54, 3.5/2.54), dpi=500)
            for i, (o, tag) in enumerate(
                zip(
                    [true_outvar[k], pred_outvar[k] * factor, true_outvar[k] - pred_outvar[k] * factor],
                    ["true", "pred", "diff"],
                )
            ):

                if i == 0 or i == 1:
                    plt.rcParams['font.size'] = 7

                    plt.subplot(1, 3, 1 + i)

                    if k == "u":
                        plt.imshow(o.T, origin="lower", extent=extent, vmin=-0.1, vmax=0.2)
                        cbar = plt.colorbar()
                        cbar.set_ticks([-0.1, 0.0, 0.1, 0.2])

                    elif k == "v":
                        plt.imshow(o.T, origin="lower", extent=extent, vmin=-0.5, vmax=0.5)
                        cbar = plt.colorbar()
                        cbar.set_ticks([-0.5, 0.00, 0.5])

                    elif k == "w":
                        plt.imshow(o.T, origin="lower", extent=extent, vmin=-0.5, vmax=0.1)
                        cbar = plt.colorbar()
                        cbar.set_ticks([-0.5, -0.2, 0.1])

                    # Impeller Geometry
                    blade = Rectangle((-radius, height - 0.015), 2 * radius, 0.03, fill=True, facecolor='white')
                    shaft = Rectangle((-0.0025, -0.0325), 0.005, 0.1325, fill=True, facecolor='white')
                    plt.gca().add_patch(blade)
                    plt.gca().add_patch(shaft)

                    plt.xticks([-0.1, 0.0, 0.1])
                    plt.yticks([-0.1, 0.0, 0.1])
                    plt.xlabel("x (m)", fontsize=8)
                    plt.ylabel("z (m)", fontsize=8)
                    plt.title(f"{k}_{tag}", fontsize=8)

                else:
                    plt.rcParams['font.size'] = 7

                    plt.subplot(1, 3, 1 + i)
                    plt.imshow(np.abs(o).T, origin="lower", extent=extent, vmin=0.00, vmax=0.15)

                    # Impeller Geometry
                    blade = Rectangle((-radius, height - 0.015), 2 * radius, 0.03, fill=True, facecolor='white')
                    shaft = Rectangle((-0.0025, -0.0325), 0.005, 0.1325, fill=True, facecolor='white')
                    plt.gca().add_patch(blade)
                    plt.gca().add_patch(shaft)

                    plt.xticks([-0.1, 0.0, 0.1])
                    plt.yticks([-0.1, 0.0, 0.1])
                    plt.xlabel("x (m)", fontsize=8)
                    plt.ylabel("z (m)", fontsize=8)

                    cbar = plt.colorbar()
                    cbar.set_ticks([0.00, 0.05, 0.10, 0.15])

                    plt.title(f"{k}_{tag}", fontsize=8)

            plt.tight_layout()
            fs.append((f, k))

        return fs