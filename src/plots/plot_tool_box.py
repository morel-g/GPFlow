import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
import matplotlib
from pathlib import Path
import imageio
import torch
import colorsys
import math
import os
from os.path import exists


FIG_DIR = "/figures/"


def save_figure(dir_path, fig, name):
    name = (
        name if os.path.splitext(name)[-1].lower() == ".png" else name + ".png"
    )
    # Create dir if not exists.
    Path(dir_path + FIG_DIR + "/".join(name.split("/")[:-1])).mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(dir_path + FIG_DIR + name)


def save_video(dir_path, figs, name):
    name = (
        name if os.path.splitext(name)[-1].lower() == ".gif" else name + ".gif"
    )
    # Create dir if not exists.
    Path(dir_path + FIG_DIR + "/".join(name.split("/")[:-1])).mkdir(
        parents=True, exist_ok=True
    )
    fps = 7 if len(figs) > 20 else 5

    imageio.mimsave(
        dir_path + FIG_DIR + name,
        [np.swapaxes(np.swapaxes(f, 0, 1), 1, -1) for f in figs],
        fps=fps,
    )


def add_logger_figure(logger, fig, name=""):
    # fig.savefig(logger.log_dir + "/test_img.png")
    if not name:
        logger.experiment.add_figure("matplotlib", fig)
    else:
        logger.experiment.add_figure(name, fig)
    logger.experiment.close()


def add_logger_video(logger, figures, name=""):
    if len(figures) > 20:
        fps = 7
    elif len(figures) > 10:
        fps = 5
    else:
        fps = 3

    logger.experiment.add_video(name, torch.tensor([figures]), fps=fps)
    logger.experiment.close()


def figure_to_data(fig):
    """
    A function to convert matplotlib fig to numpy array
    Return a numpy array.
    Args:
        fig: The figure to convert.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    X = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    X = np.transpose(X, (2, 0, 1))
    return X


def make_meshgrid(bounds, Nx=50):
    """Generate a mesh of 2d points within bounds.

    Args:
        bounds: the boundaries of the grid. Can be either a scalar
        (i.e. the maximal absolute value for all dimensions) or a list with shape
        (2, 2) containing the lower and upper bounds for each dimension.
        Nx: Number of points for each axis. Can be either a scalar or a list of
        size 2. Defaults to 50.
    Returns:
        The mesh with shape (Nx*Ny, 2) and the values both of x and y with
        shape (Ny, Nx).
    """
    if not hasattr(bounds, "__len__"):
        b = abs(bounds)
        bounds = ((-b, b), (-b, b))
    if not hasattr(Nx, "__len__"):
        Nx = 2 * [Nx]

    lsp = [
        np.linspace(*bi, Nx[i], endpoint=True) for i, bi in enumerate(bounds)
    ]
    x_y = np.meshgrid(*lsp)

    mesh = np.c_[x_y[0].ravel(), x_y[1].ravel()]
    return mesh, x_y[0], x_y[1]


def mesh_from_trajectories(
    x_trajectories,
    y_trajectories,
    global_bound=0,
    tol=0.0,
    Nx=18,
):
    x_min, x_max = (
        min(x_trajectories.min(), -global_bound) - tol,
        max(x_trajectories.max(), global_bound) + tol,
    )
    y_min, y_max = (
        min(y_trajectories.min(), -global_bound) - tol,
        max(y_trajectories.max(), global_bound) + tol,
    )

    return make_meshgrid([[x_min, x_max], [y_min, y_max]], Nx=Nx)


def plot_function(x, y, f, output_dir, name="func", contour=False):
    fig, ax = plt.subplots()
    if not contour:
        p = ax.pcolormesh(x, y, f, cmap=plt.cm.bwr, shading="gouraud")
    else:
        p = ax.contourf(x, y, f)
    fmt = lambda x, pos: "{:.2f}".format(x)
    fig.colorbar(p, ax=ax, format=FuncFormatter(fmt))

    save_figure(output_dir, fig, name)
    plt.close()


def plot_samples(z, output_dir, n_pts=1000, hist=True, name=""):
    fig = plt.figure()
    if hist:
        plt.hist2d(z[:, 0], z[:, 1], bins=n_pts, cmap=plt.cm.jet)
    else:
        plt.scatter(z[:, 0], z[:, 1], cmap=plt.cm.jet)
    # add_logger_figure(logger, fig, name)
    save_figure(output_dir, fig, name)
    plt.close()


def save_velocity_fields(
    V,
    mesh_x_y,
    output_dir,
    points=None,
    plt_mesh=False,
    normalize=True,
    boundary_bounds=None,
    name="velocity_fields",
):
    """Save a gif of multiple 2D velocity fields.

    Args:
        V: The values of the velocity fields with shape (nb_points, 2).
        mesh_x_y: Tuple made of a grid of abscissas and ordinates with shape (Nx, Ny) at which
        the velocity field is evaluated.
        output_dir: Directory to save the gif.
        points: List of points to plot on top of the velocity field.
         Defaults to None.
        plt_mesh:  True if the points are plot as a mesh. Defaults to False.
        normalize: True if the velocity field is normalized. Defaults to True.
        boundary_bounds: Boundary bounds to plot the boundaries. Defaults to
         None.
        name: Name of the gif. Defaults to "".
    """
    figures = []

    if boundary_bounds is not None:
        norm_v_max = 0.0
        xx, yy = mesh_x_y
        x_min, x_max, y_min, y_max = boundary_bounds
        eps = 1e-3
        out_of_boundary_ids = np.logical_or(
            np.logical_or((xx < x_min - eps), (xx > x_max + eps)),
            np.logical_or((yy < y_min - eps), (yy > y_max + eps)),
        )
        V_norm = np.sqrt(V[:, :, 0] ** 2 + V[:, :, 1] ** 2).reshape(
            (V.shape[0],) + xx.shape
        )
        norm_v_max = V_norm[:, np.logical_not(out_of_boundary_ids)].max()
    else:
        norm_v_max = np.linalg.norm(V, axis=-1).max()

    for i in range(len(V)):
        v = V[i]
        p = points[i] if points is not None else None
        fig = plt.figure()
        plot_velocity_field(
            v,
            mesh_x_y,
            points=p,
            plt_mesh=plt_mesh,
            normalize=normalize,
            colorbar_range=[0.0, norm_v_max],
            # colorbar_range=[0.0, 5.2],
            boundary_bounds=boundary_bounds,
        )

        # fig = plt.gcf()
        figures.append(figure_to_data(fig))
        plt.close()

    # add_logger_video(logger, figures, name)
    save_video(output_dir, figures, name)


def save_velocity_field(
    V,
    mesh_x_y,
    output_dir,
    normalize=True,
    boundary_bounds=None,
    name="velocity_field",
):
    """Save a 2D velocity field to an image.

    Args:
        V: The values of the velocity field with shape (nb_points, 2).
        mesh_x_y: Tuple made of a grid of abscissas and ordinates at which the
         velocity field is evaluated.
        output_dir: Directory to save the image.
        normalize: True if the velocity field is normalized. Defaults to True.
        boundary_bounds: Boundary bounds to plot the boundaries. Defaults to
         None.
    """
    fig = plt.figure()
    plot_velocity_field(
        V,
        mesh_x_y,
        normalize=normalize,
        boundary_bounds=boundary_bounds,
        # colorbar_range=[0.0, 1.05],
    )

    save_figure(output_dir, fig, name)
    plt.close()


def plot_velocity_field(
    V,
    velocity_mesh_xy,
    points=None,
    plt_mesh=False,
    normalize=True,
    boundary_bounds=None,
    colorbar_range=None,
):
    """Plot a 2D velocity field.

    Args:
        V: The values of the velocity field with shape (nb_points, 2).
        velocity_mesh_xy: Tuple made of a grid of abscissas and ordinates at which the
         velocity field is evaluated.
        points: Points to plot on top of the velocity field.
         Defaults to None.
        plt_mesh: True if the points are plot as a mesh. Defaults to False.
        normalize: True if the velocity field is normalized. Defaults to True.
        boundary_bounds: Boundary bounds to plot the boundaries. Defaults to
         None.
        colorbar_range: A tuple of the min and max values of the velocity field
         in the colorbar. Defaults to None.
    """
    xx, yy = velocity_mesh_xy
    Vx = V[:, 0].reshape(xx.shape)
    Vy = V[:, 1].reshape(yy.shape)
    # fig = plt.figure()
    V_norm = np.sqrt(Vx**2 + Vy**2)

    if normalize:
        Vx = Vx / (V_norm + 1e-5)
        Vy = Vy / (V_norm + 1e-5)

    if colorbar_range is None:
        cf = plt.contourf(xx, yy, V_norm, cmap="YlGn")
    else:
        vmin, vmax = colorbar_range

        levels = np.linspace(vmin, vmax + (vmax - vmin) / 20.0, 7)
        cmap = matplotlib.cm.get_cmap("YlGn")

        matplotlib.colors.Colormap(cmap(1.0))
        cf = plt.contourf(
            xx, yy, V_norm, cmap="YlGn", levels=levels, extend="both"
        )

    plt.colorbar().remove()
    fmt = lambda x, pos: "{:.2f}".format(x)
    plt.colorbar(cf, format=FuncFormatter(fmt))

    plt.quiver(xx, yy, Vx, Vy)
    if points is not None:
        plot_points(points)
    if plt_mesh:
        plot_mesh(points)
    if boundary_bounds is not None:
        x_min, x_max, y_min, y_max = boundary_bounds
        rectangle = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, color="b", fill=False
        )
        plt.gca().add_patch(rectangle)


def save_scatter_motion(x_traj, output_dir, color, name="scatter_motion"):
    figures = []
    for i in range(len(x_traj)):
        x = x_traj[i]
        fig = plt.figure()
        plt.scatter(x[:, 0], x[:, 1], color=color)
        bound = max(abs(x_traj.min()), x_traj.max())
        plt.xlim(-bound, bound)
        plt.ylim(-bound, bound)
        figures.append(figure_to_data(fig))
        plt.close()
    save_video(output_dir, figures, name)
    # add_logger_video(logger, figures, name)


def make_animation(x_start, x_end, output_dir, nb_frames=40, name=""):
    t = np.linspace(0.0, 1.0, endpoint=True).reshape(-1, 1, 1)
    x_traj = (1 - t) * x_start + t * x_end
    save_scatter_motion(x_traj, output_dir, color="blue", name=name)


def save_transformation(X, output_dir, name=""):
    fig = plt.figure()

    plot_mesh(X)

    save_figure(output_dir, fig, name)
    # add_logger_figure(logger, fig, name)
    plt.close()


def save_transformations(X1, X2, output_dir, name=""):
    fig = plt.figure()

    plot_mesh(X1, color="red")
    plot_mesh(X2, color="blue")

    save_figure(output_dir, fig, name)
    # add_logger_figure(logger, fig, name)
    plt.close()


def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    ax = plt.gca()


def plot_mesh(X, color="blue"):
    shape_X = int(np.sqrt(X.shape[0]))
    xx_traj = X[:, 0].reshape(shape_X, shape_X)
    yy_traj = X[:, 1].reshape(shape_X, shape_X)
    plot_grid(xx_traj, yy_traj, color=color)


def plot_points(X):
    """Plot 2D points of shape (n_points, 2).

    Args:
        X: The points to plot.
    """
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c="red",
        cmap=cm_bright,
        edgecolors="k",
        s=10,
        linewidths=0.5,
    )


def pos2color(X, cx=0, cy=0, radius=1, brightness=0.9):
    res_colors = []
    for i in range(X.shape[0]):
        rx = X[i, 0] - cx
        ry = X[i, 1] - cy
        s = (rx**2.0 + ry**2.0) ** 0.5 / radius
        if s <= 1.0:
            h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0
            rgb = colorsys.hsv_to_rgb(h, s, 0.8)
            res_colors.append([int(round(c * 255.0)) for c in rgb])
        else:
            res_colors.append([0, 0, 0])
    return np.array(res_colors) / 255


def save_color_distributions(
    output_dir, model_name, name="Color distribution", black_background=False
):
    X = np.load(output_dir + "/initial_points.npy")
    # y = np.load("class_points.npy")
    X_init = X  # StandardScaler().fit_transform(X)
    # X= 0.9*X
    if black_background:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    # colors = pos2color_withlabels(X,y,radius=3.7)
    colors = pos2color(X, radius=np.linalg.norm(X, axis=-1).max())

    fig = plt.figure()
    plt.scatter(X_init[:, 0], X_init[:, 1], s=10, color=colors)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xticks(())
    plt.yticks(())

    save_figure(output_dir, fig, name + "_initial_distribution.png")
    plt.close()

    X = np.load(output_dir + "/map_points.npy")
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xticks(())
    plt.yticks(())
    save_figure(output_dir, fig, name + "_" + model_name)
    plt.close()

    if exists(output_dir + "/gp_points.npy"):
        X = np.load(output_dir + "/gp_points.npy")

        fig = plt.figure()
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xticks(())
        plt.yticks(())

        save_figure(output_dir, fig, name + " " + model_name + " + GP")
        plt.close()
