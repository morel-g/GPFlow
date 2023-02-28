import numpy as np
import torch
from .plot_tool_box import (
    make_meshgrid,
    plot_function,
    plot_samples,
    save_transformation,
    save_velocity_fields,
    save_scatter_motion,
    save_velocity_field,
    save_figure,
    save_color_distributions,
    get_color_distribution,
)
from ..precision import torch_float_precision
from ..case import Case
import scipy
from src.tool_box import backward_func
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def neg_log_likelihood(x_end, log_det, probability_distribution):
    return probability_distribution.log_prob(x_end) + log_det


def compute_flow_trajectories(net, mesh_x, mesh_v):
    """Compute the trajectories of the GP flow associated with a given network.

    Args:
        net: A Network object.
        mesh_x: Mesh used to compute the trajectories of particles.
        mesh_v: Mesh used to compute the trajectories of the velocity.

    Returns:
        Particles trajectories moved by the GP flow and velocity
        trajectories of the GP flow.
    """
    with torch.no_grad():
        net.flow(mesh_v, save_trajectories=True)
        _, v_traj = net.get_flow_trajectories()
        net.flow(mesh_x, save_trajectories=True)
        x_traj, _ = net.get_flow_trajectories()
    return x_traj, v_traj


def plot_density_nf(net, Nx, output_dir, bound=2):
    """Plot several densities associated with a Network obejct.

    Args:
        net: the Network object.
        Nx: discretization integer.
        output_dir: output directory.
        bound: bound to plot the densities. Defaults to 2.
    """
    data = net.data
    mesh, xx, yy = make_meshgrid(
        (np.array([-bound, bound]), np.array([-bound, bound])), Nx=Nx
    )
    mesh_tensor = torch.tensor(
        mesh, dtype=torch_float_precision, device=net.device
    )
    x_gauss = net.probability_distribution.sample([int(5e4)])
    x_gauss_reduce = net.probability_distribution.sample([int(1e3)])

    with torch.no_grad():
        if backward_func(net.map, net.data.use_backward):
            x_sample = net.map.backward(x_gauss)
            x_sample_reduce = net.map.backward(x_gauss_reduce)
            x_sample_gp = net.ot_map(x_gauss, reverse=True)
            x_sample_reduce_gp = net.ot_map(x_gauss_reduce, reverse=True)
        else:
            x_sample, x_sample_reduce, x_sample_gp = None, None, None
        y_gp, log_det_gp = net.ot_map(mesh_tensor, return_log_det=True)
        y_map, log_det_map = net.map(mesh_tensor)

    # Sampling with backward
    if backward_func(net.map, net.data.use_backward):
        plot_samples(
            x_sample.cpu().detach().numpy(),
            output_dir,
            n_pts=1000,
            name="Flow samples/map.png",
        )
        plot_samples(
            x_sample_reduce.cpu().detach().numpy(),
            output_dir,
            hist=False,
            name="Flow samples/reduce_map.png",
        )
        if data.train_dict["gp_opt_type"] == Case.train_gp:
            plot_samples(
                x_sample_gp.cpu().detach().numpy(),
                output_dir,
                n_pts=1000,
                name="Flow samples/gp_map.png",
            )
            plot_samples(
                x_sample_reduce_gp.cpu().detach().numpy(),
                output_dir,
                hist=False,
                name="Flow samples/reduce_gp_map.png",
            )

    # Plotting density
    nll_map = (
        neg_log_likelihood(y_map, log_det_map, net.probability_distribution)
        .cpu()
        .detach()
        .numpy()
    )
    nll_gp_map = (
        neg_log_likelihood(y_gp, log_det_gp, net.probability_distribution)
        .cpu()
        .detach()
        .numpy()
    )
    # nll_id = (
    #     neg_log_likelihood(
    #         mesh_tensor,
    #         torch.zeros(mesh_tensor.shape[0]).type_as(mesh_tensor),
    #         net.probability_distribution,
    #     )
    #     .cpu()
    #     .detach()
    #     .numpy()
    # )
    if data.train_dict["gp_opt_type"] == Case.train_gp:
        plot_function(
            xx,
            yy,
            np.exp(nll_gp_map).reshape(xx.shape),
            output_dir,
            name="Density/estimate_gp_density.png",
        )
    plot_function(
        xx,
        yy,
        np.exp(nll_map).reshape(xx.shape),
        output_dir,
        name="Density/estimate_map_density.png",
    )
    # plot_function(
    #     xx,
    #     yy,
    #     np.exp(nll_id).reshape(xx.shape),
    #     output_dir,
    #     name="Density/gaussian_density.png",
    # )

    mesh, _, _ = make_meshgrid(
        (np.array([-3.5, 3.5]), np.array([-3.5, 3.5])), Nx=20
    )
    mesh_tensor = torch.tensor(
        mesh, dtype=torch_float_precision, device=net.device
    )
    with torch.no_grad():
        y_map = net.predict_map(mesh_tensor)
        y_gp_map = net.ot_map(mesh_tensor)
    save_transformation(
        mesh_tensor.cpu().detach().numpy(),
        output_dir,
        name="Transformation/initial_mesh.png",
    )
    save_transformation(
        y_map.cpu().detach().numpy(),
        output_dir,
        name="Transformation/mesh_transformation.png",
    )
    if data.train_dict["gp_opt_type"] == Case.train_gp:
        save_transformation(
            y_gp_map.cpu().detach().numpy(),
            output_dir,
            name="Transformation/mesh_gp_transformation.png",
        )
    # if backward_func(net.map, net.data.use_backward):
    #     with torch.no_grad():
    #         y_map_back = net.map.backward(mesh_tensor)
    #         y_gp_map_back = net.ot_map(mesh_tensor, reverse=True)

    #     save_transformation(
    #         y_map_back.cpu().detach().numpy(),
    #         output_dir,
    #         name="Transformation/backward_mesh_transformation.png",
    #     )
    # if data.train_dict["gp_opt_type"] == Case.train_gp:
    # save_transformation(
    #     y_gp_map_back.cpu().detach().numpy(),
    #     output_dir,
    #     name="Transformation/backward_mesh_gp_transformation.png",
    # )


def save_velocity_field_2D(net, output_dir):
    """Save two dimensional velocity field of GP flow.

    Args:
        net: a network object.
        output_dir: Output directory.
    """
    mesh_v, xx_v, yy_v = make_meshgrid(
        (np.array([-1.001, 1.001]), np.array([-1.001, 1.001])), Nx=30
    )
    mesh_x, xx, yy = make_meshgrid(
        (np.array([-1.0, 1.0]), np.array([-1.0, 1.0])), Nx=15
    )
    mesh_x_tensor = torch.tensor(
        mesh_x, dtype=torch_float_precision, device=net.device
    )
    mesh_v_tensor = torch.tensor(
        mesh_v, dtype=torch_float_precision, device=net.device
    )
    x_traj, v_traj = compute_flow_trajectories(
        net, mesh_x_tensor, mesh_v_tensor
    )
    np.save(output_dir + "/velocity_field.npy", v_traj)
    save_velocity_fields(
        v_traj,
        [xx_v, yy_v],
        output_dir,
        plt_mesh=False,
        boundary_bounds=(-1.0, 1.0, -1.0, 1.0),
        name="Transformation/velocity_field.gif",
    )
    save_velocity_fields(
        v_traj,
        [xx_v, yy_v],
        output_dir,
        points=x_traj,
        plt_mesh=True,
        boundary_bounds=(-1.0, 1.0, -1.0, 1.0),
        name="Transformation/velocity_field_with_mesh.gif",
    )
    save_transformation(
        x_traj[-1],
        output_dir,
        name="Transformation/incompressible_mesh_transformation.png",
    )
    save_velocity_field(
        v_traj[0],
        [xx_v, yy_v],
        output_dir,
        boundary_bounds=(-1.0, 1.0, -1.0, 1.0),
        name="Transformation/initial_velocity_field.png",
    )
    # div = compute_div(net.flow.v, mesh_x_tensor).cpu().detach().numpy()
    # plot_function(xx, yy, div.reshape(xx.shape), output_dir,
    # name='Transformation/Velocity field divergence')


def save_gaussian_motion(net, output_dir, use_color_distribution=True):
    def color_array(x):
        arr = [None] * x.shape[0]
        for i in range(len(arr)):
            if x[i, 0] > 0 and x[i, 1] > 0:
                c = "r"
            elif x[i, 0] > 0 and x[i, 1] < 0:
                c = "b"
            elif x[i, 0] < 0 and x[i, 1] > 0:
                c = "g"
            else:
                c = "y"
            arr[i] = c
        return arr

    with torch.no_grad():
        if not use_color_distribution:
            x_gauss = net.probability_distribution.sample([int(1e4)])
            c = color_array(x_gauss)
        else:
            x_gauss = torch.tensor(np.load(output_dir + "/map_points.npy"))
            c = get_color_distribution(output_dir)[
                torch.norm(x_gauss, dim=-1) <= 4.2
            ]
            x_gauss = x_gauss[torch.norm(x_gauss, dim=-1) <= 4.2]

        x_gp, _ = net.gp_flow(x_gauss, save_trajectories=True)
    x_traj, _ = net.get_flow_trajectories()
    for i in range(len(x_traj)):
        x_traj[i] = np.sqrt(2) * net.sigma * scipy.special.erfinv(x_traj[i])

    # np.save(output_dir + "/particles_gaussian motion.npy", x_traj)
    save_scatter_motion(
        x_traj, output_dir, c, name="Transformation/gaussian_motion.gif"
    )

    mesh_gauss, _, _ = make_meshgrid(
        (np.array([-4.0, 4.0]), np.array([-4.0, 4.0])), Nx=15
    )
    with torch.no_grad():
        mesh_gauss, _ = net.gp_flow(
            torch.tensor(
                mesh_gauss, dtype=torch_float_precision, device=net.device
            )
        )
    save_transformation(
        mesh_gauss.cpu().detach().numpy(),
        output_dir,
        name="Transformation/gaussian_mesh_transformation.png",
    )


def compute_outputs_2d(net, X, output_dir):
    plt.rcParams.update({"font.size": 11})

    bound = 1.2 * torch.norm(X, dim=-1).max().cpu().numpy()
    print("- Plotting density nf...")
    plot_density_nf(net, Nx=100, output_dir=output_dir, bound=bound)

    save_color_distributions(
        output_dir,
        net.data.nf_model_dict["name"],
        name="Density/color_distributions",
        black_background=False,
    )
    if net.data.train_dict["gp_opt_type"] == Case.train_gp:
        print("- Saving particles motion...")
        save_velocity_field_2D(net, output_dir)
        save_gaussian_motion(net, output_dir)


def save_trajectories(
    trajectories, T, output_dir, name="trajectories_2d_projection.png"
):
    """
    Save 2d projection of the trajecories.

    Args:
        trajectories: the trajectories
        T: The discrete times of the trajectories.
        output_dir: Output directory.
        name: a string to name the plots.
    """

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax = Axes3D(fig)
    Tn = np.array(T).reshape(-1, 1)
    T_np = np.array([Ti.repeat(trajectories[0].shape[0]) for Ti in Tn])
    T_np = np.swapaxes(T_np, 0, 1)
    traj = np.swapaxes(np.array(trajectories), 0, 1)

    for i in range(traj.shape[0]):
        current_traj = traj[i, :, :]
        color = "blue"
        ax.scatter(
            current_traj[0, 0],
            T_np[i, 0],
            current_traj[0, 1],
            alpha=1,
            c=color,
        )
        ax.scatter(
            current_traj[-1, 0],
            T_np[i, -1],
            current_traj[-1, 1],
            alpha=0.1,
            c=color,
        )
        ax.plot(
            current_traj[1:-1, 0],
            T_np[i, 1:-1],
            current_traj[1:-1, 1],
            alpha=0.1,
            c=color,
        )

    save_figure(output_dir, fig, name=name)
    # add_logger_figure(logger, fig, name=name)
