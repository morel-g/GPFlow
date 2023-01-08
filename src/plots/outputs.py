import torch
import numpy as np
from ..tool_box import apply_fn_batch, backward_func
from .plot_tool_box import make_meshgrid


def compute_ot_costs(
    net,
    x_test,
    logger=None,
    output_dir=None,
    device=None,
    only_ot_costs=False,
    save_latent_points=False,
):
    def neg_log_likelihood(x_end, log_det, probability_distribution):
        return (
            -probability_distribution.log_prob(x_end).mean() - log_det.mean()
        )

    if device is not None:
        x_test = x_test.to(device)
    if logger is not None:
        output_dir = logger.log_dir
    save_outputs = output_dir is not None

    data = net.data
    dim = data.dim
    batch_size = data.batch_size
    map_only = (
        net.flow is None
    )  # train_map(data.train_dict) or net.flow is None

    with torch.no_grad():
        y_map, log_det_map = apply_fn_batch(
            net.map.forward, x_test, batch_size, device=device
        )
        if not map_only:
            y_ot, log_det_ot = apply_fn_batch(
                lambda x: net.ot_map(x, return_log_det=True),
                x_test,
                batch_size,
                device=device,
            )
            if dim == 2 and save_outputs:
                np.save(output_dir + "/gp_points.npy", y_ot.cpu().numpy())

        x_gauss = net.probability_distribution.sample([x_test.shape[0]])
        if (dim == 2 and save_outputs) or save_latent_points:
            np.save(output_dir + "/map_points.npy", y_map.cpu().numpy())
            np.save(output_dir + "/initial_points.npy", x_test.cpu().numpy())
            np.save(output_dir + "/gaussian_points.npy", x_gauss.cpu().numpy())
        if not map_only:

            def flow_lagrangian(x):
                net.flow(x)
                return torch.tensor([net.flow.lagrangian])

            if x_test.shape[1] == 2:
                # In 2d we compute the lagrangian over a uniform mesh
                mesh, _, _ = make_meshgrid(
                    (np.array([-1.0, 1.0]), np.array([-1.0, 1.0])), Nx=100
                )
                #  Apply_fn_batch(net.flow, torch.tensor(mesh).type_as(x_gauss)
                # , batch_size, device=device)
                lagrangian = apply_fn_batch(
                    flow_lagrangian,
                    torch.tensor(mesh).type_as(x_gauss),
                    batch_size,
                    device=device,
                ).mean()
            else:
                # For large dimension we compute lagrangian over random
                # points. The flow is directly applied in the hypercube
                # [-1,1]^d.
                x_cube = 2 * (
                    torch.rand(x_gauss.shape, device=device) - 0.5
                ).type_as(x_test)
                lagrangian = apply_fn_batch(
                    flow_lagrangian, x_cube, batch_size, device=device
                ).mean()
            # x_gauss_gp, _ = apply_fn_batch(
            #     net.gp_flow, x_gauss, batch_size, device=device
            # )
            # log_prob_gauss = apply_fn_batch(
            #     net.probability_distribution.log_prob,
            #     x_gauss,
            #     batch_size,
            #     device=device,
            # )
            # log_prob_gp = apply_fn_batch(
            #     net.probability_distribution.log_prob,
            #     x_gauss_gp,
            #     batch_size,
            #     device=device,
            # )
            # diff_log_prob = (
            #     (log_prob_gauss - log_prob_gp).cpu().detach().numpy().mean()
            # )

        if backward_func(net.map, data.use_backward):
            y_map_back = apply_fn_batch(
                net.map.backward, x_gauss, batch_size, device=device
            )
            finv_f = apply_fn_batch(
                net.map, y_map_back, batch_size, device=device
            )[0]
            finv_f_err = np.linalg.norm(
                (finv_f.cpu() - x_gauss.cpu()).detach().numpy(), axis=1
            ).mean()
            cost_back = (
                ((y_map_back.cpu() - x_gauss.cpu()) ** 2)
                .sum(-1)
                .mean()
                .detach()
                .numpy()
            )
            if not map_only:
                y_gp_back = apply_fn_batch(
                    lambda x: net.ot_map(x, reverse=True),
                    x_gauss,
                    batch_size,
                    device=device,
                )
                finv_f_ot = apply_fn_batch(
                    net.ot_map, y_gp_back, batch_size, device=device
                )
                finv_f_gp_err = np.linalg.norm(
                    (finv_f_ot.cpu() - x_gauss.cpu()).detach().numpy(), axis=1
                ).mean()
                cost_gp_back = (
                    ((y_gp_back.cpu() - x_gauss.cpu()) ** 2)
                    .sum(-1)
                    .mean()
                    .detach()
                    .numpy()
                )

        # nll_init = (
        #     neg_log_likelihood(
        #         x_test,
        #         torch.ones(x_test.shape[0]).to(x_test.device),
        #         net.probability_distribution,
        #     )
        #     .mean()
        #     .cpu()
        #     .detach()
        #     .numpy()
        # )
        nll_map = (
            neg_log_likelihood(
                y_map, log_det_map, net.probability_distribution
            )
            .mean()
            .cpu()
            .detach()
            .numpy()
        )
        cost_map = (
            ((y_map.cpu() - x_test.cpu()) ** 2).sum(-1).mean().detach().numpy()
        )
        if not map_only:
            nll_ot = (
                neg_log_likelihood(
                    y_ot, log_det_ot, net.probability_distribution
                )
                .mean()
                .cpu()
                .detach()
                .numpy()
            )
            cost_gp_map = (
                ((y_ot.cpu() - x_test.cpu()) ** 2)
                .sum(-1)
                .mean()
                .detach()
                .numpy()
            )

    eval_str = "  \n  "

    eval_str += "  ** Standard map: ** " + "  \n  "
    if not only_ot_costs:
        eval_str += (
            "- Loss map test dataset = "
            + str("{:.3f}".format(nll_map))
            + "  \n  "
        )
    eval_str += (
        "- OT cost map test dataset = "
        + str("{:.3f}".format(cost_map))
        + "  \n  "
    )
    if backward_func(net.map, data.use_backward):
        eval_str += (
            "- OT cost map backward random data = "
            + str("{:.3f}".format(cost_back))
            + "  \n  "
        )
        eval_str += (
            "- finv_f error map = "
            + str("{:.5f}".format(finv_f_err))
            + "  \n  "
        )
    if not map_only:
        eval_str += "  ** GP map: ** " + "  \n  "
        if not only_ot_costs:
            eval_str += (
                "- Loss GP test dataset = "
                + str("{:.3f}".format(nll_ot))
                + "  \n  "
            )
        eval_str += (
            "- OT cost GP test dataset = "
            + str("{:.3f}".format(cost_gp_map))
            + "  \n  "
        )
        if backward_func(net.map, data.use_backward):
            eval_str += (
                "- OT cost GP backward random data = "
                + str("{:.3f}".format(cost_gp_back))
                + "  \n  "
            )
            eval_str += (
                "- finv_f error GP map = "
                + str("{:.5f}".format(finv_f_gp_err))
                + "  \n  "
            )
        # eval_str += (
        #     "- Diff log prob on gaussian samples = "
        #     + str("{:.3f}".format(diff_log_prob))
        #     + "  \n  "
        # )
        eval_str += (
            "- GP flow lagrangian = "
            + str("{:.3f}".format(lagrangian))
            + "  \n  "
        )

    # logger.experiment.add_text("Costs", "   \n  ")
    if logger is not None:
        logger.experiment.add_text("Costs", eval_str)
    print("")
    if save_outputs:
        with open(output_dir + "/ot_costs.txt", "w") as f:
            f.write(eval_str)

    print(eval_str)
    # logger.experiment.add_text("Costs3",re.sub("<br/>", " \n ", eval_str) )
    # print(re.sub("<br/>", "\n", eval_str))
