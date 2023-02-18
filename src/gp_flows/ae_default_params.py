from src.case import Case


def set_ae_default_params(data, model_name):
    if data.dim == 10:
        if data.train_dict["gp_opt_type"] == Case.train_nf:
            if model_name == Case.ffjord:
                data.epochs = 200
            elif model_name == Case.bnaf:
                data.epochs = 5000
        if data.train_dict["gp_opt_type"] == Case.train_gp:
            if model_name == Case.ffjord:
                data.epochs = 200
            elif model_name == Case.bnaf:
                data.epochs = 800
