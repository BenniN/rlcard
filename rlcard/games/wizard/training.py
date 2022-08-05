import os


def save_args_params(args):
    if not os.path.exists(args["log_dir"]):
        os.makedirs(args["log_dir"])

    with open(args["log_dir"] + '/model_params.txt', 'w') as f:
        for key, value in args.items():
            f.write("{}: {}\n".format(key, value))