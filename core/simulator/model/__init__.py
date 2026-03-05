from . import doppler

def create(args):
    return eval(args.model).Model(args)
