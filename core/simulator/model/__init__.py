from . import flat_fading, doppler

def create(args):
    return eval(args.model).Model(args)
