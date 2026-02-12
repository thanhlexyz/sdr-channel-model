from . import basic

def create(name, args):
    return eval(args.transceiver).Transmitter(name, args)
