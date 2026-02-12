from . import basic

def create(name, args):
    return eval(args.transceiver).Receiver(name, args)
