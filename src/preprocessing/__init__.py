from __future__ import print_function, absolute_import

def init_dataset(name, **kwargs):
    """Initializes an image dataset."""
    from dataset.market1501 import MARKET
    from dataset.msmt17 import MSMT17, MSMT17Extra
    from dataset.duke import DUKE

    __datasets = {
        'market1501': MARKET,
        'dukemtmcreid': DUKE,
        'msmt17': MSMT17,
        'msmt17extra': MSMT17Extra,
    }

    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __datasets[name](**kwargs)
