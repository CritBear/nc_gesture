
def get_dataset(name='nc_mocap'):
    if name == 'nc_mocap':
        from .nc_mocap import NCMocapDataset
        return NCMocapDataset


def get_datasets(dict_options):
    name = dict_options['dataset']

    data = get_dataset(name)
    dataset = data(split='train', **dict_options)

    train_dataset = dataset

    from copy import copy
    test_dataset = copy(train_dataset)
    test_dataset.split = test_dataset

    datasets = {
        'train': train_dataset,
        'test': test_dataset
    }

    dataset.update_parameters(dict_options)

    return datasets