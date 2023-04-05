

def get_model_and_data(dict_options):
    datasets = get_datasets(dict_options)

    model = get_gen_model(dict_options)

    return model, datasets