from autosklearn.pipeline.classification import SimpleClassificationPipeline


import ConfigSpace


def get_search_space():

    pipeline = SimpleClassificationPipeline()

    return pipeline.get_hyperparameter_search_space()


def config_sampler(
    global_seed_gen, config_id: int = None
) -> tuple[ConfigSpace.Configuration, int]:

    if config_id is None:
        random_state = next(global_seed_gen)
    else:
        for _ in range(config_id):
            random_state = next(global_seed_gen)

    search_space = get_search_space()
    search_space.seed(random_state)

    config = search_space.sample_configuration()

    for hp in search_space.get_hyperparameters():
        if "text_transformer" in hp.name:
            if hp.name in config and hasattr(hp, "default_value"):
                config[hp.name] = hp.default_value

    return config, random_state
