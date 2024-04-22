# Define settings for building and training the transformer model
from pathlib import Path


def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 20,
        'lr': 10 ** -4,
        'seq_len': 350,
        'd_model': 512,  # Dimensions of the embeddings in the Transformer.
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }


# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']  # Extracting model folder from the config
    model_basename = config['model_basename']  # Extracting the base name for model files
    model_filename = f"{model_basename}{epoch}.pt"  # Building filename
    # # Combining current directory, the model folder, and the model filename
    return str(Path('..') / model_folder / model_filename)
