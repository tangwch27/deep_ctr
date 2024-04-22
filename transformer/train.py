# https://ai.plainenglish.io/building-and-training-a-transformer-from-scratch-fdbf3db00df4
# Importing library of warnings
import warnings
# Pathlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
# HuggingFace libraries
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split

from config import get_config, get_weights_file_path
from dataset import BilingualDataset
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter

# Library for progress bars in loops
from tqdm import tqdm


# Iterating through dataset to extract the original sentence and its translation
def get_all_sentences(ds, lang):
    for pair in ds:
        yield pair['translation'][lang]


# There are several Tokenization strategies. We will use the word-level tokenization to transform each word in a sentence into a token.
# Tokenizer
'''
Transformers use special tokens for specific purposes. 
-[UNK]: unknown word in the sequence.
-[PAD]: Padding token to ensure that all sequences in a batch have the same length, 
    so we pad shorter sentences with this token. We use attention masks to “tell” the model to ignore the padded tokens 
    during training since they don’t have any real meaning to the task.
-[SOS]: signal the Start of Sentence.
-[EOS]: signal the End of Sentence.
'''


# define Tokenizer
def build_tokenizer(config, ds, lang):
    # Crating a file path for the tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    # Checking if Tokenizer already exists
    if not Path.exists(tokenizer_path):

        # If it doesn't exist, we create a new one
        print("Creating a new tokenizer_path")
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))  # Initializing a new world-level tokenizer
        tokenizer.pre_tokenizer = Whitespace()  # We will split the text into tokens based on whitespace

        # Creating a trainer for the new tokenizer
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]",
                                                   "[SOS]", "[EOS]"],
                                   min_frequency=2)  # Defining Word Level strategy and special tokens

        # Training new tokenizer on sentences from the dataset and language specified

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(
            str(tokenizer_path))  # Saving trained tokenizer to the file path specified at the beginning of the function
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))  # If the tokenizer already exist, we load it
    return tokenizer  # Returns the loaded tokenizer or the trained tokenizer


'''
The get_ds function is defined to load and prepare the dataset for training and validation. 
In this function, we build or load the tokenizer, split the dataset, and create DataLoaders, 
so the model can successfully iterate over the dataset in batches. 

The result of these functions is tokenizers for the source and target languages plus the DataLoader objects.
'''


def get_ds(config):
    # Loading the train portion of the OpusBooks dataset.
    # The Language pairs will be defined in the 'config' dictionary we will build later
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Building or loading tokenizer for both the source and target languages
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Splitting the dataset for training and validation
    train_ds_size = int(0.9 * len(ds_raw))  # 90% for training
    val_ds_size = len(ds_raw) - train_ds_size  # 10% for validation
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])  # Randomly splitting the dataset

    # Processing data with the BilingualDataset class, which we will define below
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])

    # Iterating over the entire dataset and printing the maximum length found in the sentences of both the source and target languages
    max_len_src = 0
    max_len_tgt = 0
    for pair in ds_raw:
        src_ids = tokenizer_src.encode(pair['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(pair['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Creating dataloaders for the training and validadion sets
    # Dataloaders are used to iterate over the dataset in batches during training and validation
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  shuffle=True)  # Batch size will be defined in the config dictionary
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt  # Returning the DataLoader objects and tokenizers


# Training Loop
# We pass as parameters the config dictionary, the length of the vocabylary of the source language and the target language
def get_model(config, vocab_src_len, vocab_tgt_len):
    # Loading model using the 'build_transformer' function.
    # We will use the lengths of the source language and target language vocabularies, the 'seq_len', and the dimensionality of the embeddings
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    # Setting up device to run on GPU to train faster
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Creating model directory to store weights
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Retrieving dataloaders and tokenizers for source and target languages using the 'get_ds' function
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Initializing model on the GPU using the 'get_model' function
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Setting up the Adam optimizer with the specified learning rate from the '
    # config' dictionary plus an epsilon value
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0

    # Checking if there is a pre-trained model to load
    # If true, loads it
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)  # Loading model

        # Sets epoch to the saved in the state plus one, to resume from where it stopped
        initial_epoch = state['epoch'] + 1
        # Loading the optimizer state from the saved model
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # Loading the global step state from the saved model
        global_step = state['global_step']

    # Initializing CrossEntropyLoss function for training
    # We ignore padding tokens when computing loss, as they are not relevant for the learning process
    # We also apply label_smoothing to prevent overfitting
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Initializing training loop

    # Iterating over each epoch from the 'initial_epoch' variable up to
    # the number of epochs informed in the config
    for epoch in range(initial_epoch, config['num_epochs']):

        # Initializing an iterator over the training dataloader
        # We also use tqdm to display a progress bar
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        # For each batch...
        for batch in batch_iterator:
            model.train()  # Train the model

            # Loading input data and masks onto the GPU
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Running tensors through the Transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Loading the target labels onto the GPU
            label = batch['label'].to(device)

            # Computing loss between model's output and true labels
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # Updating progress bar
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Performing backpropagation
            loss.backward()

            # Updating parameters based on the gradients
            optimizer.step()

            # Clearing the gradients to prepare for the next batch
            optimizer.zero_grad()

            global_step += 1  # Updating global step count

        # We run the 'run_validation' function at the end of each epoch
        # to evaluate model performance
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
        #                lambda msg: batch_iterator.write(msg), global_step, writer)

        # Saving model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        # Writting current model state to the 'model_filename'
        torch.save({
            'epoch': epoch,  # Current epoch
            'model_state_dict': model.state_dict(),  # Current model state
            'optimizer_state_dict': optimizer.state_dict(),  # Current optimizer state
            'global_step': global_step  # Current global step
        }, model_filename)


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')  # Filtering warnings
    config = get_config()  # Retrieving config settings
    train_model(config)  # Training model with the config arguments
