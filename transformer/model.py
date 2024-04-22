# https://ai.plainenglish.io/building-and-training-a-transformer-from-scratch-fdbf3db00df4
# pytorch
import math
import torch
import torch.nn as nn


# Input Embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # To prevent that our input embeddings become tiny,
        # we normalize them by multiplying them by the √𝑑_𝑚𝑜𝑑𝑒𝑙
        return self.embedding(x) * math.sqrt(self.d_model)


# Creating the Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = max_seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)

        # Creating a positional encoding matrix of shape (seq_len, d_model) filled with zeros
        pe = torch.zeros(max_seq_len, d_model)

        # Creating a tensor representing positions (0 to seq_len - 1)
        # Transforming 'position' into a 2D tensor['seq_len, 1']
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Creating the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in pe
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adding an extra dimension at the beginning of pe matrix for batch handling
        pe = pe.unsqueeze(0)

        # Registering 'pe' as buffer. Buffer is a tensor not considered as a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adding positional encoding to the input tensor X
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # redundant since have registered as a buffer?
        return self.dropout(x)  # Dropout for regularization


# Layer Norm
# nn.LayerNorm
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        # LN(x) = \gamma(std_norm(x)) + \beta
        # learnable scale variable
        self.gamma = nn.Parameter(torch.ones(1))
        # learnable shift variable
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mu) / (self.eps + sigma) + self.beta


class RMSNorm(nn.Module):
    def __init__(self, eps: 1e-9):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(0))

    def forward(self, x):
        x = x.squeeze(-2)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms + self.bias


# feed-forward
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # return self.dropout(self.fc2(self.relu(self.fc1(x))))
        # where to apply dropout ?
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


# multiple-head attention
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super(MultiHeadAttentionBlock, self).__init__()
        assert d_model % num_head == 0
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_head(self, x):
        # input: batch_size, seq_len, d_model
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_head, -1).transpose(1, 2).contiguous()

    def combine_head(self, x):
        # input: batch_size, num_head, seq_len, d_k
        # output: batch_size, seq_len, d_model
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def scaled_dot_product_attention(self, Q, K, V, mask=None, dropout=None):
        # Q,K,V: batch_size, num_head, seq_len, d_model
        # attn_scores: batch_size, num_head, seq_len, seq_len
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        if dropout is not None:
            attn_scores = dropout(attn_scores)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        # return batch_size, num_head, seq_len, d_k
        return torch.matmul(attn_probs, V)

    # output: batch_size, seq_len, d_model
    def forward(self, Q, K, V, mask=None):
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        attn_output = self.W_o(self.combine_head(attn_output))

        return attn_output


# Add and Norm Block
class ResidentialAddAndNorm(nn.Module):
    def __init__(self, dropout):
        super(ResidentialAddAndNorm, self).__init__()
        self.layer_norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    # add and norm
    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


# EncoderBlock
class EncoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attn = self_attn_block
        self.add_norm1 = ResidentialAddAndNorm(dropout)
        self.feed_forward = feed_forward_block
        self.add_norm2 = ResidentialAddAndNorm(dropout)

    def forward(self, x, src_mask):
        x = self.add_norm1(x, lambda x: self.self_attn(x, x, x, src_mask))  # TODO ??
        x = self.add_norm2(x, self.feed_forward)
        return x


# Building Decoder Block
class DecoderBlock(nn.Module):

    # The DecoderBlock takes in two MultiHeadAttentionBlock. One is self-attention, while the other is cross-attention.
    # It also takes in the feed-forward block and the dropout rate
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidentialAddAndNorm(dropout) for _ in range(3)])  # List of three Residual Connections with dropout rate

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-Attention block with query, key, and value plus the target language mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # The Cross-Attention block using two 'encoder_ouput's for key and value plus the source language mask. It also takes in 'x' for Decoder queries
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))

        # Feed-forward block with residual connections
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


# Encoder
# Building Encoder
# An Encoder can have several Encoder Blocks
class Encoder(nn.Module):

    # The Encoder takes in instances of 'EncoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Storing the EncoderBlocks
        self.norm = LayerNormalization()  # Layer for the normalization of the output of the encoder layers

    def forward(self, x, mask):
        # Iterating over each EncoderBlock stored in self.layers
        for layer in self.layers:
            x = layer(x, mask)  # Applying each EncoderBlock to the input tensor 'x'
        return x  # self.norm(x)  # Normalizing output -- no need?


# Building Decoder
# A Decoder can have several Decoder Blocks
class Decoder(nn.Module):

    # The Decoder takes in instances of 'DecoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Storing the 'DecoderBlock's
        self.layers = layers
        self.norm = LayerNormalization()  # Layer to normalize the output

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Iterating over each DecoderBlock stored in self.layers
        for layer in self.layers:
            # Applies each DecoderBlock to the input 'x' plus the encoder output and source and target masks
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x  # self.norm(x)  # Returns normalized output


# Building Linear Layer
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:  # Model dimension and the size of the output vocabulary
        super().__init__()
        # Linear layer for projecting the feature space of 'd_model' to the output space of 'vocab_size'
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Applying the log Softmax function to the output
        # When combined with nn.NLLLoss(), torch.log_softmax() becomes very effective as nn.NLLLoss() expects log probabilities as input.
        # This pairing is equivalent to using nn.CrossEntropyLoss() combined (unnecessary) with torch.softmax()
        return torch.log_softmax(self.proj(x), dim=-1)


# Creating the Transformer Architecture
class Transformer(nn.Module):

    # This takes in the encoder and decoder, as well the embeddings for the source and target language.
    # It also takes in the Positional Encoding for the source and target language, as well as the projection layer
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Encoder
    def encode(self, src, src_mask):
        src = self.src_embed(src)  # Apply source embeddings to the input source language
        src = self.src_pos(src)  # Apply source positional encoding to the source embeddings
        return self.encoder(src, src_mask)  # Return the source embeddings plus a source mask to prevent over-attention

    # Decoder
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)  # Apply target embeddings to the input target language (tgt)
        tgt = self.tgt_pos(tgt)  # Apply target positional encoding to the target embeddings

        # Return the target embeddings, the output of the encoder, and both source and target masks
        # The target mask ensures that the model won't 'see' future elements of the sequence
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    # Applying Projection Layer with the Softmax function to the Decoder output
    def project(self, x):
        return self.projection_layer(x)


# define a function called build_transformer, in which we define the parameters and everything we need to
# have a fully operational Transformer model for the task of machine translation.
# Building & Initializing Transformer

# Define function and its parameter, including model dimension, number of encoder and decoder stacks, heads, etc.
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      num_layer: int = 6, num_head: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Creating Embedding layers
    # Source language (Source Vocabulary to 512-dimensional vectors)
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    # Target language (Target Vocabulary to 512-dimensional vectors)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    # Creating Positional Encoding layers
    # Positional encoding for the source & target language embeddings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Creating EncoderBlocks
    encoder_blocks = []  # Initial list of empty EncoderBlocks
    for _ in range(num_layer):  # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_head, dropout)  # Self-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # FeedForward

        # Combine layers into an EncoderBlock
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)  # Append EncoderBlock to the list of EncoderBlocks

    # Creating DecoderBlocks
    decoder_blocks = []  # Initial list of empty DecoderBlocks
    for _ in range(num_layer):  # Iterate 'N' times to create 'N' DecoderBlocks (N = 6)
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_head, dropout)  # Self-Attention
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_head, dropout)  # Cross-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # FeedForward

        # Combining layers into a DecoderBlock
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block,
                                     dropout)
        decoder_blocks.append(decoder_block)  # Append DecoderBlock to the list of DecoderBlocks

    # Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating projection layer
    # Map the output of Decoder to the Target Vocabulary Space
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer by combining everything above
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            # nn.init.kaiming_uniform_(p)

    return transformer  # Assembled and initialized Transformer. Ready to be trained and validated!








