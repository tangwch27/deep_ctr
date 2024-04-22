# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
# Building a Transformer with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import math


# multiple head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()  # Call initializer of nn.Module
        assert d_model % num_head == 0  # "d_model must be divisible by num_heads"

        # init dimensions
        self.d_model = d_model
        self.num_head = num_head

        # dimension of each head's key, query and value
        self.d_k = self.d_model // self.num_head

        # ful linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # softmax(Q^K / sqrt(d_k)) * v
        # Q, K, V: batch_size, num_head, len_word, head_embd_dim
        # attn_scores: batch_size, num_head, len_word, len_word
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        # This is a boolean tensor (often with the same shape as attn_scores) where
        # a value of 1 indicates positions that should be considered (i.e., unmasked)
        # and a value of 0 indicates positions that should be ignored or masked out in the attention mechanism.
        if mask is not None:
            # make these positions extremely unattractive for the softmax operation that comes after the masking,
            # effectively "ignoring" these positions because softmax of a very large negative number approaches zero.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)  # for each query, norm the attn_scores
        # batch_size, num_head, len_word, head_embd_dim
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, len_word, d_model = x.size()
        # transpose along len_word and num_head, to allow later matrix operations
        # batch_size, num_head, len_word, head_embd_dim
        return x.view(batch_size, len_word, self.num_head, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        input: batch_size, num_head, len_word, head_embd_dim
        output: combine to batch_size, len_word, embd_dim
        - moves the len_word axis to be immediately after batch_size and before num_heads,
        preparing the tensor for the merging of the head embeddings.
        -  transposing a tensor returns a view of the original tensor, which might not be contiguous in memory.
        Making the tensor contiguous rearranges the storage so that it is contiguous
        - view reshapes the tensor to combine the last two dimensions, concatenating the embeddings from all heads for each word
        """
        batch_size, _, len_word, head_embd_dim = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, len_word, self.d_model)  # or batch_size, len_word, -1

    def forward(self, Q, K, V, mask=None):
        # apply linear transformations and split head
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # perform scaled dot-production attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.W_o(self.combine_heads(attn_output))
        return attn_output


# simpler self-attention implementation
# number of network parameters are totally the same as multi-head
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        assert d_model == 0
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scale_product(self, Q, K, V):
        # Q, K, V batch_size, len_word, embd_dim
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        # torch softmax, not nn softmax
        # batch_size, len_word, len_word
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # batch_size, len_word, embd_dim
        return torch.matmul(attn_probs, V)

    def forward(self, Q, K, V):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        attn_output = self.scale_product(Q, K, V)
        attn_output = self.W_o(attn_output)
        return attn_output


# Position-wise Feed-Forward Networks
# In the context of transformer models, this feed-forward network is applied to each position separately and identically.
# It helps in transforming the features learned by the attention mechanisms within the transformer,
# acting as an additional processing step for the attention outputs.
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input/output: batch_size, len_word, embd_dim
        return self.fc2(self.relu(self.fc1(x)))  # or can use self.fc2(torch.relu(self.fc1(x)))


# Positional Encoding
# used to inject the position information of each token in the input sequence.
# It uses sine and cosine functions of different frequencies to generate the positional encoding.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        # max_len_word, 1
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # p_{i, 2j} = sin(i/10000^{2j/d}), p_{i, 2j+1} = cos(i/10000^{2j/d})
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # div_term = torch.pow(1/10000, torch.arange(0, d_model, 2).float()/d_model)

        # max_len_word, d_model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 1,max_len_word, d_model

        # register as a buffer, which means it will be part of the module's state
        # but will not be considered a trainable parameter.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # [batch, len_word, embd_dim] + [1, len_word, embd_dim]
        # if no unsqueeze above, use x + self.pe[:x.size(1), :]
        return x + self.pe[:, :x.size(1)]


# Build the encoding block
# dropout setting here
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_head)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # The dropout rate used for regularization.

    def forward(self, x, mask):
        # x: batch_size, seq_len, d_model
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Build the decoding block
class DecodeLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout):
        super(DecodeLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_head)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        # queries are from the outputs of the decoder’s self-attention sublayer,
        # and the keys and values are from the Transformer encoder outputs
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# Combine Encoder and Decoder layers to create the complete Transformer network
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_head, num_layer, d_ff, max_seq_len, dropout):
        super(Transformer, self).__init__()
        # init embeds
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layer_list = nn.ModuleList([EncoderLayer(d_model, num_head, d_ff, dropout)] * num_layer)
        self.decoder_layer_list = nn.ModuleList(
            [DecodeLayer(d_model, num_head, d_ff, dropout) for _ in range(num_layer)])

        self.fc_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # used to prevent the model from attending to irrelevant positions
    # ensure padding tokens are ignored and that future tokens are not visible during training for the target sequence.
    # The target mask additionally prevents future information from being used in the prediction,
    # which is crucial for the autoregressive property of the model
    def generate_mask(self, src, tgt):
        # src: batch_size, len_src -> batch_size, 1, 1, len_src
        # hide padding
        # unsqueeze to make it compatible with the attention mechanism's expected input dimensions
        # src attn: batch_size, num_head, seq_len, seq_len
        # padding 0s -> give any word (including pad word): attn_score with any pad word is 0
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # typically want your mask to have a shape that matches [batch_size, 1, tgt_len, tgt_len]
        # when it's going to be used in a self-attention mechanism where it is applied to an attention score matrix of
        # shape [batch_size, num_heads, tgt_len, tgt_len].
        # tgt: batch_size, len_tgt -> batch_size, 1, len_tgt, 1
        # since no padding word in the target? unsqueeze 1 -> 3
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        no_peak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & no_peak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embd = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embd = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embd
        for enc_layer in self.encoder_layer_list:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embd
        for dec_layer in self.decoder_layer_list:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        # batch_size, len_word, tgt_vocab_size
        output = self.fc_layer(dec_output)
        # ok to not apply softmax since later will apply cross entropy
        return output  # torch.softmax(output, dim=-1)


if __name__ == "__main__":
    # Train a transformer model

    # hyperparameters
    src_vocab_cnt = 100
    tgt_vocab_cnt = 100
    sample_cnt = 500

    dim_model = 64
    head_cnt = 4
    layer_cnt = 2
    dim_feedforward = 32
    max_len_word = 25
    dropout_rate = 0.1

    # creates an instance of the Transformer class, initializing it with the given hyperparameters.
    transformer = Transformer(src_vocab_cnt, tgt_vocab_cnt, dim_model, head_cnt, layer_cnt, dim_feedforward,
                              max_len_word, dropout_rate)

    # generate random sample data
    # (batch_size, seq_length)
    # used as inputs to the transformer model, simulating a batch of data with 32 examples and sequences of length 20.
    src_data = torch.randint(1, src_vocab_cnt, (sample_cnt, max_len_word))
    tgt_data = torch.randint(1, tgt_vocab_cnt, (sample_cnt, max_len_word))

    # train the model
    # ignore_index = 0, means the loss will not consider targets with an index of 0 (typically reserved for padding tokens).
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()
    print("Start to train the Transformer")
    num_epoch = 20
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        # excluding the last target token in each sequence:
        # common in sequence-to-sequence tasks where the target is shifted by one token: next word prediction
        # tgt_data[:, :-1] is used as input to the model, to predict the last token
        output_data = transformer(src_data, tgt_data[:, :-1])
        # output: flatten the batch and sequence x tgt_vocab_size
        # tgt_data: , matching the first dimension of output_data after it has been reshaped.
        # if the model receives the token at position i from tgt_data[:, :-1],
        # it should predict the token at position i+1, which is given by tgt_data[:, 1:]
        loss = criterion(output_data.contiguous().view(-1, tgt_vocab_cnt), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # Evaluate a transformer model
    # Puts the transformer model in evaluation mode. It turns off certain behaviors like dropout that are only used during training.
    transformer.eval()

    # Generate random sample validation data
    val_src_data = torch.randint(1, src_vocab_cnt, (32, max_len_word))  # (batch_size, seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_cnt, (32, max_len_word))  # (batch_size, seq_length)

    # Disables gradient computation, which is no need during validation.
    # This can reduce memory consumption and speed up computations.
    with torch.no_grad():
        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_cnt), val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")
    '''
    Save & Load the model

    # 1. save the entire model
    torch.save(transformer, 'transformer.pth')
    transformer = torch.load('transformer.pth')

    2. save only the model state_dict -- recommended
    saves the model parameters (weights and biases) and requires recreating the model structure using the class definition
    before loading the state dictionary.
    torch.save(transformer.state_dict(), 'transformer_state_dict.pth')

    Loading the model state_dict
    load_model = Transformer()  # Example class name of your transformer
    load_model.load_state_dict(torch.load('transformer_state_dict.pth'))

    3. view the model
    import torch.onnx
    torch.onnx.export(transformer, (val_src_data, val_tgt_data[:, :-1]), "model.onnx", verbose=True)
    # then can view the model using Netron Desktop App
    
    '''
