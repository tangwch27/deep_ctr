import torch
import torch.nn as nn

from deep_ctr.models import DeepNetwork, ItemEmbeddingLayer, DeepInterestEvolutionNetwork

"""
Search-based Interest Model (SIM)
SIM follows from MIMN in attempting to use long user behaviour sequence (up to 1000 items)
## Key Ideas
General Search Unit (GSU) 
cut off the length of raw sequential behaviours to meet the strict limitation of time and computation. 
This would filter the noise from the long term UB sequence.

1. Soft Search
    Do embedding look up first and use the inner product of user behaviour embedding and candidate embedding as the relevance score. 
    Use sublinear time maximum inner product search method (ALSH) to search the top-K behaviours with target item. 
    A separate training module for soft search training is implemented with its own auxiliary loss.
    
2. Hard Search
    Simple idea - choose only behaviours belonging to the same category (can be item category, intention/NE cluster, etc)

Soft search may have better offline performance, whereas hard search balances the performance gain and resource consumption
Build an two-level structured index for each user - user behavior tree (UBT). UBT is implemented as an distributed system 
follows the Key-Key-Value data structure: the first key is user id, the second keys are category ids and 
the last values are the specific behavior items that belong to each category. 
Take the category of target item as our hard-search query.


Exact Search Unit (ESU) - takes the filtered behaviours and further capture the precise user interest.
Implemented as multi-head attention to capture diverse user interest.
"""


# soft search
# distributions of long-term and short-term data are different -
# train soft-search model under an auxiliary CTR prediction task based on long-term behavior data
class SIMSoftSearchModel(nn.Module):
    def __init__(self, embd_dim, fc_layers_list, dropout=0.1):
        super(SIMSoftSearchModel, self).__init__()
        len_layer = len(fc_layers_list)
        assert len_layer > 0
        assert fc_layers_list[-1] == 1
        self.embd_dim = embd_dim
        # Query (target_item) and key (hist_item) transformations for attention
        # introduce more parameters
        self.query_transform = nn.Linear(embd_dim, embd_dim)
        self.key_transform = nn.Linear(embd_dim, embd_dim)

        # refer to deep_fm.DeepNetwork
        self.fc_layers = nn.ModuleList([nn.Linear(embd_dim * 2, fc_layers_list[0])])
        for i in range(1, len_layer):
            self.fc_layers.append(nn.Linear(fc_layers_list[i - 1], fc_layers_list[i]))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    # or can directly use dien.AttentionLayer
    def attention(self, item_embd, hist_item_embd):
        # transform
        # note that save the value transform here
        item_embd = self.query_transform(item_embd)
        hist_item_embd = self.key_transform(hist_item_embd)
        # inner product: batch_size, 1, seq_len
        attn_scores = torch.matmul(item_embd.unsqueeze(1), hist_item_embd.transpose(-2, -1)) / self.embd_dim ** 0.5
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # sum pooling: batch_size, 1, embd_dim => batch_size, embd_dim
        output = torch.matmul(attn_probs, hist_item_embd).squeeze(1)
        return output

    def forward(self, item_embd, hist_item_embd):
        # batch_size, embd_dim
        attn_embd = self.attention(item_embd, hist_item_embd)
        input_embd = torch.cat([item_embd, attn_embd], dim=1)
        for fc in self.fc_layers:
            input_embd = self.dropout(self.prelu(fc(input_embd)))
        return torch.sigmoid(input_embd.squeeze())


# Create the model
SIMSoftSearchModel(embd_dim=10, fc_layers_list=[20, 10, 1], dropout=0.1)


# exact search
# multi-head over dien.AttentionLayer
class MultiHeadAttentionLayer(nn.Module):
    # attention mechanism for adapting interest representation
    def __init__(self, embd_dim, num_head):
        super(MultiHeadAttentionLayer, self).__init__()
        assert embd_dim % num_head == 0
        self.embd_dim = embd_dim
        self.num_head = num_head
        self.d_k = embd_dim // num_head

        self.W_q = nn.Linear(embd_dim, embd_dim)
        self.W_k = nn.Linear(embd_dim, embd_dim)
        self.W_v = nn.Linear(embd_dim, embd_dim)

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, len_word, d_model = x.size()
        # transpose along len_word and num_head, to allow later matrix operations
        # batch_size, num_head, len_word, head_embd_dim
        return x.view(batch_size, len_word, self.num_head, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, _ = x.size()
        # batch_size, num_head * d_k = batch_size, embd_dim
        return x.view(batch_size, -1)

    def scaled_dot_product(self, query, keys, values):
        # softmax(QK^)V
        # query: batch_size, num_head, d_k
        # keys: batch_size, num_head, seq_len, d_k
        # attn_scores: batch_size, num_head, 1, seq_len

        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / self.d_k ** 0.5
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # batch_size, num_head, 1, seq_len * batch_size, num_head, seq_len, d_k =>
        # batch_size, num_head, 1, d_k => batch_size, num_head, d_k
        output = torch.matmul(attn_probs, values).squeeze(2)
        return output

    def forward(self, raw_query, raw_keys, raw_values):
        # batch_size, 1, embd_dim
        query = self.split_heads(self.W_q(raw_query).unsqueeze(1))
        # batch_size, seq_len, embd_dim
        keys = self.split_heads(self.W_k(raw_keys))
        values = self.split_heads(self.W_v(raw_values))

        # batch_size, num_head, d_k
        attn_embd = self.scaled_dot_product(query, keys, values)
        return self.combine_heads(attn_embd)


class SIM(nn.Module):
    def __init__(self, item_size, embd_dim, hidden_dim, num_head, mid_layer_list):
        super(SIM, self).__init__()
        # embedding layer
        self.item_embd = ItemEmbeddingLayer(item_size, embd_dim)
        # dien and multi-head block
        self.dien_layer = DeepInterestEvolutionNetwork(embd_dim, hidden_dim)
        self.multi_head_attn_layer = MultiHeadAttentionLayer(embd_dim, num_head)
        # fc
        # target item embd, attn_embd, dien_embd
        fc_input_dim = embd_dim + embd_dim + hidden_dim
        self.fc_layers = DeepNetwork(fc_input_dim, mid_layer_list)
        self.final_layer = nn.Linear(mid_layer_list[-1], 1)

    def forward(self, item_id, hist_item_ids):
        item_embd = self.item_embd(item_id)
        hist_item_embd = self.item_embd(hist_item_ids)
        interest_evolution = self.dien_layer(item_embd, hist_item_embd)
        attn_embd = self.multi_head_attn_layer(item_embd, hist_item_embd, hist_item_embd)
        combined_embd = torch.cat([item_embd, attn_embd, interest_evolution], dim=-1)
        # batch_size, 1
        output = self.final_layer(self.fc_layers(combined_embd))
        # squeeze from batch_size, 1 to batch_size to match the label size
        return torch.sigmoid(output.squeeze())


if __name__ == "__main__":
    # Simplified Dataset Example
    # User IDs, Item IDs (target), and Historical Items (behavior)
    sample_cnt = 3
    item_cnt = 10
    items = torch.randint(1, item_cnt, (sample_cnt,))

    # 0 is padding token (history item)
    hist_items = torch.LongTensor([
        [1, 5, 4, 0],
        [2, 6, 9, 8],
        [3, 7, 0, 0]
    ])
    embd_dims = 32
    hidden_dims = 16
    n_head = 2
    fc_mid_layer = [64]

    # whether user like the target item or not
    labels = torch.FloatTensor([1, 0, 1])

    # Model Training
    model = SIM(item_cnt, embd_dims, hidden_dims, n_head, fc_mid_layer)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99), weight_decay=1e-5)
    criterion = nn.BCELoss()
    print("Start to train the SIM model")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(items, hist_items)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # dummy eval step
        model.eval()
        with torch.no_grad():
            # print(f"Validation Loss: {val_loss.item()}")
            pass
