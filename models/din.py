import torch
import torch.nn as nn

from deep_ctr.layers import SwishGLU

'''
Deep Interest Network (DIN) is for capturing user interest based on historical behaviors. 
It adjusts the representation of user interests as per the candidate items through the use of an attention mechanism.

Key component of DIN: the attention mechanism that dynamically computes the user's interest representation by 
considering the relevance of historical behaviors to the target item.
'''


class ActivationUnit(nn.Module):
    def __init__(self, embd_dim, attn_hidden_dim, dropout=0.1):
        super(ActivationUnit, self).__init__()
        # ActivationUnit Layer: To compute the weighted sum of past behaviors.
        # input: target item embd, hist item embd, out product between the two
        self.attention_v2 = nn.Sequential(
            nn.Linear(embd_dim * 3, attn_hidden_dim),
            SwishGLU(attn_hidden_dim, embd_dim),
            nn.Dropout(dropout),
            nn.Linear(embd_dim, 1)
        )

        self.attention_v1 = nn.Sequential(
            nn.Linear(embd_dim * 3, attn_hidden_dim),
            # nn.ReLU(), nn.LeakyReLU(), PRelu allow the slop alpha to be learned during training
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden_dim, 1)
        )

    def forward(self, item_embd, hist_embd):
        # attention between target item and hist item
        # item_embd: batch_size, 1, embd -> batch_size, hist_cnt, embd_dim
        # to facilitate element-wise operations between the target item and each historical item
        exp_item_embd = item_embd.unsqueeze(1).expand_as(hist_embd)
        # print(item_embd.size(), item_embed.size(), hist_embd.size())
        # outer product in the Activation Unit from the paper:
        # include element-wise multiplication between the target and hist item embedding as an explicit interaction term
        interaction = exp_item_embd * hist_embd  # Hadamard product. Further can add a subtraction part
        # batch_size, hist_cnt, embd * 3
        attention_input = torch.cat([exp_item_embd, hist_embd, interaction], dim=-1)
        # three  versions of attention mechanisms
        # batch_size, hist_cnt, 1
        # attention_weight = self.attention_v1(attention_input)
        attention_weight = self.attention_v2(attention_input)

        # softmax to norm to the prob
        # batch_size, hist_cnt, 1
        # from the paper: norm with softmax on the output is abandoned to reserve the intensity of user interests.
        attention_prob = torch.softmax(attention_weight, dim=1)
        # [batch_size, hist_cnt, 1] * [batch_size, hist_cnt, embd_dim] => sum(dim=1) => batch_size, embd
        user_interest = (attention_prob * hist_embd).sum(dim=1)
        return user_interest, attention_weight


class AttentionLayer(nn.Module):
    # attention mechanism for adapting interest representation
    # better than DIN to use DNN
    def __init__(self, embd_dim):
        super(AttentionLayer, self).__init__()
        self.embd_dim = embd_dim
        self.W_q = nn.Linear(embd_dim, embd_dim)
        self.W_k = nn.Linear(embd_dim, embd_dim)
        self.W_v = nn.Linear(embd_dim, embd_dim)

    def forward(self, raw_query, raw_keys):
        # batch_size, embd_dim
        query = self.W_q(raw_query)
        # batch_size, seq_len, embd_dim
        keys = self.W_k(raw_keys)
        values = self.W_v(raw_keys)

        # batch_size, 1, seq_len
        attn_scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1)) / self.embd_dim ** 0.5
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # batch_size, 1, seq_len * batch_size, seq_len, embd => batch_size, 1, embd => batch_size, embd
        output = torch.matmul(attn_probs, values).squeeze(1)  # sum(1)
        return output


class DeepInterestNetwork(nn.Module):
    def __init__(self, user_size, item_size, embd_dim, attn_hidden_dim, context_dim, dropout=0.1):
        super(DeepInterestNetwork, self).__init__()
        # Embedding layers: For users, items, and historical behaviors.
        self.user_embd = nn.Embedding(user_size, embd_dim)
        self.item_embd = nn.Embedding(item_size, embd_dim)
        # other context features' dimension
        self.activation_unit = ActivationUnit(embd_dim, attn_hidden_dim)

        # directly use attention mechanism
        self.attention = AttentionLayer(embd_dim)
        # Fully Connected layers: To predict the outcome based on the combined representation of
        # the user's current interest and other features.
        # user interest, user embd, item embd, context
        fc_input_dim = embd_dim + embd_dim + embd_dim + context_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_id, item_id, hist_item_ids, context_feature):
        # user, item: batch_size, embd
        # hist_item: batch_size, hist_cnt, embd
        user_embd = self.user_embd(user_id)
        item_embd = self.item_embd(item_id)
        hist_embd = self.item_embd(hist_item_ids)

        # batch_size, embd
        user_interest_v1, _ = self.activation_unit(item_embd, hist_embd)
        # or can directly compute user_interest by AttentionLayer
        user_interest_v2 = self.attention(item_embd, hist_embd)

        # combine user embd, user interest embd and others
        combined_embd_v1 = torch.cat([user_embd, user_interest_v1, item_embd, context_feature], dim=1)
        combined_embd_v2 = torch.cat([user_embd, user_interest_v2, item_embd, context_feature], dim=1)
        # batch_size, 1
        fc_output = self.fc_layers(combined_embd_v2)
        # squeeze from batch_size, 1 to batch_size to match the label size
        return torch.sigmoid(fc_output.squeeze())


if __name__ == "__main__":
    # Simplified Dataset Example
    # User IDs, Item IDs (target), and Historical Items (behavior)
    batch_size = 3
    user_cnt = 5
    item_cnt = 10
    users = torch.arange(0, batch_size)
    # item = 0 for padding
    items = torch.randint(1, item_cnt, (batch_size,))
    d_model = 32
    interest_hidden_unit = 32

    # 0 is padding token (history item)
    hist_items = torch.LongTensor([
        [1, 5, 4, 0],
        [2, 6, 9, 8],
        [3, 7, 0, 0]
    ])
    dense_fea_dim = 15
    dense_feature = torch.randn(3, dense_fea_dim)

    # whether user like the target item or not
    labels = torch.FloatTensor([1, 0, 1])

    # Model Training
    model = DeepInterestNetwork(user_cnt, item_cnt, d_model, interest_hidden_unit, dense_fea_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    criterion = nn.BCELoss()
    print("Start to train the DIN model")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(users, items, hist_items, dense_feature)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # dummy eval step
        model.eval()
        with torch.no_grad():
            # print(f"Validation Loss: {val_loss.item()}")
            pass

'''
nn.BCELoss()
- Expects the output to be probabilities (after applying a sigmoid activation), 
and typically the model should have one output neuron per instance.
- Requires targets in the same shape as the outputs, i.e., each target is a binary value (0 or 1).

nn.CrossEntropyLoss()
- Expects raw scores for each class (logits), and typically the model should have as many output neurons as there are classes.
- Requires targets to be class indices (0 to C-1, where C is the number of classes), not one-hot encoded vectors.
'''
