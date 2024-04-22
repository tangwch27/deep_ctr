import torch
import torch.nn as nn

from deep_ctr.models import ItemEmbeddingLayer, ActivationUnit, DeepInterestEvolutionNetwork, DeepNetwork

# Deep Interest with Hierarchical Attention Network (DHAN)

'''
For HAN, item embedding and item weights are grouped based on their attributes to generate each attribute-level feature representation
'''


class DHAN(nn.Module):
    def __init__(self, item_size, embd_dim, gru_hidden_unit, activate_hidden_dim, mid_layer_list):
        super(DHAN, self).__init__()
        # embedding layer
        self.item_embd = ItemEmbeddingLayer(item_size, embd_dim)
        self.activation_unit = ActivationUnit(embd_dim, activate_hidden_dim)
        # dien and multi-head block
        self.dien_layer = DeepInterestEvolutionNetwork(embd_dim, gru_hidden_unit)

        # fc
        # attn module2, attn_module1, dien_embd, target item embd
        fc_input_dim = embd_dim + embd_dim + activate_hidden_dim + embd_dim
        self.fc_layers = DeepNetwork(fc_input_dim, mid_layer_list)
        self.final_layer = nn.Linear(mid_layer_list[-1], 1)

    def attribute_group(self, hist_embd, hist_weight, hist_attribute):
        # hist_embd: [batch_size, hist_cnt, embd_dim]
        # hist_weight: [batch_size, hist_cnt, 1]
        # hist_attribute: [batch_size, hist_cnt], k distinct attributes, sum_k hist_cnt_k = hist_cnt
        # return a tensor with size [batch_size, k, embd_dim], each embd at k = weighed sum of hist_embd_k
        batch_size, hist_cnt, embd_dim = hist_embd.size()
        unique_attrs = torch.unique(hist_attribute, sorted=True)
        k = len(unique_attrs)

        # prepare the output tensor
        output = torch.zeros(batch_size, k, embd_dim, device=hist_embd.device, dtype=hist_embd.dtype)
        for i, attr in enumerate(unique_attrs):
            # batch_size, hist_cnt, 1
            mask = (hist_attribute == attr).unsqueeze(-1)

            # use the mask to select embeddings and apply weights
            selected_embds = hist_embd * mask
            selected_weights = hist_weight * mask
            # output[:, i, :] = (selected_weights * selected_embds).sum(dim=1) / (selected_weights.sum(dim=1) + 1e-5)
            output[:, i, :] = (
                    torch.softmax(selected_weights.masked_fill(mask == 0, -1e9), dim=1) * selected_embds).sum(dim=1)
        return output

    def forward(self, item_id, hist_item_ids, hist_item_attributes):
        item_embd = self.item_embd(item_id)
        hist_item_embd = self.item_embd(hist_item_ids)
        # batch_size, embd_dim
        interest_evolution = self.dien_layer(item_embd, hist_item_embd)

        # Attention Module 1
        # batch_size, hist_cnt
        # user_interest is the softmax normalized, user_interest_v2 is just weighted sum
        user_interest_l1, activation_weights_l1 = self.activation_unit(item_embd, hist_item_embd)
        user_interest_l1_v2 = (activation_weights_l1 * hist_item_embd).sum(dim=1)

        # batch_size, k, embd_dim
        hist_cluster_embd = self.attribute_group(hist_item_embd, activation_weights_l1, hist_item_attributes)

        # Attention Module 2
        # batch_size, embd
        user_interest_l2, activation_weights_l2 = self.activation_unit(item_embd, hist_cluster_embd)
        user_interest_l2_v2 = (activation_weights_l2 * hist_cluster_embd).sum(dim=1)

        # concat
        combined_embd = torch.cat([user_interest_l1_v2, user_interest_l2_v2, interest_evolution, item_embd], dim=1)
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
    attr_cnt = 2

    # 0 is padding token (history item)
    hist_items = torch.LongTensor([
        [1, 5, 4, 0],
        [2, 6, 9, 8],
        [3, 7, 0, 0]
    ])
    hist_item_attrs = torch.randint(0, attr_cnt+1, (sample_cnt, 4))
    embd_dims = 32
    gru_hidden_dims = 16
    hidden_dims = 16
    n_head = 2
    fc_mid_layer = [64]

    # whether user like the target item or not
    labels = torch.FloatTensor([1, 0, 1])

    # Model Training
    model = DHAN(item_cnt, embd_dims, gru_hidden_dims, hidden_dims, fc_mid_layer)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99), weight_decay=1e-5)
    criterion = nn.BCELoss()
    print("Start to train the DHAN model")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(items, hist_items, hist_item_attrs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # dummy eval step
        model.eval()
        with torch.no_grad():
            # print(f"Validation Loss: {val_loss.item()}")
            pass
