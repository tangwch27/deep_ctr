import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deep_ctr.models import DeepNetwork, EmbeddingLayer


class DSSMSingleTower(nn.Module):
    def __init__(self, obj_size, obj_embd_dim, obj_dense_fea_dim, hidden_unit_list, fc_output_dim):
        super(DSSMSingleTower, self).__init__()
        self.obj_embd = EmbeddingLayer(obj_size, obj_embd_dim)
        input_dim = obj_embd_dim + obj_dense_fea_dim

        self.deep_net = DeepNetwork(input_dim, hidden_unit_list)
        self.final_fc = nn.Linear(hidden_unit_list[-1], fc_output_dim)

    def forward(self, obj_id, dense_feature):
        input_feature = torch.cat([self.obj_embd(obj_id), dense_feature], dim=-1)
        embd_out = self.final_fc(self.deep_net(input_feature))
        return F.normalize(embd_out, p=2, dim=-1)


class DSSM(nn.Module):
    def __init__(self, item_size, user_size, item_embd_dim, user_embd_dim, item_dense_fea_dim, user_dense_fea_dim,
                 item_hidden_list, user_hidden_list, fc_output_dim):
        super(DSSM, self).__init__()
        self.item_tower = DSSMSingleTower(item_size, item_embd_dim, item_dense_fea_dim, item_hidden_list, fc_output_dim)
        self.user_tower = DSSMSingleTower(user_size, user_embd_dim, user_dense_fea_dim, user_hidden_list, fc_output_dim)

    def forward(self, item_id, user_id, item_dense_feature, user_dense_feature):
        # user: batch_size, embd_dim
        # item: batch_size, seq_len, embd_dim
        item_out = self.item_tower(item_id, item_dense_feature)
        user_out = self.user_tower(user_id, user_dense_feature)
        sim = torch.matmul(user_out.unsqueeze(1), item_out.transpose(-2, -1)).squeeze(1)
        # batch_size, seq_len
        return sim


if __name__ == "__main__":
    def generate_full_data(user_size, item_size, num_negatives, user_dense_fea_dim, item_dense_fea_dim):
        users = torch.randint(0, user_size, (user_size,))
        items = torch.randint(0, item_size, (user_size, num_negatives + 1))

        user_dense_fea = torch.randn(user_size, user_dense_fea_dim)
        item_dense_fea = torch.randn(user_size, num_negatives + 1, item_dense_fea_dim)
        return TensorDataset(users, items, user_dense_fea, item_dense_fea)


    def info_nce_loss(sim_score, temperature=0.1):
        """
        Compute the InfoNCE loss given similarities between the user representations and
        the concatenated positive and negative item representations.
    
        similarities: Tensor of shape [batch_size, 1 + num_negatives]
        where similarities[:, 0] should be the similarities with the positive examples.
    
        temperature: A scaling factor (often called tau in literature) used to control
        the sharpness of the distribution.
    
        The function expects that the positive examples are at index 0 along dimension 1.
        """
        # Scale the similarities by the temperature
        logits = sim_score / temperature
        # Zero index is positive, all others are negatives
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        # The CrossEntropyLoss automatically applies log_softmax on the logits
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels)

    # sample
    batch_size = 32
    num_item = 200
    num_user = 100
    num_neg = 3
    user_dense_dim = 10
    item_dense_dim = 20
    embd_dim = 32

    # Create the dataset
    dataset = generate_full_data(num_user, num_item, num_neg, user_dense_dim, item_dense_dim)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    item_fc_layers = [64, 32]
    user_fc_layers = [32, 16]
    fc_out_dim = 8

    model = DSSM(num_item, num_user, embd_dim, embd_dim, item_dense_dim, user_dense_dim,
                 item_fc_layers, user_fc_layers, fc_out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer.zero_grad()

    print("Start to train the DSSM model")
    batch_accumulate = 2
    for epoch in range(10):
        model.train()
        total_loss = 0
        batch_idx = 0
        for user_ids, item_ids, user_features, item_features in loader:
            batch_idx += 1
            # Forward pass
            similarities = model(item_ids, user_ids, item_features, user_features)

            # Compute InfoNCE loss
            loss = info_nce_loss(similarities)
            loss.backward()
            total_loss += loss.item()

            if batch_idx % batch_accumulate == 0:
                optimizer.step()  # Update weights
                optimizer.zero_grad()  # Reset gradients only after N batches

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(loader)}')

'''
Applying optimizer.zero_grad() every N batches instead of every batch is a technique sometimes used to simulate a 
larger effective batch size or to stabilize the training updates. 
This method accumulates the gradients over multiple mini-batches and then updates the parameters once after 
these N mini-batches have been processed.
'''