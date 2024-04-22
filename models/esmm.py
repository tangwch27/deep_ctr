import torch
import torch.nn as nn
import torch.optim as optim


# ESMM
class ESMM(nn.Module):
    # CTR and CVR task share the input embeddings
    # cat_feature_size, cat_fea_cnt: refer n, m from the FM model
    def __init__(self, cat_fea_size, embd_dim, task_hidden_unit, cat_fea_cnt, dense_fea_dim=0, dropout=0.1):
        super(ESMM, self).__init__()
        self.embedding = nn.Embedding(cat_fea_size, embd_dim)
        input_dim = cat_fea_cnt * embd_dim + dense_fea_dim
        self.ctr_layer = nn.Sequential(
            nn.Linear(input_dim, task_hidden_unit),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_unit, 1)
        )
        self.cvr_layer = nn.Sequential(
            nn.Linear(input_dim, task_hidden_unit),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_unit, 1)
        )

    # refer to the deep_fm model
    def forward(self, cat_fea_list, dense_features):
        # batch_size, feature_cnt
        embeddings = self.embedding(cat_fea_list)
        # batch_size, feature_cnt * embd_dim
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten the embeddings
        fc_input = torch.cat([embeddings, dense_features], dim=1)

        ctr_output = torch.sigmoid(self.ctr_layer(fc_input).squeeze())
        cvr_output = torch.sigmoid(self.cvr_layer(fc_input).squeeze())
        return ctr_output, cvr_output


if __name__ == "__main__":
    # Generate synthetic data
    sample_cnt = 100
    cat_feature_size = 64
    cat_feature_cnt = 4  # on average each feature have 16 cats
    input_dense_dim = 15
    embd_dim_cnt = 32
    hidden_unit = 64
    input_sparse_features = torch.randint(0, cat_feature_size, (sample_cnt, cat_feature_cnt))
    input_dense_features = torch.randn(sample_cnt, input_dense_dim)

    # collect labels
    clicks = torch.randint(0, 2, (sample_cnt,)).float()  # Binary labels
    conversions = torch.randint(0, 2, (sample_cnt,)).float() * clicks  # Only have conversions where there's a click

    # Create model instance
    model = ESMM(cat_feature_size, embd_dim_cnt, hidden_unit, cat_feature_cnt, input_dense_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCELoss()

    # Training loop
    num_epochs = 10
    cvr_loss_weight = 2
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        ctr_score, cvr_score = model(input_sparse_features, input_dense_features)
        ctr_loss = criterion(ctr_score, clicks)

        # Masking the CVR loss calculation
        valid_cvr_indices = clicks.bool()  # Indices where there was a click
        cvr_loss = criterion(cvr_score[valid_cvr_indices], conversions[valid_cvr_indices])

        # Weighted loss computation
        # Only add CVR loss if there are valid indices
        loss = ctr_loss + cvr_loss_weight * cvr_loss if valid_cvr_indices.any() else ctr_loss

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")