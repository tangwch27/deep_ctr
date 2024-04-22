import torch
import torch.nn as nn

from deep_ctr.models import DeepNetwork


# much better than v1 for simulation loss
class CrossLayerV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossLayerV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: batch_size, input_dim
        x0 = x  # Initial input

        # Perform cross layer operations
        for layer in self.cross_layers:
            # Apply cross layer operation
            # batch_size, input_dim
            x = x0 * layer(x) + x

        return x


# bad simulation loss with high num_layers
class CrossLayer(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossLayer, self).__init__()
        self.num_layers = num_layers
        # self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)])
        # self.biases = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)])

        self.weights = nn.Parameter(torch.randn(num_layers, input_dim, 1))
        self.biases = nn.Parameter(torch.randn(num_layers, input_dim, 1))

    def forward(self, x):
        # x: batch_size, input_dim
        output = x  # Initial input
        # x_{l+1} = x_0 x_l'w_l + b_l + x_l = f(x_l, w_l, b_l) + x_l, w_l b_l
        # Perform cross layer operations
        for i in range(self.num_layers):
            # Apply cross layer operation
            # [batch_size, input_dim, 1] * [batch_size, 1, input_dim] * [1, input_dim, 1] + [1, input_dim, 1]
            output = (x.unsqueeze(-1) @ output.unsqueeze(1) @ self.weights[i].unsqueeze(0)
                      + self.biases[i].unsqueeze(0)).squeeze(-1) + output
        return output


class DCN(nn.Module):
    def __init__(self, m, n, k, dense_fea_dim, cross_layers, deep_hidden_units, dcn_ver=1, output_dim=1):
        super(DCN, self).__init__()
        input_dim = m * k + dense_fea_dim
        self.embd_layer = nn.Embedding(n, k)
        if dcn_ver == 1:
            self.cross_layers = CrossLayer(input_dim, cross_layers)
        else:
            self.cross_layers = CrossLayerV2(input_dim, cross_layers)
        self.deep_layers = DeepNetwork(input_dim, deep_hidden_units)
        self.output_layer = nn.Linear(input_dim + deep_hidden_units[-1], output_dim)

    def forward(self, cat_features, dense_features):
        # cat_features: batch_size, m
        # dense_features: batch_size, dense_fea_dim
        embd = self.embd_layer(cat_features)  # batch_size, input_dim, k
        embd_flat = embd.view(embd.size(0), -1)
        x = torch.cat([embd_flat, dense_features], dim=1)

        # Cross Network
        x_cross = self.cross_layers(x)

        # Deep Network
        x_deep = self.deep_layers(x)

        # Concatenate Cross and Deep outputs
        x_concat = torch.cat([x_cross, x_deep], dim=1)

        # Output layer
        output = torch.sigmoid(self.output_layer(x_concat).squeeze())

        return output


if __name__ == "__main__":
    # Example usage of DCN
    sample_cnt = 10

    # input dense features
    input_dense_dim = 15
    input_dense_data = torch.randn(sample_cnt, input_dense_dim)

    # sparse feature
    input_fea_dim = 8
    cat_fea_dim = 128
    latent_k = 10  # Number of latent factors
    input_cat_data = torch.randint(0, cat_fea_dim, (sample_cnt, input_fea_dim))

    # structure
    cross_layer_cnt = 2
    deep_layers_list = [64, 32]
    dcn_v = 2

    dcn_model = DCN(input_fea_dim, cat_fea_dim, latent_k, input_dense_dim, cross_layer_cnt, deep_layers_list, dcn_v)
    optimizer = torch.optim.Adam(dcn_model.parameters(), lr=0.001)

    labels = torch.randint(0, 2, (sample_cnt,)).float()
    criterion = nn.BCELoss()
    # Training loop
    print(f"Start to train the DCN model, version={dcn_v}")
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        dcn_model.train()
        optimizer.zero_grad()
        dcn_output = dcn_model(input_cat_data, input_dense_data)
        loss = criterion(dcn_output, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
