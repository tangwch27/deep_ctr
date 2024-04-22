import torch
import torch.nn as nn


# DEEP
class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_unit_list, dropout=0.1):
        super(DeepNetwork, self).__init__()
        len_layer = len(hidden_unit_list)
        assert len_layer > 0
        input_layer = nn.Linear(input_dim, hidden_unit_list[0])
        self.layers = nn.ModuleList([input_layer])
        for i in range(1, len_layer):
            self.layers.append(nn.Linear(hidden_unit_list[i - 1], hidden_unit_list[i]))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        return x


# FM
# The model is capable of estimating interactions between any two features in a sparse setting efficiently,
# making it powerful for dealing with categorical data in scenarios like collaborative filtering.
class FactorizationMachine(nn.Module):
    """
    n: int - number of unique feature tokens
    k: int - number of latent factors
    y_fm = <w, x> + sum_{i,j, 1<=i<j<=N} <V_i, V_j> x_i * x_j
         = <w, x> + 0.5 sum_{k} { [sum_{1<=i<=N} V_{ik}*x_{i}]^2 - sum_{1<=i<=N}V_{ik}^2*x^2_{i} }
    cross term: square of sum - sum of square
    """

    def __init__(self, n, k):
        super(FactorizationMachine, self).__init__()
        # Initialize the weight for linear terms
        self.linear = nn.Embedding(n, 1)
        # Initialize the weights for pairwise interaction terms
        self.V = nn.Embedding(n, k)

    """
    x: LongTensor of size (batch_size, input_dim) - list of **index** of each feature category
    """

    def forward(self, x):
        # linear part
        # batch_size, m, 1 -> batch_size, 1
        linear_part = self.linear(x).sum(dim=1)  # torch.sum(self.linear(x), dim=1, keepdim=True)

        # cross interaction part: batch_size, input_dim, k
        embd = self.V(x)
        square_of_sum = (embd.sum(dim=1)) ** 2  # = torch.sum(embd, dim=1, keepdim=True) ** 2
        sum_of_square = (embd ** 2).sum(dim=1)  # = torch.sum(embd ** 2, dim=1, keepdim=True)

        interaction_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        return torch.sigmoid(linear_part + interaction_part), linear_part, interaction_part


# Attention FM
class FMAttentionLayer(nn.Module):
    def __init__(self, embd_dim, attn_factor):
        super(FMAttentionLayer, self).__init__()

        self.attn_fc = nn.Sequential(
            nn.Linear(embd_dim, attn_factor),
            nn.ReLU(),
            nn.Linear(attn_factor, 1)
        )

    def forward(self, interactions):
        # interactions: batch_size, num_field, num_field, embd_dim
        # Compute the attention scores for each interaction
        # batch_size, num_field, num_field, 1
        attn_scores = self.attn_fc(interactions)
        attn_probs = torch.softmax(attn_scores, dim=2)
        return attn_probs


class AFM(nn.Module):
    # refer to the deep_fm
    # n: number of unique feature tokens
    # k: number of latent factors
    def __init__(self, n, k, attn_factor):
        super(AFM, self).__init__()
        # Initialize the weight for linear terms
        self.linear = nn.Embedding(n, 1)
        # Initialize the weights for pairwise interaction terms
        self.V = nn.Embedding(n, k)
        self.attention_layer = FMAttentionLayer(k, attn_factor)  # Attention mechanism for interactions
        self.fc_final = nn.Linear(k, 1)  # Final prediction layer

    def attn_interaction(self, embd):
        # Pairwise Interactions
        # Attentional Pooling
        # [batch_size, num_fields, 1, k] * [batch_size, 1, num_field, k] => [batch_size, num_field, num_field, k]
        pairwise_interactions = embd.unsqueeze(2) * embd.unsqueeze(1)  # Outer product

        # attention_weights: [batch_size, num_field, num_fields, 1]
        attention_weights = self.attention_layer(pairwise_interactions)  # Get attention weights for each pair
        # weighted sum of interactions: [batch_size, num_field, k]
        weighted_interactions = torch.sum(attention_weights * pairwise_interactions, dim=2)
        # sum over fields: [batch_size, k]
        weighted_embd = torch.sum(weighted_interactions, dim=1)

        # Final Prediction
        # batch_size, num_field => batch_size, 1 => batch_size
        # to replace the interaction_part in the normal FM model
        total_interaction = self.fc_final(weighted_embd)
        return total_interaction

    def forward(self, x):
        # Linear Part
        linear_part = self.linear(x).sum(dim=1)

        # batch_size, num_fields, k
        embeddings = self.V(x)
        interaction_part = self.attn_interaction(embeddings)

        # Adding up the components
        result = torch.sigmoid(linear_part + interaction_part).squeeze(1)
        return result


# DeepFM
class DeepFM(nn.Module):
    # m: input_dim, categorical features -> number of distinct feature fields
    # n: number of unique cat-feature tokens
    def __init__(self, m, n, k, deep_hidden_units, dense_fea_dim=0, dropout=0.1):
        super(DeepFM, self).__init__()
        self.fm = FactorizationMachine(n, k)
        self.deep = DeepNetwork(m * k + dense_fea_dim, deep_hidden_units, dropout)
        # concat Deep and FM part and then fc
        self.fc = nn.Linear(deep_hidden_units[-1] + 2, 1)  # deep output, fm linear and interaction part

    # x: categorical features
    def forward(self, cat_features, dense_features):
        _, linear_part, interaction_part = self.fm(cat_features)
        # take note
        embd = self.fm.V(cat_features)  # batch_size, input_dim, k
        embd_flat = embd.view(embd.size(0), -1)
        deep_input = torch.cat([embd_flat, dense_features], dim=1)
        deep_part = self.deep(deep_input)

        concat_fc_input = torch.cat([linear_part, interaction_part, deep_part], dim=1)
        result = torch.sigmoid(self.fc(concat_fc_input).squeeze())
        return result


if __name__ == "__main__":
    # Example usage of DEEP
    sample_cnt = 10
    input_fea_dim = 8
    layers = [64, 32, 2]
    model = DeepNetwork(input_fea_dim, layers)
    # input dense features
    input_dense_data = torch.randn(sample_cnt, input_fea_dim)
    deep_output = model(input_dense_data)
    # print(f"Deep model output:\n {deep_output}")  # batch_size * layers[-1]

    # Example usage of FM
    cat_fea_dim = 128
    latent_k = 10  # Number of latent factors
    model = FactorizationMachine(cat_fea_dim, latent_k)
    # Example data: 10 batch, 8 features, avg fea_dim = 128/8=16
    # e.g. on average each feature has 16 categories, [0, 16), [16, 32), ...
    input_cat_data = torch.randint(0, cat_fea_dim, (sample_cnt, input_fea_dim))
    fm_output = model(input_cat_data)  # sigmoid output, linear part, interaction part
    # print(f"FM model output:\n {fm_output[0]}")
    labels = torch.randint(0, 2, (sample_cnt,)).float()
    criterion = nn.BCELoss()
    num_epochs = 10

    # Example usage of AFM
    attn_hidden = 64
    attn_fm_model = AFM(cat_fea_dim, latent_k, attn_hidden)

    attn_fm_output = attn_fm_model(input_cat_data)
    optimizer = torch.optim.Adam(attn_fm_model.parameters(), lr=0.003)
    # Training loop
    print("Start to train the Attention FM model")
    for epoch in range(num_epochs):
        attn_fm_model.train()
        optimizer.zero_grad()
        attn_fm_output = attn_fm_model(input_cat_data)
        loss = criterion(attn_fm_output, labels)
        loss.backward()
        optimizer.step()
        print(f"AFM Epoch {epoch + 1}: Loss = {loss.item()}")

    # Example usage of DeepFM
    input_dense_dim = 15

    deep_layers_list = [64, 32, 5]  # Deep layers configuration
    deep_fm_model = DeepFM(input_fea_dim, cat_fea_dim, latent_k, deep_layers_list, input_dense_dim)
    input_dense_data = torch.randn(sample_cnt, input_dense_dim)
    deep_fm_output = deep_fm_model(input_cat_data, input_dense_data)
    # print(f"DeepFM model output:\n {deep_fm_output}")
    optimizer = torch.optim.Adam(deep_fm_model.parameters(), lr=0.003)
    print("Start to train the DeepFM model")

    # Training loop
    for epoch in range(num_epochs):
        deep_fm_model.train()
        optimizer.zero_grad()
        deep_fm_output = deep_fm_model(input_cat_data, input_dense_data)
        loss = criterion(deep_fm_output, labels)
        loss.backward()
        optimizer.step()
        print(f"DeepFM Epoch {epoch + 1}: Loss = {loss.item()}")
