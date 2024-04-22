import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from deep_ctr.models import ItemEmbeddingLayer, DeepInterestEvolutionNetwork, MultiHeadAttentionLayer

'''
The Multi-gate Mixture-of-Experts (MMoE) is a machine learning model designed for multi-task learning where 
different tasks can benefit from shared representations but may require task-specific adjustments. 

The model uses several shared experts, each being a sub-network, and gates that learn to 
weigh the contributions of each expert for specific tasks.
'''


class MMoE(nn.Module):
    def __init__(self, input_dim, num_expert, expert_hidden_unit, num_task, task_hidden_units, dropout=0.1):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            num_expert (int): Number of experts in the MMoE layer.
            expert_hidden_unit (int): Number of hidden units in each expert.
            num_task (int): Number of tasks to predict.
            task_hidden_units (list of int): A list containing the sizes of each task-specific layer.
        """
        super(MMoE, self).__init__()
        self.num_expert = num_expert
        self.num_task = num_task

        # Define the experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_unit),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_unit, expert_hidden_unit),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_expert)
        ])

        # Define the gates for each task
        '''
        set bias=False: the weighting of the experts should be solely based on the information provided by
        the input features, without any constant shift or scaling.
        Forces the gates to focus purely on the relationships and interactions between the input features and 
        the expert weights. This can lead to a more interpretable model where the gate's decisions are directly 
        related to the input, without any inherent preference for one expert over another when all inputs are zero.
        '''

        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_expert, bias=False)  # Softmax over experts
            for _ in range(num_task)
        ])

        # Define task-specific layers
        # different task may have diff hidden unit
        self.task_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_unit, task_hidden_unit),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden_unit, 1)
            ) for task_hidden_unit in task_hidden_units
        ])

    def forward(self, x):
        # x: batch_size, input_dim
        # Compute expert outputs
        # expert(x): batch_size, expert_hidden_unit
        expert_outputs = [expert(x) for expert in self.experts]

        # Stack expert outputs: [batch_size, num_experts, expert_hidden_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Task-specific computations
        final_outputs = []
        for i, gate in enumerate(self.gates):
            # Gate values: batch_size, num_expert
            gate_values = F.softmax(gate(x), dim=1)

            # [batch_size, 1, num_expert] * [batch_size, num_experts, expert_hidden_dim]
            # => [batch_size, 1, expert_hidden_dim] => [batch_size, expert_hidden_dim]
            # Weighted sum of expert outputs: [batch_size, expert_hidden_dim]
            # weighted_expert_output = torch.matmul(gate_values.unsqueeze(1), expert_outputs).squeeze(1)
            weighted_expert_output = torch.einsum('be,bep->bp', gate_values, expert_outputs)

            # Task-specific operations
            task_output = self.task_layers[i](weighted_expert_output).squeeze()  # => BCEWithLogitsLoss
            # torch.sigmoid(self.task_layers[i](weighted_expert_output))
            final_outputs.append(task_output)

        return final_outputs


class SIMPlusMMoE(nn.Module):
    def __init__(self, item_size, embd_dim, dien_hidden_unit, num_head, dense_fea_dim, num_expert, expert_hidden_unit, num_task, task_hidden_units):
        super(SIMPlusMMoE, self).__init__()
        # embedding layer
        self.item_embd = ItemEmbeddingLayer(item_size, embd_dim)
        # dien and multi-head
        self.dien_layer = DeepInterestEvolutionNetwork(embd_dim, dien_hidden_unit)
        self.multi_head_attn_layer = MultiHeadAttentionLayer(embd_dim, num_head)
        # item embd, attn_embd, dien_hidden, dense_feature
        fc_input_dim = embd_dim + embd_dim + dien_hidden_unit + dense_fea_dim
        self.mmoe = MMoE(fc_input_dim, num_expert, expert_hidden_unit, num_task, task_hidden_units)

    def forward(self, item_id, hist_item_ids, dense_feature):
        item_embd = self.item_embd(item_id)
        hist_item_embd = self.item_embd(hist_item_ids)
        interest_evolution = self.dien_layer(item_embd, hist_item_embd)
        attn_embd = self.multi_head_attn_layer(item_embd, hist_item_embd, hist_item_embd)
        combined_embd = torch.cat([item_embd, attn_embd, interest_evolution, dense_feature], dim=-1)
        # batch_size, 1
        return self.mmoe(combined_embd)


if __name__ == "__main__":
    # Example Usage
    batch_size = 5
    item_cnt = 10
    embd_dims = 32
    hidden_dims = 16
    n_head = 2
    input_dense_dim = 15

    num_experts = 3
    expert_hidden_dim = 64
    num_tasks = 2
    task_hidden_dims = [32, 24]

    model = SIMPlusMMoE(item_cnt, embd_dims, hidden_dims, n_head, input_dense_dim, num_experts, expert_hidden_dim,
                        num_tasks, task_hidden_dims)

    # prepare input data
    items = torch.randint(1, item_cnt, (batch_size,))
    hist_items = torch.LongTensor([
        [1, 5, 4, 0],
        [2, 6, 9, 8],
        [3, 7, 0, 0],
        [6, 9, 0, 0],
        [4, 2, 8, 5]
    ])
    input_dense_features = torch.randn(batch_size, input_dense_dim)

    # collect labels
    clicks = torch.randint(0, 2, (batch_size,)).float()  # Binary labels
    conversions = torch.randint(0, 2, (batch_size,)).float() * clicks  # Only have conversions where there's a click
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 10
    cvr_loss_weight = 2
    print("Start to train the MMoE model")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(items, hist_items, input_dense_features)
        ctr_score, cvr_score = outputs[0], outputs[1]
        ctr_loss = criterion(ctr_score, clicks)

        # Masking the CVR loss calculation
        valid_cvr_indices = clicks.bool()  # Indices where there was a click
        cvr_loss = criterion(cvr_score[valid_cvr_indices], conversions[valid_cvr_indices])

        # Weighted loss computation
        # Only add CVR loss if there are valid indices
        loss = ctr_loss + cvr_loss_weight * cvr_loss if valid_cvr_indices.any() else ctr_loss

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: CTR Loss = {ctr_loss}, CVR Loss = {cvr_loss}, Total Loss = {loss}")
