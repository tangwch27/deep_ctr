import math

import torch
import torch.nn as nn

'''
DIEN (Alibaba 2018) is an improvement over DIN in that it considers both the interest evolution of users and the dynamic influence of 
external factors (e.g., time). This is because user interests are diverse, and leads to 
interest drifting phenomenon: user’s intentions can be very different in adjacent visits, and one behaviour can depend on user visits long ago. 

Comparatively, DIN is weak in capturing the dependencies between sequential behaviours.
In other words, DIN still treat user behaviour as independent signals but DIEN treat them as sequential signals.

Key Ideas
1. Introduces a GRU (Gated Recurrent Unit) module to extract a series of interest states from sequential user behaviours
2. Introduces another modified GRU module called AUGRU to model user interests' temporal dynamics. 
    Idea is to model relevant interest evolution, while weakening irrelevant interests caused by interest drifting.
3. Add auxiliary loss which uses the next behaviour to supervise the learning of current hidden state, 
    or “interest state”, to mimic the temporal nature of user interests
    
    
GRU: uses fewer gates and does not use a memory unit
- Update Gate
- Reset Gate
Args: input_size, hidden_size, num_layers, etc
Inputs: input, h_0
Outputs: output, h_n
'''


class AUGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AUGRUCell, self).__init__()
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.new_state = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev, attn_score):
        combined = torch.cat([x, h_prev], dim=1)
        # update
        z = torch.sigmoid(self.update_gate(combined))
        # reset
        r = torch.sigmoid(self.reset_gate(combined))
        combined_new = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.new_state(combined_new))
        attn_score = attn_score.view(-1, 1)  # broadcast
        # print(z.size(), attn_score.size(), attn_score)
        z = z * attn_score
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next


class AttentionLayer(nn.Module):
    # attention mechanism for adapting interest representation
    # query is the target item, with embd_dim
    # keys and values are output of GRUs, with hidden_dim
    def __init__(self, embd_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.embd_dim = embd_dim
        self.W_q = nn.Linear(embd_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, raw_query, raw_keys, raw_values):
        # batch_size, hidden_dim
        query = self.W_q(raw_query)
        # batch_size, seq_len, hidden_dim
        keys = self.W_k(raw_keys)
        values = self.W_v(raw_values)

        # batch_size, 1, seq_len
        attn_scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1)) / math.sqrt(self.embd_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # batch_size, seq_len, embd => batch_size, embd
        attn_embd = torch.matmul(attn_probs, values).squeeze(1)
        return attn_embd, attn_scores


#  auxiliary loss to help with effective negative sampling during training.
class AuxiliaryNet(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(embd_dim * 2, embd_dim),
            nn.PReLU(),
            nn.Linear(embd_dim, 1)
        )

    def forward(self, pos_embd, neg_embd):
        x = torch.cat([pos_embd, neg_embd], dim=-1)
        output = self.fc_layers(x)
        return torch.sigmoid(output)


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, item_size, embd_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.item_embd = nn.Embedding(item_size, embd_dim)

    def forward(self, item_id):
        return self.item_embd(item_id)


class EmbeddingLayer(nn.Module):
    def __init__(self, obj_size, embd_dim):
        super(EmbeddingLayer, self).__init__()
        self.obj_embd = nn.Embedding(obj_size, embd_dim)

    def forward(self, obj_id):
        return self.obj_embd(obj_id)


class DeepInterestEvolutionNetwork(nn.Module):
    def __init__(self, embd_dim, hidden_dim):
        super(DeepInterestEvolutionNetwork, self).__init__()
        # attention layer
        self.attn_layer = AttentionLayer(embd_dim, hidden_dim)
        # gru: batch_size, seq_len, embd_dim => batch_size, seq_len, hidden_dim
        self.gru_layer = nn.GRU(embd_dim, hidden_dim, batch_first=True)
        self.auxiliary_net = AuxiliaryNet(embd_dim)
        self.au_gru = AUGRUCell(hidden_dim, hidden_dim)

    def forward(self, target_item_embd, hist_item_embd):
        # hist_item_embd: batch_size, hist_cnt, embd
        """
        interest extractor layer (GRU)
        utilizes a standard GRU to process sequences of user interactions over time,
        capturing the evolving nature of user interests based on their behavior history.
        The output is a sequence of hidden states representing the interest states at each timestep.

        GRU inherently processes data sequentially along the sequence dimension (time steps).
        For each element in a sequence, the GRU uses the current input and the previous hidden state to
        calculate the next hidden state.
        """
        # gru_hidden_output: batch_size, hist_cnt, embd_dim => batch_size, hist_cnt, hidden_dim
        # second gru output is the final_state: 1, batch_size, hidden_dim
        gru_hidden_output, _ = self.gru_layer(hist_item_embd)
        '''
        Interest Evolving Layer with AUGRU
        AUGRU integrates the attention mechanism directly into the GRU's update gate. 
        This allows the model to dynamically focus on different parts of the interest state sequence depending on 
        the current target item, leading to a more nuanced understanding of how a user's interests evolve.
        
        The update gate's computation is modified by incorporating an attention mechanism that allows the model to 
        weigh the importance of each input differently based on an external attention score derived from 
        the context (usually the output of an attention layer).
        '''
        # GRU hidden outputs as the key and value, target item embd as the key
        # context: batch_size, hidden_dim
        # attn_scores: batch_size, seq_len
        attn_embd, attn_scores = self.attn_layer(target_item_embd, gru_hidden_output, gru_hidden_output)
        attn_scores = attn_scores.squeeze(1)

        # Applying AUGRU for each timestamp (hist_cnt)
        # Interest Evolving Layer: Process each sequence step with AUGRU
        # gru_outputs: batch_size, hist_cnt, embd_dim
        # batch_size, embd_dim
        '''
        # has in place issue
        interest_evolution = torch.zeros_like(gru_hidden_output)
        for t in range(gru_hidden_output.size(1)):
            if t == 0:
                interest_evolution[:, t, :] = self.augru(gru_hidden_output[:, t, :], gru_hidden_output[:, t, :], attn_scores[:, t])
            else:
                interest_evolution[:, t, :] = self.augru(gru_hidden_output[:, t, :], interest_evolution[:, t - 1, :], attn_scores[:, t])

        combined_embd = torch.cat([item_embd, interest_evolution[:, -1, :]], dim=-1)
        '''

        interest_evolution = gru_hidden_output[:, 0, :]
        for t in range(gru_hidden_output.size(1)):
            interest_evolution = self.au_gru(gru_hidden_output[:, t, :], interest_evolution, attn_scores[:, t])

        return interest_evolution


class DIEN(nn.Module):
    def __init__(self, item_size, embd_dim, hidden_dim):
        super(DIEN, self).__init__()
        # embedding layer
        self.item_embd = ItemEmbeddingLayer(item_size, embd_dim)
        # fc
        self.dien_layer = DeepInterestEvolutionNetwork(embd_dim, hidden_dim)

        fc_input_dim = embd_dim + hidden_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, item_id, hist_item_ids):
        target_item_embd = self.item_embd(item_id)
        hist_item_embd = self.item_embd(hist_item_ids)
        interest_evolution = self.dien_layer(target_item_embd, hist_item_embd)
        combined_embd = torch.cat([target_item_embd, interest_evolution], dim=-1)
        # batch_size, 1
        output = self.fc_layers(combined_embd)
        # squeeze from batch_size, 1 to batch_size to match the label size
        return torch.sigmoid(output.squeeze())


if __name__ == "__main__":
    # Simplified Dataset Example
    # User IDs, Item IDs (target), and Historical Items (behavior)
    batch_size = 3
    items = torch.randint(1, 10, (batch_size,))

    # 0 is padding token (history item)
    hist_items = torch.LongTensor([
        [1, 5, 4, 0],
        [2, 6, 9, 8],
        [3, 7, 0, 0]
    ])
    embd_dims = 32
    hidden_dims = 16

    # whether user like the target item or not
    labels = torch.FloatTensor([1, 0, 1])

    # Model Training
    model = DIEN(10, embd_dims, hidden_dims)
    '''
    Configure the Optimizer with L2 Regularization: 
    When setting up the optimizer, specify the weight_decay parameter, which controls the amount of L2 penalty.
    During training, the L2 regularization is automatically applied to the weights through the optimizer step function.
    
    The weight_decay in the optimizer adds an additional term to the loss during the backward pass. 
    This term is the derivative of 0.5*λ*|weights|^2 = λ*weights, which is subtracted from the weights during 
    the optimizer update, effectively implementing the L2 regularization. 
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99), weight_decay=1e-5)
    criterion = nn.BCELoss()
    print("Start to train the DIEN model")
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


'''
MIMN: Multi-channel user Interest Memory Network 
is mainly built on top of the DIEN framework, which has an issue of latency. The system latency and storage costs would 
increase linearly with the length of sequential user behaviour. "Long" UB means up to 1000 items.

Key Ideas

MIMN embeds user long term interest into fixed sized memory network to solve the problem of large storage of user behaviour data. 
UIC module is designed to record the new user behaviours incrementally to deal with latency limitation.

1. Decouple the User Interest Center (UIC) module from the entire CTR prediction system
    Update latent user interest vector based on real time behaviour trigger events -> tackles latency
    
2. Multi-channel user Interest Memory Network (MIMN) 
    Build on Neural Tuning Machines (NTM) and implemented with the UIC module -> tackles storage

'''