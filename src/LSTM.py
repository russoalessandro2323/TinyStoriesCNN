# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.nnetwork import EmbeddingLayer

# class LSTM(nn.Module):

#     def __init__(self, fan_in, fan_out, vocab_size):

#         """
#             fan_in: input dimention 
#             fan_out: output dimention
#         """

#         super().__init__()

#         self.embedding = EmbeddingLayer(vocab_size, fan_in)
#         self.conv = nn.Conv1d(in_channels = fan_in, out_channels = vocab_size, kernel_size = 1)

#         self.W = nn.Parameter(torch.randn(4, fan_out, fan_out) / fan_out)
#         self.U = nn.Parameter(torch.randn(4, fan_in, fan_out) / fan_out)

#         self.fan_out = fan_out
    
#     def forward(self, x):
#         """
#             x: embedded batch, shape = (batch_size, embedded_length, L)
#         """

#         current_input = self.embedding(x)                                   # (B, L, emb_dim)
#         batch_size, L, _ = current_input.shape

#         hidden_state = current_input.new_zeros(batch_size, self.fan_out)    # (B, emb_dim)
#         cell_update  = current_input.new_zeros(batch_size, self.fan_out)    # (B, emb_dim)
#         hidden_state_list = []

#         for i in range(L):
#             # x[:, i, :] = Select the i-embedded vectors for each sentence in batch  (batch_size, L)

#             input_gate =   torch.sigmoid( (current_input[:, i, :] @ self.U[0]) + (hidden_state @ self.W[0]) ) 
#             forget_gate =  torch.sigmoid( (current_input[:, i, :] @ self.U[1]) + (hidden_state @ self.W[1]) ) 
#             output_gate =  torch.sigmoid( (current_input[:, i, :] @ self.U[2]) + (hidden_state @ self.W[2]) ) 
#             candidate_state = torch.tanh( (current_input[:, i, :] @ self.U[3]) + (hidden_state @ self.W[3]) )

#             cell_update = (cell_update * forget_gate) + (candidate_state * input_gate)

#             hidden_state = torch.tanh(cell_update) * output_gate
#             hidden_state_list.append(hidden_state)

#         return self.conv(torch.stack(hidden_state_list, dim=2)) #(batch_size, vocab_size, L)

import torch.nn as nn
import torch.nn.functional as F
from src.nnetwork import EmbeddingLayer
import torch

class LSTM(nn.Module):

    def __init__(self, fan_in, fan_out, vocab_size):

        """
            fan_in: input dimention 
            fan_out: output dimention
        """

        super().__init__()

        self.embedding = EmbeddingLayer(vocab_size, fan_in)
        self.conv = nn.Conv1d(in_channels = fan_in, out_channels = vocab_size, kernel_size = 1)
        self.rnn = nn.LSTM(fan_in, fan_out, batch_first=True)  
        
    
    def forward(self, x):
        """
            x: embedded batch, shape = (batch_size, embedded_length, L)
        """

        y, _ = self.rnn(self.embedding(x)) # (batch_size, L, embedded_length)

        return self.conv(y.transpose(1, 2)) # (batch_size, vocab_size, L)

    def generate_story(self, prompt_text, t, L, device, top_k=40, temperature=0.8, max_new_tokens=None):
        self.eval()

        formatted_text = "<NWS> " + t.replace_punctuation(prompt_text)
        ids = [t.word2idx.get(tok, t.word2idx["<UNK>"]) for tok in formatted_text.split()]
        idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        start = t.word2idx["<NWS>"]

        with torch.no_grad():
            for _ in range(max_new_tokens or L):
                logits = self(idx[:, -L:])[0, :, -1].float() / temperature

                logits[[0, 2]] = float("-inf")                                  # no <PAD> or <UNK>
                top_k_th = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < top_k_th, float("-inf"))   # select only top_k

                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)

                if nxt.item() == start:
                    break

                idx = torch.cat((idx, nxt.unsqueeze(0)), dim=1)

        story = t.decode(idx[0].tolist()).replace("<NWL>", "\n")
        print(f"\n{story}\n")