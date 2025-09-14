from torch import nn
import torch.nn.functional as F
import random
import torch

class MLPNet(nn.Module):
    """_summary_

    Args:
    - emb_dim: int, the dimension of the intermediate embeddings
    - num_classes: int, the number of classes to predict
    - dropout: float, the dropout rate

    Returns:
    - z: tensor, the output of the network
    """

    def __init__(self, emb_dim, num_classes, dropout=0.3):

        super().__init__()
        self.desc_skip_connection = True
        print('dropout is {}'.format(dropout))

        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(emb_dim, emb_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.relu3 = nn.GELU()
        self.fc4 = nn.Linear(emb_dim, emb_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.relu4 = nn.GELU()
        self.fc5 = nn.Linear(emb_dim, emb_dim)
        self.dropout5 = nn.Dropout(dropout)
        self.relu5 = nn.GELU()
        self.final = nn.Linear(emb_dim, num_classes)

    def forward(self, inter_emb):
        x_out = self.fc1(inter_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        x_out = x_out + inter_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        z = self.final(z + x_out)
        
        z = self.fc3(x_out)
        z = self.dropout3(z)
        z = self.relu3(z)
        z = self.final(z + x_out)
        
        z = self.fc4(x_out)
        z = self.dropout4(z)
        z = self.relu4(z)
        z = self.final(z + x_out)
        
        z = self.fc5(x_out)
        z = self.dropout5(z)
        z = self.relu5(z)
        z = self.final(z + x_out)
        return z
def batch_maker(peptide_seq_list, protein_seq_list, peptide_interactive_res_list, protein_interactive_res_list, batch_size=10, random_shuffle=True):
    if batch_size == 1:
        # Even for a single batch, pad to the maximum sequence length
        max_peptide_len = max(len(seq) for seq in peptide_seq_list)
        max_protein_len = max(len(seq) for seq in protein_seq_list)
        
        peptide_seq_batch = []
        protein_seq_batch = []
        peptide_interactive_res_batch = []
        protein_interactive_res_batch = []
        
        for i in range(len(peptide_seq_list)):
            # Pad sequences
            peptide_seq_padded = peptide_seq_list[i] + 'X' * (max_peptide_len - len(peptide_seq_list[i]))
            protein_seq_padded = protein_seq_list[i] + 'X' * (max_protein_len - len(protein_seq_list[i]))
            
            peptide_seq_batch.append(peptide_seq_padded)
            protein_seq_batch.append(protein_seq_padded)
            
            # Pad labels
            peptide_padded = peptide_interactive_res_list[i] + [0.0] * (max_peptide_len - len(peptide_interactive_res_list[i]))
            protein_padded = protein_interactive_res_list[i] + [0.0] * (max_protein_len - len(protein_interactive_res_list[i]))
            
            peptide_interactive_res_batch.append(peptide_padded)
            protein_interactive_res_batch.append(protein_padded)
        
        return [[peptide_seq_batch, protein_seq_batch, peptide_interactive_res_batch, protein_interactive_res_batch]]
    
    zip_list = list(zip(peptide_seq_list, protein_seq_list, peptide_interactive_res_list, protein_interactive_res_list))
    batch_list = []
    if random_shuffle:
        random.shuffle(zip_list)
    
    for batch_ind in range(int(len(peptide_seq_list)/batch_size)):
        batch_data = zip_list[batch_ind*batch_size:(batch_ind+1)*batch_size]
        
        # Compute the maximum sequence length within the batch
        max_peptide_len = max(len(data[0]) for data in batch_data)
        max_protein_len = max(len(data[1]) for data in batch_data)
        
        peptide_seq_batch = []
        protein_seq_batch = []
        peptide_interactive_res_batch = []
        protein_interactive_res_batch = []
        
        for data in batch_data:
            peptide_seq, protein_seq, peptide_interactive_res, protein_interactive_res = data
            
            # Pad sequences
            peptide_seq_padded = peptide_seq + 'X' * (max_peptide_len - len(peptide_seq))
            protein_seq_padded = protein_seq + 'X' * (max_protein_len - len(protein_seq))
            
            peptide_seq_batch.append(peptide_seq_padded)
            protein_seq_batch.append(protein_seq_padded)
            
            # Pad labels (fill with 0)
            peptide_padded = peptide_interactive_res + [0.0] * (max_peptide_len - len(peptide_interactive_res))
            protein_padded = protein_interactive_res + [0.0] * (max_protein_len - len(protein_interactive_res))
            
            peptide_interactive_res_batch.append(peptide_padded)
            protein_interactive_res_batch.append(protein_padded)
        
        batch_list.append([
            peptide_seq_batch,
            protein_seq_batch,
            peptide_interactive_res_batch,
            protein_interactive_res_batch
        ])
    
    return batch_list

