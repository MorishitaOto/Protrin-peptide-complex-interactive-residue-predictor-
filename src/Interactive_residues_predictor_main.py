import pandas as pd
from sequence_data_encoders import *
import util_for_model_construct
import torch
from torch import nn
from torch.nn import functional as F

class CrossAttention(nn.Module):
    """
    peptide: (batch_size, peptide_len, peptide_dim)
    protein: (batch_size, protein_len, protein_dim)
    calculate attention score between peptide and protein
    """

    def __init__(self, peptide_dim=768, protein_dim=2560, heads=8, dim_head=96):
        super().__init__()
        self.peptide_dim = peptide_dim
        self.protein_dim = protein_dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(peptide_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(protein_dim, heads * dim_head, bias=False)
        self.to_v_protein = nn.Linear(protein_dim, heads * dim_head, bias=False)
        self.to_v_peptide = nn.Linear(peptide_dim, heads * dim_head, bias=False)
        self.to_out_protein = nn.Linear(heads * dim_head, peptide_dim)
        self.to_out_peptide = nn.Linear(heads * dim_head, protein_dim)
        self.layer_norm = nn.LayerNorm(peptide_dim)

    def forward(self, peptide, protein, peptide_mask, pro_mask):
        b, n, _, h = *peptide.shape, self.heads
        
        # Project peptide into query space
        q = self.to_q(peptide).view(b, n, self.heads, -1).transpose(1, 2)
        
        # Project protein into key and value space
        protein_len = protein.shape[1]
        peptide_len = peptide.shape[1]
        k = self.to_k(protein).view(b, protein_len, self.heads, -1).transpose(1, 2)
        v_protein = self.to_v_protein(protein).view(b, protein_len, self.heads, -1).transpose(1, 2) #(B, H, L_protein, D_head)
        v_peptide = self.to_v_peptide(peptide).view(b, peptide_len, self.heads, -1).transpose(1, 2) #(B, H, L_peptide, D_head)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale #Calculate the matrix multiplication in the last two dimensions of a 4D tensor. Shape of dots: (B, H, L_q, L_k)
        

        
        # Apply masks
        peptide_mask = peptide_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.heads, n, protein_len) #(B, L_peptide) >> unsqueeze(1).unsqueeze(-1) >> (B, 1, L_peptide, 1) >> expand >> (B, H, L_peptide, L_protein) = (32, 16, 1500, 1000) 
        masked_dots = dots.masked_fill(peptide_mask == 0, -1e6) #Positions where peptide_mask == 0 (padding) are set to -∞
        pro_mask = pro_mask.unsqueeze(1).unsqueeze(-2).expand(-1, self.heads, n, protein_len) #(B, L_protein)>> ... >> (B, H, L_peptide, L_protein) = (32, 16, 1500, 1000) 
        masked_dots = masked_dots.masked_fill(pro_mask == 0, -1e6)
        
        # Apply softmax to compute attention weights
        attn = F.softmax(masked_dots, dim=-1) #Normalize each row of L_peptide into probabilities (important components in the L_protein dimension are emphasized)
        
        
        # Compute out_protein
        out_protein = torch.matmul(attn, v_protein) #(B, H, L_peptide, L_protein) * (B, H, L_protein, D_head) = (B, H, L_peptide, D_head)
        out_protein = out_protein.transpose(1, 2).contiguous().view(b, n, -1) #(B, H, L_peptide, D_head) >> (B, L_peptide, D_head * H) = (32, 1500, 768)

        out_protein = self.to_out_protein(out_protein) #(B, L_peptide, D_head * H) >> (B, L_peptide, D_peptide)
        
        assert peptide.shape == out_protein.shape, "Shape mismatch between peptide and out_protein"
        out_protein = self.layer_norm(out_protein + peptide) #Added for skip connection
            
        # Compute out_peptide
        attn_t = attn.transpose(-2, -1) # (B, H, L_protein, L_peptide))
        out_peptide = torch.matmul(attn_t, v_peptide) #(B, H, L_protein, L_peptide) * (B, H, L_peptide, D_head) = (B, H, L_protein, D_head)
        out_peptide = out_peptide.transpose(1, 2).contiguous().view(b, protein_len, -1) #(B, H, L_protein, D_head) >> (B, L_protein, D_head * H) = 

        out_peptide = self.to_out_peptide(out_peptide) #(B, L_protein, D_head * H) >> (B, L_protein, D_protein) = 
        
        assert protein.shape == out_peptide.shape, "Shape mismatch between protein and out_peptide"
        out_peptide = self.layer_norm(out_peptide + protein) #skip connection
        
        return out_protein, out_peptide, attn


###############
# model definition
###############
class InteractionResiduesPredictor(nn.Module):
    def __init__(self, peptide_seq_encoder, protein_seq_encoder, peptide_seq_emb_dim=2560, protein_seq_emb_dim=2560, device='cuda'):
        super(InteractionResiduesPredictor, self).__init__()
        self.device = device
        self.peptide_seq_encoder = peptide_seq_encoder
        self.protein_seq_encoder = protein_seq_encoder
        self.MLP_for_peptide_seq = util_for_model_construct.MLPNet(peptide_seq_emb_dim, peptide_seq_emb_dim)
        self.dim_adjuster     = nn.Linear(peptide_seq_emb_dim, protein_seq_emb_dim)
        self.MLP_for_protein_seq = util_for_model_construct.MLPNet(protein_seq_emb_dim, protein_seq_emb_dim)
        self.cross_Attn       = CrossAttention(peptide_dim=protein_seq_emb_dim, protein_dim=protein_seq_emb_dim, heads=16, dim_head=160)
        self.MLP_1 = util_for_model_construct.MLPNet(protein_seq_emb_dim, int(protein_seq_emb_dim/8))
        self.MLP_2 = util_for_model_construct.MLPNet(int(protein_seq_emb_dim/8), 2)

    def forward(self, peptide_seq_list, protein_seq_list):
        #Vectorize sequence data
        protein_seq_tensor, protein_seq_attention_mask = self.protein_seq_encoder.protein_seqs2embeded_tokens(protein_seq_list)
        peptide_seq_tensor, peptide_seq_attention_mask = self.peptide_seq_encoder.peptide_seqs2embeded_tokens(peptide_seq_list)
        #Apply linear transformation to the sequence vectors
        peptide_seq_x = self.MLP_for_peptide_seq(peptide_seq_tensor) #Transform amino acid sequence vectors using MLP
        peptide_seq_x = self.dim_adjuster(peptide_seq_x) #Adjust amino acid sequence vectors to match nucleotide dimensions
        protein_seq_x = self.MLP_for_protein_seq(protein_seq_tensor) #Transform nucleotide sequence vectors using MLP
        #Process sequence vectors with the attention mechanism
        interaction_output_protein, interaction_output_peptide, weight = self.cross_Attn(peptide_seq_x, protein_seq_x, peptide_seq_attention_mask, protein_seq_attention_mask) #attention_result(v_protein, v_peptide), masked_dots with softmax
        # (B, L_peptide, D_peptide), (B, L_protein, D_protein), (B, H, L_peptide, L_protein) 
        
        #todo: consider how to use interaction_output_peptide as input
        #todo: consider the shape of last output
        
        #ppm prediction
        x_protein = self.MLP_1(interaction_output_protein)
        x_protein = x_protein * peptide_seq_attention_mask.unsqueeze(-1)  # Apply mask by broadcasting
        
        x_peptide = self.MLP_1(interaction_output_peptide)
        x_peptide = x_peptide * protein_seq_attention_mask.unsqueeze(-1)  # Apply mask by broadcasting

        logits_protein = self.MLP_2(x_protein)  # shape: (B, L_peptide, 2)
        logits_peptide = self.MLP_2(x_peptide)  # shape: (B, L_protein, 2)
        # Insert a very small value into masked positions to deactivate them (however, handling with ignore_index etc. in CrossEntropyLoss is more precise)
        logits_protein = logits_protein.masked_fill(peptide_seq_attention_mask.unsqueeze(-1) == 0, -1e6)
        logits_peptide = logits_peptide.masked_fill(protein_seq_attention_mask.unsqueeze(-1) == 0, -1e6)

        return logits_protein, logits_peptide, weight  
        # shape: (B, L_peptide, 2) , (B, L_protein, 2), weight      
