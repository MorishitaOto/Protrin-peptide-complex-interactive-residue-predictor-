import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess as sb
import esm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class peptide_seq2representation():
    """
    (ESMのモデル名)       : (各アミノ酸に対応したベクトル表現の次元数)
    esm2_t48_15B_UR50D  : 5120
    esm2_t36_3B_UR50D   : 2560
    esm2_t33_650M_UR50D : 1280
    esm2_t30_150M_UR50D : 640
    esm2_t12_35M_UR50D  : 480
    esm2_t6_8M_UR50D    : 320
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.lm, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.lm
        self.lm.eval()
        self.batch_tokens = None
        self.batch_converter = self.alphabet.get_batch_converter()

        for name, param in self.lm.named_parameters():
            param.requires_grad = False
    def peptide_seqs2embeded_tokens(self, seq_list):
        '''
        input : アミノ酸配列のlist
        output:
            vector_outputs: [batch_size, max_seq_len, embed_dim] のテンソル
            attention_masks: [batch_size, max_seq_len] のアテンションマスク (0: PAD, 1: 実トークン)
        '''
        if not seq_list:
            raise ValueError("Empty sequence list provided")   
        try:
            last_layer_ind = 36  # モデルによって変更する

            len_list = [len(seq) for seq in seq_list]
            max_len = max(len_list)
            seq_input = [(index, seq) for index, seq in enumerate(seq_list)]
            batch_labels, batch_strs, self.batch_tokens = self.batch_converter(seq_input)
            
            self.lm.to(self.device)

            with torch.no_grad():
                results = self.lm(self.batch_tokens.to(self.device), repr_layers=[last_layer_ind], return_contacts=False)

            embed_dim = results["representations"][last_layer_ind].shape[-1]
            batch_size = len(seq_list)

            # パディング用のゼロテンソルを作成
            vector_outputs = torch.zeros((batch_size, max_len, embed_dim))
            attention_masks = torch.zeros((batch_size, max_len), dtype=torch.int32)

            for i, (rep, length) in enumerate(zip(results["representations"][last_layer_ind], len_list)):
                vector_outputs[i, :length, :] = rep[1:length+1, :]
                attention_masks[i, :length] = 1

            self.lm.to('cpu')

            return vector_outputs.to(self.device), attention_masks.to(self.device)
        
        except Exception as e:
            raise RuntimeError(f"Error in peptide sequence embedding: {str(e)}")


class protein_seq2representation():
    """
    (ESMのモデル名)       : (各アミノ酸に対応したベクトル表現の次元数)
    esm2_t48_15B_UR50D  : 5120
    esm2_t36_3B_UR50D   : 2560
    esm2_t33_650M_UR50D : 1280
    esm2_t30_150M_UR50D : 640
    esm2_t12_35M_UR50D  : 480
    esm2_t6_8M_UR50D    : 320
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.lm, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()

        self.lm
        self.lm.eval()
        self.batch_tokens = None
        self.batch_converter = self.alphabet.get_batch_converter()

        for name, param in self.lm.named_parameters():
            param.requires_grad = False

    def protein_seqs2embeded_tokens(self, seq_list):
        '''
        input : アミノ酸配列のlist
        output:
            vector_outputs: [batch_size, max_seq_len, embed_dim] のテンソル
            attention_masks: [batch_size, max_seq_len] のアテンションマスク (0: PAD, 1: 実トークン)
        '''
        last_layer_ind = 36  # モデルによって変更する

        len_list = [len(seq) for seq in seq_list]
        max_len = max(len_list)
        seq_input = [(index, seq) for index, seq in enumerate(seq_list)]
        batch_labels, batch_strs, self.batch_tokens = self.batch_converter(seq_input)
        
        self.lm.to(self.device)

        with torch.no_grad():
            results = self.lm(self.batch_tokens.to(self.device), repr_layers=[last_layer_ind], return_contacts=False)

        embed_dim = results["representations"][last_layer_ind].shape[-1]
        batch_size = len(seq_list)

        # パディング用のゼロテンソルを作成
        vector_outputs = torch.zeros((batch_size, max_len, embed_dim))
        attention_masks = torch.zeros((batch_size, max_len), dtype=torch.int32)

        for i, (rep, length) in enumerate(zip(results["representations"][last_layer_ind], len_list)):
            vector_outputs[i, :length, :] = rep[1:length+1, :]
            attention_masks[i, :length] = 1

        self.lm.to('cpu')

        return vector_outputs.to(self.device), attention_masks.to(self.device)


