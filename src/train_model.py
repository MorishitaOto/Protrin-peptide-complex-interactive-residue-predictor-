import pandas as pd
import sequence_data_encoders
import Interactive_residues_predictor_main
from util_for_model_construct import batch_maker, custom_ppm_loss, custom_ppm_loss_MSEver
import torch.nn.functional as F
import tqdm
from torch import nn
import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import logging
from datetime import datetime

# Logging settings
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

epochs = 3
batch_size = 20
lr   = 1e-4
device = 'cuda'


###########################
# Load training data      #
###########################
data_csv_path = '../data/example_for_train.csv'

df = pd.read_csv(data_csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
peptide_seq_list = []
protein_seq_list = []
peptide_interactive_res_list = []
protein_interactive_res_list = []

for peptide_seq, protein_seq, peptide_interactive_res, protein_interactive_res in zip(df['peptide_sequence'], df['receptor_sequence'], df['peptide_vector'], df['receptor_vector']):
    peptide_seq_list.append(peptide_seq)
    protein_seq_list.append(protein_seq)
    peptide_interactive_res_list.append([float(x) for x in peptide_interactive_res.split(',')])
    protein_interactive_res_list.append([float(x) for x in protein_interactive_res.split(',')])
    
peptide_seq_encoder = sequence_data_encoders.peptide_seq2representation()
protein_seq_encoder = sequence_data_encoders.protein_seq2representation()

###########################
# Load evaluation data    #
###########################
data_csv_path = '../data/example_for_eval.csv'

df = pd.read_csv(data_csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
peptide_seq_list_test = []
protein_seq_list_test = []
peptide_interactive_res_list_test = []
protein_interactive_res_list_test = []

for peptide_seq, protein_seq, peptide_interactive_res, protein_interactive_res in zip(df['peptide_sequence'], df['receptor_sequence'], df['peptide_vector'], df['receptor_vector']):
    peptide_seq_list_test.append(peptide_seq)
    protein_seq_list_test.append(protein_seq)
    peptide_interactive_res_list_test.append([float(x) for x in peptide_interactive_res.split(',')])
    protein_interactive_res_list_test.append([float(x) for x in protein_interactive_res.split(',')])

###############################
# Instantiate prediction model #
###############################
model = Interactive_residues_predictor_main.InteractionResiduesPredictor(peptide_seq_encoder, protein_seq_encoder).to(device)

# After this, peptide_seq_encoder is loaded in the initializer of InteractiveResiduesPredictor,
# and functions are called to input seq_list and use it.

'''
##################################
# Reload trained model           #
##################################
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    trust_remote_code=True
)
pth_path = '../pth/Expression_level_prediction_1_4.pth'

torch.serialization.add_safe_globals([Expression_predictor_main.Expression_predictor])

model = torch.load(pth_path, weights_only=False)
'''

########################################
# Define optimizer and loss function   #
########################################
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

#########################################
# Training loop                        #
#########################################

# Create save directory before training
save_dir = '../pth'
os.makedirs(save_dir, exist_ok=True)

# Variable to save the best model
best_val_loss = float('inf')
best_model_path = os.path.join(save_dir, 'best_model.pth')

for epoch in range(epochs):
    training_loss = 0.0
    validation_loss = 0.0
    batch_list = batch_maker(peptide_seq_list, protein_seq_list, peptide_interactive_res_list,protein_interactive_res_list, batch_size)
    batch_list_test = batch_maker(peptide_seq_list_test, protein_seq_list_test, peptide_interactive_res_list_test, protein_interactive_res_list_test, batch_size)
    
    # Training phase
    model.train()
    logger.info(f"\nEpoch {epoch+1}/{epochs}")
    logger.info("Training phase:")
    training_metrics = []  # List to save metrics for the training phase
    
    for batch_idx, batch in enumerate(tqdm.tqdm(batch_list, desc="Training")):
        peptide_seq_batch = batch[0]
        protein_seq_batch = batch[1]
        peptide_interactive_res_data_batch = torch.tensor(batch[2]).to(device)
        protein_interactive_res_data_batch = torch.tensor(batch[3]).to(device)
        
        output_protein, output_peptide, atten_weight = model(peptide_seq_batch, protein_seq_batch)
        atten_weight.to('cpu')
        
        batch_size, L_peptide, _ = output_protein.shape
        _         , L_protein, _ = output_peptide.shape
        
        peptide_interactive_res_data_batch = peptide_interactive_res_data_batch.reshape(batch_size, L_peptide)
        protein_interactive_res_data_batch = protein_interactive_res_data_batch.reshape(batch_size, L_protein)
        
        output_protein_flat = output_protein.view(-1, 2)
        output_peptide_flat = output_peptide.view(-1, 2)
        labels_peptide_flat = peptide_interactive_res_data_batch.view(-1).long()
        labels_protein_flat = protein_interactive_res_data_batch.view(-1).long()
        
        loss = criterion(output_protein_flat, labels_peptide_flat) + criterion(output_peptide_flat, labels_protein_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        
        # Calculate metrics for each batch
        protein_pred = torch.softmax(output_protein_flat, dim=1)[:, 1].cpu().numpy()
        peptide_pred = torch.softmax(output_peptide_flat, dim=1)[:, 1].cpu().numpy()
        protein_true = labels_protein_flat.cpu().numpy()
        peptide_true = labels_peptide_flat.cpu().numpy()
        
        # Concatenate predictions for protein and peptide
        true_combined = np.concatenate([protein_true, peptide_true])
        pred_combined = np.concatenate([protein_pred, peptide_pred])
        
        # Calculate AUC and F1 score
        batch_auc = roc_auc_score(true_combined, pred_combined)
        pred_binary = (pred_combined >= 0.5).astype(int)
        batch_f1 = f1_score(true_combined, pred_binary)
        
        # Save batch metrics
        training_metrics.append({
            'batch_idx': batch_idx,
            'loss': loss.item(),
            'auc': batch_auc,
            'f1_score': batch_f1
        })
        
        # Output batch log
        batch_log = f"\nTraining Batch {batch_idx + 1}/{len(batch_list)}:\n" \
                   f"  Loss: {loss.item():.4f}\n" \
                   f"  AUC: {batch_auc:.4f}\n" \
                   f"  F1 Score: {batch_f1:.4f}"
        logger.info(batch_log)
        
        # Display average loss every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss_so_far = training_loss / (batch_idx + 1)
            logger.info(f"\nAverage Loss after {batch_idx + 1} batches: {avg_loss_so_far:.4f}")
    
    # Save training phase metrics to CSV
    training_metrics_df = pd.DataFrame(training_metrics)
    training_metrics_df.to_csv(f'../test_result/training_metrics_epoch_{epoch+1}.csv', index=False)
    
    # Validation phase
    model.eval()
    logger.info("\nValidation phase:")
    validation_loss = 0
    batch_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(batch_list_test, desc="Validation")):
            peptide_seq_batch = batch[0]
            protein_seq_batch = batch[1]
            peptide_interactive_res_data_batch = torch.tensor(batch[2]).to(device)
            protein_interactive_res_data_batch = torch.tensor(batch[3]).to(device)
            
            output_protein, output_peptide, atten_weight = model(peptide_seq_batch, protein_seq_batch)
            atten_weight.to('cpu')
            
            batch_size, L_peptide, _ = output_protein.shape
            _         , L_protein, _ = output_peptide.shape
            
            peptide_interactive_res_data_batch = peptide_interactive_res_data_batch.reshape(batch_size, L_peptide)
            protein_interactive_res_data_batch = protein_interactive_res_data_batch.reshape(batch_size, L_protein)
            
            output_protein_flat = output_protein.view(-1, 2)
            output_peptide_flat = output_peptide.view(-1, 2)
            labels_peptide_flat = peptide_interactive_res_data_batch.view(-1).long()
            labels_protein_flat = protein_interactive_res_data_batch.view(-1).long()
            
            loss = criterion(output_protein_flat, labels_peptide_flat) + criterion(output_peptide_flat, labels_protein_flat)
            validation_loss += loss.item()
            
            # Calculate metrics for each batch
            protein_pred = torch.softmax(output_protein_flat, dim=1)[:, 1].cpu().numpy()
            peptide_pred = torch.softmax(output_peptide_flat, dim=1)[:, 1].cpu().numpy()
            protein_true = labels_protein_flat.cpu().numpy()
            peptide_true = labels_peptide_flat.cpu().numpy()
            
            # Concatenate predictions for protein and peptide
            true_combined = np.concatenate([protein_true, peptide_true])
            pred_combined = np.concatenate([protein_pred, peptide_pred])
            
            # Calculate AUC and F1 score
            batch_auc = roc_auc_score(true_combined, pred_combined)
            pred_binary = (pred_combined >= 0.5).astype(int)
            batch_f1 = f1_score(true_combined, pred_binary)
            
            # Save batch metrics
            batch_metrics.append({
                'batch_idx': batch_idx,
                'loss': loss.item(),
                'auc': batch_auc,
                'f1_score': batch_f1
            })
            
            # Output batch log
            batch_log = f"\nValidation Batch {batch_idx + 1}/{len(batch_list_test)}:\n" \
                       f"  Loss: {loss.item():.4f}\n" \
                       f"  AUC: {batch_auc:.4f}\n" \
                       f"  F1 Score: {batch_f1:.4f}"
            logger.info(batch_log)
    
    # Calculate epoch summary
    avg_training_loss = training_loss / len(batch_list)
    avg_validation_loss = validation_loss / len(batch_list_test)
    avg_training_auc = np.mean([m['auc'] for m in training_metrics])
    avg_training_f1 = np.mean([m['f1_score'] for m in training_metrics])
    avg_validation_auc = np.mean([m['auc'] for m in batch_metrics])
    avg_validation_f1 = np.mean([m['f1_score'] for m in batch_metrics])
    
    # Output epoch summary
    epoch_summary = f"\nEpoch {epoch+1}/{epochs} Summary:\n" \
                   f"Training:\n" \
                   f"  Average Loss: {avg_training_loss:.4f}\n" \
                   f"  Average AUC: {avg_training_auc:.4f}\n" \
                   f"  Average F1 Score: {avg_training_f1:.4f}\n" \
                   f"Validation:\n" \
                   f"  Average Loss: {avg_validation_loss:.4f}\n" \
                   f"  Average AUC: {avg_validation_auc:.4f}\n" \
                   f"  Average F1 Score: {avg_validation_f1:.4f}"
    logger.info(epoch_summary)
    
    # Save batch metrics to CSV
    validation_metrics_df = pd.DataFrame(batch_metrics)
    validation_metrics_df.to_csv(f'../test_result/validation_metrics_epoch_{epoch+1}.csv', index=False)
    
    # Save best model
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, best_model_path)
        logger.info(f"Best model saved with validation loss: {best_val_loss:.4f}")
    
    # Periodic checkpoint saving
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_validation_loss,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at epoch {epoch+1}")

# Save final model
final_model_path = os.path.join(save_dir, 'final_model.pth')
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_validation_loss,
}, final_model_path)
logger.info("Training completed. Final model saved.")


