import pandas as pd
import torch
import sequence_data_encoders
from Interactive_residues_predictor_main import InteractionResiduesPredictor
import numpy as np
import os
import argparse

def load_test_data(file_path):
    """Load test data from CSV file."""
    df = pd.read_csv(file_path)
    return df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on protein-peptide pairs')
    
    # Path arguments
    parser.add_argument('--test_data_path', type=str, default='../data/test_data.csv',
                      help='Path to test data CSV file')
    parser.add_argument('--model_path', type=str, default='../pth/best_model.pth',
                      help='Path to model checkpoint file')
    parser.add_argument('--output_dir', type=str, default='../test_result',
                      help='Directory to save results')
    parser.add_argument('--output_file', type=str, default='all_predictions.csv',
                      help='Name of output CSV file')
    
    # Model parameters
    parser.add_argument('--output_type', type=str, default='binary',
                        help='binary or probability')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Probability threshold for interaction prediction')
    parser.add_argument('--top_k', type=int, default=5,
                      help='Number of top predictions to display')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize encoders and model
    peptide_seq_encoder = sequence_data_encoders.peptide_seq2representation(device=args.device)
    protein_seq_encoder = sequence_data_encoders.protein_seq2representation(device=args.device)
    model = InteractionResiduesPredictor(peptide_seq_encoder, protein_seq_encoder).to(args.device)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_data = load_test_data(args.test_data_path)
    
    # List to store results
    all_results = []
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, args.output_file)
    
    # Process each row
    for idx, row in test_data.iterrows():
        pair_key = row['pair_key']
        receptor_chain = row.get('receptor_chain', '')  # optional
        peptide_chain = row.get('peptide_chain', '')    # optional
        receptor_sequence = row['receptor_sequence']
        peptide_sequence = row['peptide_sequence']
        
        print(f"\nProcessing pair: {pair_key}")
        print(f"Receptor sequence: {receptor_sequence}")
        print(f"Peptide sequence: {peptide_sequence}")
        
        # Prepare input sequences
        peptide_seq = [peptide_sequence]
        protein_seq = [receptor_sequence]
        
        # Run inference
        with torch.no_grad():
            _, peptide_mask = peptide_seq_encoder.peptide_seqs2embeded_tokens(peptide_seq)
            _, protein_mask = protein_seq_encoder.protein_seqs2embeded_tokens(protein_seq)
            
            output_protein, output_peptide, attention_weights = model(peptide_seq, protein_seq)
            
            # Convert logits to probabilities
            protein_probs = torch.softmax(output_peptide, dim=-1)
            peptide_probs = torch.softmax(output_protein, dim=-1)
            
            # Get predictions (class 1 probabilities) and apply masks
            protein_pred = protein_probs[0, :, 1].cpu().numpy()
            peptide_pred = peptide_probs[0, :, 1].cpu().numpy()
            
            # Apply masks to predictions
            protein_pred = protein_pred * protein_mask[0].cpu().numpy()
            peptide_pred = peptide_pred * peptide_mask[0].cpu().numpy()
            
            # Convert predictions to binary interaction vector
            if args.output_type == 'binary':
                receptor_vector = ",".join(map(lambda x: str(int(x > args.threshold)), protein_pred))
                peptide_vector = ",".join(map(lambda x: str(int(x > args.threshold)), peptide_pred))
            
            elif args.output_type == 'probability':
                receptor_vector = ",".join(map(str, protein_pred))
                peptide_vector = ",".join(map(str, peptide_pred))
            
            
            else:
                raise ValueError(f"Invalid output_type: {args.output_type}. Must be either 'binary' or 'probability'")
                
            all_results.append({
                'pair_key': pair_key,
                'receptor_chain': receptor_chain,
                'peptide_chain': peptide_chain,
                'receptor_sequence': receptor_sequence,
                'peptide_sequence': peptide_sequence,
                'receptor_vector': receptor_vector,
                'peptide_vector': peptide_vector
            })

    # Save all results to a single CSV file
    results_df = pd.DataFrame(all_results, columns=[
        'pair_key', 'receptor_chain', 'peptide_chain',
        'receptor_sequence', 'peptide_sequence',
        'receptor_vector', 'peptide_vector'
    ])
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nAll predictions saved to {output_csv_path}")

if __name__ == "__main__":
    main()
