import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import os


def load_data(true_path: str, pred_path: str):
    """
    CSVファイルを読み込んでNumPy配列として返す
    """
    true_data = pd.read_csv(true_path)
    predicted_data = pd.read_csv(pred_path)
    
    # Convert string vectors to lists of floats (not numpy arrays)
    true_protein_labels = [list(map(float, vec.split(','))) for vec in true_data['receptor_vector']]
    true_peptide_labels = [list(map(float, vec.split(','))) for vec in true_data['peptide_vector']]
    pred_protein_labels = [list(map(float, vec.split(','))) for vec in predicted_data['receptor_vector']]
    pred_peptide_labels = [list(map(float, vec.split(','))) for vec in predicted_data['peptide_vector']]
    
    return true_data, predicted_data, true_protein_labels, true_peptide_labels, pred_protein_labels, pred_peptide_labels

def compute_roc_auc(true_labels, predicted_labels):
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_pair_metrics(true_protein, true_peptide, pred_protein, pred_peptide, threshold=0.5):
    """
    各ペアのAUCとF1スコアを計算
    """
    # タンパク質とペプチドの予測を結合
    true_combined = np.concatenate([true_protein, true_peptide])
    pred_combined = np.concatenate([pred_protein, pred_peptide])
    
    # AUCを計算
    fpr, tpr, _ = roc_curve(true_combined, pred_combined)
    pair_auc = auc(fpr, tpr)
    
    # F1スコアを計算（予測を二値化）
    pred_binary = (pred_combined >= threshold).astype(int)
    pair_f1 = f1_score(true_combined, pred_binary)
    
    return pair_auc, pair_f1

def plot_roc(fpr, tpr, roc_auc, output_path=None, title='ROC Curve'):
    """
    ROC曲線を描画し、必要に応じてファイルに保存
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"ROC curve saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Draw ROC curve from prediction scores and true labels CSVs.")
    parser.add_argument("--true_csv", required=True, help="Path to true label CSV file")
    parser.add_argument("--pred_csv", required=True, help="Path to predicted score CSV file")
    parser.add_argument("--output", default=None, help="Path to save ROC curve plot (e.g., 'roc.png')")
    parser.add_argument("--title", default="ROC Curve", help="Title of the ROC curve plot")
    parser.add_argument("--auc_output", default="auc_scores.csv", help="Path to save AUC scores for each pair")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")

    args = parser.parse_args()

    # 入力ファイル読み込み
    true_data, pred_data, true_protein_labels, true_peptide_labels, pred_protein_labels, pred_peptide_labels = load_data(args.true_csv, args.pred_csv)

    # 各ペアのAUCとF1スコアを計算
    pair_metrics = []
    for i in range(len(true_data)):
        pair_key = true_data.iloc[i]['pair_key']
        receptor_chain = true_data.iloc[i].get('receptor_chain', '')
        peptide_chain = true_data.iloc[i].get('peptide_chain', '')
        
        # このペアのメトリクスを計算
        pair_auc, pair_f1 = calculate_pair_metrics(
            np.array(true_protein_labels[i]),
            np.array(true_peptide_labels[i]),
            np.array(pred_protein_labels[i]),
            np.array(pred_peptide_labels[i]),
            args.threshold
        )
        
        pair_metrics.append({
            'pair_key': pair_key,
            'receptor_chain': receptor_chain,
            'peptide_chain': peptide_chain,
            'auc_score': pair_auc,
            'f1_score': pair_f1
        })
    
    # メトリクスをDataFrameに変換して保存
    metrics_df = pd.DataFrame(pair_metrics)
    metrics_df.to_csv(args.auc_output, index=False)
    print(f"Metrics saved to {args.auc_output}")
    
    # 全体のROC曲線を計算（タンパク質とペプチドの両方）
    true_labels_all = np.concatenate([np.array(vec).flatten() for vec in true_protein_labels + true_peptide_labels])
    pred_labels_all = np.concatenate([np.array(vec).flatten() for vec in pred_protein_labels + pred_peptide_labels])
    fpr_all, tpr_all, roc_auc_all = compute_roc_auc(true_labels_all, pred_labels_all)

    # タンパク質のみのROC曲線を計算
    true_labels_protein = np.concatenate([np.array(vec).flatten() for vec in true_protein_labels])
    pred_labels_protein = np.concatenate([np.array(vec).flatten() for vec in pred_protein_labels])
    fpr_protein, tpr_protein, roc_auc_protein = compute_roc_auc(true_labels_protein, pred_labels_protein)

    # ペプチドのみのROC曲線を計算
    true_labels_peptide = np.concatenate([np.array(vec).flatten() for vec in true_peptide_labels])
    pred_labels_peptide = np.concatenate([np.array(vec).flatten() for vec in pred_peptide_labels])
    fpr_peptide, tpr_peptide, roc_auc_peptide = compute_roc_auc(true_labels_peptide, pred_labels_peptide)

    # グラフ描画（両方のROC曲線を同じプロットに表示）
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_all, tpr_all, color='darkorange', lw=2, 
            label=f'All residues (AUC = {roc_auc_all:.2f})')
    plt.plot(fpr_protein, tpr_protein, color='blue', lw=2, 
            label=f'Protein residues only (AUC = {roc_auc_protein:.2f})')
    plt.plot(fpr_peptide, tpr_peptide, color='green', lw=2, 
            label=f'Peptide residues only (AUC = {roc_auc_peptide:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(args.title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300)
        print(f"ROC curves saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

