import pandas as pd
import ast
import argparse


def sort_indices(vector_str):
    """
    ベクトル文字列をパースし、値の降順でソートされたインデックスを返す。
    """
    try:
        vector = ast.literal_eval(vector_str)
        sorted_indices = sorted(range(len(vector)), key=lambda i: vector[i], reverse=True)
        sorted_indices = [i + 1 for i in sorted_indices]
        return sorted_indices
    except Exception as e:
        print(f"Error parsing vector: {e}")
        return []


def process_dataframe(input_path, output_path):
    """
    入力CSVを処理し、sorted_indices列を追加して出力する。
    """
    df = pd.read_csv(input_path)

    if 'receptor_vector' not in df.columns or 'peptide_vector' not in df.columns:
        raise ValueError("CSVに必要な列（receptor_vector, peptide_vector）が存在しません。")

    # 各列に対してインデックスのソート
    df['receptor_sorted_indices'] = df['receptor_vector'].apply(sort_indices)
    df['peptide_sorted_indices'] = df['peptide_vector'].apply(sort_indices)

    # 出力
    df.to_csv(output_path, index=False)
    print(f"保存完了: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="自信度ベクトルを基に残基インデックスをソートし、CSVに保存するスクリプト")
    parser.add_argument("--input", type=str, required=True, help="入力CSVファイルパス")
    parser.add_argument("--output", type=str, required=True, help="出力CSVファイルパス")

    args = parser.parse_args()
    process_dataframe(args.input, args.output)


if __name__ == "__main__":
    main()

