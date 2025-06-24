import os
import pandas as pd
from sklearn.metrics import f1_score
import ast
import numpy as np


def get_acc1_f1(df):
    acc1 = (df['prediction'] == df['ground_truth']).sum() / len(df)
    f1 = f1_score(df['ground_truth'], df['prediction'], average='weighted')
    return acc1, f1


def get_is_correct(row):
    pred_list = row['prediction']
    if row['ground_truth'] in pred_list:
        row['is_correct'] = True
    else:
        row['is_correct'] = False

    return row


def get_is_correct10(row):
    pred_list = row['top10']
    if row['ground_truth'] in pred_list:
        row['is_correct10'] = True
    else:
        row['is_correct10'] = False

    pred_list = row['top5']
    if row['ground_truth'] in pred_list:
        row['is_correct5'] = True
    else:
        row['is_correct5'] = False

    pred = row['top1']
    if pred == row['ground_truth']:
        row['is_correct1'] = True
    else:
        row['is_correct1'] = False

    return row


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def get_ndcg(prediction, targets, k=10):
    """
    Calculates the NDCG score for the given predictions and targets.

    Args:
        prediction (Nxk): list of lists. the softmax output of the model.
        targets (N): torch.LongTensor. actual target place id.

    Returns:
        the sum ndcg score
    """
    for _, xi in enumerate(prediction):
        if len(xi) < k:
            # print(f"the {i}th length: {len(xi)}")
            xi += [-5 for _ in range(k - len(xi))]
        elif len(xi) > k:
            xi = xi[:k]
        else:
            pass

    n_sample = len(prediction)
    prediction = np.array(prediction)
    targets = np.broadcast_to(targets.reshape(-1, 1), prediction.shape)
    hits = first_nonzero(prediction == targets, axis=1, invalid_val=-1)
    hits = hits[hits >= 0]
    ranks = hits + 1
    ndcg = 1 / np.log2(ranks + 1)
    return np.sum(ndcg) / n_sample

def bubble_sort(arr_str):
    n = len(arr_str)
    for i in range(n):
        for j in range(0, n-i-1):
            if int(arr_str[j].split('.csv')[0]) > int(arr_str[j + 1].split('.csv')[0]):
                arr_str[j], arr_str[j+1] = arr_str[j+1], arr_str[j]
    return arr_str

if __name__ == '__main__':
    # Calculate the metric for all user
    # output_dir = 'output/ft_fsq/top1'
    # output_dir = 'output/ft_ori/top1'
    # output_dir = 'output/ft_dest/top1'

    # output_dir = 'output/ft_pissa_n4_dest/top1'

    output_dir = 'output/ft_olora_dest_hk_incident/top1'
    # output_dir = 'output_api/llmmob_gpt-4o-mini_ori_hk_incident/top1'

    file_list = [file for file in os.listdir(output_dir) if file.endswith('.csv')]

    file_list = bubble_sort(file_list)
    print(file_list)
    file_path_list = [os.path.join(output_dir, file) for file in file_list]

    df = pd.DataFrame({
        'user_id': None,
        'ground_truth': None,
        'prediction': None,
        'reason': None
    }, index=[])

    hall_rate = 0
    # 1191, 2701, 3171
    # user 479, 312
    for file_path in file_path_list:
        iter_df = pd.read_csv(file_path)
        df = pd.concat([df, iter_df], ignore_index=True)

    df.fillna(-100, inplace=True)

    pred_results = []
    for i in range(len(df)):
        try:
            pred_i = int(df['prediction'].iloc[i])
        except:
            pred_i = -100
        pred_results.append(pred_i)

    # df['prediction'] = df['prediction'].apply(lambda x: int(x))
    df['prediction'] = pred_results
    df['ground_truth'] = df['ground_truth'].apply(lambda x: int(x))



    acc1, f1 = get_acc1_f1(df)
    print('sample size:', len(df))
    print(F'acc, F1: {acc1 * 100:.2f}, {f1:.4f}' )

 
