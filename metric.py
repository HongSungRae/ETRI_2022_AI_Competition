import torch
import numpy as np
import pandas as pd


def concordance_correlation_coefficient(y_pred, y_true):
    """Concordance correlation coefficient."""
    # Remove NaNs
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator



# 7by 7 metrix를 만들어서 바로 계산하면되지 않을까?
def get_metric(pred,target):
    '''
    pred : (Batch, num_class) shape의 tensor, 답이 1개부터 num_class개 까지 다양해도 OK
    target : (Batch, num_class) shape의 tensor, num_class 차원의 벡터에 softmax된 logit이 있다
    '''
    batch_size = pred.shape[0]
    num_class = pred.shape[-1]
    confusion_matrix = torch.zeros((num_class, num_class))
    
    # confusion_matrix 구하기
    for batch in range(batch_size):
        gt_index = torch.where(target[batch]==1)[0].tolist() # [num_class]에서 정답인 class의 index
        num_gt = len(gt_index) # 정답의 개수
        pred_index = torch.flip(torch.argsort(pred[batch],dim=-1),dims=(-1,))[0:num_gt].tolist() # pred에서 top-num_gt개의 top logit의 index를 가져옴
        for prediction in pred_index: # confusion matrix에 할당
            if prediction in gt_index:
                confusion_matrix[prediction, prediction] += 1
                gt_index.remove(prediction)
            else:
                confusion_matrix[prediction, gt_index[0]] += 1

    # precision, recall, f1구하기 : row는 model_pred, column은 GT이다
    eps = 1e-3
    precision = 0
    recall = 0
    for row in range(num_class):
        tp = confusion_matrix[row,row].item()
        fp = torch.sum(confusion_matrix[row]).item() - tp
        fn = torch.sum(confusion_matrix[:,row]).item() - tp
        precision += (1/num_class)*(tp/(tp+fp+eps))
        recall += (1/num_class)*(tp/(tp+fn+eps))
        
    f1 = 2*precision*recall/(precision+recall+eps)
    return confusion_matrix, (precision, recall, f1)





def get_recall(pred,target,k):
    recall = []
    for i in range(len(pred)):
        inter = 0
        query = pred[i]
        y_q = target[i]
        idx = torch.argsort(query,dim=0).tolist()
        idx.reverse()
        idx = idx[0:k] # k개의 top index
        for j in idx:
            if y_q.tolist()[j] == 1:
                inter += 1
        recall.append(inter/len(torch.where(y_q==1)[0]))
    
    return sum(recall)/len(recall)



def get_precision(pred, target, k):
    precision = []
    for i in range(len(pred)):
        inter = 0
        query = pred[i]
        y_q = target[i]
        idx = torch.argsort(query,dim=0).tolist()
        idx.reverse()
        idx = idx[0:k] # k개의 top index
        for j in idx:
            if y_q.tolist()[j] == 1:
                inter += 1
            query[j] = 1
        precision.append(inter/k)
    
    return sum(precision)/len(precision)



def get_f1(precision, recall):
    eps = 1e-3
    return 2*precision*recall/(precision+recall+eps)


if __name__ == '__main__':
    # CCC : 1~5까지의 continuous 변수
    y_true = [5, 4.1, 3.9, 4.1, 1.2]
    y_pred = [4.2, 4.4, 3.0, 2.1, 1.9]
    ccc = concordance_correlation_coefficient(y_pred, y_true)
    print(f'CCC : {ccc}')

    # Recall, Precision : 7차원 discrete 벡터
    y_true = torch.FloatTensor([[1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 1]]) # (B, 7)
    y_pred = torch.rand((3,7))
    print(f'Recall@1 : {get_recall(y_pred, y_true, 1)}')
    print(f'Precision@1 : {get_precision(y_pred, y_true, 1)}')
    print(f'F1@1 : {get_f1(get_recall(y_pred, y_true, 1),get_precision(y_pred, y_true, 1))}')
    print(f'Recall@2 : {get_recall(y_pred, y_true, 2)}')
    print(f'Precison@2 : {get_precision(y_pred, y_true, 2)}')
    print(f'F1@2 : {get_f1(get_recall(y_pred, y_true, 2),get_precision(y_pred, y_true, 2))}')

    # multi-class confusion-matrix 기반 계산
    pred = torch.rand((3,7))
    print(f'pred :\n{pred}')
    confusion_matrix, (precision, recall, f1) = get_metric(pred, y_true)
    print(f'confusion_matrix : \n{confusion_matrix}')
    print(f'precision : {precision}')
    print(f'recall : {recall}')
    print(f'f1 : {f1}')