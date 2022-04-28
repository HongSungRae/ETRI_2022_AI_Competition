import torch
import torch.nn as nn
import ast
from dataset import EmotionDataset


class DoRaeMiSoLoss(nn.Module):
    def __init__(self, lam=0.66):
        super().__init__()
        self.lam = lam
        self.kld_emotion = nn.KLDivLoss(reduction = "batchmean")
        self.mse_arousal = nn.MSELoss()
        self.mse_valence = nn.MSELoss()

    def forward(self, emotion_pred, emotion, arousal_pred, arousal, valence_pred, valence):
        loss = self.lam*self.kld_emotion(emotion_pred, emotion) +\
               (1-self.lam)*self.mse_arousal(arousal_pred, arousal) +\
               (1-self.lam)*self.mse_valence(valence_pred, valence)
        return loss


class CBDoRaeMiSoLoss(nn.Module):
    def __init__(self, lam=0.9, beta=0.99, df=None, ws_df=None):
        '''
        df 는 dataset 선언후에 train data담고있는 self.df
        '''
        super().__init__()
        label_dict = {'disgust':0, 'angry':1, 'sad':2, 'fear':3, 'surprise':4,'neutral':5, 'happy':6, 'disqust':0}
        self.lam = lam
        self.beta = beta
        self.mse_arousal = nn.MSELoss()
        self.mse_valence = nn.MSELoss()
        beta = torch.tensor([beta for _ in range(7)])
        n_y = torch.zeros((7))
        for idx in range(len(ws_df)):
            _ws_list = ws_df['seq'][idx] # str이다 : '['padding','padding',...]'
            last_talk = ast.literal_eval(_ws_list)[-1] # list로 변형
            emotion_list = df[df['Segment ID']==last_talk]['Emotion'].values[0].split(';')
            for emotion in emotion_list:
                n_y[label_dict[emotion]] += 1
        n_y = (n_y/min(n_y)).int() # 간단한 정수비
        numerator = torch.ones((7)) - beta
        denominator = torch.ones((7)) - torch.pow(beta, n_y)
        weight = numerator / denominator
        self.cb_emotion = nn.BCELoss(weight=weight) # 정답이 2개 이상일 수도 있다면 각 class별로 CELoss를 구한다

    def forward(self, emotion_pred, emotion, arousal_pred, arousal, valence_pred, valence):
        loss = self.lam*self.cb_emotion(emotion_pred, emotion) +\
               (1-self.lam)*self.mse_arousal(arousal_pred, arousal) +\
               (1-self.lam)*self.mse_valence(valence_pred, valence)
        return loss



if __name__ == '__main__':
    # Normal Loss
    criterion = DoRaeMiSoLoss()
    emotion_pred = torch.rand((16,7))
    emotion = torch.ones((16,7))
    arousal_pred = torch.randn((16,1))
    arousal = torch.randn((16,1))
    valence_pred = torch.randn((16,1))
    valence = torch.randn((16,1))
    loss = criterion(emotion_pred.log(), emotion, arousal_pred, arousal, valence_pred, valence)
    print(f'DoRaeMiSoLoss : {loss.item()}')


    # CBLoss
    train_dataset = EmotionDataset('train','speaker',8)
    criterion = CBDoRaeMiSoLoss(lam=0.9, 
                                beta=0.99,
                                df=train_dataset.df,
                                ws_df=train_dataset.ws_df)
    loss = criterion(emotion_pred, emotion, arousal_pred, arousal, valence_pred, valence)
    print(f'CBLoss : {loss.item()}')