import torch
import torch.nn as nn
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import ast
import torchaudio
import torchaudio.transforms
from constants import F_MAX, F_MIN, HOP_SIZE, N_FFT, N_MELS, SAMPLE_RATE





class EmotionDataset(Dataset):
    def __init__(self, split='train', test_split=[1,2,3,4], SorL=None, ws=8, text_dim=768):
        '''
        split : train / test
        test_split : k-fold에서 test에 사용할 set들
        SorL : speaker Or listener
        ws : Window Size
        '''
        assert SorL in ['listener','speaker']
        assert split in ['train','test']
        self.label_dict = {'disgust':0, 'angry':1, 'sad':2, 'fear':3, 'surprise':4,'neutral':5, 'happy':6, 'disqust':0}
        self.text_emb_dic = np.load(f'./data/KEMDy19/embedding_{text_dim}.npy', allow_pickle=True).item()
        self.split  = split
        self.SorL = SorL
        self.ws = ws
        self.text_dim = text_dim
        self.df = pd.read_csv('./data/KEMDy19/df_'+SorL+'.csv')
        self.random = np.random.RandomState(42)
        if split == 'train':
            self.ws_df = self._get_ws_df(self.df,
                                           ws,
                                           [session for session in [i for i in range(1,21)] if session not in test_split])
        elif split == 'test':
            self.ws_df = self._get_ws_df(self.df,ws,test_split) # index:0번부터
        
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                    n_fft=N_FFT,
                                                    hop_length=HOP_SIZE,
                                                    f_min=F_MIN,
                                                    f_max=F_MAX,
                                                    n_mels=N_MELS,
                                                    normalized=False)

        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        self.audio_embedding = nn.Sequential(self.spec,
                                             self.to_db,
                                             self.spec_bn
                                            )
    
    
    
    def _get_ws_df(self,df,ws,split_list):
        ws_df = pd.DataFrame(columns=['seq'])
        ws_list = ['padding' for _ in range(0,ws-1)]
        seed_id = df['Segment ID'][1]
        for id in df['Segment ID']:
            if int(seed_id.split('_')[0][-2:]) in split_list:
                if seed_id.split('_')[0:2] == id.split('_')[0:2]: #같은 dialog라면
                    ws_list.append(id)
                    seed_id = id
                else:
                    # ws 만큼 잘라서 ws_df에 append
                    for i in range(0,len(ws_list)-ws+1):
                        ws_df = ws_df.append({'seq' : ws_list[i:i+ws]},ignore_index=True)
                    ws_list = ['padding' for _ in range(0,ws-1)]
                    seed_id = id
                    if int(id.split('_')[0][-2:]) in split_list:
                        ws_list.append(id)
            else:
                seed_id = id
        else: # 마지막 다이얼로그가 위의 else문에 들어갈 수 없으므로 따로 처리
            # print(ws_list)
            for i in range(0,len(ws_list)-ws+1):
                ws_df = ws_df.append({'seq' : ws_list[i:i+ws]},ignore_index=True)
            
        # ws_df = ws_df.sample(frac=1)
        # ws_df = ws_df.sample(frac=1).reset_index(drop=True)
        ws_df.to_csv('./data/KEMDy19/df_'+self.SorL+'_'+str(ws)+'_'+self.split+'.csv',index=False,encoding='utf-8')
        ws_df = pd.read_csv('./data/KEMDy19/df_'+self.SorL+'_'+str(ws)+'_'+self.split+'.csv')
        return ws_df

    def __len__(self):
        return len(self.ws_df)

    def __getitem__(self, idx):
        # 무언가 stack할 tensor나 ndarray를 선언
        audio = torch.zeros((self.ws,1,229,313))
        text = torch.zeros((self.ws,1,self.text_dim))
        emotion = torch.zeros((7))
        valence = torch.FloatTensor([2.5])
        arousal = torch.FloatTensor([2.5])
        _ws_list = self.ws_df['seq'][idx] # str이다 : '['padding','padding',...]'
        ws_list = ast.literal_eval(_ws_list) # list로 변형
        for i,talk in enumerate(ws_list):
            if talk == 'padding':
                pass
            else:
                # 이하 audio
                wave, sr = librosa.load(f'./audio/{talk}.wav', sr=SAMPLE_RATE)
                wave_len = wave.shape[0]
                if wave_len <= sr*10: # 10초가 안되는 음성이면
                    wave = librosa.util.fix_length(wave, size=sr*10)
                else: # 10초가 넘는 음성이면
                    start = self.random.randint(wave_len - sr*10)
                    end = start + sr*10
                    wave = wave[start:end]
                wave = torch.FloatTensor(wave) # (160000)
                wave = torch.unsqueeze(torch.unsqueeze(wave,0),0) # (1,1,160000)
                audio[i] = self.audio_embedding(torch.FloatTensor(wave)) # (1,229,313)
                # 이하 text
                try:
                    text[i] = torch.FloatTensor(self.text_emb_dic[talk+'.txt'])
                except:
                    pass
                          
        else:# 이하 emotion
            if self.split == 'train': # soft label
                for people in range(1,10+1):
                    emotion[self.label_dict[self.df[self.df['Segment ID']==talk]['Emotion.'+str(people)].values[0]]] += 1
                emotion = emotion/10
            elif self.split == 'test': # hard label ex) [1,0,1,0,0,1,0]
                emotion_list = self.df[self.df['Segment ID']==talk]['Emotion'].values[0].split(';')
                for emo in emotion_list:
                    emotion[self.label_dict[emo]] += 1
            valence = torch.FloatTensor(self.df[self.df['Segment ID']==talk]['Valence'].values)
            arousal = torch.FloatTensor(self.df[self.df['Segment ID']==talk]['Arousal'].values)
            if valence.shape[0]>1: # 이유는 모르겠는데 평균값이 2개나 있는 애들이 있다.
                valence = torch.FloatTensor([float(valence[0])])
            if arousal.shape[0]>1:
                arousal = torch.FloatTensor([float(arousal[0])])
        audio =  audio.detach()
        return audio, text, emotion, arousal, valence



if __name__ == '__main__':
    dataset = EmotionDataset('train',[5,6,7,8],'listener',8)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,drop_last=False)
    audio, text, emotion, arousal, valence = next(iter(dataloader))

    def property(name,data):
        print(f'{name} : {data.shape} : {type(data)} : {data.requires_grad}')
    property('audio',audio)
    property('text',text)
    property('emotion',emotion)
    property('arousal',arousal)
    property('valence',valence)
    for i, (audio, text, emotion, arousal, valence) in enumerate(dataloader):
        print(i)
    '''
    audio : torch.Size([B, 8, 1, 160000])
    text : torch.Size([B, 8, 1, 64])
    emotion : torch.Size([B, 7])
    arousal : torch.Size([B, 1])
    valence : torch.Size([B, 1])
    '''