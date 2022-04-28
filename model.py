import torch
import torch.nn as nn
from constants import F_MAX, F_MIN, HOP_SIZE, N_FFT, N_MELS, SAMPLE_RATE
from torchinfo import summary



class DoRaeMiSol_lstm(nn.Module): #Wrapper
    def __init__(self,embed_d=32, hidden_size=512, num_layers=4, text_dim=768, ws=8, bidirectional=False):
        '''
        conv1d_channel : ws와 똑같아야합니다
        '''
        super().__init__()
        if bidirectional:
            bi = 2
        else: bi = 1
        self.embed_d = embed_d
        self.ws = ws
        self.audio_embedding = nn.Sequential(BasicConv2d(ws,32,3),
                                             BasicConv2d(32, 48, 3,max_pool=2),
                                             BasicConv2d(48, 32, 5,max_pool=2),
                                             BasicConv2d(32, ws, 7,max_pool=2)
                                             )
        self.lstm = nn.LSTM(input_size=816+text_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.mlp_emotion = nn.Sequential(BasicLinear(ws*hidden_size*bi,2048),
                                         BasicLinear(2048,512),
                                         BasicLinear(512,256),
                                         BasicLinear(256,64),
                                         nn.Linear(64,7)
                                         )
        self.softmax = nn.Softmax(dim=-1)
        self.mlp_av = nn.Sequential(BasicLinear(ws*hidden_size*bi,2048),
                                    BasicLinear(2048,1024),
                                    BasicLinear(1024,512))
        self.mlp_arousal = nn.Sequential(BasicLinear(512+7,128),
                                         BasicLinear(128,32),
                                         nn.Linear(32,1),
                                         nn.Sigmoid())
        self.mlp_valence = nn.Sequential(BasicLinear(512+7,128),
                                          BasicLinear(128,32),
                                          nn.Linear(32,1),
                                          nn.Sigmoid())
    def shape(self,tensor):
        print(tensor.shape)

    def forward(self,audio,text):
        text = torch.squeeze(text) # (B,ws,text_dim)
        audio = torch.squeeze(audio) # (B,ws,229,313)
        audio = self.audio_embedding(audio) # (B, ws, 24, 34)
        audio = torch.reshape(audio,(audio.shape[0],self.ws,-1)) # (B, ws, 816)
        x = torch.cat((audio,text),dim=-1) # (B,ws,816+text_dim)
        x,_ = self.lstm(x) # (B, ws, hidden_size*bi) (num_layers, B, hidden_size) == ()
        x = torch.reshape(x,(x.shape[0],-1))
        emotion = self.mlp_emotion(x) # (B,7)
        emotion = self.softmax(emotion)
        av = self.mlp_av(x)
        av = torch.cat((av,emotion),dim=-1)
        arousal = 4*self.mlp_arousal(av) + 1
        valance = 4*self.mlp_valence(av) + 1
        return emotion, arousal, valance





class BasicLinear(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_features,out_features),
                                nn.ReLU(),
                                nn.BatchNorm1d(out_features))

    def forward(self,x):
        return self.fc(x)




class BasicConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation=1, stride=1, padding=0, max_pool=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(max_pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


if __name__ == '__main__':
    text_dim = 768

    audio = torch.randn((16,8,1,229,313)).cuda()
    text = torch.randn((16,8,1,text_dim)).cuda()
    model = DoRaeMiSol_lstm(embed_d=32,hidden_size=256, text_dim=text_dim, bidirectional=True).cuda()
    emotion, arousal, valence = model(audio,text)
    print(f'emotion : {emotion.shape}')
    print(f'arousal : {arousal.shape}')
    print(f'valance : {valence.shape}')
    summary(model,[(16,8,1,229,313),(16,8,1,text_dim)])