# ETRI - 2022 휴먼이해 인공지능 경진대회
> 본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다

## Abstract


## 1. 소개
### 1.1 대회 소개
#### 멀티모달 감정 데이터셋 활용 감정 인식 기술 분야
- 인간과 교감할 수 있는 인공지능 구현을 위해서는 인간의 행동과 감정을 이해하는 기술이 필요합니다. 이러한 기술 연구를 위해 구축한 데이터셋을 활용한 휴먼이해 인공지능 기술 연구를 위해 ETRI에서 금번 대회를 개최했습니다.

- Task는 다음과 같습니다.
    - 우리는 본 연구에서 성우 대상 상황극:[KEMDy19](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR) 데이터셋을 활용하여 감정의 레이블(기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔)에 대한 분류 정확도(F1)와 1~5단계의 각성도와 긍/부정도의 예측정확도(CCC;Concordance Correlation Coefficient)를 제시합니다.
    - 멀티모달 데이터를 혼합합니다. 발화음성과 발화텍스트를 사용하여 멀티모달 데이터 감정인식 모델을 구축했습니다.
    - 데이터 class 불균형 해소를 위한 Loss를 도입하고 그 효과를 비교했습니다.


### 1.2 Methodolgy
#### Model Architecture
![model_architecture](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F02e2f67e-59e5-4051-8543-b16a004e4a64%2FUntitled.png?table=block&id=6c9bb92c-881d-49d7-96a7-1ed675230d13&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)

#### Audio Spectrogram
![](https://miro.medium.com/max/1200/1*V2mgZ7y0ngd3q4DZ01xkEQ.png)
- Audio를 sr마다 sampling하여 spectrogram으로 표현

#### Korean Sentence Embedding[1]
- 성우들의 한국어 script를 embedding하기 위해 KoBert기반의 sentence embedding 

#### Class Balanced Loss[2]
- Class imbalance한 KEMDy19의 특성에 맞는 Loss 도입

![](http://latex.codecogs.com/gif.latex?\mathrm{CB}(p,y)=\frac{1}{E_{n_{y}}}\mathcal{L}(p,y)=\frac{1-\beta}{1-\beta^{n_{y}}}\mathcal{L}(p,y))

### 1.3 코드 설명

```constants.py``` : Hyper parameters
```dataset.py``` : torch.utils.data Dataset class를 상속한 KEMDy19 dataset class
```loss.py``` : Emotion classification과 Valence, Arousal MSE Loss를 정의하는 class. CBLoss를 정의하는 class
```main.py``` : main 함수
```metric.py``` : f1, recall, precision, ccc
```model.py``` : 우리의 모델
```utils.py``` : 작동에 필요한 함수를 정의
```preprocessing.ipynb``` : 데이터셋 전처리, split등 전처리에 사용된 코드


## 2. How To Use?
- 이 코드를 사용하는 방법을 다룹니다
- 순서와 지시를 __그대로__ 따라 사용해주세요

### 2.1 환경설정
0. 여러분의 PC나 서버에 GPU가 있고 cuda setting이 되어있어야합니다.
1. 여러분의 환경에 이 repo를 clone합니다 : ```git clone <this_repo>```
2. requirements libraries를 확인합니다 : ```pip install -r requirements.txt```

### 2.2 데이터셋 다운로드
1. [KEMDy19](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR) dataset을 다운로드하여 '2022-휴먼이해-인공지능-경진대회/data' 폴더에 넣으세요. 다운로드 권한을 신청해야할 수도 있습니다.
2. [Google_Drive]()에서 미리 가공된 데이터들을 다운로드하여 '2022-휴먼이해-인공지능-경진대회/data/KEMDy19' 폴더에 넣으세요.

- 최종적으로 structure가 이렇게 되어있다면 모든 준비가 끝났습니다!
```
<2022-휴먼이행-인공지능-경진대회>
                            ├ <data>
                                └ <KEMDy19>
                                    ├ <annotation>
                                    ├ <ECG>
                                    ├ <EDA>
                                    ├ <TEMP>
                                    ├ <wav>
                                    ├ annotation.csv
                                    ├ df_listener.csv
                                    ├ df_speaker.csv
                                    └ embedding_768.npy
                            ├ constants.py
                            ├ dataset.py
                            ├ loss.py
                            ├ main.py
                            ├ metric.py
                            ├ model.py
                            ├ utils.py
                            ├ EDA.ipynb
                            ├ prerprocessing.ipynb
                            ├ LICENSE
                            ├ requirements.txt
                            └ README.md                           
```

### 2.3 학습+추론
#### Speaker 감정 추론 baseline
```
python main.py --SorL speaker
               --test_split [13,14,15,16]
               --lam 0.66
               --epochs 100
```

#### Listener 감정 추론 baseline
```
python main.py --SorL listener
               --test_split [13,14,15,16]
               --lam 0.66
               --epochs 100
```
- argparser parameter 소개
    - gup_id : 사용할 GPU의 id
    - save_path : 실험결과가 저장될 경로 -> 만지지 마시오
    - backbone : Backbone network -> 만지지 마시오
    - text_dim : sentence embedding의 dimension -> 만지지 마시오
    - bidirectional : 양방향 RNN옵션. Default는 False. True로 만드려면 ```--bidirectional```
    - ws : sliceing window size -> 만지지 마시오
    - SorL : 추론할 감정. ```speaker``` 또는 ```listener```
    - sr : audio의 sampling rate -> 만지지 마시오
    - test_split : 20개의 session중에서 test split으로 나눌 session. 예시) ```[1,8,9,13]```
    - batch_size : Batch size ```64```
    - optim : optimizer. choices=sgd,adam,adagrad
    - loss : Default는 ```normal```로 MSE와 KLDiv Loss를 사용합니다. ```cbloss```로 설정하면 Class Balanced Loss가 사용됩니다.
    - lam : Default는 ```0.66```으로 감정을 맞추는 가중치(lam)와 각성도,긍부정도를 맞추는 가중치(1-lam) 사이의 비율을 결정합니다.
    - beta : Default는 ```0.99```. CBLoss의 가중치 beta를 결정합니다.
    - lr_decay : lr decay term
    - lr : learning rate
    - weight_decay : weight decay term(L2 regularization)
    - epochs : total training epochs



### 2.4 추론만 하기
```
python main.py --test_only
               --./exp에있으면서_test할_모델이_있는_폴더_이름
```
예를 들어서 ```exp/lstm_speaker_adam_8/model.pth```가 있었고 이 모델을 테스트만 하고싶다면 아래와같이 명령하세요.
```
python main.py --test_only
               --lstm_speaker_adam_8
```

## 3. 성능
### 3.1 기존 성능[3]

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| SPSL | 0.608 | - | 0.599 | - | - |
| MPSL | 0.591 | - | 0.584 | - | - |
| MPGL | 0.608 | - | 0.598 | - | - |

### 3.2 Baseline 성능

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker[1,2,3,4] | 0.687 | 0.663 | 0.674 | 0.771 | 0.845 |
| Speaker[5,6,7,8] | 0.685 | 0.663 | 0.673 | 0.781 | 0.777 |
| Speaker[9,10,11,12] | 0.719 | 0.691 | 0.704 | 0.802 | 0.891 |
| Speaker[13,14,15,16] | 0.748 | 0.719 | 0.733 | 0.745 | 0.860 |
| Speaker[17,18,19,20] | 0.718 | 0.688 | 0.702 | 0.751 | 0.872 |

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Listener[1,2,3,4] | 0.696 | 0.669 | 0.681 | 0.691 | 0.852 |
| Listener[5,6,7,8] | 0.671 | 0.651 | 0.661 | 0.767 | 0.816 |
| Listener[9,10,11,12] | 0.660 | 0.632 | 0.645 | 0.724 | 0.860 |
| Listener[13,14,15,16] | 0.744 | 0.713 | 0.728 | 0.756 | 0.865 |
| Listener[17,18,19,20] | 0.710 | 0.683 | 0.695 | 0.566 | 0.866 |

### 3.3 양방향 RNN
- configuration은 모두 default setting

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker_bi | 0.755 | 0.726 | 0.740 | 0.784 | 0.884 |
| Listener_bi | 0.741 | 0.710 | 0.725 | 0.711 | 0.880 |

### 3.4 Emotion 정보를 concat한 것이 도움이 되었을까?
- configuration은 모두 default setting
- emotion 정보 concat을 모두 제거 후 실험

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker_noCat | 0.719 | 0.690 | 0.704 | 0.751 | 0.827 |
| Listener_noCat | 0.722 | 0.692 | 0.706 | 0.689 | 0.876 |

### 3.5 CB Loss when ![](http://latex.codecogs.com/gif.latex?\lambda=0.9)

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker(![](http://latex.codecogs.com/gif.latex?\beta=0.8)) | 0.716 | 0.686 | 0.700 | 0.755 | 0.796 |
| Speaker (![](http://latex.codecogs.com/gif.latex?\beta=0.9)) | 0.621 | 0.594 | 0.607 | 0.623 | 0.785 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\beta=0.99)) | 0.721 | 0.691 | 0.705 | 0.612 | 0.829 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\beta=0.999)) | 0.670 | 0.643 | 0.656 | 0.730 | 0.869 |

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Linstener(![](http://latex.codecogs.com/gif.latex?\beta=0.8)) | 0.727 | 0.697 | 0.711 | 0.701 | 0.857 |
| Listener(![](http://latex.codecogs.com/gif.latex?\beta=0.9)) | 0.745 | 0.715 | 0.729 | 0.723 | 0.877 |
| Listener(![](http://latex.codecogs.com/gif.latex?\beta=0.99))| 0.711 | 0.681 | 0.695 | 0.725 | 0.854 |
| Listener(![](http://latex.codecogs.com/gif.latex?\beta=0.999)) | 0.698 | 0.669 | 0.682 | 0.741 | 0.868 |


### 3.6 ![](http://latex.codecogs.com/gif.latex?\lambda)에 따른 baseline ablation study

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.5))(1:1:1) | 0.738 | 0.709 | 0.722 | 0.780 | 0.824 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.66))(2:1:1) | 0.748 | 0.719 | 0.733 | 0.745 | 0.860 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.75))(3:1:1) | 0.759 | 0.731 | 0.744 | 0.791 | 0.869 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.8))(4:1:1) | 0.696 | 0.670 | 0.682 | 0.783 | 0.803 |

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.5))(1:1:1) | 0.712 | 0.683 | 0.696 | 0.746 | 0.880 |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.66))(2:1:1) | 0.744 | 0.713 | 0.728 | 0.756 | 0.865 |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.75))(3:1:1) | 0.740 | 0.710 | 0.724 | 0.712 | 0.870 |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.8))(4:1:1) | 0.709 | 0.680 | 0.694 | 0.688 | 0.862 |


## License & citiation
### License
MIT License 하에 공개되었습니다. 모델 및 코드를 사용시 첨부된 ```LICENSE```를 참고하세요.
### Citiation
Under Review 이므로 이 git을 참조부탁드립니다.


## Contact
- SungRae Hong : sun.hong@kaist.ac.kr
- TaeMi Kim : taemi_kim@kaist.ac.kr
- Sol Lee : leesol4553@kaist.ac.kr
- JongWoo Kim : gimjongu81@gmail.com

## Reference
[1] https://github.com/SKTBrain/KoBERT
</br>
[2] Cui, Yin, et al. "Class-balanced loss based on effective number of samples." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
</br>
[3] Noh, K.J.; Jeong, C.Y.; Lim, J.; Chung, S.; Kim, G.; Lim, J.M.; Jeong, H. Multi-Path and Group-Loss-Based Network for Speech Emotion Recognition in Multi-Domain Datasets. Sensors 2021, 21, 1579. https://doi.org/10.3390/s21051579