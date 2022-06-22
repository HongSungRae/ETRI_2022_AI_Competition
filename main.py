# library
from torch.utils.data import DataLoader
import os
import torch
import argparse
import torch.optim as optim
import json
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
from model import DoRaeMiSol_lstm
import sys
import ast

# local
from utils import *
from dataset import EmotionDataset
from model import DoRaeMiSol_lstm
from loss import DoRaeMiSoLoss, CBDoRaeMiSoLoss
from metric import concordance_correlation_coefficient, get_recall, get_precision, get_f1, get_metric


# parser
parser = argparse.ArgumentParser(description='ETRI_DoRaeMiSol')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--backbone', default='lstm', type=str,
                    help='backbone network for simsiam',
                    choices=['lstm'])
parser.add_argument('--embed_d', default=32, type=int,
                    help='lstm 할때 임베딩 dim')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='hidden size of lstm')
parser.add_argument('--text_dim', default=768, type=int,
                    help='text embedding dimension')
parser.add_argument('--bidirectional', action='store_true',
                    help='How To Make TRUE? : --bidirectional, Flase : default')
parser.add_argument('--ws', default=8, type=int,
                    help='window size')
parser.add_argument('--SorL', default='speaker', type=str,
                    help='누구의 감정을 맞출까요',
                    choices=['speaker','listener'])
parser.add_argument('--sr', default=16000, type=int,
                    help='sampling rate')
parser.add_argument('--test_split', default='[1,2,3,4]', type=str,
                    help='k-fold 알고리즘에서 k=5를 사용합니다. test split으로 사용할 set을 설정합니다.')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adam', type=str,
                    help='optimizer', choices=['sgd','adam','adagrad'])
parser.add_argument('--loss', default='normal', type=str,
                    help='loss function', choices=['normal','cbloss'])
parser.add_argument('--lam', default=0.66, type=float,
                    help='emotion에 lam의 가중치를, arousal과 valence에 1-lam의 가중치를 줍니다.')
parser.add_argument('--beta', default=0.99, type=float,
                    help='CBLoss의 hyper parameter인 Beta')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--lr_decay', default=1e-3, type=float,
                    help='learning rate decay')
parser.add_argument('--weight_decay', default=0.00001, type=float,
                    help='weight_decay')
parser.add_argument('--epochs', default=50, type=int,
                    help='train epoch')

# For test
parser.add_argument('--test_only', action='store_true',
                    help='How To Make TRUE? : --test_only, Flase : default')
parser.add_argument('--test_path', default='DeFaUlT', type=str,
                    help='test할 model이 있는 폴더의 이름. 해당 폴더는 ./exp에 있어야한다')

# GPU srtting							
parser.add_argument('--gpu_id', default='1', type=str,
                    help='How To Check? : cmd -> nvidia-smi')
args = parser.parse_args()
start = time.time()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(model, train_loader, criterion, optimizer, epoch, num_epoch, train_logger):
    model.train()
    train_loss = AverageMeter()
    for i, (audio, text, emotion, arousal, valence) in enumerate(train_loader):
        audio, text, emotion, arousal, valence = audio.cuda(), text.cuda(), emotion.cuda(), arousal.cuda(), valence.cuda()
        emotion_pred, arousal_pred, valence_pred = model(audio, text)

        if args.loss == 'normal':
            loss = criterion(emotion_pred.log(), emotion, arousal_pred, arousal, valence_pred, valence)
        else:
            loss = criterion(emotion_pred, emotion, arousal_pred, arousal, valence_pred, valence)

        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(train_loader), loss=loss))
    train_logger.write([epoch, train_loss.avg])



def test(model, test_loader, save_path):
    print("=================== Test Start ====================")
    
    # metrics
    precision_av = AverageMeter()
    recall_av = AverageMeter()
    f1_av = AverageMeter()
    precision_at_1_av = AverageMeter()
    recall_at_1_av = AverageMeter()
    f1_at_1_av = AverageMeter()
    precision_at_2_av = AverageMeter()
    recall_at_2_av = AverageMeter()
    f1_at_2_av = AverageMeter()
    arousal_gt_list = []
    arousal_pred_list = []
    valence_gt_list = []
    valence_pred_list = []
    total_confusion_matrix = torch.zeros((7,7))

    # model test
    model.eval()
    with torch.no_grad():
        for i, (audio, text, emotion, arousal, valence) in enumerate(test_loader):
            audio, text, emotion, arousal, valence = audio.cuda(), text.cuda(), emotion.cuda(), arousal.cuda(), valence.cuda()
            emotion_pred, arousal_pred, valence_pred = model(audio, text)
            # CCC
            arousal_gt_list += torch.reshape(arousal,(-1,)).tolist()
            arousal_pred_list += torch.reshape(arousal_pred,(-1,)).tolist()
            valence_gt_list += torch.reshape(valence,(-1,)).tolist()
            valence_pred_list += torch.reshape(valence_pred,(-1,)).tolist()
            # multi-class metric
            confusion_matrix, (precision, recall, f1) = get_metric(emotion_pred, emotion)
            precision_av.update(precision)
            recall_av.update(recall)
            f1_av.update(f1)
            total_confusion_matrix += confusion_matrix
            # @1
            precision_at_1 = get_precision(emotion_pred, emotion, 1)
            recall_at_1 = get_recall(emotion_pred, emotion, 1)
            f1_at_1 = get_f1(precision_at_1, recall_at_1)
            precision_at_1_av.update(precision_at_1)
            recall_at_1_av.update(recall_at_1)
            f1_at_1_av.update(f1_at_1)
            # @2
            precision_at_2 = get_precision(emotion_pred, emotion, 2)
            recall_at_2 = get_recall(emotion_pred, emotion, 2)
            f1_at_2 = get_f1(precision_at_2, recall_at_2)
            precision_at_2_av.update(precision_at_2)
            recall_at_2_av.update(recall_at_2)
            f1_at_2_av.update(f1_at_2)
        ccc_arousal = concordance_correlation_coefficient(arousal_pred_list, arousal_gt_list)
        ccc_valence = concordance_correlation_coefficient(valence_pred_list, valence_gt_list)

        result = {
                  'Precision' : f'{precision_av.avg:3f}+-{precision_av.std:.3f}',
                  'Recall' : f'{recall_av.avg:3f}+-{recall_av.std:.3f}',
                  'F1' : f'{f1_av.avg:3f}+-{f1_av.std:.3f}',
                  'Precision@1' : f'{precision_at_1_av.avg:.3f}+-{precision_at_1_av.std:.3f}',
                  'Recall@1' : f'{recall_at_1_av.avg:.3f}+-{recall_at_1_av.std:.3f}',
                  'F1@1' : f'{f1_at_1_av.avg:.3f}+-{f1_at_1_av.std:.3f}',
                  'Precision@2' : f'{precision_at_2_av.avg:.3f}+-{precision_at_2_av.std:.3f}',
                  'Recall@2' : f'{recall_at_2_av.avg:.3f}+-{recall_at_2_av.std:.3f}',
                  'F1@2' : f'{f1_at_2_av.avg:.3f}+-{f1_at_2_av.std:.3f}',
                  'ccc_arousal' : ccc_arousal,
                  'ccc_valence' : ccc_valence
                  }

    # Save result, confusion matrix
    with open(save_path + '/result.json', 'w') as f:
        json.dump(result, f, indent=2)
    try:
        f = open('./exp/'+args.test_path+'/confusion.txt','w')
    except:
        f = open(save_path+'/confusion.txt','w')
    f.write(str(total_confusion_matrix))
    print(result)
    print(f'Confusion Matrix : \n{total_confusion_matrix}')
    print("=================== Test End ====================")



def main():
    # define model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # wav audio file preprocessing
    if not os.path.exists('./audio'):
        print(' Audio 전처리부터 실행합니다... ')
        os.makedirs('./audio')
        for audio in glob('./data/KEMDy19/wav/*/*/*.wav'):
            new_audio = audio.split('\\')[-1]
            shutil.move(audio, f'./audio/{new_audio}')

    
    # Train and Test
    if args.test_only: # Test
        print(' Test만 진행됩니다... ')
        
        # dir에서 model 불러오기
        save_path = './exp/' + args.test_path
        if not os.path.exists(save_path):
            sys.exit('해당 경로의 학습된 모델이 존재하지 않습니다. 코드를 종료합니다.')

        with open(save_path+'/configuration.json', 'r') as f: 
            configuration = json.load(f)
        model = DoRaeMiSol_lstm(embed_d=configuration['embed_d'],
                                hidden_size=configuration['hidden_size'],
                                text_dim=configuration['text_dim'],
                                ws=configuration['ws'],
                                bidirectional=configuration['bidirectional']).cuda()
        model.load_state_dict(torch.load(save_path+'/model.pth'))
        test_dataset = EmotionDataset(split='test',
                                      test_split=ast.literal_eval(configuration['test_split']),
                                      SorL=configuration['SorL'],
                                      ws=configuration['ws'],
                                      text_dim=configuration['text_dim'])
        
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=1,shuffle=False,drop_last=False)
        test(model, test_loader, save_path)
        
    else: # Train and Test
        # Define model
        model = DoRaeMiSol_lstm(embed_d=args.embed_d, 
                                hidden_size=args.hidden_size, 
                                text_dim=args.text_dim, 
                                ws=args.ws, 
                                bidirectional=args.bidirectional).cuda()

        # Save path
        save_path = './exp/'+args.backbone+'_'+args.SorL+'_'+args.optim+'_'+str(args.ws)+'_'+str(time.time()).split('.')[-1]
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # dataset loading
        train_dataset = EmotionDataset(split='train',
                                       test_split=ast.literal_eval(args.test_split),
                                       SorL=args.SorL,
                                       ws=args.ws,
                                       text_dim=args.text_dim)
        test_dataset = EmotionDataset(split='test',
                                      test_split=ast.literal_eval(args.test_split),
                                      SorL=args.SorL,
                                      ws=args.ws,
                                      text_dim=args.text_dim)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=1,shuffle=True,drop_last=True)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=1,shuffle=False,drop_last=False)
        print(f'=== DataLoader R.e.a.d.y | Length : {len(train_dataset)} | {len(test_dataset)} ===')

        # define criterion
        if args.loss == 'normal':
            criterion = DoRaeMiSoLoss(lam=args.lam).cuda()
        else:
            criterion = CBDoRaeMiSoLoss(lam=args.lam, 
                                        beta=args.beta,
                                        df=train_dataset.df,
                                        ws_df=train_dataset.ws_df).cuda()
        # define Optimizer
        if args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        milestones = [int(args.epochs/3),int(args.epochs/2)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)

        # logger
        train_logger = Logger(os.path.join(save_path, 'train_loss.log'))

        # 학습시작
        for epoch in tqdm(range(1,args.epochs+1)):
            train(model, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
            scheduler.step()
            # 모델저장
            if epoch == args.epochs :
                path = '{0}/model.pth'.format(save_path)
                torch.save(model.state_dict(), path)    
        draw_curve(save_path, train_logger, train_logger)

        # 테스트시작
        test(model, test_loader, save_path)

    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))

if __name__ == '__main__':
    main()
