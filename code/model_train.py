import pandas as pd
import mxnet
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import train_test_split

#=========================================================#

##나중에 argpaser로 변경
max_grad_norm = 1
log_interval = 1000
warmup_ratio = 0.1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type = int, default=5)
parser.add_argument('--batch_size', type = int, default=16)
parser.add_argument('--max_len', type = int, default=512)
parser.add_argument('--learning_rate', type = float, default=5e-5)

args = parser.parse_args()

print(f"build model with {args.num_epochs}epochs and {args.batch_size}batch size")
#########################


#GPU사용
device = torch.device("cuda:1")
print(f"Using {device}")


data = pd.read_pickle("../result/sample5_tokenized.pkl")
##############

print(data.info())

#Label
label_to_int = {}
for i, item in enumerate(data['접수기관'].unique()):
    label_to_int[item] = i


data['접수기관'] = data['접수기관'].apply(lambda x : label_to_int[x])
data = data[['token', '접수기관']]


dataset_train, dataest_test = train_test_split(data, test_size=0.1, random_state=42)


print(f"train len : {len(dataset_train)}, test len : {len(dataest_test)}")



class BERTDataset(Dataset):
    def __init__(self, dataset,bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([" ".join(dataset.iloc[i]['token'])]) for i in range(len(dataset))]
        self.labels = [np.int32(dataset.iloc[i]['접수기관']) for i in range(len(dataset))]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))



print('get bertmodel and vocab')
bertmodel, vocab = get_pytorch_kobert_model()


print("data setting")
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

train_data = BERTDataset(dataset_train, tok, args.max_len, True, False)
test_data = BERTDataset(dataest_test, tok, args.max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size = args.batch_size, num_workers = 8)

test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size = args.batch_size, num_workers = 8)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = len(dataset_train['접수기관'].unique()),   ##클래스 수 조정##
                 dr_rate = None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(
            input_ids = token_ids, token_type_ids = segment_ids.long(), 
            attention_mask = attention_mask.float().to(token_ids.device)
        )
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


    #BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * args.num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
    
train_dataloader


print("Train Start")
for e in range(args.num_epochs):
    train_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} / {} loss {} train acc {}".format(e+1, batch_id+1 , len(train_dataloader), loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
    model.eval()
    test_acc = 0.0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))



###########################SAVE

PATH = '../result/kobert/' # google 드라이브 연동 해야함. 관련코드는 뺐음
torch.save(model, PATH + 'KoBERT_0705_e15_py.pt')  # 전체 모델 저장
torch.save(model.state_dict(), PATH + 'Kobert_0705_e15_py_state_dict.pt')  # 모델 객체의 state_dict 저장
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all.tar') 