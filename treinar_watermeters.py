"""VC001: treinar e avaliar os quatro classificadores de dígitos WaterMeters.
Execute na raiz do repositório após gerar digits/index.csv e os PNG.
O CSV de entrada pode conter orientações e recortes ainda não validados;
os resultados NÃO são uma garantia da qualidade dos dados.
"""
import argparse
import copy
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
FIELDS = ['model','acc_digit','f1_macro','acc_reading','params_M','s_per_epoch','ms_per_image','ckpt_MB','best_epoch']
MODELS = ['smallcnn','alexnet','vgg16_bn','resnet18']
CONFIG = {
    'smallcnn': {'epochs': 10, 'batch_cpu':128, 'batch_cuda':256, 'head_lr':1e-3,'backbone_lr':None,'size':(64,48)},
    'alexnet': {'epochs': 8, 'batch_cpu':64, 'batch_cuda':128, 'head_lr':1e-3,'backbone_lr':1e-4,'size':(128,128)},
    'vgg16_bn': {'epochs': 8, 'batch_cpu':16, 'batch_cuda':32, 'head_lr':1e-3,'backbone_lr':1e-5,'size':(128,128)},
    'resnet18': {'epochs': 8, 'batch_cpu':64, 'batch_cuda':128, 'head_lr':1e-3,'backbone_lr':1e-4,'size':(128,128)},
}

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

class Digits(Dataset):
    def __init__(self, data, model, training=False):
        self.frame = data.reset_index(drop=True)
        c=CONFIG[model]
        pipeline=[transforms.Resize(c['size']), transforms.RandomAffine(degrees=5,translate=(0.05,0.05),scale=(0.95,1.05)), transforms.ColorJitter(brightness=.3,contrast=.3)] if training else [transforms.Resize(c['size'])]
        pipeline += [transforms.ToTensor()]
        if model != 'smallcnn': pipeline += [transforms.Normalize((.485,.456,.406),(.229,.224,.225))]
        self.transform=transforms.Compose(pipeline)
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r=self.frame.iloc[i]
        with Image.open(r['crop_path']) as im: x=self.transform(im.convert('RGB'))
        return x, int(r['digit']), int(i)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,3)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128*4*3,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,10))
    def forward(self,x):return self.head(self.features(x))

def build_model(name):
    if name=='smallcnn': return SmallCNN(), None
    if name=='alexnet':
        m=models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6]=nn.Linear(m.classifier[6].in_features,10)
        head=list(m.classifier.parameters())
    elif name=='vgg16_bn':
        m=models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        m.classifier[6]=nn.Linear(m.classifier[6].in_features,10)
        head=list(m.classifier.parameters())
    elif name=='resnet18':
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc=nn.Linear(m.fc.in_features,10)
        head=list(m.fc.parameters())
    return m, {id(p) for p in head}

def evaluate(model, loader, device, criterion):
    model.eval(); loss_sum=0; ys=[]; preds=[]; inds=[]; conf=[]
    with torch.inference_mode():
        for x,y,idx in loader:
            x=x.to(device);y=y.to(device)
            out=model(x); loss_sum+=criterion(out,y).item()*len(y)
            prob=out.softmax(1);p=prob.argmax(1)
            ys.extend(y.cpu().tolist()); preds.extend(p.cpu().tolist()); inds.extend(idx.tolist()); conf.extend(prob.max(1).values.cpu().tolist())
    return {'loss':loss_sum/len(loader.dataset), 'acc':accuracy_score(ys,preds), 'f1':f1_score(ys,preds,labels=list(range(10)),average='macro',zero_division=0), 'y':ys, 'p':preds, 'indices':inds, 'confidence':conf}

def create_optimizer(model, name, head_ids):
    c=CONFIG[name]
    if head_ids is None:
        return torch.optim.AdamW(model.parameters(),lr=c['head_lr'])
    head=[p for p in model.parameters() if id(p) in head_ids]
    backbone=[p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([{'params':head,'lr':c['head_lr'],'max_lr':c['head_lr']},{'params':backbone,'lr':c['backbone_lr'],'max_lr':c['backbone_lr']}])

def lr_list(name):
    c=CONFIG[name]
    return c['head_lr'] if name=='smallcnn' else [c['head_lr'],c['backbone_lr']]

def train_one(name, df, device, outdir, workers=0):
    c=CONFIG[name]
    bs=c['batch_cuda'] if device.type=='cuda' else c['batch_cpu']
    loaders={s:DataLoader(Digits(df[df['split']==s],name,training=s=='train'),batch_size=bs if s=='train' else min(bs,64),shuffle=s=='train',num_workers=workers,pin_memory=device.type=='cuda') for s in ['train','val','test']}
    print(f'\nMODELO {name}: epochs={c["epochs"]}, batch={bs}, device={device}',flush=True)
    model,head_ids=build_model(name);model.to(device)
    optimizer=create_optimizer(model,name,head_ids)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr_list(name),epochs=c['epochs'],steps_per_epoch=len(loaders['train']))
    criterion=nn.CrossEntropyLoss(label_smoothing=.1)
    best=-1;best_epoch=0;no_improve=0;history=[]
    ckpt=outdir/'checkpoints'/f'{name}.pth';ckpt.parent.mkdir(parents=True,exist_ok=True)
    if head_ids is not None:
        for p in model.parameters():
            if id(p) not in head_ids:p.requires_grad_(False)
    for epoch in range(1,c['epochs']+1):
        if epoch==2 and head_ids is not None:
            for p in model.parameters():p.requires_grad_(True)
        model.train();total_loss=0;tr_pred=[];tr_y=[]
        start=time.perf_counter()
        for x,y,_ in loaders['train']:
            x=x.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out=model(x); loss=criterion(out,y);loss.backward();optimizer.step();scheduler.step()
            total_loss+=loss.item()*len(y);tr_pred.extend(out.argmax(1).detach().cpu().tolist());tr_y.extend(y.cpu().tolist())
        seconds=time.perf_counter()-start
        val=evaluate(model,loaders['val'],device,criterion)
        history.append({'epoch':epoch,'train_loss':total_loss/len(tr_y),'train_acc':accuracy_score(tr_y,tr_pred),'val_loss':val['loss'],'val_acc':val['acc'],'val_f1':val['f1'],'seconds':seconds})
        pd.DataFrame(history).to_csv(outdir/f'history_{name}.csv',index=False)
        print(f'  {name} ep {epoch}: train_acc={history[-1]["train_acc"]:.4f} val_f1={val["f1"]:.4f} train_s={seconds:.1f}',flush=True)
        if val['f1']>best+1e-8:
            best=val['f1'];best_epoch=epoch;no_improve=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'val_f1':best},ckpt)
        else:no_improve+=1
        if no_improve>=3: print('  early stopping',flush=True);break
    try: state=torch.load(ckpt,map_location=device,weights_only=True)
    except TypeError: state=torch.load(ckpt,map_location=device)
    model.load_state_dict(state['model']);test=evaluate(model,loaders['test'],device,criterion)
    testframe=df[df['split']=='test'].reset_index(drop=True).iloc[test['indices']].copy()
    testframe['pred']=test['p'];testframe['confidence']=test['confidence']
    correct=testframe.assign(ok=testframe['digit'].astype(int)==testframe['pred']).groupby('photo_name')['ok'].agg(['all','size'])
    acc_reading=float(correct.loc[correct['size']==8,'all'].mean()) if (correct['size']==8).any() else float('nan')
    (outdir/'reports').mkdir(exist_ok=True)
    report=classification_report(test['y'],test['p'],labels=list(range(10)),zero_division=0)
    (outdir/'reports'/f'classification_report_{name}.txt').write_text(report,encoding='utf-8')
    testframe.to_csv(outdir/'reports'/f'predictions_{name}.csv',index=False)
    cm=confusion_matrix(test['y'],test['p'],labels=list(range(10)),normalize='true')
    np.save(outdir/'reports'/f'confusion_{name}.npy',cm)
    per_position=testframe.assign(ok=testframe['digit'].astype(int)==testframe['pred']).groupby('position')['ok'].mean()
    per_position.to_csv(outdir/'reports'/f'accuracy_by_position_{name}.csv')
    errors=testframe[testframe['pred']!=testframe['digit'].astype(int)].sort_values('confidence',ascending=False).head(16)
    fig,axes=plt.subplots(4,4,figsize=(11,11))
    for ax,(_,row) in zip(axes.flat,errors.iterrows()):
        with Image.open(row['crop_path']) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f"real {row['digit']} / pred {row['pred']} / {row['confidence']:.2f}",fontsize=8);ax.axis('off')
    for ax in axes.flat[len(errors):]:ax.axis('off')
    fig.tight_layout();(outdir/'data'/'error_galleries').mkdir(parents=True,exist_ok=True);fig.savefig(outdir/'data'/'error_galleries'/f'errors_{name}.png',dpi=120);plt.close(fig)
    # Benchmark batch 1: 20 warmup + 200 timed trials; sync GPU for meaningful time.
    sample_x,_,_=loaders['test'].dataset[0];sample=sample_x.unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(20):model(sample)
        if device.type=='cuda':torch.cuda.synchronize()
        timings=[]
        for _ in range(200):
            t=time.perf_counter();model(sample)
            if device.type=='cuda':torch.cuda.synchronize()
            timings.append((time.perf_counter()-t)*1000)
    row={'model':name,'acc_digit':float(test['acc']),'f1_macro':float(test['f1']),'acc_reading':acc_reading,'params_M':sum(p.numel() for p in model.parameters())/1e6,'s_per_epoch':float(np.median([h['seconds'] for h in history])),'ms_per_image':float(np.median(timings)),'ckpt_MB':ckpt.stat().st_size/1024**2,'best_epoch':best_epoch}
    return row

def create_figures(outdir, names):
    if not names:return
    (outdir/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,len(names),figsize=(5*len(names),4),squeeze=False)
    for i,name in enumerate(names):
        cm=np.load(outdir/'reports'/f'confusion_{name}.npy')
        axes[0,i].imshow(cm,vmin=0,vmax=1,cmap='Blues');axes[0,i].set_title(name);axes[0,i].set_xlabel('Predito');axes[0,i].set_ylabel('Real');axes[0,i].set_xticks(range(10));axes[0,i].set_yticks(range(10))
    fig.tight_layout();fig.savefig(outdir/'figures'/'confusion_matrices.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for name in names:
        s=pd.read_csv(outdir/'reports'/f'accuracy_by_position_{name}.csv');ax.plot(s.iloc[:,0],s.iloc[:,1],marker='o',label=name)
    ax.set(xlabel='Posição (0–7)',ylabel='Acurácia',ylim=(0,1),title='Acurácia por posição');ax.legend();fig.tight_layout();fig.savefig(outdir/'figures'/'accuracy_by_position.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in names:
        h=pd.read_csv(outdir/f'history_{name}.csv')
        axes[0].plot(h['epoch'],h['train_loss'],label=f'{name} treino');axes[0].plot(h['epoch'],h['val_loss'],linestyle='--',label=f'{name} val')
        axes[1].plot(h['epoch'],h['train_acc'],label=f'{name} treino');axes[1].plot(h['epoch'],h['val_acc'],linestyle='--',label=f'{name} val')
    axes[0].set(xlabel='Época',ylabel='Loss');axes[1].set(xlabel='Época',ylabel='Acurácia',ylim=(0,1))
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(outdir/'figures'/'training_curves.png',dpi=150);plt.close(fig)
    r=pd.read_csv(outdir/'results.csv');fig,ax=plt.subplots(figsize=(7,5))
    for _,v in r.iterrows():ax.scatter(v['ms_per_image'],v['f1_macro'],s=90);ax.annotate(v['model'],(v['ms_per_image'],v['f1_macro']))
    ax.set(xlabel='Inferência (ms/imagem)',ylabel='F1 macro',title='Desempenho × custo')
    fig.tight_layout();fig.savefig(outdir/'figures'/'performance_vs_cost.png',dpi=150);plt.close(fig)

def main():
    a=argparse.ArgumentParser();a.add_argument('--index',default='digits/index.csv');a.add_argument('--out',default='.');a.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);a.add_argument('--workers',type=int,default=0);args=a.parse_args()
    set_seed(); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    data=pd.read_csv(args.index,dtype={'photo_name':'string','image_id':'string','split':'string','crop_path':'string'})
    assert len(data)==9944, f'Esperado 9944 recortes, encontrados {len(data)}'
    assert data['photo_name'].nunique()==1243
    assert not data.duplicated(['photo_name','position']).any()
    assert data.groupby('photo_name')['split'].nunique().eq(1).all()
    assert data.groupby('photo_name')['position'].agg(lambda s: set(s)==set(range(8))).all()
    for p in data['crop_path']:
        if not Path(p).is_file():raise FileNotFoundError(p)
    print(f'Iniciando com {len(data)} recortes. Device={device}. AVISO: orientações e visores com 5 dígitos não foram corrigidos automaticamente.',flush=True)
    rows=[];existing=out/'results.csv'
    if existing.exists():rows=pd.read_csv(existing).to_dict('records')
    for name in args.models:
        rows=[r for r in rows if r['model']!=name]
        row=train_one(name,data,device,out,args.workers);rows.append(row)
        pd.DataFrame(rows,columns=FIELDS).to_csv(existing,index=False)
        create_figures(out,[r['model'] for r in rows]);print('RESULTADO',row,flush=True)
    print('Concluído. Galerias de erros salvas em data/error_galleries (não versionar).',flush=True)

if __name__=='__main__':main()
