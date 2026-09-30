"""Consolidação dos módulos 39 e 40. Execute: python src/analise.py."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl-cache'))
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score, confusion_matrix)

FIG = ROOT / 'reports' / 'figures'
REPORT = ROOT / 'reports'
NAVY, TEAL, BLUE, ORANGE, GREY = '#16324f', '#007f78', '#4074b2', '#bd6132', '#6b7785'
COLORS = {'Referência majoritária': GREY, 'SVM Linear': BLUE,
          'SVM Polinomial': ORANGE, 'XGBoost': TEAL}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.labelcolor':NAVY,
    'text.color':NAVY, 'axes.titlecolor':NAVY, 'figure.facecolor':'#f6f8fb',
    'axes.facecolor':'white', 'savefig.facecolor':'#f6f8fb'})

def load_data():
    path = ROOT / 'data' / 'raw' / 'car_data.csv'
    raw = pd.read_csv(path, encoding='utf-8-sig')
    expected = ['User ID', 'Gender', 'Age', 'AnnualSalary', 'Purchased']
    if list(raw.columns) != expected:
        raise ValueError('O esquema da base mudou. Revise a fonte antes de executar.')
    if raw.isna().any().any() or raw.duplicated().any() or not raw['User ID'].is_unique:
        raise ValueError('Ausências ou registros repetidos exigem revisão explícita.')
    if set(raw.Gender) != {'Female','Male'} or set(raw.Purchased) != {0,1}:
        raise ValueError('Categorias inesperadas.')
    if not raw.Age.between(18,100).all() or not raw.AnnualSalary.ge(0).all():
        raise ValueError('Valores incompatíveis com o esquema esperado.')
    # Mapeamento fixo equivalente ao LabelEncoder do M40; não aprende categorias do teste.
    X = raw[['Age','AnnualSalary']].copy()
    X['Gender_encoded'] = raw.Gender.map({'Female':0,'Male':1})
    return raw, X, raw.Purchased, hashlib.sha256(path.read_bytes()).hexdigest()

def estimators():
    return {
      'Referência majoritária': DummyClassifier(strategy='most_frequent'),
      'SVM Linear': Pipeline([('scale', StandardScaler()), ('model', SVC(
          kernel='linear', C=1.0, probability=True, random_state=1))]),
      'SVM Polinomial': Pipeline([('scale', StandardScaler()), ('model', SVC(
          kernel='poly', C=1.0, degree=3, gamma='scale', coef0=0.0,
          probability=True, random_state=1))]),
      'XGBoost': xgboost.XGBClassifier(objective='binary:logistic',
          n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.90,
          colsample_bytree=0.90, eval_metric='logloss', random_state=42, n_jobs=1)
    }

def save(fig, filename, note):
    fig.text(.04,.025,note,fontsize=10,color=GREY)
    fig.savefig(FIG / filename,dpi=160)
    plt.close(fig)

def figures(raw, metrics, folds, cms, selected):
    fig, ax = plt.subplots(1,2,figsize=(14,7))
    fig.suptitle('Quem compra carros? Evidências do histórico',fontsize=23,fontweight='bold',y=.97)
    groups=raw.assign(faixa=pd.cut(raw.Age,[17,29,39,49,63],labels=['18–29','30–39','40–49','50–63']))
    age_summary=groups.groupby('faixa',observed=True).Purchased.agg(['mean','count'])
    bars_age=ax[0].bar(age_summary.index.astype(str),age_summary['mean']*100,color=TEAL,width=.6)
    ax[0].set(xlabel='Faixa de idade (anos)',ylabel='Compradores na faixa (%)',ylim=(0,115),
              title='Proporção de compradores por faixa de idade')
    for bar,(_,row) in zip(bars_age,age_summary.iterrows()):
        ax[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,
                   f"{row['mean']:.1%}\nn = {int(row['count'])}",ha='center',fontsize=11)
    count=raw.Purchased.value_counts().sort_index()
    bars=ax[1].bar(['Não comprou','Comprou'],count.values,color=[BLUE,TEAL],width=.55)
    ax[1].set(title='40,2% da amostra realizou a compra',ylabel='Clientes',ylim=(0,720))
    for bar,val in zip(bars,count.values):
        ax[1].text(bar.get_x()+bar.get_width()/2,val+15,f'{val} clientes\n{val/len(raw):.1%}',ha='center')
    fig.subplots_adjust(top=.82,bottom=.15,wspace=.25)
    save(fig,'01_eda.png','Fonte: Kaggle / Gabriel Santello | 1.000 registros | Valores agregados; associação não demonstra causalidade.')

    fig,ax=plt.subplots(figsize=(14,7))
    fig.suptitle('Comparação no teste: o equilíbrio entre encontrar e acertar',fontsize=22,fontweight='bold',y=.97)
    names=metrics.Modelo.tolist(); y=np.arange(len(names)); h=.23
    for k,(metric,label) in enumerate([('Precisão','Precisão'),('Recall','Recall'),('F1','F1')]):
        vals=metrics[metric].to_numpy()
        ax.barh(y+(k-1)*h,vals,h,label=label,color=[BLUE,ORANGE,TEAL][k])
        for pos,val in zip(y+(k-1)*h,vals): ax.text(val+.01,pos,f'{val:.1%}',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=names,xlim=(0,1.12),xlabel='Pontuação (0 a 1)')
    ax.invert_yaxis(); ax.legend(loc='lower right',frameon=False); ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.23,top=.83,bottom=.15)
    save(fig,'02_comparacao.png','Mesmo teste: 200 clientes, sendo 80 compradores | F1 considera precisão e recall da classe compradora.')

    fig,axes=plt.subplots(1,3,figsize=(15,6))
    fig.suptitle('Os erros revelam o custo da escolha do modelo',fontsize=23,fontweight='bold',y=.98)
    for ax,name in zip(axes,['SVM Linear','SVM Polinomial','XGBoost']):
        cm=np.array(cms[name]); ax.imshow(cm,cmap='Blues',vmin=0,vmax=120)
        for (i,j),val in np.ndenumerate(cm):
            ax.text(j,i,str(val),ha='center',va='center',fontsize=25,
                    color='white' if val>65 else NAVY)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Não compra','Compra'],
               yticklabels=['Não comprou','Comprou'],ylabel='Real',title=name)
        ax.text(.5,-.24,f'{cm[1,0]} compradores não identificados\n{cm[0,1]} contatos com não compradores',
                transform=ax.transAxes,ha='center',fontsize=12)
    fig.subplots_adjust(top=.8,bottom=.28,wspace=.4)
    fig.text(.5,.075,'Colunas: previsão do modelo',ha='center',fontsize=11)
    save(fig,'03_erros.png','Classe positiva: Purchased = 1 | Erros observados no teste; custos monetários não disponíveis.')

    fig,ax=plt.subplots(figsize=(14,7))
    fig.suptitle('Validação cruzada no treino: consistência antes do teste',fontsize=22,fontweight='bold',y=.97)
    for i,name in enumerate(metrics.Modelo):
        vals=folds.loc[folds.Modelo==name,'F1'].to_numpy()
        ax.scatter(np.arange(1,6)+(i-1.5)*.035,vals,label=name,color=COLORS[name],s=65)
        ax.plot(np.arange(1,6),vals,color=COLORS[name],alpha=.5)
    ax.set(xticks=np.arange(1,6),xlabel='Partição de validação',ylabel='F1 da classe compradora',ylim=(0,1))
    ax.grid(alpha=.15); ax.legend(loc='lower right',frameon=False)
    ax.set_title(f'Modelo indicado pelo maior F1 médio: {selected}',fontsize=14,pad=20)
    fig.subplots_adjust(top=.79,bottom=.14)
    save(fig,'04_validacao.png','5 partições estratificadas apenas nos 800 registros de treino | Sem busca de hiperparâmetros nesta consolidação.')

    m=metrics.set_index('Modelo').loc[selected]
    cm=np.array(cms[selected]); svm=np.array(cms['SVM Polinomial'])
    fig=plt.figure(figsize=(14,8)); fig.suptitle('Compra de carros | resultado da consolidação',fontsize=26,fontweight='bold',y=.94)
    cards=[('MODELO INDICADO',selected),('ACURÁCIA NO TESTE',f'{m.Acurácia:.1%}'),
           ('COMPRADORES ENCONTRADOS',f'{cm[1,1]} de {cm[1].sum()}'),('F1 NO TESTE',f'{m.F1:.3f}')]
    for i,(label,value) in enumerate(cards):
        x=.055+i*.24
        fig.text(x,.79,label,fontsize=10,color=GREY)
        fig.text(x,.72,value,fontsize=23,color=TEAL,fontweight='bold')
    ax=fig.add_axes([.17,.28,.31,.33]); sub=metrics[metrics.Modelo!='Referência majoritária']
    ax.barh(sub.Modelo,sub.F1,color=[COLORS[n] for n in sub.Modelo]);ax.invert_yaxis()
    ax.set(xlim=(0,1.1),xlabel='F1 da classe compradora',title='Comparação nos mesmos 200 clientes')
    for i,v in enumerate(sub.F1): ax.text(v+.015,i,f'{v:.3f}',va='center')
    fig.text(.56,.55,f'{cm[1,1]-svm[1,1]} compradores a mais identificados',fontsize=20,color=TEAL,fontweight='bold')
    fig.text(.56,.47,'em relação ao SVM polinomial, neste teste.\nA diferença não representa vendas adicionais.',fontsize=13,linespacing=1.6)
    fig.text(.56,.34,f'{cm[1,0]} compradores ficaram sem alerta.\n{cm[0,1]} não compradores receberam alerta.',fontsize=15,linespacing=1.6)
    fig.text(.055,.15,'Decisão: candidato a um piloto de priorização comercial, condicionado a validação com dados atuais.\nEfeito sobre conversão e retorno financeiro exige teste controlado; não foi medido nesta base.',fontsize=13,linespacing=1.8)
    save(fig,'05_resumo.png','Projeto acadêmico: módulos 39 e 40, consolidado para o módulo 41 | Fonte: Kaggle / Gabriel Santello.')

def run():
    FIG.mkdir(parents=True,exist_ok=True)
    raw,X,y,sha=load_data()
    train_idx,test_idx=train_test_split(np.arange(len(raw)),test_size=.2,random_state=42,stratify=y)
    Xt,Xv=X.iloc[train_idx],X.iloc[test_idx];yt,yv=y.iloc[train_idx],y.iloc[test_idx]
    # Pré-processamento faz parte do Pipeline e é reajustado em cada fold.
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    fold_rows=[]; fitted={}; predictions={}; cms={}; metric_rows=[]
    models=estimators()
    for name,model in models.items():
        scores=cross_validate(model,Xt,yt,cv=cv,n_jobs=1,
          scoring={'F1':'f1','Acurácia':'accuracy','ROC_AUC':'roc_auc','Balanced_Accuracy':'balanced_accuracy'})
        for fold in range(5):
            fold_rows.append({'Modelo':name,'Fold':fold+1,**{key:float(scores['test_'+key][fold])
                for key in ['F1','Acurácia','ROC_AUC','Balanced_Accuracy']}})
    folds=pd.DataFrame(fold_rows)
    cv_summary=folds.groupby('Modelo',sort=False)['F1'].agg(['mean','std']).reset_index()
    selected=cv_summary.loc[cv_summary['mean'].idxmax(),'Modelo']
    # Só após a comparação no treino, os quatro modelos são avaliados no teste para comunicação.
    for name,model in models.items():
        fitted[name]=clone(model).fit(Xt,yt)
        pred=fitted[name].predict(Xv); prob=fitted[name].predict_proba(Xv)[:,1]
        predictions[name]=(pred,prob)
        cms[name]=confusion_matrix(yv,pred,labels=[0,1]).tolist()
        metric_rows.append({'Modelo':name,'Acurácia':accuracy_score(yv,pred),
          'Precisão':precision_score(yv,pred,zero_division=0),'Recall':recall_score(yv,pred),
          'F1':f1_score(yv,pred),'ROC_AUC':roc_auc_score(yv,prob),
          'Balanced_Accuracy':balanced_accuracy_score(yv,pred)})
    metrics=pd.DataFrame(metric_rows)
    folds.to_csv(REPORT/'validacao_folds.csv',index=False)
    cv_summary.to_csv(REPORT/'validacao_resumo.csv',index=False)
    metrics.to_csv(REPORT/'metricas_teste.csv',index=False)
    pred_frame=pd.DataFrame({'User ID':raw.iloc[test_idx]['User ID'].to_numpy(),'real':yv.to_numpy()})
    for name,(pred,prob) in predictions.items():
        pred_frame[name+'_classe']=pred; pred_frame[name+'_probabilidade']=prob
    pred_frame.to_csv(REPORT/'previsoes_teste.csv',index=False)
    pd.DataFrame({'User ID':raw['User ID'],'particao':np.where(np.isin(np.arange(len(raw)),test_idx),'teste','treino')}).to_csv(REPORT/'particoes.csv',index=False)
    historical={'SVM Linear':{'Acurácia':.855,'Precisão':.8592,'Recall':.7625,'F1':.8079,'ROC_AUC':.9239},
       'SVM Polinomial':{'Acurácia':.875,'Precisão':.9104,'Recall':.7625,'F1':.8299,'ROC_AUC':.9310},
       'XGBoost':{'Acurácia':.915,'Precisão':.8889,'Recall':.9,'F1':.8944,'ROC_AUC':.9666}}
    quality={'linhas':len(raw),'colunas':len(raw.columns),'nulos':int(raw.isna().sum().sum()),
      'duplicados_completos':int(raw.duplicated().sum()),'ids_unicos':int(raw['User ID'].nunique()),
      'perfis_repetidos_sem_id':int(raw.drop(columns='User ID').duplicated().sum()),
      'perfis_X_teste_presentes_treino':int(Xv.apply(tuple,axis=1).isin(set(Xt.apply(tuple,axis=1))).sum()),
      'classes':{str(k):int(v) for k,v in y.value_counts().sort_index().items()},
      'treino':len(train_idx),'teste':len(test_idx),
      'teste_compradores':int(yv.sum()),'sha256':sha,'modelo_indicado_cv':selected,
      'matrizes_confusao':cms,'resultados_historicos_m40':historical,
      'versoes':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                 'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'matplotlib':matplotlib.__version__}}
    (REPORT/'resultados.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf8')
    raw.describe(include='all').to_csv(REPORT/'estatisticas_descritivas.csv')
    pd.concat([X,y],axis=1).corr().to_csv(REPORT/'correlacoes.csv')
    figures(raw,metrics,folds,cms,selected)
    print(metrics.round(4).to_string(index=False))
    print('\nCV F1:\n'+cv_summary.round(4).to_string(index=False))
    print('\nModelo indicado:',selected,'| SHA256:',sha)
    return quality,metrics,folds

if __name__=='__main__': run()
