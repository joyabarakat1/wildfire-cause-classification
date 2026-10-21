"""Train proposal baselines from scratch with NumPy; never predict test rows.
Run: python train_numpy_baselines.py --database PATH --output baseline_results
Requires numpy, pandas; keep baseline.py beside this file for its audited loader.
"""
from pathlib import Path
import argparse, ast, json, sqlite3, time, sys
import numpy as np
import pandas as pd

def encode(frame, location=False, mean=None, sd=None):
    # Reference categories January and Monday; explicit intercept.
    blocks=[np.ones((len(frame),1))]
    blocks += [(frame.month.to_numpy()[:,None]==np.arange(2,13)).astype(float)]
    blocks += [(frame.weekday.to_numpy()[:,None]==np.arange(1,7)).astype(float)]
    names=['intercept']+[f'month_{i}' for i in range(2,13)]+[f'weekday_{i}' for i in range(1,7)]
    if location:
        coordinates=frame[['LATITUDE','LONGITUDE']].to_numpy(dtype=float)
        if mean is None:
            mean=coordinates.mean(axis=0)
            sd=coordinates.std(axis=0)
        blocks.append((coordinates-mean)/sd)
        names+=['latitude_standardized','longitude_standardized']
    return np.column_stack(blocks),mean,sd,names

def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-z))

def loss(X,y,w):
    z=X@w
    return float(np.mean(np.logaddexp(0.,z)-y*z))

def fit(X,y,Xv,yv,l2=.001,max_iter=100,tolerance=1e-8):
    # Mean binary cross-entropy + (lambda/2)*sum(non-intercept weights squared).
    n,p=X.shape
    penalty=np.eye(p)*l2;penalty[0,0]=0
    w=np.zeros(p)
    history=[]
    objective=lambda q:loss(X,y,q)+.5*float(q@penalty@q)
    converged=False
    for iteration in range(1,max_iter+1):
        probability=sigmoid(X@w)
        gradient=X.T@(probability-y)/n+penalty@w
        gnorm=float(np.max(np.abs(gradient)))
        history.append({'iteration':iteration-1,'train_cross_entropy':loss(X,y,w),
                        'validation_cross_entropy':loss(Xv,yv,w),'gradient_inf_norm':gnorm})
        if gnorm<tolerance:
            converged=True;break
        curvature=probability*(1-probability)
        hessian=penalty.copy()
        for start in range(0,n,20000):
            part=X[start:start+20000]
            hessian+=part.T@(part*curvature[start:start+20000,None])/n
        direction=np.linalg.solve(hessian+np.eye(p)*1e-12,gradient)
        step=1.
        old=objective(w)
        while objective(w-step*direction)>old-1e-4*step*float(gradient@direction):
            step*=.5
            if step<1e-10:raise RuntimeError('Line search failed')
        w-=step*direction
    if not converged:raise RuntimeError('Model did not converge')
    return w,history

def metrics(y,pred):
    matrix=np.array([[np.sum((y==a)&(pred==b)) for b in [0,1]] for a in [0,1]])
    tp=np.diag(matrix).astype(float)
    precision=np.divide(tp,matrix.sum(0),out=np.zeros(2),where=matrix.sum(0)!=0)
    recall=tp/matrix.sum(1)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(2),where=(precision+recall)!=0)
    return {'accuracy':float(tp.sum()/matrix.sum()),'balanced_accuracy':float(recall.mean()),
            'macro_f1':float(f1.mean()),'natural_precision':float(precision[0]),
            'natural_recall':float(recall[0]),'natural_f1':float(f1[0]),
            'human_precision':float(precision[1]),'human_recall':float(recall[1]),
            'human_f1':float(f1[1]),'confusion_matrix':matrix.tolist()}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('baseline_results'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('baseline.py').read_text(encoding='utf-8')
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='load_data')
    exec(compile(ast.Module(body=[function],type_ignores=[]),'audited_loader','exec'))
    data,audit=load_data(args.database)
    train=data[data.FIRE_YEAR<=2014].copy()
    validation=data[data.FIRE_YEAR.between(2015,2017)].copy()
    assert len(train)==180656 and len(validation)==17559
    y=train.target.to_numpy(dtype=float);yv=validation.target.to_numpy(dtype=float)
    rows=[];details={}
    majority=int(np.bincount(y.astype(int)).argmax())
    for split,labels in [('Training',y),('Validation',yv)]:
        result=metrics(labels,np.full(len(labels),majority))
        details[f'Majority / {split}']=result
        rows.append({'model':'Majority','split':split,**{k:v for k,v in result.items() if k!='confusion_matrix'}})
    for location in [False,True]:
        name='Logistic: calendar + location' if location else 'Logistic: calendar only'
        X,mean,sd,names=encode(train,location)
        Xv,_,_,_=encode(validation,location,mean,sd)
        start=time.perf_counter()
        weights,history=fit(X,y,Xv,yv)
        elapsed=time.perf_counter()-start
        print(name,'converged after',len(history)-1,'updates; seconds:',round(elapsed,2),flush=True)
        stem='calendar_location' if location else 'calendar'
        pd.DataFrame(history).to_csv(out/(stem+'_loss.csv'),index=False)
        pd.DataFrame({'coefficient':names,'weight':weights}).to_csv(out/(stem+'_weights.csv'),index=False)
        if location:(out/'coordinate_scaling.json').write_text(json.dumps({'mean':mean.tolist(),'sd':sd.tolist()}))
        for split,inputs,labels in [('Training',X,y),('Validation',Xv,yv)]:
            prediction=(sigmoid(inputs@weights)>=.5).astype(int)
            result=metrics(labels,prediction)
            details[f'{name} / {split}']={**result,'cross_entropy':loss(inputs,labels,weights)}
            rows.append({'model':name,'split':split,**{k:v for k,v in result.items() if k!='confusion_matrix'}})
    table=pd.DataFrame(rows)
    table.to_csv(out/'baseline_metrics.csv',index=False)
    (out/'details.json').write_text(json.dumps(details,indent=2))
    (out/'run_metadata.json').write_text(json.dumps({'implementation':'NumPy Newton method with backtracking',
        'l2_lambda':.001,'objective':'mean cross entropy + lambda/2 * squared non-intercept weights',
        'threshold':.5,'class_weighting':'none','gradient_tolerance':1e-8,
        'test_predictions_computed':False,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},indent=2))
    print(table.to_string(index=False))
