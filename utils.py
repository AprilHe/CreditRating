'''
  
  
'''
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing,metrics 


# convert to time
def convert_date(data, date_field):
    for col in [col_name for col_name in data.columns if col_name in date_field]:
        data[col] = pd.to_datetime(data[col])   
        
    return data


'''

Categorical Fields: made up of character strings or being presented as factors
(1) count plot
(2) a plot of default rate by charactor 

'''
def plot_cat(col_name, full_name, label, df, i):
    
    f,ax=plt.subplots(1,2, figsize=(12,4))

    sns.countplot(df[col_name], color='#5975A4', saturation=1, ax=ax[0])
    ax[0].set_title('Distribution of ' + full_name, fontsize = 14)
    ax[0].set_ylabel('Count', fontsize = 14)
    ax[0].set_xlabel('')
    
    
    sns.barplot(col_name,label, data=df, ax=ax[1])
    ax[1].set_title('Default rate by ' + full_name, fontsize = 14)
    ax[1].set_ylabel('Fraction', fontsize = 14)
    ax[1].set_xlabel(full_name, fontsize = 14)
    
    f.autofmt_xdate(rotation=45)
    plt.tight_layout()
    # plt.show()
    # if i % 10 == 1: plt.show()
    f.savefig("plots/categorical/{}-{}.png".format(i,full_name))
    plt.close()
    

    
    
'''
Numerical Fields: discrete numeric or continus numeric.
For continuous numeric, the output is 
(1) a histogram, 
(2) the ROC plot, 
(3) a plot of default rate by quantile 
(4) a plot of smoothed bad rate by using a degree 2 loess smother with 95% confidence intervals

For discrete numeric,
(1) a histogram, 
(2) the ROC plot, 
(3) a single bad rate over value plot, again showns on log-odds scale but labbelled in percent.
This plot can also optionally be overlayed with a single-variables logistic regression fit.

'''

def plot_num(i, col_name, full_name, label, df, path):
    
    df = df[[col_name, label]].sort_values(col_name)
    tmp = df.dropna(axis=0)
    
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2, 2)
    fig.subplots_adjust(top=0.3)

    # 1) Histogram Plot
    ax = fig.add_subplot(gs[0, :1])
    ax = sns.distplot(tmp[col_name], kde=False)
    ax.set(xlabel=col_name, title='Histogram')
    
    # 2) ROC Plot
    ax = fig.add_subplot(gs[0, 1:])
    clf = LogisticRegression()
    X = tmp[col_name].values.reshape(-1,1)
    y = tmp[label].values
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y,  y_pred)
    auc = metrics.roc_auc_score(y, y_pred)
    plt.plot(fpr,tpr,label=  full_name + '(AUC = %0.2f)' %auc)
    plt.legend(loc=4)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curve')
    
    
    # 3) Default rate by quantile
    if tmp[col_name].nunique() > 30:
        # continuous numeric
        df['group'] = pd.qcut(df[col_name], 10, duplicates='drop')
        df['group'] = np.where(df.group.isna(), 'missing', df.group).astype(str)
        df_plot = df.groupby('group').agg({col_name: min,label: np.mean}).reset_index().rename(columns={'loan_status':'default_rate'}).sort_values(col_name).reset_index(drop = True)
        df_plot['x'] = df_plot.index
        ax = fig.add_subplot(gs[1, :])
        sns.lineplot(data=df_plot, x="x", y='default_rate')
        plt.xticks(df_plot.x, df_plot.group, rotation=60)
        ax.set(xlabel=col_name, ylabel = 'Default Rate', title='Bad Rate by Quantile')
    else:
        # discrete numeric
        df_plot = df.groupby(col_name)[label].mean().reset_index().rename(columns={'loan_status':'default_rate'})
        ax = fig.add_subplot(gs[1, :])
        sns.lineplot(data=df_plot, x=col_name, y='default_rate')
        ax.set(xlabel=col_name, ylabel = 'Default Rate', title='Bad Rate by Group')
    
    
    plt.tight_layout()
    fig.suptitle('Feature: {}, N: {}, Min: {}, Max: {}, NAs: {:.0%}'.format(col_name, df.shape[0], df[col_name].min(), df[col_name].max(), df[col_name].isna().mean()  ), y =0.98)
    fig.savefig("plots/{}/{}-{}.png".format(path,i,full_name))
    plt.close()
    # plt.show()
    
    


import seaborn as sns
sns.set('talk', 'whitegrid', 'dark', font_scale=1, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
def plotAUC(truth, pred, lab):
    fpr, tpr, _ = metrics.roc_curve(truth,pred)
    roc_auc = metrics.auc(fpr, tpr)
    lw = 2
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color= c,lw=lw, label= lab +'(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve') #Receiver Operating Characteristic 
    plt.legend(loc="lower right")
    
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model, normalize=False): # This function prints and plots the confusion matrix.
    cm = confusion_matrix(y_test, model, labels=[0, 1])
    classes=["Will Pay", "Will Default"]
    cmap = plt.cm.Blues
    title = "Confusion Matrix"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')  
