# importing libraries
import plotly.express as px
import pandas as pd
import numpy as np


# ROC curve
def roc_plot(df, y_true, y_pred):
    """
    df - датафрейм
    y_true - название колонки с целевой переменной
    y_pred - название колонки с предсказаной вероятностью
    """
    tpr = [0]
    fpr = [0]
    
    # your code here
    cutoff = .5  # cut-off value
    pred = []

    # comapring logreg results with cutoff value
    for i in range(df.shape[0]):
        if df[y_pred].iloc[i] < cutoff:
            pred.append(0)
        else:
            pred.append(1)
    df['pred'] = pred

    tp = []
    p = []
    fp = []
    f = []

    for i in range(df.shape[0]):
        # finding true positives
        tp.append(df[y_true].iloc[i])
        tp_sum = sum(tp)

        # find all positives
        if df['pred'].iloc[i] == 1:
            p.append(1)
        else:
            p.append(0)
        p_sum = sum(p)

        # calc tpr
        if np.isnan(tp_sum/(tp_sum + p_sum)):
             tpr.append(0)
        else:
            tpr.append(tp_sum/(tp_sum + p_sum + 0.0000001))

        # finding a number of false positive predictions
        if df['pred'].iloc[i] == 1 and df[y_true].iloc[i] != 1:
            fp.append(1)
        else:
            fp.append(0)
        fp_sum = sum(fp)

         # finding a number of all negatives predictions
        if df[y_true].iloc[i] == 0:
            f.append(1)
        else:
            f.append(0)
        f_sum = sum(f)

        # calc fpr
        fpr.append(fp_sum/(fp_sum + f_sum + 0.0000001))
            
    # adding last points
    fpr.append(1)
    tpr.append(1)
    
    # converting to series to plot
    fprs = pd.Series(fpr)
    tprs = pd.Series(tpr)

    fig = px.line(x=fprs, y=tprs, width=500, height=500,  title='ROC кривая')
    fig.update_xaxes(title_text='FPR')
    fig.update_yaxes(title_text='TPR')
    
    fig.show()
    
    return tpr, fpr


# Information Value
def IV(df,feature, target, num_buck = 4):
    """
    df - датафрейм
    feature - название признака для которого считать IV
    num_buck - число бакетов
    """
    iv = []
    # assigning buckets
    df = df.assign(bucket = np.ceil(df[feature].rank(pct = True) * num_buck))
    
    
    # your code here
    # looping through buckets and calculating woe & iv
    for i in range(1, num_buck+1):
        df_t = df[df['bucket'] == i] 
        
        feature00_cnt = df_t[df_t[target] == 0][feature].count()
        target00_cnt = df[df[target] == 0][target].count()
        feature01_cnt = df_t[df_t[target] == 1][feature].count()
        target01_cnt = df[df[target] == 1][target].count()

        woe0 = np.log((feature00_cnt/target00_cnt) / (feature01_cnt/target01_cnt) + 0.00001)

        iv0 = (feature00_cnt/target00_cnt - feature01_cnt/target01_cnt) * woe0
        iv.append(iv0)
        
        print(f'Bucket #{i}: WoE = {woe0: <21}| IV = {iv0}')

    return iv


# Hosmer–Lemeshow Statistics
def custom_HL(df, predict, target, n_buck = 10):
    """
    predict - массив предсказний
    target - массив истинных значений флага дефолта
    n_buck - число бакетов, по умолчанию 10
    Разбиение на бакеты реализовано в plot_gain_chart
    """
    # your code here
    h = []
    
    # bucketting 
    df = df.assign(bucket = np.ceil(df[predict].rank(pct = True) * num_buck))
    
    # looping through buckets
    for i in range(1, num_buck+1):
        df_t = df[df['bucket'] == i]
        
        pd = df_t[predict].mean()
        badrate = df_t[target].sum() / df_t.shape[0]

        h.append(((pd - badrate) ** 2 / (pd * (1-pd))) * df_t.shape[0])
    
    H = sum(h)
    
    return H