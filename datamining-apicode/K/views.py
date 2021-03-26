from django.shortcuts import render
import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow 
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
import warnings 


# Create your views here.

def Home(request):
    return render(request , 'Form.html')


def Result(request):
    agee = int(request.GET.get('age'))
    sexe = int(request.GET.get('sex'))
    cpe = int(request.GET.get('cp'))
    trestpbse = int(request.GET.get('trestpbs'))
    restecge = int(request.GET.get('restecg'))
    thalache = int(request.GET.get('thalach'))
    exange = int(request.GET.get('exang'))
    oldpeake = int(request.GET.get('oldpeak'))
    slopee = int(request.GET.get('slope'))
    cae = int(request.GET.get('ca'))
    thale = int(request.GET.get('thal'))
    targeet = int(request.GET.get('target'))


  
    with open('heart.csv', mode='w') as csv_file:
        fieldnames = ['agee', 'sexe', 'cpe' ,'trestpbse', 'restecge', 'thalache' , 'exange', 'oldpeake', 'slopee' , 'cae' , 'thale', 'targeet', ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'agee': agee , 'sexe': sexe, 'cpe':
        cpe ,'trestpbse': trestpbse , 'restecge': restecge, 'thalache': 
        thalache ,'exange': exange , 'oldpeake': oldpeake, 'slopee':
        slopee ,'cae': cae , 'thale': thale, 'targeet' :targeet })



    df=pd.read_csv("heart.csv") 
    model_rfc = 'Random Forest Classfier'
    rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)
    rf.fit(X_train,y_train)
    rf_predicted = rf.predict(X_test)
    rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
    rf_acc_score = accuracy_score(y_test, rf_predicted)
    print("confussion matrix")
    print(rf_conf_matrix)
    print("-------------------------------------------")
    print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
    print("-------------------------------------------")
    print(classification_report(y_test,rf_predicted))



    return render(request , 'Result.html' ,{'form' : agee })
