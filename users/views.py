# Create your views here.
import os.path

import pandas as pd
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    path = os.path.join(settings.MEDIA_ROOT, 'datasets', '08 disease-burden-rates-by-cancer-types.csv')
    if request.method == 'POST':
        countyCode = request.POST.get('countryCode')
        df = pd.read_csv(path)
        country = df['Entity'].unique()
        df = df[df.Entity.isin([countyCode])]
        df = df.rename(columns={df.columns[3]: 'Other pharyx', df.columns[4]: 'Liver', df.columns[5]: 'Breast',
                                df.columns[6]: 'Tracheal', df.columns[7]: 'Gallbladder & bilary tract',
                                df.columns[8]: 'Kidney',
                                df.columns[9]: 'Larynx', df.columns[10]: 'Esophageal', df.columns[11]: 'Nasopharynx',
                                df.columns[12]: 'Colon & rectum', df.columns[13]: 'Non-melanoma skin',
                                df.columns[14]: 'lip & oral',
                                df.columns[15]: 'Malignant skin melanoma', df.columns[16]: 'Other malignant neoplasms',
                                df.columns[17]: 'Mesothelioma', df.columns[18]: 'Hodgkin lymphoma',
                                df.columns[19]: 'Non-Hodgkin lymphoma'})
        df = df.drop(df.columns[[3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]], axis=1)
        df.dropna(inplace=True)
        df = df.to_html(index=False)

        return render(request, 'users/viewdataset.html', {'data': df, 'county': country})

    else:
        df = pd.read_csv(path)
        country = df['Entity'].unique()
        df = df.rename(columns={df.columns[3]: 'Other pharyx', df.columns[4]: 'Liver', df.columns[5]: 'Breast',
                                df.columns[6]: 'Tracheal', df.columns[7]: 'Gallbladder & bilary tract',
                                df.columns[8]: 'Kidney',
                                df.columns[9]: 'Larynx', df.columns[10]: 'Esophageal', df.columns[11]: 'Nasopharynx',
                                df.columns[12]: 'Colon & rectum', df.columns[13]: 'Non-melanoma skin',
                                df.columns[14]: 'lip & oral',
                                df.columns[15]: 'Malignant skin melanoma', df.columns[16]: 'Other malignant neoplasms',
                                df.columns[17]: 'Mesothelioma', df.columns[18]: 'Hodgkin lymphoma',
                                df.columns[19]: 'Non-Hodgkin lymphoma'})

        df = df.drop(df.columns[[3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]], axis=1)
        df.dropna(inplace=True)
        df = df.to_html(index=False)

        return render(request, 'users/viewdataset.html', {'data': df, 'county': country})


def UserRegressions(request):
    if request.method == 'POST':
        path = os.path.join(settings.MEDIA_ROOT, 'datasets', '08 disease-burden-rates-by-cancer-types.csv')
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        countryCode = request.POST.get('countryCode')
        cancerType = request.POST.get('cancerType')
        df = df[df.Code.isin([countryCode])]
        # print(df.head())

        X = df['Year'].to_list()
        y = df[cancerType].to_list()
        X_X = []
        y_y = []
        for i in X:
            X_X.append([i])
        for j in y:
            y_y.append([j])

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_X, y_y, test_size=0.2, random_state=42)

        from .utility import CancerRegressionModel
        lr_dict = CancerRegressionModel.process_LinearRegression(X_train, X_test, y_train, y_test)
        dt_dict = CancerRegressionModel.process_decesionTree(X_train, X_test, y_train, y_test)
        rf_dict = CancerRegressionModel.process_randomForest(X_train, X_test, y_train, y_test)
        pf_dict = CancerRegressionModel.process_polynomialRegressor(X_train, X_test, y_train, y_test)
        print("My Predictions:", lr_dict)
        # return render(request, 'users/cl_reports.html',
        #               {'rf': rf_report.to_html, 'dt': dt_report.to_html, 'nb': nb_report.to_html, 'gb': gb_report.to_html})
        path = os.path.join(settings.MEDIA_ROOT, 'datasets', '08 disease-burden-rates-by-cancer-types.csv')
        df = pd.read_csv(path)
        countryCode = df['Code'].unique()
        Type_of_cancer = df.columns[3:].to_list()
        return render(request, 'users/regressionModelResults.html',
                      {'lr_dict': lr_dict, 'dt_dict': dt_dict, 'rf_dict': rf_dict, 'pf_dict': pf_dict,
                       'countryCode': countryCode,
                       'cancerType': Type_of_cancer})

    else:
        path = os.path.join(settings.MEDIA_ROOT, 'datasets', '08 disease-burden-rates-by-cancer-types.csv')
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        countryCode = df['Code'].unique()
        Type_of_cancer = df.columns[3:].to_list()
        # print(type(Type_of_cancer), Type_of_cancer)
        return render(request, 'users/regressionModel.html', {'countryCode': countryCode, 'cancerType': Type_of_cancer})


def ForecastAnalysis(request):
    if request.method == 'POST':
        path = os.path.join(settings.MEDIA_ROOT, 'datasets', '08 disease-burden-rates-by-cancer-types.csv')
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        countryCode = request.POST.get('countryCode')
        cancerType = request.POST.get('cancerType')
        df = df[df.Code.isin([countryCode])]
        # print(df.head())

        X = df['Year'].values
        y = df[cancerType].values
        # X_X = []
        # y_y = []
        # for i in X:
        #     X_X.append([i])
        # for j in y:
        #     y_y.append([j])
        myDf = pd.DataFrame(list(zip(X, y)),columns=['year', 'val'])

        from .utility.predections import FuturePredImpl
        fut = FuturePredImpl()
        pred_ci = fut.startFuturePrediction(myDf)
        print('Am Which type ', type(pred_ci))
        pred_ci['lower val'] = pred_ci['lower val'].astype(float)
        pred_ci['upper val'] = pred_ci['upper val'].astype(float)
        pred_ci['lower val'] = pred_ci['lower val'] / 800
        pred_ci['upper val'] = pred_ci['upper val'] / 800
        print(pred_ci.head())
        pred_ci = pred_ci.tail(600)
        pred_ci = pred_ci.to_html

        path = os.path.join(settings.MEDIA_ROOT, 'datasets', '08 disease-burden-rates-by-cancer-types.csv')
        df = pd.read_csv(path)
        countryCode = df['Code'].unique()
        Type_of_cancer = df.columns[3:].to_list()

        return render(request, 'users/forecastModel.html',
                      {'data': pred_ci, 'countryCode': countryCode,
                       'cancerType': Type_of_cancer})

    else:
        path = os.path.join(settings.MEDIA_ROOT, 'datasets', '08 disease-burden-rates-by-cancer-types.csv')
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        countryCode = df['Code'].unique()
        Type_of_cancer = df.columns[3:].to_list()
        return render(request, 'users/forecastModel.html', {'countryCode': countryCode, 'cancerType': Type_of_cancer})
