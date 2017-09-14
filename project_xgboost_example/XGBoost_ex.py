#!/usr/bin/env python
# -*- coding: UTF-8
# http://danielhnyk.cz/how-to-use-xgboost-in-python/
# http://www.xavierdupre.fr/app/pymyinstall/helpsphinx/notebooks/example_xgboost.html


# prepare for data
from sklearn import datasets  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score

X, y = datasets.make_classification(n_samples=10000, n_features=20,  
                                    n_informative=2, n_redundant=10,
                                    random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  
                                                    random_state=42)

# fit data
from xgboost.sklearn import XGBClassifier  
from xgboost.sklearn import XGBRegressor

xclas = XGBClassifier()  # and for classifier  
xclas.fit(X_train, y_train)  
xclas.predict(X_test)  

# score
cross_val_score(xclas, X_train, y_train)  



# XGBoost with RandomizedSearchCV
from xgboost.sklearn import XGBRegressor  
import scipy.stats as st

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgbreg = XGBRegressor(nthreads=-1)  

from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
gs.fit(X_train, y_train)  
gs.best_model_  