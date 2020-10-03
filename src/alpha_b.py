#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
'''
Xs = #insieme di x
y = #y

lasso = Lasso()

parameters = {'alpha' : [0, 1]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring= 'neg_mean_squared_error', cv =5 '''?''')

lasso_regressor.fit(Xs, y)

print(lasso_regressor.best_params_)
'''

def alpha(Xs, Y, lam):

    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'), threshold= lam )
    sel_.fit(Xs, np.ravel(Y,order='C'))
    sel_.get_support()

    removed_feats = X.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
    return removed_feats