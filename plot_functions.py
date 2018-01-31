import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Source : https://www.kaggle.com/helgejo/titanic/an-interactive-data-science-tutorial
def plot_correlation_map(df, figsize=(13, 12), fontsize=10, savefig=False):
    """
    """
    corr = df.corr()
    fig , ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette( 220 , 10 , as_cmap=True )
    sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : fontsize }
    )
    
    fig.tight_layout()
    fig.savefig(savefig)

def plot_variable_importance(X, y, figsize=(10, 7), random_state=None, savefig=None):
    """
    """
    tree = DecisionTreeClassifier(random_state=random_state)
    tree.fit(X, y)
    plot_model_var_imp(tree, X.columns, figsize=figsize, savefig=savefig)
    

def plot_model_var_imp(model, features, figsize=(10, 7), savefig=None):
    """
    """
    imp = pd.DataFrame( 
        model.feature_importances_,
        columns = [ 'Importance' ],
        index = features
    )

    imp = imp.sort_values(['Importance'], ascending=False)
    plot = imp[:10].plot(kind='barh', figsize=figsize)

    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(savefig)