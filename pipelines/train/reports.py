from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def confusion_matrix_pretty(y_true, y_pred):
    conf_m = confusion_matrix(y_true, y_pred, labels=[0,1])

    table = PrettyTable(['', '1__Predicted', '0__Predicted'])
    table.add_row(['1__True'] + conf_m[0].tolist())
    table.add_row(['0__True'] + conf_m[1].tolist())

    return table

def classification_report_pretty(y_true, y_pred):
    clf_rep = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(clf_rep).iloc[:-1, :].T

    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        title = [ row[0] ]
        values = [ round(v, 4) for v in row[1:] ]
        table.add_row(title + values)

    return table
