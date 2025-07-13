# G6 Performs operations on the predictions files
# Usage: python pred_helper.py {run_identifier}
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import sys

# Month columns
csv_output_columns = []
first_col = pd.DataFrame({"phase": ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep"]})
csv_output_columns.append(first_col)
identifier = sys.argv[1]
pred_dir = f'multiple_data/g6data/predictions/pred_{identifier}/'

test_arr = []
for i in range(12):
    test_arr.append(np.load(f'{pred_dir}test_{i}.npy'))

# Can add metrics here to quickly change which is calculated by using indexes
metric_cat_quickArr = ["macro avg", "weighted avg"]
metric_name_quickArr = ["f1-score"]

# SETTINGS HERE
metric_cat = metric_cat_quickArr[0]
metric_name = metric_name_quickArr[0]

print(f'\nReporting the {metric_cat} of {metric_name}')

j = 0
second_col_data = []
for test in test_arr:

    test_df = pd.DataFrame(test)

    # Calculate macro f1
    y_pred = test_df[0]
    y_true = test_df[1]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
    j += 1

    # Print to terminal and build a csv file
    metric = report[metric_cat][metric_name]
    print(f'{metric}')
    second_col_data.append(metric)

# Save the calculations
second_col = pd.DataFrame({"Macro F1": second_col_data})
csv_output_columns.append(second_col)
print("\n")

final_df = pd.concat(csv_output_columns, axis=1)
final_df.to_csv(f'{pred_dir}MF1_score_{identifier}.csv', index=False)