# This helps with consolidating the .csv files
import pandas as pd

filename_base = "multiple_reports/bluehex_multiclass/gbdt_multiclass_report_bluehex_families_40_test_2019-10_2020-09_r10_"
#identifiers = ["base", "family_pop_gt10", "family_pop_gt40", "family_pop_lt20", "full_mix", "puresynth"]
identifiers = ["base", "10000_1150"]
columns = []
first_col = pd.DataFrame({"phase": ["Val", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep"]})
columns.append(first_col)
for identity in identifiers:
    filename = f'{filename_base}{identity}.csv'
    df = pd.read_csv(filename)

    replacement_name = f'known_topacc_2_{identity}'

    top2_col = df[["inclass_topacc_2"]].copy()

    top2_col.rename(columns={"inclass_topacc_2": replacement_name}, inplace=True)

    columns.append(top2_col)

final_df = pd.concat(columns, axis=1)
final_df.to_csv("multiple_reports/combined.csv", index=False)

'''

    replacement_name1 = f'{identity}_known'
    replacement_name2 = f'{identity}_all'

    top2_known_col = df[["topacc_2"]].copy()
    top2_all_col = df[["inclass_topacc_2"]].copy()

    top2_known_col.rename(columns={"topacc_2": replacement_name1}, inplace=True)
    top2_all_col.rename(columns={"inclass_topacc_2": replacement_name2}, inplace=True)

    columnsKnown.append(top2_known_col)
    columnsAll.append(top2_all_col)

for col in columnsKnown:
    columns.append(col)

for col in columnsAll:
    columns.append(col)'''