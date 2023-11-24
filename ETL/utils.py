import pandas as pd

def data_overview(df):
    '''
    Analyzes the DataFrame and provides detailed insights into data types, null values, and general statistics.
    
    :param df: pandas DataFrame to be analyzed.
    :return: DataFrame containing information on data types, null percentages, quantities, and general statistics.
    '''
    mi_dict = {"Column": [], "dType": [], "No_Null_%": [], "No_Null_Qty": [], "Null_%": [], "Null_Qty": []}
    
    duplicated_rows = df[df.duplicated()]
    count_duplicated_rows = len(duplicated_rows)

    for column in df.columns:
        no_null_porcent = (df[column].count() / len(df)) * 100
        mi_dict["Column"].append(column)
        mi_dict["dType"].append(df[column].apply(type).unique())
        mi_dict["No_Null_%"].append(round(no_null_porcent, 2))
        mi_dict["No_Null_Qty"].append(df[column].count())
        mi_dict["Null_%"].append(round(100 - no_null_porcent, 2))
        mi_dict['Null_Qty'].append(df[column].isnull().sum())

    overview = pd.DataFrame(mi_dict)
    
    print("\nTotal rows: ", len(df))
    print("\nTotal full null rows: ", df.isna().all(axis=1).sum())
    print("\nTotal duplicated rows:", count_duplicated_rows)
    
    return overview


def check_none_values(df):

    '''
    '''

    none_percentage = {}

    for column in df.columns:
        none_count = df[column].apply(lambda x: x == "None").sum()
        total_count = len(df[column])
        percentage = (none_count / total_count) * 100
        none_percentage[column] = percentage

    result_df = pd.DataFrame(list(none_percentage.items()), columns=['Columna', 'Porcentaje None'])
    print(result_df)