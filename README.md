# BackEnd Development and Cosine Similarity ML Model for Steam

This project is a practical knowledge recopilation for the development, deploy and testing of an API using FastAPI framework for hosting multiple endpoint queries aswell as a Cosine Similarity Machine Learning model.

Please refer to the following table for check all the files content:

| ID | Name          | Type   | Description                                                           |
|----|---------------|--------|-----------------------------------------------------------------------|
| 1  | simplified-data | .csv | Contains summarized data for endpoints queries|
| 2  | ETL | .ipynb | Contains Notebooks with the ETL process for each Steam API Json                 |
| 3  | ETL/utils | .py | Contains utility functions used for the ETL process                |
| 4  | .gitignore | .git | Contains ignored raw data (Too heavy for be uploaded to Render) and a testing notebook|
| 5  | EDA | .ipynb |It contains a view of the cleaned data and statistical analysis|
| 6  | requirements | .txt | Libraries required for backend functioning |
| 7  | querys | .py | Contains functions used by the BackEnd (main) file and its end points|
| 9  | main | .py | Contains HTTP request methods for BackEnd functionality|
| 10 | Things to improve | .txt | Some options that occurred to me to improve project management |
  11 | CSV-query-engine | .ipynb | Contains generation of datasets for use in queries|


Link to raw datasets: https://drive.google.com/file/d/1CeBmBvvr043S-oL7ihuGSz8eLPFeNJFE/view?usp=sharing


## Datasets

This project involved 3 big Json files extracted from Steam API about Australian users and games data published at the time. In general terms the information contained in this file were:

* User Reviews
* User Items Acquired
* Steam Games Data

The following section contains detailed information about every dataset and their initial state of art (Data types and Null values).



### Steam games dataset

| Column         | dType                                  | No_Null_% | No_Null_Qty | Null_% | Null_Qty |
|----------------|----------------------------------------|-----------|-------------|--------|----------|
| publisher      | [<class 'float'>, <class 'str'>]       | 20.00     | 24083       | 80.00  | 96362    |
| genres         | [<class 'float'>, <class 'str'>]       | 23.95     | 28852       | 76.05  | 91593    |
| app_name       | [<class 'float'>, <class 'str'>]       | 26.68     | 32133       | 73.32  | 88312    |
| title          | [<class 'float'>, <class 'str'>]       | 24.98     | 30085       | 75.02  | 90360    |
| url            | [<class 'float'>, <class 'str'>]       | 26.68     | 32135       | 73.32  | 88310    |
| release_date   | [<class 'float'>, <class 'str'>]       | 24.96     | 30068       | 75.04  | 90377    |
| tags           | [<class 'float'>, <class 'str'>]       | 26.54     | 31972       | 73.46  | 88473    |
| reviews_url    | [<class 'float'>, <class 'str'>]       | 26.68     | 32133       | 73.32  | 88312    |
| specs          | [<class 'float'>, <class 'str'>]       | 26.12     | 31465       | 73.88  | 88980    |
| price          | [<class 'float'>, <class 'str'>]       | 25.54     | 30758       | 74.46  | 89687    |
| early_access   | [<class 'float'>, <class 'bool'>]      | 26.68     | 32135       | 73.32  | 88310    |
| id             | [<class 'float'>, <class 'str'>]       | 26.68     | 32133       | 73.32  | 88312    |
| developer      | [<class 'float'>, <class 'str'>]       | 23.94     | 28836       | 76.06  | 91609    |

Total rows: 120445
Total full null rows: 88310
Total duplicated rows: 88309


### Reviews dataset

| Column   | dType            | No_Null_% | No_Null_Qty | Null_% | Null_Qty |
|----------|------------------|-----------|-------------|--------|----------|
| user_id  | <class 'str'>    | 100.0     | 25799       | 0.0    | 0        |
| user_url | <class 'str'>    | 100.0     | 25799       | 0.0    | 0        |
| reviews  | <class 'str'>    | 100.0     | 25799       | 0.0    | 0        |

Total rows: 25799
Total full null rows: 0
Total duplicated rows: 313


### Items dataset

| Column       | dType            | No_Null_% | No_Null_Qty | Null_% | Null_Qty |
|--------------|------------------|-----------|-------------|--------|----------|
| user_id      | [<class 'str'>]  | 100.0     | 88310       | 0.0    | 0        |
| items_count  | [<class 'int'>]  | 100.0     | 88310       | 0.0    | 0        |
| steam_id     | [<class 'str'>]  | 100.0     | 88310       | 0.0    | 0        |
| user_url     | [<class 'str'>]  | 100.0     | 88310       | 0.0    | 0        |
| items        | [<class 'str'>]  | 100.0     | 88310       | 0.0    | 0        |

Total rows: 88310
Total full null rows: 0
Total duplicated rows: 657


### The basic conventions for ETL are as follows:

Float values in the majority of columns represent null values and were managed considering the following criteria:

* Columns with more than 95% null values - Columns Eliminated
* Records with 80% or more of their data being null were deleted.
* Columns with null values between 5% and 30% - Imputed Values ('Not specified' for str and mean value for float and int)
* In the items dataset, only the columns within the 'items' column were retained, as they are the only ones relevant to the project. A process was carried out to unnest them

### The goal is to be able to provide these querys.

## Note:"It was decided to handle the entire project in English; however, the function signatures and their return will be maintained in the requested language, as it is not a good practice to change their names. In the function that uses machine learning, the parameter name is changed because having a variable with spaces would not be valid, as requested in the instructions."


## Game Analytics Functions

```python
def developer(desarrollador: str):
    """
    Returns the number of items and the percentage of Free content per year for the given developer.
    Example return: 
    Año    Cantidad de Items    Contenido Free
    2023   50                  27%
    2022   45                  25%
    xxxx   xx                  xx%
    """

def userdata(User_id: str):
    """
    Returns the amount of money spent by the user, the percentage of recommendation based on reviews.recommend,
    and the number of items.
    Example return: 
    {"Usuario X": "us213ndjss09sdf", "Dinero gastado": "200 USD", "% de recomendación": "20%", "cantidad de items": 5}
    """

def UserForGenre(genero: str):
    """
    Returns the user with the most played hours for the given genre and a list of accumulated played hours per release year.
    Example return: 
    {"Usuario con más horas jugadas para Género X": "us213ndjss09sdf", 
     "Horas jugadas": [{"Año": 2013, "Horas": 203}, {"Año": 2012, "Horas": 100}, {"Año": 2011, "Horas": 23}]}
    """

def best_developer_year(año: int):
    """
    Returns the top 3 developers with the MOST recommended games by users for the given year.
    (reviews.recommend = True and positive comments)
    Example return: 
    [{"Puesto 1": X}, {"Puesto 2": Y}, {"Puesto 3": Z}]
    """

def developer_reviews_analysis(desarrolladora: str):
    """
    According to the developer, returns a dictionary with the developer's name as the key
    and a list with the total number of user review records categorized with sentiment analysis as the value.
    Example return: 
    {'Valve': {'Negative': 182, 'Positive': 278}}
    """
```

 ## Recommendation system.

Finally, a machine learning model based on cosine similarity is developed to create a recommendation system. The function game_recommendation(product_id) is implemented.

```python
def recomendacion_juego(product_id):
    # By entering the product ID, receives a list of 5 recommended games similar to the one entered.
```

 #### Queries are made available through the FastAPI framework, and Render is used to enable the API to be consumed from the web.






Link to Render deployment: https://steam-rec-system-deploy.onrender.com

Note: Due to updates in the input datasets, some output from the queries may differ from what is shown in the video; however, the functionality remains the same.