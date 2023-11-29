# BackEnd Development and Cosine Similarity ML Model for Steam

This project is a practical knowledge recopilation for the development, deploy and testing of an API using FastAPI framework for hosting multiple endpoint queries aswell as a Cosine Similarity Machine Learning model.

Please refer to the following table for check all the files content:

| ID | Name          | Type   | Description                                                           |
|----|---------------|--------|-----------------------------------------------------------------------|
| 1  | simplified-data | .csv | Contains summarized data for endpoints queries|
| 2  | ETL | .ipynb | Contains Notebooks with the ETL process for each Steam API Json                 |
| 3  | ETL/utils | .py | Contains utility functions used for the ETL process                |
| 4  | .gitignore | .git | Contains ignored raw data (Too heavy for be uploaded to Render) and a testing notebook|
| 5  | EDA | .ipynb |It contains a view of the cleaned data and statistical analysis. Finally, the dataset is generated for use in queries. |
| 6  | requirements | .txt | Libraries required for backend functioning |
| 7  | querys | .py | Contains functions used by the BackEnd (main) file and its end points|
| 9  | main | .py | Contains HTTP request methods for BackEnd functionality|
| 10 | requirements | .txt | Libraries required for backend functioning |


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


## Game Analytics Functions

### PlayTimeGenre

```python
def PlayTimeGenre(genre: str):
    # Returns the year with the most played hours for the given genre.
    # Example return: {"Year of release with most played hours for Genre X": 2013}

def UserForGenre(genre: str):
    # Should return the user who has accumulated the most played hours for the given genre
    # and a list of the accumulated played hours per year.
    # Example return: {"User with most played hours for Genre X": "us213ndjss09sdf", "Played hours": [{"Year": 2013, "Hours": 203}, {"Year": 2012, "Hours": 100}, {"Year": 2011, "Hours": 23}]}

def UsersRecommend(year: int):
    # Returns the top 3 games MOST recommended by users for the given year.
    # (reviews.recommend = True and positive/neutral comments)
    # Example return: [{"Rank 1": X}, {"Rank 2": Y}, {"Rank 3": Z}]

def UsersWorstDeveloper(year: int):
    # Returns the top 3 developers with the LEAST recommended games by users for the given year.
    # (reviews.recommend = False and negative comments)
    # Example return: [{"Rank 1": X}, {"Rank 2": Y}, {"Rank 3": Z}]

def sentiment_analysis(developer_company: str):
    # According to the developer company, returns a dictionary with the company name as the key
    # and a list with the total number of user review records categorized with sentiment analysis as the value.
    # Example return: {'Valve': {'Negative': 182, 'Neutral': 120, 'Positive': 278}}
```

 ## Recommendation system.

Finally, a machine learning model based on cosine similarity is developed to create a recommendation system. The function game_recommendation(product_id) is implemented.

```python
def game_recommendation(product_id):
    # By entering the product ID, receives a list of 5 recommended games similar to the one entered.
```

 #### Queries are made available through the FastAPI framework, and Render is used to enable the API to be consumed from the web.
