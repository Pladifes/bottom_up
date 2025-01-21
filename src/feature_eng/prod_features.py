from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
import json
try:
    from src.feature_engineering import get_plant_age, get_electricity_intensity, get_steel_use_capita, get_carbon_price
    from src.get_sklearn_pipeline import generate_pipelines
except:
    from feature_engineering import get_plant_age, get_electricity_intensity, get_steel_use_capita, get_carbon_price
    from get_sklearn_pipeline import generate_pipelines
from tqdm import tqdm
import pickle
import joblib

import pandas as pd
import numpy as np

def fit_model(pipelines,
                X_cv, 
              y_cv, 
              groups, 
              test_size,
              n_splits,
              n_iter,
              refit,
              verbose,
              random_state,
              save_dir,
              n_jobs=None):
    res = {}
    for pipeline_name, pipeline, param_grid in (pbar := tqdm(pipelines)):
        # Print regressor name
        regressor_name = str(pipeline['estimator']).split("(")[0]
        pbar.set_description(regressor_name)
        pbar.refresh()
        
        # Initialize GroupShuffleSplit with the number of splits
        group_splits = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        # Define the list of scoring metrics
        scoring = ["r2", 
                   "neg_mean_squared_error",
                   "neg_mean_absolute_error", 
                   "neg_mean_absolute_percentage_error",
                   "neg_root_mean_squared_error"]
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            scoring=scoring,
            cv=group_splits,
            n_iter=n_iter,  # Number of iterations for random search
            verbose=verbose,
            refit=refit,
            n_jobs=n_jobs,
            random_state=random_state,
            error_score='raise'
        )

        # Perform the random search
        random_search.fit(X_cv, y_cv, groups=groups)

        if random_search.estimator['feature_selection'] != "passthrough":
            pass
        # save results
        with open(save_dir / f"{pipeline_name}.pkl", "wb") as file:
            joblib.dump(random_search, file)
        res[pipeline_name] = random_search

    return res


def get_X_y_groups(df, target):    
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    idx_cols = ['Plant ID', "year"]
    df = df.set_index(idx_cols)
    df["is_BF"] = df["Primary production route"] == "BF-BOF"
    country_cols = [c for c in df.columns if "Country_" in c]
    product_cols = [c for c in df.columns if "product_" in c]
    techno_cols = ["is_BF"]
    feature_cols =  ["log_capa", "prod_costs", "ord_year", "log_age", "steel_use_capita", "log_steel_use_capita"] + country_cols + product_cols + techno_cols
    X = df.loc[:, list(set(feature_cols) - set(['groups', "log_prod", target]))]
    y = df.loc[:, [target]]
    groups = df[['groups']]
    return X, y, groups

def get_prod_dataset(plants):
    # columns to keep
    country_cols = [c for c in plants.columns if "Country_" in c]
    product_cols = [c for c in plants.columns if "product_" in c]
    feature_cols =  ["log_prod", "log_capa", "Nominal crude steel capacity (ttpa)", "prod_costs", "ord_year", "log_age"] + country_cols + product_cols
    df = plants.loc[plants['Crude steel production (ttpa)'].notna(), feature_cols]
    return df
    
def get_plant_features(plants, costs_path, energy_mix_path, carbon_price_path, steel_use_path, steel_use_map_path):
    len_in = len(plants)
    plants = process_steel_products(plants)
    plants = binarize_steel_products(plants)
    plants = binarize_country_column(plants, "Country")
    plants = merge_costs_to_plants(plants=plants, costs_path=costs_path)
    plants = get_electricity_intensity(plants, energy_mix_path=energy_mix_path)
    plants = get_carbon_price(plants, carbon_price_path=carbon_price_path)
    plants = get_steel_use_capita(plants=plants, steel_use_path=steel_use_path, steel_use_map_path=steel_use_map_path)
    len_out = len(plants)
    assert len_in == len_out

    plants['ord_year'] = plants['year'] - 2018
    plants['log_prod'] = np.log(plants['Crude steel production (ttpa)'])
    plants['log_capa'] = np.log(plants['Nominal crude steel capacity (ttpa)'])
    plants['Plant age'] = plants.groupby("year").apply(get_plant_age).values
    plants['log_age'] = np.log(plants['Plant age'].replace(to_replace=0, value=1))
    plants['log_steel_use_capita'] = np.log(plants['steel_use_capita'])
    
    # set index
    plants = plants.set_index(['Plant ID', "year"])
    
    # add groups column
    plants = get_groups(plants)
    return plants

def train_test_split_group(df, test_size=0.2, random_state=None):
    """Helper for splitting a dataset into a train and test set based on group 
    information. We prevent information from the same group to be present both
    during training and testing.

    Args:
        df (_type_): dataset
        test_size (float, optional): proportion of groups to be included in the test set. Defaults to 0.2.
        random_state (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Initialize GroupShuffleSplit with the number of splits
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Loop over the splits and get the indices for train and test
    for train_idx, test_idx in gss.split(df, groups=df['groups']):
        df_cv = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

    return df_cv, df_test

def get_groups(plants):
    plant_names = plants.index.get_level_values("Plant ID")
    name2int = {name: i for i, name in enumerate(list(plant_names.unique()))}
    groups = plant_names.map(name2int)
    plants['groups'] = groups
    return plants


def merge_costs_to_plants(plants, costs_path):
    """
    Merge cost data from an Excel file to the plants DataFrame.

    Args:
        plants (pandas.DataFrame): The DataFrame containing plant information.
        costs_path (str): Path to the Excel file containing cost data.

    Returns:
        pandas.DataFrame: The plants DataFrame with cost information merged.
    """
    # Read cost data from the Excel file
    costs = pd.read_excel(costs_path, sheet_name="Detailed Data")
    
    # Filter relevant cost components (Total BF/BOF and Total EAF)
    total_costs = costs.loc[costs["Component"].isin(["Total BF/BOF", "Total EAF"])]
    
    plants['Primary production route'] = plants['Main production process'].apply(lambda x: "EAF" if x == "electric" else "BF-BOF")
    # Merge cost data to the plants DataFrame based on common columns
    plants = plants.merge(
        total_costs[["Plant ID", "Model Year", "Value", "Primary production route"]],
        left_on=["Plant ID", "year", "Primary production route"],
        right_on=["Plant ID", "Model Year", "Primary production route"],
        how='left'
    )
    
    # average costs
    avg_costs = total_costs.groupby(["Model Year", "Primary production route"])["Value"].mean().to_frame("avg_global_costs").reset_index()
    
    
    # impute global average costs to plants with missing values
    plants = plants.merge(avg_costs,
                          left_on=["Primary production route", "year"],
                          right_on=["Primary production route", "Model Year"],
                          how='left')
    plants['Value'] = plants['Value'].fillna(plants['avg_global_costs'])
    
    # Rename the merged cost value column
    plants = plants.rename(columns={"Value": "prod_costs"})
    
    return plants


def binarize_country_column(df, column_name):
    """
    Binarize a categorical column using LabelBinarizer.

    Args:
        df (pandas.DataFrame): The DataFrame containing the categorical column.
        column_name (str): The name of the column to be binarized. (Country, Region)

    Returns:
        pandas.DataFrame: The DataFrame with the binarized columns.
    """
    lb = LabelBinarizer()
    binarized_countries = lb.fit_transform(df[column_name])
    
    # Create a DataFrame with the binarized columns and column names
    binarized_df = pd.DataFrame(binarized_countries, columns=[f'{column_name}_{class_}' for class_ in lb.classes_], index=df.index)
    
    # Concatenate the binarized DataFrame with the original DataFrame
    result_df = pd.concat([df, binarized_df], axis=1)
    
    return result_df


def binarize_steel_products(df):
    """
    Binarize the 'Category steel product' column using MultiLabelBinarizer.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'Category steel product' column.

    Returns:
        pandas.DataFrame: The DataFrame with the binarized 'Category steel product' columns.
    """
    mlb = MultiLabelBinarizer()
    binarized_categories = mlb.fit_transform(df['Category steel product'])
    
    # Create a DataFrame with the binarized columns and column names
    product_cols = ['product_' + name for name in mlb.classes_]
    binarized_df = pd.DataFrame(binarized_categories, columns=product_cols, index=df.index)
    
    # Concatenate the binarized DataFrame with the original DataFrame
    result_df = pd.concat([df, binarized_df], axis=1)
    
    return result_df

def process_steel_products(df):
    """
    Process the 'Category steel product' column in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'Category steel product' column.

    Returns:
        pandas.DataFrame: The DataFrame with the processed 'Category steel product' column.
    """
    df['Category steel product'] = df['Category steel product'].str.replace(' finished rolled', 'finished rolled')
    df['Category steel product'] = df['Category steel product'].str.replace(' semi-finished', 'semi-finished')
    df['Category steel product'] = df['Category steel product'].fillna("nan")
    df['Category steel product'] = df['Category steel product'].str.split(';')
    return df


