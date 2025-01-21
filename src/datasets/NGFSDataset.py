import pandas as pd
from pathlib import Path
from functools import lru_cache

# Conversion factors
EJ2KWH = 277777777777.78

SECTORS_MAP = {"Electricity": "Final Energy|Electricity"}

ID_VARS = ["Model", "Scenario", "Region", "Variable"]

def calculate_cagr(group):
    """_summary_

    Args:
        group (_type_): melted data to pivot to melt.

    Returns:
        _type_: _description_
    """
    # The original dataset is melted
    # Pivot to compute cagr
    group = group.pivot(index= ["Model", "Scenario", "Region", "Year"], columns="Variable", values="Value")
    group = group.reset_index()
    group["Year"] = group["Year"].astype(int)
    # Ensure the data is sorted by year
    group = group.sort_values('Year', ascending=True)
    min_year = group["Year"].iloc[0]
    max_year = group["Year"].iloc[-1]
    all_years = pd.DataFrame({'Year': range(min_year, max_year+1)})

    cols = {"elec_int": "CO₂ intensity of electricity generation (g CO₂ per kWh)"}
    c = "elec_int"
    group = group.rename(columns={cols[c]: c})
    group["Year_diff"] = group["Year"].diff(1)
    group[f"{c}_shift"] = group[c].shift(1)
    group[f"{c}_cagr"] = (group[c] / group[f"{c}_shift"]) ** (1/group["Year_diff"]) - 1
    group = group[["Year", "Model", "Scenario", "Region", f"{c}_cagr"]]
    mergedf = pd.merge(all_years, group, how="left", on="Year")
    mergedf = mergedf.fillna(method='bfill').fillna(method='ffill')
    
    # Melt back for consistency with original data format
    mergedf["Variable"] = f"{c}_cagr"
    mergedf = mergedf.rename(columns={"elec_int_cagr": "Value"})
    mergedf = mergedf[["Year", "Model", "Scenario", "Region", "Variable", "Value"]]
    return mergedf

class NGFSDataset:
    def __init__(self, file_path):
        """
        Initialize the NGFSDataReader with the path to the Excel file.

        Data download (requires a login): https://data.ene.iiasa.ac.at/ngfs/#/login?redirect=%2Fworkspaces
        
        :param file_path: str, path to the Excel file "IAM_data.xlsx"
        """
        self.file_path = file_path
        self.data = self.read_data()
        self.usevars = ["Emissions|Kyoto Gases|Electricity",
                        "Emissions|CO2|Energy|Supply|Electricity",
                "Final Energy|Electricity",
                "Production|Steel",
                "Emissions|CO2|Energy|Demand|Industry|Steel"
                ]
        if self.usevars is not None:
            self.data = self.data.loc[self.data["Variable"].isin(self.usevars)]
        self.data = self.melt_data(id_vars=ID_VARS, region="World")

        self._set_ghg_intensity(sector="Electricity")

            # Interpolated annual values
        self.agg_dict = {
            "CO₂ intensity of electricity generation (g CO₂ per kWh)": "cagr",
            "Production|Steel": "lin_inter"
        }
    
    def __repr__(self) -> str:
        return "V4.2-NGFS-Phase-4"
    
    def transform_columns(self) -> pd.DataFrame:
        """
        Applies specified transformations to columns in the input DataFrame and merges the results into a single DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing the data to be transformed.
        agg_dict : dict
            A dictionary where the keys are column names from the DataFrame and 
            the values are the transformation methods to be applied. Supported methods are:
            - "cagr": Compound Annual Growth Rate transformation
            - "lin_inter": Linear interpolation
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame with the transformed columns merged together.

        Raises:
        -------
        ValueError
            If an unknown transformation method is passed in agg_dict.
        
        Example:
        --------
        agg_dict = {
            "CO₂ intensity of electricity generation (g CO₂ per kWh)": "cagr",
            "Production|Steel": "lin_inter"
        }
        transformed_df = transform_columns(df, agg_dict)
        """
        # region_mask = melted["Region"] == region # a single region
        # melted = melted.loc[region_mask]

        # Initialize an empty list to store the transformed DataFrames or Series
        transformed_dfs = []

        # Loop through each column and its corresponding transformation method
        for col, method in self.agg_dict.items():
            # Apply the appropriate transformation based on the method
            if method == "cagr":
                transfo = self.get_cagr(df=self.data, col=col)
            elif method == "lin_inter":
                transfo = self.interpolate_var(df=self.data, col=col)
            else:
                raise ValueError(f"Unknown transformation method: {method}")
            
            # Add the transformed column as a DataFrame or Series to the list
            transformed_dfs.append(transfo)

        # Merge all the transformed DataFrames/Series into a single DataFrame
        transformed_df = pd.concat(transformed_dfs, axis=0, ignore_index=True)
        
        return transformed_df


    def get_cagr(self, df, col: str):
        sub_df = df.loc[df["Variable"] == col]
        cagr = sub_df.groupby(["Model", "Scenario"]).apply(calculate_cagr)
        cagr = cagr.drop(columns=["Model", "Scenario"]).reset_index()
        cagr = cagr.drop(columns="level_2")
        return cagr

    
    def interpolate_var(self, df, col):
        """Dataframe with a single variable for multiple (model, scenario) couples.

        Args:
            melted_prod (_type_): cols (Model, Scenario, Year, Value)

        Returns:
            _type_: _description_
        """
        # 1. Create a range of all years between 2020 and 2030
        all_years = pd.DataFrame({'Year': range(2020, 2031)})

        # 2. For each `id`, merge with all years and interpolate
        df_interpolated = df.groupby(["Model", "Scenario"]).apply(
            lambda group: pd.merge(all_years, group, on='Year', how='left').interpolate()
        ).reset_index(drop=True)

        # 3. Fill missing values (if any remain, like at the start/end)
        df_interpolated = df_interpolated.fillna(method='bfill').fillna(method='ffill')

        return df_interpolated

    def _set_ghg_intensity(self, sector: str):
        ghg_intensity = self.get_ghg_intensity(sector=sector, region="World")
        self.data = pd.concat([self.data, ghg_intensity], axis=0, ignore_index=True)
    
    def get_ghg_intensity(self, sector: str, region: str = "World"):
        ghg_var = f"Emissions|CO2|Energy|Supply|{sector}" #f"Emissions|Kyoto Gases|{sector}"
        prod_var = SECTORS_MAP[sector]

        pivot = self.data.pivot(index= ["Model", "Scenario", "Region", "Year"], columns="Variable", values="Value")
        pivot = pivot.reset_index()

        # Calculate ghg intensity of electricity generation
        # 1 EJ (exajoule) = f"{EJ2kwh} kwh"
        elec_colname = "CO₂ intensity of electricity generation (g CO₂ per kWh)"
        pivot["Variable"] = elec_colname
        pivot["Value"] = (pivot[ghg_var] * 1E12) / (pivot[prod_var] * EJ2KWH)
        pivot = pivot[["Model", "Scenario", "Region", "Year", "Variable", "Value"]]
        return pivot

    def melt_data(self, id_vars: list, region: str = None):
        if region is not None:
            df = self.data.loc[self.data["Region"] == region]
        else:
            df = self.data

        # Get value columns for each year
        value_cols = df.select_dtypes("number").columns.to_list()

        # ghg_prod_df = self.data.loc[(ghg_mask | prod_mask) & region_mask, id_vars + value_cols]
        ghg_prod_df = df.loc[:, id_vars + value_cols]

        melted = ghg_prod_df.melt(id_vars=id_vars, var_name="Year", value_name="Value")
        return melted.astype({"Year": int, "Value": float})
    
    def read_data(self):
        return pd.read_excel(self.file_path, engine="calamine")

