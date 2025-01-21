"""Helpers for reading IEA scenario data (WEO_Extended_Data.xlsx).
Granularity of data is at the country-level."""
    
import pandas as pd
import json

try:
    from datasets.utils import get_cagr
except:
    from utils import get_cagr
    
from typing import Tuple


def merge_iea_var(plants: pd.DataFrame, iea_var: pd.DataFrame, gspt2iea_countries_path: "Path") -> pd.DataFrame:
    """

    Args:
        plants (pd.DataFrame): _description_
        iea_var (pd.DataFrame): _description_
        gspt2iea_countries_path (Path): _description_

    Returns:
        pd.DataFrame: _description_
    """
    assert "year" in plants.columns
    
    # Read dictionary mapping gspt countries to IEA regions
    with open(gspt2iea_countries_path, 'r', encoding="utf-8") as file:
        gspt2iea_countries = json.load(file)
    
    # map GSPT country to IEA region to assign 
    # projected values at a granular level (e.g. country or region)
    plants["iea_region"] = plants["Country"].map(gspt2iea_countries)
    assert plants["iea_region"].notna().all()
    
    
    # merge IEA variable based on previous regional mapping
    to_merge = ["iea_region", "year"] if "year" in iea_var.columns else ['iea_region']
    
    plants = pd.merge(plants,
                      iea_var,
                      on=to_merge,
                      how='left')
    return plants
        
    
def get_iea_vars(world_fpath, regions_fpath, var, scenario, end_year=2030, interp=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        world_fpath (str): path to WEO2023_Extended_Data_World.csv
        regions_fpath (str): path to WEO2023_Extended_Data_Regions.csv
        var (_type_): _description_
        country (_type_): _description_
        scenario (_type_): _description_
        end_year (int, optional): _description_. Defaults to 2030.
        interp (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    usecols = ("SCENARIO", "CATEGORY", "PRODUCT", "FLOW", "REGION", "YEAR", "VALUE")
    world_data = pd.read_csv(world_fpath, usecols=usecols)
    regions_data = pd.read_csv(regions_fpath, usecols=usecols)
    data = pd.concat([world_data, regions_data], axis=0, ignore_index=True)
    countries = list(data["REGION"].unique())
    
    traj, cagr = [], []    
    for country in countries:
        iea_var, r = get_iea_var(data=data, var=var, country=country, scenario=scenario, end_year=end_year, interp=interp)
        traj.append(iea_var)
        cagr.append(pd.DataFrame({"start_year": 2022, "end_year": 2030, "iea_region": [country], "var": var, "cagr": [r]}))
    
    traj = pd.concat(traj, axis=0, ignore_index=True)
    cagr = pd.concat(cagr, axis=0, ignore_index=True)
    return traj, cagr
    
def get_iea_var(data, var, country, scenario, end_year=2030, interp=False) -> pd.DataFrame:
    """Get projected variable from IEA scenarios for a given
    country-scenario pair.
    
    Data may be (linearly) interpolated between latest historical value and chosen end year.
    
    Country-level emissions are available for Industry (not Iron & Steel).

    Args:
        var (str): elec_int, steel_prod
        country (str): region
        scenario (str): APS, STEPS
        end_year (int, optional): _description_. Defaults to 2030.
        interp (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    assert scenario in ["APS", "STEPS"]
    assert var in ["elec_int", "steel_prod"]
    
    if var == "elec_int":
        category, product, flow = ("CO2 total intensity", "Electricity", "Electricity generation")
    elif var == "steel_prod":
        category, product, flow = ("Industrial material production", "Crude steel", "Iron and steel")
    elif var == "emissions":
        category, product, flow = ("CO2 total", "Total", "Total energy supply")
    else:
        raise ValueError("Unknown IEA variable")
    
    base_year = 2022
    
    # Full name of scenario
    scn = "Stated Policies Scenario" if scenario == "STEPS" else ("Announced Pledges Scenario" if scenario == "APS" else "NON_SCENARIO")
    m1 = data["SCENARIO"] == scn
    m2 = data["REGION"] == country
    m3 = (data["CATEGORY"] == category) & (data["PRODUCT"] == product) & (data["FLOW"] == flow) 
    
    base_mask = (data['SCENARIO'] == "Stated Policies Scenario") &\
                m2 & m3 & (data['YEAR'] == base_year)
                
    end_mask = m1 & m2 & m3 & (data['YEAR'] == end_year)
    iea_var = data.loc[base_mask | end_mask]
    
    if interp:
        span_years = pd.DataFrame({"YEAR": range(base_year, end_year+1)})
        iea_var = pd.merge(span_years,
                            iea_var,
                            on="YEAR",
                            how='left')
        iea_var["VALUE"] = iea_var["VALUE"].interpolate(method="linear")
        iea_var['REGION'] = iea_var['REGION'].fillna(method="ffill")
        
        iea_var = iea_var.loc[:, ("YEAR", "SCENARIO", "REGION", "VALUE")]
    
    # Compute annualised growth rate of variable
    iea_var = iea_var.sort_values(by="YEAR", ascending=True)
    base_value = float(iea_var.loc[iea_var["YEAR"] == base_year, "VALUE"])
    end_value = float(iea_var.loc[iea_var["YEAR"] == end_year, "VALUE"])
    n = end_year - base_year
    cagr = get_cagr(base_value=base_value, end_value=end_value, n=n)
    
    # Rename columns
    iea_var.columns = iea_var.columns.str.lower()
    iea_var = iea_var.rename(columns={"value": f"{scenario}_{var}",
                                      "region": "iea_region"})
        
    return iea_var, cagr

