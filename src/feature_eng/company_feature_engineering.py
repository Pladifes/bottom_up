import pandas as pd
import numpy as np
import math

from pathlib import Path
project_dir = Path().cwd()
while project_dir.stem != "bottom_up_alignment": 
    project_dir = project_dir.parent
import sys
sys.path.append(str(project_dir / "src"))

from datasets.GSPTDataset import GSPTDataset
from datasets.utils import get_group_col



def get_X_y(gspt, 
            impute_prod, 
            average,
            EF,
            energy_mix_path,
            carbon_price_path,
            gspt2refi,
            gspt2gspt_path,
            parent_group_map,
            refinitiv,
            gspt2cdp,
            cdp):
    """Get training set (2019-2021) for group companies. Some companies are currently labeled as outliers
    since there seems to be a matching issue between bottom-up and top-down (share of steel activity in overall portfolio?,
    or unidentified plants or subsidiaries?)

    Args:
        gspt (_type_): _description_
        years (list): desired time coverage
        impute_prod (_type_): geographic level of aggregation
        average (_type_): utilisation rate averagering method

    Returns:
        _type_: _description_
    """
    years = [2019,2020, 2021]
    panel = gspt.get_panel_data(years=years, entity="country", impute_prod=impute_prod, average=average)
    X_y = get_features(dataset=panel,
                EF=EF,
                energy_mix_path=energy_mix_path,
                carbon_price_path=carbon_price_path,
                gspt2refi=gspt2refi,
                gspt2gspt_path=gspt2gspt_path,
                parent_group_map=parent_group_map,
                refinitiv=refinitiv,
                gspt2cdp=gspt2cdp,
                cdp=cdp,
                ef_level="country",
                ef_source="sci")

    X_y["TD_BU_ratio"] = X_y["max_CF12"] / X_y["Attributed emissions"]
    X_y["log_TD_BU_ratio"] = np.log(X_y["TD_BU_ratio"])
    # remove outliers based on interquartile distance
    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(X_y["log_TD_BU_ratio"], 25)
    Q3 = np.percentile(X_y["log_TD_BU_ratio"], 75)
    IQR = Q3 - Q1
    X_y_no_outliers = X_y.loc[X_y["log_TD_BU_ratio"].between(Q1-1.5*IQR, Q3+1.5*IQR)]

    return X_y_no_outliers

def get_X_y_groups(X_y: pd.DataFrame, feature_set: dict):
    """Get features (X), target (y) and row-wise groups. Each group corresponds to observations of the same company over multiple years.
    This helper function is used to prepare the data before running cross validation.

    Args:
        X_y (pd.DataFrame): dataframe with features and target
        feature_set (dict): choice of input features and target variable

    Returns:
        _type_: _description_
    """
    X_y = X_y.copy()
    y_col, X_cols = feature_set['y'], feature_set['X']
    X = X_y.loc[:, X_cols]
    y = X_y[[y_col]]
    
    #
    company_names = X.index.get_level_values("GSPT Name")
    name2int = {name: i for i, name in enumerate(list(company_names.unique()))}
    groups = company_names.map(name2int)
    return X, y, groups


def get_features(dataset: pd.DataFrame, 
                 EF,
                 energy_mix_path,
                 carbon_price_path,
                 gspt2refi: pd.DataFrame,
                 gspt2cdp: pd.DataFrame,
                 gspt2gspt_path: "Path",
                 parent_group_map: dict, 
                 refinitiv: pd.DataFrame,
                 cdp: pd.DataFrame,
                 ef_level,
                 ef_source,
                 ) -> pd.DataFrame:
    """Merge panels from GSPT and Refinitiv. Compute different features at company level based on bottom up information.

    Args:
        dataset (GSPTDataset): plant level data
        EF (pd.DataFrame): emissions factors
        mapping (pd.DataFrame): map between gspt and refinitiv company names
        gspt2cdp (pd.DataFrame): map between gspt and cdp company names
        refinitiv (pd.DataFrame): refinitiv dataset with reported data
        cdp (pd.DataFrame): "data" / "intermediate" / "cdp_clean.xlsx"

    Returns:
        pd.DataFrame: _description_
    """
    # datasets
    X = dataset.copy()
    init_len = X.shape[0]
    
    ## ASSET LEVEL FEATURES
    # Group agnostic features
    X['Attributed crude steel capacity (ttpa)'] = X["Nominal crude steel capacity (ttpa)"] * X["Share"]
    X['Attributed production (ttpa)'] = X["Estimated crude steel production (ttpa)"] * X["Share"]
    X[["lat", "lon"]] = X["Coordinates"].str.split(",", expand=True)
    X.replace({"lon": {None: np.nan}}, inplace=True)
    X[['lat', 'lon']] = X[['lat', 'lon']].astype(float) 
    
    # Group related features
    X['Plant Age'] = X.groupby("year").apply(get_plant_age).values
    X = EF.match_emissions_factors(X, level=ef_level, source=ef_source, techno_col="Main production process")
    
    # interactions
    X['Attributed emissions (ttpa)'] = X['Attributed production (ttpa)'] * X['EF']
    
    # electricity intensity
    X = get_electricity_intensity(X, energy_mix_path=energy_mix_path)

    X = get_carbon_price(X=X, carbon_price_path=carbon_price_path)
    
    # EAF country electrictiy generation intensity
    X["EAF_country_elec_int"] = (X["Main production process"] == "electric") * X["country_elec_int"]
    
    assert len(X) == init_len
    
    # rename single name subsidiaries to group name
    # to align with reported data (which most often reported at a higher level)
    X["Group"] = get_group_col(gspt_parent_col=X['Parent'], 
                               gspt2gspt_path=gspt2gspt_path, 
                               parent_group_map=parent_group_map)
    
    
    ## COMPANY LEVEL FEATURES / BOTTOM-UP FEATURES
    wm_capa = lambda x: np.average(x, weights=X.loc[x.index, "Attributed crude steel capacity (ttpa)"])
    
    company = X.groupby(['Group', 'year']).agg(
                                            prod_ttpa=("Attributed production (ttpa)", "sum"),
                                            emissions_ttpa=("Attributed emissions (ttpa)", "sum"),
                                            capacity_ttpa=("Nominal crude steel capacity (ttpa)", "sum"),
                                            BU_int_micro_capa=("EF", wm_capa), # Bottom-up intensity
                                            elec_int_capa=("country_elec_int", wm_capa),
                                            EAF_elec_int_capa=("EAF_country_elec_int", wm_capa),
                                            carbon_price=("CO2Price", wm_capa),
                                            carbon_status=('CO2Status', wm_capa),
                                            avg_EF=("EF", "mean"),
                                            EF_capa=("EF", wm_capa)
                                            ).reset_index()
    # company intensity 1 is measured as the average plant intensities (e.g. emission factors)
    # weighted by plant capacity
    company.rename(columns={"prod_ttpa": "Attributed production (ttpa)",
                            "emissions_ttpa": "Attributed emissions (ttpa)",
                            "capacity_ttpa": "Nominal crude steel capacity (ttpa)"}, inplace=True)
    # company intensity 2 is measured as the ratio between aggregated plant emissions
    # and plants aggregated production
    company['BU_int_macro'] = company['Attributed emissions (ttpa)'] / company['Attributed production (ttpa)']
    
    company['UR'] = company['Attributed production (ttpa)'] / company["Nominal crude steel capacity (ttpa)"]
    
    # rescale emissions features
    company['log_Attributed production'] = np.log(company['Attributed production (ttpa)'] * 1000)
    company["Attributed emissions"] = company['Attributed emissions (ttpa)'] * 1000
    company['log_Attributed emissions'] = np.log(company['Attributed emissions'])
    
    # add reported emissions from Refinitiv and CDP
    company = get_reported_emissions(company_feats=company, 
                                     gspt2refi=gspt2refi, 
                                     refinitiv=refinitiv,
                                     gspt2cdp=gspt2cdp,
                                     cdp=cdp)

    company.set_index(["GSPT_Name", "year"], inplace=True)
    
    return company


def get_steel_use_capita(plants: pd.DataFrame, steel_use_path, steel_use_map_path):
    """
    Steel use is a proxy for national steel demand.
    
    https://worldsteel.org/wp-content/uploads/World-Steel-in-Figures-2022.pdf
    
    Args:
        plants (pd.DataFrame): _description_
        steel_use_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    steel_use = pd.read_excel(steel_use_path)
    # countries from wsa with a steel use value
    wsa_countries = steel_use[['Country']]
    wsa_countries = wsa_countries.rename(columns={"Country": "WSA_Country"})
    
    plants = pd.merge(plants,
                      wsa_countries, 
                      left_on="Country", 
                      right_on="WSA_Country", 
                      how='left')
    
    # list of countries that did not match with raw matching
    steel_use_map = pd.read_excel(steel_use_map_path)
    
    # map plant countries (GSPT) to WSA countries
    plants = pd.merge(plants,
                      steel_use_map,
                      left_on="Country",
                      right_on="GSPT_Country",
                      how='left')
    plants = plants.drop(columns=["GSPT_Country"])
    
    plants['WSA_Country'] = plants['WSA_Country'].fillna(plants['missing_WSA_Country'])
    
    plants['WSA_Country'] = plants['WSA_Country'].fillna("World")
    
    # melt steel use frame
    steel_use_melt = steel_use.loc[:, ["Country", "2017", "2018", "2019", "2020", "2021"]].melt(id_vars=["Country"], var_name=["year"], value_name="steel_use_capita")
    steel_use_melt['year'] = steel_use_melt['year'].astype(int)
    
    # merge
    plants = pd.merge(plants,
                      steel_use_melt,
                      left_on=["WSA_Country", "year"],
                      right_on=["Country", "year"],
                      how='left')
    
    # clean
    plants = plants.drop(columns=["Country_y", "missing_WSA_Country"])
    plants = plants.rename(columns={"Country_x": "Country"})
    return plants


def get_steel_use(plants: pd.DataFrame, steel_use_path):
    """
    Steel use is a proxy for national steel demand.
    
    https://worldsteel.org/wp-content/uploads/World-Steel-in-Figures-2022.pdf
    
    Args:
        plants (pd.DataFrame): _description_
        steel_use_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    steel_use = pd.read_excel(steel_use_path)
    steel_use_melt = steel_use.loc[:, ["Country", "2017", "2018", "2019", "2020", "2021"]].melt(id_vars=["Country"], var_name=["year"], value_name="steel_use")
    steel_use_melt['year'] = steel_use_melt['year'].astype(int)
    plants = pd.merge(plants,
                   steel_use_melt,
                   on=["Country", "year"],
                   how='left')
    return plants
    
def get_future_carbon_price(plants, inc_group_map_path):
    """
    Add future carbon price from NZE50 scenario for three country groups.
    Since IEA does not seem to provide an explicit list of countries for each category
    (e.g. "advanced", "emerging", "other emerging"), we use the World Bank's income group
    mapping.
    
    https://iea.blob.core.windows.net/assets/2db1f4ab-85c0-4dd0-9a57-32e542556a49/GlobalEnergyandClimateModelDocumentation2022.pdf
    """
    # map countries to ["advanced", "emerging", "other_emerging"]
    inc_group_map = pd.read_excel(inc_group_map_path)
    
    plants= pd.merge(plants,
                inc_group_map,
                on="Country",
                how='left')
    
    prices = pd.DataFrame({"iea_2022": ["advanced", "emerging", "other_emerging"],
                           "Price": [140, 90, 25], # $/t CO2e
                           "Year": 2030})
    
    plants = pd.merge(plants,
                      prices,
                      on="iea_2022")  
    
    assert plants['Price'].notna().all()  
    
    return plants


def get_carbon_price(X: pd.DataFrame, carbon_price_path: str) -> pd.DataFrame:
    """Merge 2 carbon price features with plant level data.

    Args:
        X (pd.DataFrame): plant level data
        carbon_price_path (str): path to "Carbon Price Rework 20230405.xlsx"
    Returns:
        X (pd.DataFrame): plant level data with carbon price features
    """
    carbon_price = read_carbon_pricing(carbon_price_path=carbon_price_path)
    carbon_price.rename(columns={"FiscalYear": "year"}, inplace=True)
    cp2gspt_countries = {"United States of America": "United States",
                         'Korea, Dem. People\x92s Rep.': 'North Korea',
                         'Korea; Republic (S. Korea)': 'South Korea',
                         'Macedonia': 'North Macedonia',
                         'Slovak Republic': 'Slovakia',
                         'Iran, Islamic Republic of': 'Iran',
                         'Republic of Serbia': 'Serbia',
                        } 
    carbon_price.replace({"TR Name": cp2gspt_countries}, inplace=True)
    feats = ["TR Name", "year", 'CO2Status', "Price"]
    X = pd.merge(X, carbon_price.loc[:, feats], left_on=['Country', "year"], right_on=['TR Name', "year"], how='left')
    X = X.drop(columns=["TR Name"])

    # We assume countries with missing carbon policy status have none
    # Convert to binary variable (is implemented or not)
    X['CO2Status'] = X['CO2Status'] == "Implemented"
    
    # TODO: what is the unit of the carbon price
    # We assume countries with missing carbon price have zero carbon price
    X = X.replace({"Price": {np.nan: 0.}})
    X = X.rename(columns={"Price": "CO2Price"})
    return X

def read_carbon_pricing(carbon_price_path):
    """
    This code snippet performs the following operations:
    1. Selects specific columns from a pandas DataFrame called CarbonPricing.
    2. Transposes the selected columns into rows.
    3. Replaces any instances of "No" or "TBD" in the "Status" column of the transposed DataFrame with "No" and sets the corresponding values in the "CO2Law", "CO2Scheme", "CO2Status", and "CO2Coverage" columns to None.
    """
    CarbonPricing = pd.read_excel(carbon_price_path)
    CarbonPricing = CarbonPricing[
        [
            "TR Name",
            "CO2Law",
            "CO2Scheme",
            "CO2Status",
            "CO2Coverage",
            "StartYear",
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2009",
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "price_2004",
            "price_2005",
            "price_2006",
            "price_2007",
            "price_2008",
            "price_2009",
            "price_2010",
            "price_2011",
            "price_2012",
            "price_2013",
            "price_2014",
            "price_2015",
            "price_2016",
            "price_2017",
            "price_2018",
            "price_2019",
            "price_2020",
            "price_2021",
        ]
    ]
    price_cols = CarbonPricing.filter(regex="^price").columns.tolist()
    status_cols = CarbonPricing.filter(regex="^20").columns.tolist()

    CarbonPricing_Transposed1 = pd.melt(
        CarbonPricing,
        id_vars=[
            "TR Name",
            "CO2Law",
            "CO2Scheme",
            "CO2Status",
            "CO2Coverage",
            "StartYear",
        ],
        value_vars=price_cols,
        var_name="Year",
        value_name="Price",
    )
    
    CarbonPricing_Transposed1["FiscalYear"] = (
        CarbonPricing_Transposed1["Year"].str.extract(r"price_(\d{4})").astype(int)
    )

    CarbonPricing_Transposed1 = CarbonPricing_Transposed1.drop(columns=["Year"])
    CarbonPricing_Transposed1 = CarbonPricing_Transposed1.sort_values(by=["TR Name", "FiscalYear"]).reset_index(drop=True)

    CarbonPricing_Transposed = pd.melt(
        CarbonPricing,
        id_vars=[
            "TR Name",
            "CO2Law",
            "CO2Scheme",
            "CO2Status",
            "CO2Coverage",
            "StartYear",
        ],
        value_vars=status_cols,
        var_name="FiscalYear",
        value_name="Status",

    )
    
    CarbonPricing_Transposed = CarbonPricing_Transposed.sort_values(by=["TR Name", "FiscalYear"]).reset_index(drop=True)
    CarbonPricing_Transposed["Price"] = CarbonPricing_Transposed1["Price"]

    dict_names = {
        "Korea (South)": "Korea; Republic (S. Korea)",
        "USA": "United States of America",
        "British Virgin Islands": "Virgin Islands; British",
    }

    CarbonPricing_Transposed["TR Name"] = CarbonPricing_Transposed["TR Name"].replace(dict_names)

    mask = CarbonPricing_Transposed["Status"].isin(["No", "TBD"])

    CarbonPricing_Transposed.loc[

        mask, ["CO2Law", "CO2Scheme", "CO2Status", "CO2Coverage"]

    ] = ("No", None, None, None)

    CarbonPricing_Transposed["FiscalYear"] = CarbonPricing_Transposed.FiscalYear.astype(int)

    return CarbonPricing_Transposed


def nanaverage(c_group, X, weights_col="Nominal crude steel capacity (ttpa)"):
    """Calculate average not taking nan values into account."""
    idx = c_group.index
    masked_data = np.ma.masked_array(c_group, np.isnan(c_group))
    average = np.ma.average(masked_data, weights=X.loc[idx, weights_col])
    return average

    
def get_electricity_intensity(X, energy_mix_path):
    """Merge country electricity generation intensity from Ember to feature set.

    Args:
        X (_type_): plant level dataset

    Returns:
        _type_: entire feature set
    """
    # merge country and region energy mix
    energy_mix = pd.read_csv(energy_mix_path, usecols=["Area", "Year", "Continent", "Category", "Variable", "Unit", "Value"])
    energy_mix.rename(columns={"Value": "country_elec_int"}, inplace=True)
    ember_to_rename = {
        "Russian Federation (the)": "Russia",
        "Viet Nam": "Vietnam",
        "United States of America": "United States",
        "Bosnia Herzegovina": "Bosnia and Herzegovina",
        "Syrian Arab Republic (the)":  "Syria",
        "Iran (Islamic Republic of)": "Iran",
        "Czechia": "Czech Republic",
        "Korea (the Democratic People's Republic of)": "North Korea",
        "Philippines (the)": "Philippines",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
    }
    energy_mix = energy_mix.replace({"Area": ember_to_rename})
    energy_mix = energy_mix.rename(columns={"Year": "year"})
    gspt_to_rename = {"Türkiye": "Turkey"}
    X = X.replace({"Country": gspt_to_rename})
    X_merge = pd.merge(X, 
            energy_mix.loc[:, ["country_elec_int", "Area", "year"]], 
            left_on=["Country", "year"], 
            right_on=["Area", "year"], 
            how="left")
    # for countries that have not updated their intensities in 2022
    # we impute the value from 2021
    elec_int_2021 = energy_mix.loc[energy_mix["year"] == 2021]
    elec_int_2021["country_elec_int_2021"] = elec_int_2021["country_elec_int"].copy() 
    X_merge = pd.merge(X_merge,
                       elec_int_2021[["Area", "country_elec_int_2021"]],
                       left_on="Country",
                       right_on="Area",
                       how="left")
    X_merge['country_elec_int'] = X_merge['country_elec_int'].fillna(X_merge["country_elec_int_2021"])
    X_merge = X_merge.drop(columns=['Area_x', 'Area_y', "country_elec_int_2021"])
    return X_merge


def get_reported_emissions(company_feats: pd.DataFrame, 
                           gspt2refi: pd.DataFrame, 
                           refinitiv: pd.DataFrame,
                           gspt2cdp: pd.DataFrame,
                           cdp: pd.DataFrame):
    """Add top-down emissions columns to company level feature set.

    
    61 companies over 2019-2020
    119 points over 2019-2020
    companies present once: Cleveland-Cliffs Inc (2019), Gerdau SA (2020), Steel Dynamics Inc (2019)
    
    Args:
        company_feats (pd.DataFrame): bottom-up features + exogenous variables
        gspt2refi (pd.DataFrame): map between gspt and refinitiv company names
        refinitiv (pd.DataFrame): refinitiv dataset with top-down emissions
        gspt2cdp (pd.DataFrame): map between gspt and cdp company names
        cdp (pd.DataFrame): CDP dataset with top-down emissions

    Returns:
        pd.DataFrame: company_feats with top-down emissions from different sources (Refinitiv, CDP)
    """
    parent_name = "Group" if "Group" in company_feats.columns else "Parent"
    company_feats['GSPT_Name'] = company_feats[parent_name].copy()
    # mapping contains companies that were matched between GSPT and Refinitiv
    mergedf = pd.merge(company_feats, 
                       gspt2refi.loc[:, ["GSPT_Name", "Refinitiv_Name"]], 
                       on="GSPT_Name", 
                       how='left')
    assert len(company_feats) == len(mergedf)
    
    # add top down emissions and sector from Refinitiv
    refinitiv = refinitiv.rename(columns={"Name": "Refinitiv_Name",
                                          "FiscalYear": "year"})
    refinitiv = refinitiv.dropna(subset="Refinitiv_Name")
    df = pd.merge(mergedf, 
                  refinitiv.loc[:, ["Refinitiv_Name", "GICSName", "year", "CF12"]], 
                  on=["Refinitiv_Name", "year"], 
                  how="left")
    
    # add map from GSPT to CDP
    df = pd.merge(df, 
                  gspt2cdp, 
                  on="GSPT_Name", 
                  how='left')
    # add top-down emissions from CDP
    cdp["CDP_CF12_location"] = cdp['CDP_CF1'] + cdp['CDP_CF2_location']
    df = pd.merge(df, 
                  cdp[["account_name", "accounting_year", "CDP_CF12_location"]], 
                  left_on=["CDP_Name", "year"], 
                  right_on=["account_name", "accounting_year"], 
                  how='left')
    df = df.drop(columns=['account_name', "accounting_year"])
    
    # Add different flavours of top-down emissions based on Refinitiv and CDP values
    # CDP.fillna(Refinitiv)
    df['CDP_fillna_CF12'] = df["CDP_CF12_location"].fillna(df['CF12'])
    
    # max(CDP, Refinitiv)
    df['max_CF12'] = df[["CF12", "CDP_CF12_location"]].max(axis=1)
    
    # Add log top-down emissions
    df['log_CF12'] = np.log(df["CF12"])
    df['log_CDP_fillna_CF12'] = np.log(df["CDP_fillna_CF12"])
    df['log_max_CF12'] = np.log(df["max_CF12"])

    # keep companies with reported CDP_fillna_CF12 
    # (this or max equivalently have highest coverage)
    df = df.loc[df['CDP_fillna_CF12'].notna()].reset_index(drop=True)
    to_drop = ["Refinitiv_Name", parent_name]
    df = df.drop(columns=to_drop)
    return df
    
def get_plant_age(df):
    prod_year = int(df['year'].unique())
    return prod_year - df['Start year']



