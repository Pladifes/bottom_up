import pandas as pd

def get_all_techno_ur(start_year, end_year, prod_dir, techno_capa_path):
    """Get country/techno level utilisation rates for specified years,
    based on WSA reports.

    Args:
        start_year (_type_): _description_
        end_year (_type_): _description_
        prod_dir (_type_): bottom_up_alignment/data/raw/production

    Returns:
        pd.DataFrame: (year, country) steel utilisation rates
    """
    assert start_year >= 2018
    assert end_year <= 2022
    assert start_year <= end_year
    
    techno_ur_path = prod_dir / "country_techno_ur.xlsx"
    years = [str(year) for year in range(start_year, end_year+1)]
    all_ur_dict = pd.read_excel(techno_ur_path, sheet_name=years)
    country_techno_ur = pd.concat([all_ur_dict[year] for year in all_ur_dict.keys()], axis=0, ignore_index=True)
    usecols = ["Year", "Country", "Million\ntonnes", "Oxygen\n%", "Electric\n%", "Open\nhearth %", "Other\n%", "Total\n%"]
    country_techno_ur = country_techno_ur.loc[:, usecols]
    country_techno_ur.columns = country_techno_ur.columns.str.replace('\n', " ")
    country_techno_ur = country_techno_ur.replace("-", 0.)
    techno_cols = ["Oxygen %", "Electric %", "Open hearth %", "Other %"]
    country_techno_ur = country_techno_ur.astype({col: "float" for col in techno_cols})
    country_techno_ur.loc[:, techno_cols] = country_techno_ur.loc[:, techno_cols].fillna(0.)
    
    melt_techno_ur = country_techno_ur.melt(id_vars=["Year", "Country", "Million tonnes"],
                                            value_vars=techno_cols,
                                            value_name="techno_share",
                                            var_name="Technology")
    melt_techno_ur["Technology"] = melt_techno_ur["Technology"].str.split(" ", n=1, expand=True)[0]
    melt_techno_ur["techno_share"] /= 100. 
    melt_techno_ur = melt_techno_ur.rename(columns={"Million tonnes": "Production (Mt)"})
    melt_techno_ur["Production (Mt)"] = melt_techno_ur["Production (Mt)"].astype("float")
    
    # capacity
    capa = pd.read_excel(techno_capa_path)
    capa = capa.ffill()
    capa = capa.astype({"year": "int"})
    # techno mapping
    capa["Technology"] = capa["Main production process"].replace({"integrated (BF)": "Oxygen",
                                                                  "electric": "Electric"})
    capa["Country_gspt"] = capa['Country'].copy()
    
    # country mapping
    melt_techno_ur["Country"] = melt_techno_ur["Country"].replace({"Taiwan, China": "Taiwan",
                                                                   "Turkey": "Türkiye"})
    melt_techno = pd.merge(melt_techno_ur,
                              capa,
                              left_on=["Year", "Country", "Technology"],
                              right_on=["year", "Country", "Technology"],
                              how="left")

    melt_techno['country UR'] = melt_techno['techno_share'] * (melt_techno["Production (Mt)"] * 1e3) / melt_techno['Nominal crude steel capacity (ttpa)']
    
    return melt_techno

def get_all_country_ur(start_year, end_year, prod_dir, global_capa_path):
    """Get country level utilisation rates for specified years,
    based on country level production and capacity values from (resp.) WSA and OECD.

    Args:
        start_year (_type_): _description_
        end_year (_type_): _description_
        prod_dir (_type_): bottom_up_alignment/data/raw/production
        global_capa_path (_type_): bottom_up_alignment/data/raw/capacity / "STI_STEEL_MAKINGCAPACITY_23112023172621215.csv"

    Returns:
        pd.DataFrame: (year, country) steel utilisation rates
    """
    all_country_ur = []
    for year in range(start_year, end_year+1):
        country_prod_path = prod_dir / f"country_prod_WSiF/top_50_{year}.xlsx"
        country_ur = calculate_ur(prod_path=country_prod_path, capa_path=global_capa_path, level="country")
        country_ur = country_ur.rename(columns={"UR": "country UR"})
        all_country_ur.append(country_ur)
    all_country_ur = pd.concat(all_country_ur, axis=0, ignore_index=True)
    all_country_ur = all_country_ur.dropna(subset="country capacity (mt)", axis=0)
    return all_country_ur

def calculate_ur(capa_path: str, prod_path: str, level: str) -> pd.DataFrame:
    """
    Merge capacity and production dataframes on the 'year' and 'country' columns (if 'country' level is chosen)
    and calculate the Utilization Rate (UR) as the ratio of production to capacity.

    Parameters:
    - capa_path (str): File path to the capacity data (CSV format).
    - prod_path (str): File path to the production data (CSV format).
    - level (str): Either 'global' or 'country'. Default is 'global'.

    Returns:
    - pd.DataFrame: A dataframe containing 'year' and 'UR' columns.
    """
    # Read data into dataframes
    if capa_path.suffix == ".csv":
        capa_df = pd.read_csv(capa_path)
    else:
        capa_df = pd.read_excel(capa_path)
    if prod_path.suffix == ".csv":
        prod_df = pd.read_csv(prod_path)
    else:
        prod_df = pd.read_excel(prod_path)

    prod_df = prod_df.rename(columns={"Tonnage": "Crude steel production (Mt)"})
    prod_df = filter_prod(prod_df, level=level)
    
    # filter capacity data
    capa_df = filter_capa(capa_df, level=level)
    
    # lower case column names
    prod_df.columns = prod_df.columns.str.lower()
    capa_df.columns = capa_df.columns.str.lower()
    
    if level == "global":
        # Merge dataframes on the 'year' column for global level
        merged_df = pd.merge(capa_df, prod_df, on='year')
    elif level == "country":
        # Merge dataframes on both 'year' and 'country' columns for country level
        merged_df = pd.merge(prod_df,
                             capa_df[["year", "economy", "capacity (mt)"]],
                            left_on=['year', 'country'], 
                             right_on=['year', "economy"],
                             how='left')
        merged_df = merged_df.loc[~merged_df['country'].isin(["World", "Others"])].reset_index()
    elif level == "company":
        merged_df = pd.merge(prod_df, 
                            capa_df[["year", "group", "capacity (mt)"]], 
                            left_on=['year', 'gspt_group_name'],
                            right_on=['year', "group"],
                            how='left')
    else:
        raise ValueError("Invalid 'level' argument. Use either 'global' or 'country'.")

    # Calculate UR as the ratio of production to capacity
    merged_df['UR'] = merged_df['crude steel production (mt)'] / merged_df['capacity (mt)']
    merged_df['UR'] = merged_df['UR'].clip(upper=1.0)
    # countries with prod = capa = 0 get nan ur
    m1 = (merged_df["crude steel production (mt)"] == 0)
    m2 = (merged_df["capacity (mt)"] == 0)
    merged_df.loc[m1 & m2, "UR"] = 0.
        
    # Select only the 'year' and 'UR' columns
    final_cols = ['year', "country", 'UR', "crude steel production (mt)", "capacity (mt)"] if level == "country" else \
    (['year', 'company', "gspt_group_name", 'UR', "crude steel production (mt)", "capacity (mt)"]\
        if level == "company" else ['year', "UR", "crude steel production (mt)", "capacity (mt)"])
    result_df = merged_df[final_cols]
    result_df = result_df.rename(columns={"crude steel production (mt)": f"{level} production (mt)",
                                          "capacity (mt)": f"{level} capacity (mt)"})
    return result_df

def filter_prod(prod_df, level):
    """
    
    Country-level production values in World Steel Association's "Steel Statistical Yearbook"
    https://worldsteel.org/wp-content/uploads/Steel-Statistical-Yearbook-2019-concise-version.pdf

    Otherwise, country-level production data may be retrieved from graph on this page:
    https://worldsteel.org/steel-topics/statistics/annual-production-steel-data/?ind=P1_crude_steel_total_pub/CHN/IND/LBY
    Args:
        prod_df (_type_): _description_
        level (_type_): _description_

    Returns:
        _type_: _description_
    """
    if level == "country":
        # normalise WSA names (they may change across years)
        wsa2wsa = {"Czechia": "Czech Republic",
                   "Turkey": "Türkiye",
                   "Taiwan, China": "Taiwan",
                   "Byelorussia": "Belarus",
                   "Viet Nam": "Vietnam",
                   "Slovak Republic": "Slovakia",
                   "D.P.R. Korea": "North Korea",
                   "Macedonia": "North Macedonia"}
        prod_df["Country"] = prod_df["Country"].replace(wsa2wsa)
        
        # exceptions (potential reporting errors)
        drop1 = (prod_df['Country'] == 'Switzerland') & (prod_df['Year'].isin([2019, 2020, 2021]))
        drop2 = (prod_df['Country'] == 'Philippines') & (prod_df['Year'].isin([2019, 2020, 2021]))

        prod_df = prod_df.drop(prod_df.loc[drop1 | drop2].index)
        # missing production values 
        missing_prod = [("North Korea", 2019, 680 / 1e3), # Source: https://tradingeconomics.com/north-korea/steel-production
                        ("North Korea", 2020, 730 / 1e3),
                        ("North Korea", 2021, 730 / 1e3), 
                        ("Bangladesh", 2019, 5.1),  # https://www.tbsnews.net/bangladesh/bangladeshs-steel-production-hits-3-year-low-amid-reduced-demand-664586
                        ("Venezuela", 2019, 0.05079),
                        ("Venezuela", 2020, 0.029),
                        ("Venezuela", 2021, 0.029), 
                        ('Peru', 2019, 1.2313),
                        ("Peru", 2020, 0.731),
                        ("Peru", 2021, 1.234), 
                        ("Qatar", 2020, 1.218),
                        ("Qatar", 2021, 1.002),
                        ("Hungary", 2021, 1.543), 
                        ("Syria", 2019, 0.005), # (2018 value) https://www.statista.com/statistics/1263022/syria-crude-steel-production-volume/ 
                        ("Syria", 2020, 0.005),
                        ("Syria", 2021, 0.005),
                        ("Chile", 2019, 0.94507),
                        ("Chile", 2020, 1.07934),
                        ("Bosnia and Herzegovina", 2019, 0.801), 
                        ("Bosnia and Herzegovina", 2020, 0.759), 
                        ("Bosnia and Herzegovina", 2021, 0.775), 
                        ("Morocco", 2019, 0.5), # source: steel statistical yearbook 2020
                        ("Morocco", 2020, 0.320), # source: https://min-met.com/blog/2021-steel-statistical-yearbook-published/
                        ("Morocco", 2021, 0.5), 
                        ("Libya", 2019, 0.6), # source: steel statistical yearbook
                        ("Libya", 2020, 0.495), # source: WSA graph
                        ("Libya", 2021, 0.652), 
                        ("Nigeria", 2019, 0.65),
                        ("Nigeria", 2020, 0.65),
                        ("Nigeria", 2021, 0.65),
                        ("Iraq", 2019, 0.3),
                        ("Iraq", 2020, 0.3),
                        ("Iraq", 2021, 0.3),
                        ("Bulgaria", 2019, 0.566),
                        ("Bulgaria", 2020, 0.486),
                        ("Bulgaria", 2021, 0.548),
                        ("Kuwait", 2019, 1.27),
                        ("Bahrain", 2019, 0.8),
                        ("Bahrain", 2020, 0.9),
                        ("Bahrain", 2021, 1.167),
                        ("Uzbekistan", 2019, 0.666),
                        ("Uzbekistan", 2020, 0.939),
                        ("Uzbekistan", 2021, 1.05),
                        ("Moldova", 2019, 0.392),
                        ("Moldova", 2020, 0.465),
                        ("Moldova", 2021, 0.570),
                        ("Azerbaijan", 2019, 0.2),
                        ("Azerbaijan", 2020, 0.2),
                        ("Azerbaijan", 2021, 0.286),
                        ("Ghana", 2019, 0.4),
                        ("Ghana", 2020, 0.4),
                        ("Ghana", 2021, 0.4),
                        ("Singapore", 2019, 0.55),
                        ("Singapore", 2020, 0.419),
                        ("Singapore", 2021, 0.568),
                        ("Slovenia", 2019, 0.623),
                        ("Slovenia", 2020, 0.585),
                        ("Slovenia", 2021, 0.662),
                        ("Albania", 2019, 0.),
                        ("Albania", 2020, 0.),
                        ("Albania", 2021, 0.),
                        ("Norway", 2019, 0.621),
                        ("Norway", 2020, 0.624),
                        ("Norway", 2021, 0.622),
                        ("Switzerland", 2019, 1.325),
                        ("Switzerland", 2020, 1.175),
                        ("Switzerland", 2021, 1.291),
                        ("New Zealand", 2019, 0.667),
                        ("New Zealand", 2020, 0.586),
                        ("New Zealand", 2021, 1.291),
                        ("North Macedonia", 2019, 0.239),
                        ("North Macedonia", 2020, 0.180),
                        ("North Macedonia", 2021, 0.315),
                        ("Guatemala", 2019, 0.306),
                        ("Guatemala", 2020, 0.283),
                        ("Guatemala", 2021, 0.307),
                        ("Angola", 2019, 0.290),
                        ("Angola", 2020, 0.250),
                        ("Angola", 2021, 0.275),
                        ("Philippines", 2019, 1.915),
                        ("Philippines", 2020, 0.892),
                        ("Philippines", 2021, 1.555),
                        ("Uganda", 2018, 0.6), # 2019 value
                        ("Uganda", 2019, 0.6), # https://www.researchgate.net/publication/344783277_Modelling_the_Growth_Trend_of_the_Iron_and_Steel_Industry_Case_for_Uganda
                        ("Uganda", 2020, 0.6), # 2019 value
                        ("Uganda", 2021, 0.6), # 2019 value
                        ("Georgia", 2019, 0.),
                        ("Georgia", 2020, 0.),
                        ("Georgia", 2021, 0.),
                        ("Croatia", 2019, 0.069),
                        ("Croatia", 2020, 0.045),
                        ("Croatia", 2021, 0.185),
                        ("Uganda", 2022, 0.6),
                        ("Namibia", 2022, 1E15) 
                        ]
        missing_prod = pd.DataFrame(missing_prod, columns=("Country", "Year", "Crude steel production (Mt)"))
        
        year = int(prod_df['Year'].unique())
        
        prod_df = pd.concat([prod_df,
                            missing_prod.loc[missing_prod["Year"] == year]],
                            axis=0,
                            ignore_index=True)

    else:
        pass
    return prod_df

def filter_capa(capa_df: str, level: str) -> pd.DataFrame:
    """
    Read global capacity data and filter on the 'country' column for 'WLD'.

    Parameters:
    - glob_capa_path (str): File path to the global capacity data (CSV format) from the OECD.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing only rows where 'country' is 'WLD'.
    """
    if level in ["global", "country"]:
        # duplicate year column
        capa_df = capa_df.drop(columns=["YEAR"])
        capa_df = capa_df.rename(columns={"Value": "Capacity (Mt)"})
    
        if level == "global":
            # Filter on the 'country' column for 'WLD'
            filtered_df = capa_df[capa_df['COUNTRY'] == 'WLD']
        elif level == "country":
            filtered_df = capa_df[capa_df['COUNTRY'] != 'WLD']
            # map oecd countries to gspt
            oecd2wsa = {"China (People's Republic of)": "China",
                         "Democratic People's Republic of Korea": "North Korea", 
                         "Korea": "South Korea",
                         "Viet Nam": "Vietnam",
                         "Byelorussia": "Belarus",
                         "Chinese Taipei": "Taiwan",
                         "Slovak Republic": "Slovakia",
                         "Syrian Arab Republic": "Syria",
                         "Czechia": "Czech Republic"
                         } 
            filtered_df["Economy"] = filtered_df["Economy"].replace(oecd2wsa) 
    elif level == "company":
        filtered_df = capa_df
    else:
        raise Exception
    return filtered_df
