import pandas as pd
import numpy as np

import json
from pathlib import Path
try:
    from utils import get_parents_ids, convert_plant2parent
except:
    from .utils import get_parents_ids, convert_plant2parent

from pathlib import Path
project_dir = Path().cwd()
import sys
sys.path.append(str(project_dir / "src"))
from feature_eng.plant_features import get_all_country_ur, get_all_techno_ur


class GSPTDataset:
    def __init__(self, data_path, version_year, missing_years_path, gspt2gspt_path, parent_group_map_path):
        self.data_path = Path(data_path)
        self.version_year = version_year
        if not str(self.version_year) in self.data_path.as_posix():
            raise AssertionError("Data file and version year do not match! Check path.")
        if self.version_year == 2022:
            self.prod_years = [2019, 2020]
        elif self.version_year == 2023:
            self.prod_years = [2019, 2020, 2021]
        self.missing_years_path = missing_years_path
        self.gspt2gspt_path = gspt2gspt_path
        with open(parent_group_map_path, "r") as f:
            self.parent_group_map = json.load(f)
        self.raw_data = pd.read_excel(data_path, sheet_name=['Steel Plants', 'Yearly Production', "Metadata"], engine="calamine")
        self.steel_plants = self.raw_data["Steel Plants"]

    def get_projected_data(self, parent_group_map_path, wsa_ef_path):
        """Get projected bottom-up capacity, production and emissions based on global emissions factors.
        Projections are made for years 2022-2030.

        Args:
            parent_group_map_path (str): path to mappings/parent_group_map.json used to remap some subsidiaries to group ownership.
            wsa_ef_path: path to global emission factors from world steel association
        """
        # OECD Steelmaking
        GLOBAL_UR_21 = 0.785
        with open(parent_group_map_path, "r") as f:
            parent_group_map = json.load(f)
            
        # get 2019 utilization rates
        df19_melt = self.get_melted_dataset(year=2019, impute_prod='Country')
        df19_melt['Group'] = df19_melt['Parent'].copy()
        # Vital step for getting accurate BU aggregated data
        df19_melt = df19_melt.replace({'Group': parent_group_map})
        df19_melt['Attributed crude steel capacity (ttpa)'] = df19_melt['Nominal crude steel capacity (ttpa)'] * df19_melt['Share']
        ur_df = df19_melt.groupby(["Group", "Main production process"]).agg({"Attributed crude steel capacity (ttpa)": "sum",
                                        "Estimated crude steel production (ttpa)": "sum"})
        ur_df["UR"] = ur_df["Estimated crude steel production (ttpa)"] / ur_df["Attributed crude steel capacity (ttpa)"]
        
        ur_df.reset_index(inplace=True)
        
        # get EF
        EF = pd.read_excel(wsa_ef_path, engine="calamine")
        EF.rename(columns={"Technology": "Main production process"}, inplace=True)
        
        # get operating plants between 2022 and 2030
        # this is solely based on actual open/closure dates
        # (i.e. no projection hypotheses are yet included)
        op_plants_melt = pd.concat([self.get_operating_plants(year=year) for year in range(2022, 2031)],
                                   axis=0,
                                   ignore_index=True
                                   )
        
        # Aggregate plant capacity based on group company mapping
        op_plants_melt['Group'] = op_plants_melt['Parent'].copy()
        op_plants_melt = op_plants_melt.replace({'Group': parent_group_map})
        op_plants_melt['Attributed crude steel capacity (ttpa)'] = op_plants_melt['Nominal crude steel capacity (ttpa)'] * op_plants_melt['Share']
        
        # Baseline projected group company capacity
        group_capa = op_plants_melt.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
        proj_group_capa = group_capa.copy()
        
        # Hypothesis: assuming constant steelmaking capacity
        # new electric plants should replace blast furnace plants
        start_year_mask = op_plants_melt['Start year'] >= 2022
        is_elec_mask = op_plants_melt['Main production process'] == "electric"
        new_elec_plants = op_plants_melt.loc[start_year_mask & is_elec_mask]
        new_elec_group = new_elec_plants.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
        new_elec_group = new_elec_group[['Group', 'year', 'Attributed crude steel capacity (ttpa)']]
        # Subtract additional electric capacity from BF capacity
        BF_mask = proj_group_capa['Main production process'] == "integrated (BF)"
        BF_capa = proj_group_capa.loc[BF_mask].copy()
        new_BF_capa = pd.merge(BF_capa, 
                           new_elec_group, 
                           how="left", 
                           on=["Group", "year"],
                           suffixes=["_BF", "_elec"])
        new_BF_capa['Attributed crude steel capacity (ttpa)_elec'] = new_BF_capa['Attributed crude steel capacity (ttpa)_elec'].replace(np.nan, 0.)
        new_BF_capa['new Attributed crude steel capacity (ttpa)_BF'] = new_BF_capa['Attributed crude steel capacity (ttpa)_BF'] - new_BF_capa['Attributed crude steel capacity (ttpa)_elec']
        proj_group_capa.loc[BF_mask, 'Attributed crude steel capacity (ttpa)'] = new_BF_capa['new Attributed crude steel capacity (ttpa)_BF']
        
        # Merge UR and emission factors for calculating production from capacity
        mergedf_proj = pd.merge(proj_group_capa, ur_df[["Group", "Main production process", "UR"]], on=["Group", "Main production process"], how='left')
        mergedf_proj['UR'] = mergedf_proj['UR'].fillna(GLOBAL_UR_21)
        assert len(mergedf_proj) == len(proj_group_capa)
        assert mergedf_proj['UR'].notna().all()
        
        # compute projected production
        mergedf_proj["Projected production (ttpa)"] = mergedf_proj["Attributed crude steel capacity (ttpa)"] * mergedf_proj["UR"]
        mergedf_proj = pd.merge(mergedf_proj, EF[["Main production process", "EF"]], on="Main production process", how='left')
        # compute projected bottom-up emissions
        mergedf_proj['Projected BU emissions (ttpa)'] = mergedf_proj['Projected production (ttpa)'] * mergedf_proj['EF']
        
        # final projection values used as input to model
        final_proj = mergedf_proj.groupby(["Group", "year"])['Projected BU emissions (ttpa)'].sum().reset_index()
        return final_proj
            
        
        
    
    def get_panel_data(self, years: list, entity: str, impute_prod: str, average: str):
        """Get operating plants for 'years' period. Missing production values are imputed for each plant according to 
        impute_prod method.
        
        Args:
            years (list): list of years to get plant data
        Returns:
            pd.DataFrame: plant-level panel data
        """
        panel = []
        for year in years:
            df = self.get_melted_dataset(year=year, entity=entity, impute_prod=impute_prod, average=average)
            panel.append(df)
        panel = pd.concat(panel, axis=0)
        return panel
    
    def get_melted_dataset(self, year: int, entity: str, impute_prod: str, average: str):
        """Each row is a tuple (year, company, plant). That is, nominal crude steel capacity
        will not add up to global capacity. Instead, look at attributed capacity.

        Args:
            year (int): [2019, 2020]
            impute_prod (str): impute production values method

        Returns:
            pd.DataFrame: plant level dataset melted on company column
        """
        dataset = self.get_merged_capacity_prod(year=year)
        # PLANT LEVEL FEATURES
        # impute Utilization rate at impute_prod level
        dataset["Estimated UR crude steel"] = self.get_estimated_UR(capa_prod=dataset, 
                                                                    entity=entity,
                                                                    agg_level=impute_prod, 
                                                                    average=average)
        # impute production: capacity * UR
        dataset["Estimated crude steel production (ttpa)"] = dataset["Nominal crude steel capacity (ttpa)"] * dataset["Estimated UR crude steel"]
        #dataset["Estimated crude steel production (ttpa)"] = dataset["Crude steel production (ttpa)"].fillna(filled_values) 
        #dataset["Estimated crude steel production (ttpa)"] = self.impute_prod(dataset, impute_year=year)
        
        melted_gspt = convert_plant2parent(dataset, self.gspt2gspt_path, self.parent_group_map)
        return melted_gspt
    
    def get_estimated_prod(self, year: int, entity: str, impute_prod: str, average: str):
        """Add estimated production column to data for a given year.
        Estimate plant level production for all plants in the dataset for which production is missing.

        Args:
            year (int): production year
            impute_prod (str): geographical level at which utilisation rates are imputed
        """
        def get_historical_prod(year, impute_prod, average) -> pd.DataFrame:
            if year == 2022:
                capa_prod = self.get_operating_plants(start_year=year)
            else:
                capa_prod = self.get_merged_capacity_prod(year=year)
            # PLANT LEVEL FEATURES
            # impute Utilization rate at impute_prod level
            UR_col, ur_df = self.get_estimated_UR(capa_prod=capa_prod.copy(), 
                                                                                 entity="country",
                                                                                agg_level=impute_prod, 
                                                                                average=average,
                                                                                return_all=True)
            capa_prod["Estimated UR crude steel"] = UR_col.values
            #capa_prod["Estimated UR crude steel"] = capa_prod["Estimated UR crude steel"].fillna(ur_df['UR crude steel'].mean())
            
            # impute production: capacity * UR
            #capa_prod["Estimated crude steel production (ttpa)"] = self.impute_prod(capa_prod=capa_prod, impute_year=year)
            capa_prod["Estimated crude steel production (ttpa)"] = capa_prod["Estimated UR crude steel"] * capa_prod['Nominal crude steel capacity (ttpa)']
            #capa_prod["Estimated crude steel production (ttpa)"] = capa_prod["Crude steel production (ttpa)"].fillna(capa_prod["Estimated crude steel production (ttpa)"])
            return capa_prod
        
        if entity == "country":
            dataset = get_historical_prod(year=year, impute_prod=impute_prod, average=average)
        else:
            # Estimate past production
            if year in self.prod_years:
                dataset = get_historical_prod(year=year, impute_prod=impute_prod, average=average)
            # Estimate future production
            elif max(self.prod_years) < year <= 2022:
                # estimate utilisation rate from historical data
                capa_prod = self.get_merged_capacity_prod(year=2021)
                _, ur_df = self.get_estimated_UR(capa_prod=capa_prod, 
                                                entity="plant",
                                                agg_level=impute_prod, 
                                                average=average,
                                                return_all=True)
                
                # impute utilisation rates for future years without production
                capa = self.get_operating_plants(start_year=year) #, melt=True)
                if impute_prod in ["Country", "Region"]:
                    capa_ur = pd.merge(capa, 
                                    ur_df.reset_index(), on=["Main production process", impute_prod], how='left')
                elif impute_prod == "Global":
                    capa_ur = pd.merge(capa, 
                                    ur_df.reset_index(), on=["Main production process"], how='left')
                else:
                    raise Exception
                capa_ur['Estimated UR crude steel'] = capa_ur['UR crude steel'].fillna(capa_ur['UR crude steel'].mean()) 
                capa_ur['Estimated crude steel production (ttpa)'] = capa_ur['Estimated UR crude steel'] * capa_ur['Nominal crude steel capacity (ttpa)']
                #capa_ur['Attributed crude steel production'] = capa_ur['UR crude steel'] * (capa_ur['Nominal crude steel capacity (ttpa)'] * 1e3) * capa_ur['Share']
                
                dataset = capa_ur.copy()
            else:
                raise ValueError(f"Production cannot currently be estimated for input year ({year})!")
        return dataset
    
    
    def impute_prod(self, capa_prod: pd.DataFrame, impute_year: int) -> pd.Series:
        """Calculate estimated production from estimated utilisation rate and capacity. Then impute missing production values.

        Args:
            capa_prod (pd.DataFrame): plant level data with capacity and production values. Operating plants.

        Returns:
            pd.Series: crude steel production column with imputed values
        """
        # Create flag for distinguishing original production values from estimated production values
        capa_prod['is_orig'] = capa_prod['Crude steel production (ttpa)'].notna()
        
        ## 1. For plants with at least one production value for a given year,
        ## we can apply cross-temporal imputation
        
        # Merge all plants with production values across all year
        all_prod = pd.DataFrame()
        for year in self.prod_years:
            # Plant with original production values for a given year
            orig_prod = self.get_prod_dataset(year=year)
            orig_prod = orig_prod.loc[:, ["Plant ID", "Plant name (English)", "Crude steel production (ttpa)"]]
            orig_prod = orig_prod.rename(columns={"Crude steel production (ttpa)": f"Crude steel production (ttpa) {year}"})
            if not all_prod.empty:
                all_prod = pd.merge(all_prod, orig_prod[["Plant ID", f"Crude steel production (ttpa) {year}"]], how='left', on='Plant ID')
            else:
                all_prod = orig_prod
        
        # Add crude steel production with imputed values from other years
        capa_prod = pd.merge(capa_prod, 
                             all_prod.loc[:, ["Plant ID", f"Crude steel production (ttpa) {impute_year}"]],
                             on="Plant ID",
                             how="left")
        
        # Fillna crude steel production column
        temporal_imputed_prod = capa_prod[f"Crude steel production (ttpa) {impute_year}"]
        capa_prod['Crude steel production (ttpa)'] = capa_prod[f'Crude steel production (ttpa)'].fillna(temporal_imputed_prod)
        
        ## 2. For plants without production information, we rely on geography and technology,
        ## which are accounted for when estimating the utilisation rate
        
        # Estimate production from capacity and utilisation rate
        estimated_prod = capa_prod['Estimated UR crude steel'] * capa_prod['Nominal crude steel capacity (ttpa)']
        
        # Impute missing production values at plant level
        imputed_prod = capa_prod['Crude steel production (ttpa)'].fillna(estimated_prod)
        return imputed_prod

        
    def get_estimated_UR(self, capa_prod: pd.DataFrame, entity: str, agg_level: str, average: str, return_all=False):
        """Calculate plant level utilisation rate.

        Args:
            cap_prod (pd.DataFrame): -> get_merged_capacity_prod
            entity (str): entity from which plant utilisation rates are calculated. Includes company, country, plant.
            agg_level (str): level for aggregating utilisation rates ['Global', 'Region', 'Country']. Applies to plant entity.
            average (str): Type of averaging: "micro", "macro". Applies to plant entity.
        """
        if entity == "techno":
            raw_data_dir = self.data_path.parent
            prod_dir = raw_data_dir / "production"
            techno_capa_path = raw_data_dir / "capacity" / "country_techno_capa.xlsx"
            all_country_ur = get_all_techno_ur(start_year=2019,
                                                end_year=2022,
                                                prod_dir=prod_dir,
                                                techno_capa_path=techno_capa_path
                                                )
            
            plants_ur = pd.merge(capa_prod,
                                all_country_ur[["Year", "Country_gspt", "country UR", "Main production process"]],
                                left_on=["year", "Country", "Main production process"],
                                right_on=["Year", "Country_gspt", "Main production process"], 
                                how="left")
            plants_ur = plants_ur.rename(columns={"country UR": "UR crude steel"})
            UR_col = plants_ur["UR crude steel"]
            ur_df = all_country_ur
        elif entity == "country":
            raw_data_dir = self.data_path.parent
            prod_dir = raw_data_dir / "production"
            global_capa_path = raw_data_dir / "capacity" / "STI_STEEL_MAKINGCAPACITY_23112023172621215.csv"
            all_country_ur = get_all_country_ur(start_year=2019,
                                                end_year=2022,
                                                prod_dir=prod_dir,
                                                global_capa_path=global_capa_path)
            plants_ur = pd.merge(capa_prod,
                                all_country_ur[["year", "country", "country UR"]],
                                left_on=["year", "Country"],
                                right_on=["year", "country"], 
                                how="left")
            plants_ur = plants_ur.rename(columns={"country UR": "UR crude steel"})
            UR_col = plants_ur["UR crude steel"]
            ur_df = all_country_ur
        elif entity == "plant":
            prod_col = "Crude steel production (ttpa)"
            capacity_col = "Nominal crude steel capacity (ttpa)"
            
            cols = ['Country', "Region", "Main production process", prod_col, capacity_col]
            df = capa_prod.loc[:, cols].copy()
            ur_df = self.impute_UR(df, level=agg_level, average=average)
            
            # Estimated UR for input df
            if agg_level == "Global":
                UR_col = pd.merge(capa_prod[["Main production process"]], ur_df, on=["Main production process"], how='left')['UR crude steel']
            elif agg_level == "Region":
                UR_col = pd.merge(capa_prod[["Main production process", "Region"]], ur_df, on=["Main production process", "Region"], how='left')['UR crude steel']
            elif agg_level == "Country":
                UR_col = pd.merge(capa_prod[["Main production process", "Region", "Country"]], ur_df, on=["Main production process", "Region", "Country"], how='left')['UR crude steel']
            else:
                raise ValueError(f"Specified level ({agg_level}) is not recognised!")
        else:
            raise ValueError("Unknown entity!")
        
        if not return_all:
            return UR_col
        else:
            return UR_col, ur_df

    @staticmethod
    def impute_UR(df, level, average: str) -> pd.Series:
        """Aggregate utilisation rates and impute at some level.

        Args:
            df (_type_): _description_
            level (_type_): _description_
            average (str): "micro", "macro"

        Returns:
            pd.Series: utilisation rates column
        """
        ur_col = "UR crude steel"
        techno_col = 'Main production process'
        
        df['UR crude steel'] = df["Crude steel production (ttpa)"] / df["Nominal crude steel capacity (ttpa)"]
        df['UR crude steel'] = df['UR crude steel'].clip(upper=1.0)
        
        if level == "Global":
            if average == "micro":
                # compute average utilisation rate for each technology (across all countries)
                ur_df = df.groupby(techno_col).agg(ur_col = (ur_col, "mean"), 
                                                sample = (ur_col, lambda x:x.notna().sum()))
                ur_df = ur_df.rename(columns={"ur_col": ur_col})
                
                # impute data for technologies with missing production values
                # using the average utilisation rate across technologies
                ur_df['UR crude steel'] = ur_df['UR crude steel'].fillna(ur_df['UR crude steel'].mean())
            elif average == "macro":
                notna_df = df.loc[df["Crude steel production (ttpa)"].notna()]
                ur_df = notna_df.groupby(techno_col).agg({"Crude steel production (ttpa)": "sum",
                                                    "Nominal crude steel capacity (ttpa)": "sum"})
                # compute macro UR
                ur_df[ur_col] = ur_df["Crude steel production (ttpa)"] / ur_df["Nominal crude steel capacity (ttpa)"]
                ur_df = ur_df.drop(columns=["Crude steel production (ttpa)", "Nominal crude steel capacity (ttpa)"])
        elif level == "Region":
            if average == "micro":
                # compute average utilisation rate for each technology across each Region
                ur_df = df.groupby([techno_col, level]).agg(ur_col = (ur_col, "mean"), 
                                                            sample = (ur_col, lambda x:x.notna().sum()))
                ur_df = ur_df.rename(columns={"ur_col": ur_col})

                # impute missing region UR with average per techno
                ur_df['UR crude steel'] = ur_df['UR crude steel'].fillna(ur_df.groupby('Main production process')['UR crude steel'].transform('mean'))
                # when a unique (technology, Region) has a missing value,
                # we impute the mean of the previously calculated averages (across (technology, region) pairs)
                ur_df['UR crude steel'] = ur_df['UR crude steel'].fillna(ur_df['UR crude steel'].mean())
            elif average == "macro":
                notna_df = df.loc[df["Crude steel production (ttpa)"].notna()]
                ur_df = notna_df.groupby([techno_col, level]).agg({"Crude steel production (ttpa)": "sum",
                                                    "Nominal crude steel capacity (ttpa)": "sum"})
                # compute macro UR
                ur_df[ur_col] = ur_df["Crude steel production (ttpa)"] / ur_df["Nominal crude steel capacity (ttpa)"]
                ur_df = ur_df.drop(columns=["Crude steel production (ttpa)", "Nominal crude steel capacity (ttpa)"])
        elif level == "Country":
            if average == "micro":
                # compute average utilisation rate for each technology across each country
                # groupby is performed over Region and Country
                ur_df = df.groupby([techno_col, "Region", "Country"]).agg(ur_col = (ur_col, "mean"), 
                                                                        sample = (ur_col, lambda x:x.notna().sum()))
                ur_df = ur_df.rename(columns={"ur_col": ur_col})
                # impute missing country UR with average per (techno, region)
                ur_df['UR crude steel'] = ur_df['UR crude steel'].fillna(ur_df.groupby(['Main production process', 'Region'])['UR crude steel'].transform('mean'))
                # when a unique (technology, region, country) has a missing value within a region with more than one countries
                # we impute the maximum average utilisation within that (technology, region)
                ur_df['UR crude steel'] = ur_df['UR crude steel'].fillna(ur_df.groupby(['Main production process'])['UR crude steel'].transform('mean'))
                # when a unique (technology, region, country) has a missing value within a singleton region
                # we impute the mean average utilisation across all (technology, region, country) tuples
                ur_df['UR crude steel'] = ur_df['UR crude steel'].fillna(ur_df['UR crude steel'].mean())
            elif average == "macro":
                notna_df = df.loc[df["Crude steel production (ttpa)"].notna()]
                ur_df = notna_df.groupby([techno_col, "Region", "Country"]).agg({"Crude steel production (ttpa)": "sum",
                                                    "Nominal crude steel capacity (ttpa)": "sum"})
                # compute macro UR
                ur_df[ur_col] = ur_df["Crude steel production (ttpa)"] / ur_df["Nominal crude steel capacity (ttpa)"]
                ur_df = ur_df.drop(columns=["Crude steel production (ttpa)", "Nominal crude steel capacity (ttpa)"])
        else:
            raise Exception
        
        ur_df["UR crude steel"] = ur_df['UR crude steel'].clip(upper=1.0)
        return ur_df
        
    def get_merged_capacity_prod(self, year: int):
        """Get steel capacity and production for all operating plants in a given year.

        Args:
            year (int): [2019, 2020, 2021] -> years for which real production values are available

        Returns:
            pd.DataFrame: plant level data with capacity and production
        """
        capacity_df = self.get_steel_dataset()
        prod_df = self.get_prod_dataset(year=year)
        merged_df = pd.merge(capacity_df, prod_df, how='left', on='Plant ID')

        # select operating plants for given years (according to available knowledge)
        plant_mask = self.get_current_plant_mask(year=year, plants=merged_df)
        merged_df = merged_df.loc[plant_mask, :]
            
        merged_df = merged_df.reset_index(drop=True)
        merged_df["year"] = year
        return merged_df
    
    def get_operating_plants(self, start_year: int, end_year: int = None, melt: bool = False):
        """Select operating plants for a given year (according to available knowledge).
        Args:
            start_year (int): past, present and future years
            end_year (int): if this is specified, returns operating plants between start_year and end_year
            melt (bool): Melt from (Plant ID) to (Plant ID, Parent, Share)
        Returns:
            pd.DataFrame: Operating plants for given year
        """
        def get_single_year(year: int, melt: bool):
            plants = self.get_steel_dataset()
            plant_mask = self.get_current_plant_mask(year=year, plants=plants)
            op_plants = plants.loc[plant_mask, :]
            if melt:
                parent_ids = get_parents_ids(op_plants, gspt2gspt_path=self.gspt2gspt_path)
                
                final_cols = [col for col in op_plants.columns if col not in ["Parent", "Parent PermID"]]
                
                op_plants_melted = pd.merge(
                    parent_ids, op_plants.loc[:, final_cols], on="Plant ID", how="left"
                )
                op_plants_melted.loc[:, 'year'] = year
                return op_plants_melted
            else:
                op_plants.loc[:, 'year'] = year
                return op_plants
        
        if isinstance(start_year, int) & (end_year is None):
            op_plants_melted = get_single_year(year=start_year, melt=melt)
        
        elif isinstance(start_year, int) & isinstance(end_year, int):
            op_plants_years = []
            for year in range(start_year, end_year + 1):
                op_plants_year = get_single_year(year=year, melt=melt)
                op_plants_years.append(op_plants_year)
                
            op_plants_melted = pd.concat(op_plants_years,
                                axis=0,
                                ignore_index=True
                                )
        return op_plants_melted
    
    @staticmethod
    def get_current_plant_mask(year: int, plants: pd.DataFrame):
        """Return mask for selecting operating plants for a given year based on 'Start year' and 'Closed/idled year'.

        Args:
            year (int): operating plants year. Must be smaller than 2022.
        Returns:
            mask for selecting operating plants
        """
        cols = ['Start year', "Plant age (years)", "Closed/idled year", "Nominal crude steel capacity (ttpa)"]
        df = plants.loc[:, cols].copy()
        df = df.replace({'unknown': np.nan})
                
        df['Plant age (years)'] = year - df["Start year"]
        
        # to discard
        d0 = df['Start year'] > year
        d1 = df['Start year'].isna()
        d2 = df["Plant age (years)"].isna() 
        d3 = df["Plant age (years)"] < 0
        d4 = df['Closed/idled year'] <= year
        d5 = df['Nominal crude steel capacity (ttpa)'].isna()
        
        keep_mask = ~(d0 | d1 | d2 | d3 | d4 | d5)
        
        return keep_mask
                 
    def get_prod_dataset(self, year: int):
        """Method for loading production data for years 2019, 2020, 2021.
        This dataset is separate from the capacity dataset.

        Args:
            year (int): production year

        Returns:
            pd.DataFrame: clean production data
        """
        if year not in self.prod_years:
            raise ValueError(f"Input ({year}) is not valid. Year must be between {self.prod_years[0]} and {self.prod_years[-1]}")
        
        prod_df = self.raw_data['Yearly Production'].copy()
        
        # extract 2019, 2020, 2021 steel production data
        idx_cols = prod_df.columns[:2]
        last_cols = {}
        for i, y in enumerate(["2019", "2020", "2021"]):
            try:
                usecols = prod_df.columns[2 + 7*i:9 + 7*i]
                assert not usecols.empty
                last_cols[y] = usecols 
            except:
                pass
        to_keep = idx_cols.union(last_cols[str(year)], sort=False)
        prod_df = prod_df.loc[:, to_keep]
        prod_df.columns = prod_df.iloc[0]
        prod_df = prod_df.iloc[1:]
        prod_df = self.preprocess_prod_dataset(prod_df)
        return prod_df
    
    def get_steel_dataset(self, melt=False):
        """Method for fetching the plant level dataset with capacity values and technology.

        Returns:
            pd.DataFrame: clean steel plant data
        """
        steel_df = self.steel_plants.copy()
        steel_df = self.preprocess_steel_dataset(steel_df)
        if melt:
            # permid, parent name, plant id, share
            parents_ids = get_parents_ids(steel_df, self.gspt2gspt_path)
            steel_df = pd.merge(parents_ids, steel_df, on="Plant ID", how='left')
            steel_df = steel_df.rename(columns={"Parent PermID_x": "Parent PermID",
                                                        "Parent_x": "Parent"})
            steel_df["Attributed crude steel capacity (ttpa)"] = steel_df['Nominal crude steel capacity (ttpa)'] * steel_df['Share']
        return steel_df
        
    
    def preprocess_prod_dataset(self, prod_df):
        """Replace 'unknown' and '>0' capacity values with nan.

        Args:
            prod_df (_type_): _description_

        Returns:
            pd.DataFrame: raw production data
        """
        # 2023 version has years in each column name
        # we want to remove them to stay consistent with 2022 version
        if self.version_year == 2023:
            prod_df = prod_df.rename(columns={
                "Crude steel production 2019 (ttpa)": "Crude steel production (ttpa)",
             "Crude steel production 2020 (ttpa)": "Crude steel production (ttpa)",
             "Crude steel production 2021 (ttpa)": "Crude steel production (ttpa)",
             "BOF steel production 2019 (ttpa)": "BOF steel production (ttpa)",
             "BOF steel production 2020 (ttpa)": "BOF steel production (ttpa)",
             "BOF steel production 2021 (ttpa)": "BOF steel production (ttpa)",
             "EAF steel production 2019 (ttpa)": "EAF steel production (ttpa)",
             "EAF steel production 2020 (ttpa)": "EAF steel production (ttpa)",
             "EAF steel production 2021 (ttpa)": "EAF steel production (ttpa)",
             "OHF steel production 2019 (ttpa)": "OHF steel production (ttpa)",
             "OHF steel production 2020 (ttpa)": "OHF steel production (ttpa)",
             "OHF steel production 2021 (ttpa)": "OHF steel production (ttpa)",
             "Iron production 2019 (ttpa)": "Iron production (ttpa)",
             "Iron production 2020 (ttpa)": "Iron production (ttpa)",
             "Iron production 2021 (ttpa)": "Iron production (ttpa)",
             "BF production 2019 (ttpa)": "BF production (ttpa)",
             "BF production 2020 (ttpa)": "BF production (ttpa)",
             "BF production 2021 (ttpa)": "BF production (ttpa)",
             "DRI production 2019 (ttpa)": "DRI production (ttpa)",
             "DRI production 2020 (ttpa)": "DRI production (ttpa)",
             "DRI production 2021 (ttpa)": "DRI production (ttpa)"})
        
        # replace zero production values with np.nan as it won't allow us
        # to estimate utilisation rate properly
        prod_df = prod_df.replace({"Crude steel production (ttpa)": {'unknown': np.nan, '>0': np.nan}})
        return prod_df
    
    def preprocess_steel_dataset(self, steel_df):   
        """Clean steel dataset and complete start years or closed years.
        Args:
            steel_df: gspt raw steel plants data
            years_path: path to file that contains plants from GSPT with unknown start year.
            The file was completed manually, sources vary (news articles, ...).
        """      
        if self.version_year == 2023:
            date_cols = ['Proposed date',
                         'Construction date',
                         'Start date',
                         'Closed/idled date'] 
            steel_df = steel_df.replace({"unknown": np.nan})
            
            def extract_year(value):
                import datetime
                if isinstance(value, (int, float)):
                    return value
                elif isinstance(value, datetime.datetime):
                    return value.year  
            
            steel_df[[c.replace("date", "year") for c in date_cols]] = steel_df.loc[:, date_cols].applymap(extract_year, na_action="ignore")
            to_drop = date_cols + ['Proposed year', 'Construction year'] 
            steel_df = steel_df.drop(columns=to_drop)
            to_rename = {"Parent [formula]": "Parent",
                         "Parent PermID [formula]": "Parent PermID"}
            steel_df = steel_df.rename(columns=to_rename)
            
            # Create "Start year" and "Closed/idled year" columns for consistency with old version
            
            # duplicate Plant ID, but plants seem to be at different locations
            # create new Plant ID
            steel_df.loc[steel_df['Plant name (English)'] == "Southern Steel Prai Penang plant", 
                         "Plant ID"] = "odd_plant_out"
            
        steel_df = steel_df.replace(">0", np.nan)
        steel_df = steel_df.replace({'Start year': {'unknown': np.nan},
                                    "Nominal BOF steel capacity (ttpa)": {'unknown':np.nan, ">0": np.nan},
                                    "Nominal EAF steel capacity (ttpa)": {'unknown':np.nan,
                                                                                            ">0": np.nan},
                                    "Nominal OHF steel capacity (ttpa)": {'unknown':np.nan,
                                                                                            ">0": np.nan}})
        ## Start year completion
        
        # Jindal Steel And Power
        # source: https://www.gem.wiki/JSPL_Chhattisgarh_steel_plant
        steel_df.loc[steel_df['Plant name (English)'] == "JSPL Chhattisgarh steel plant","Start year"] = 1989
        # source: https://m.economictimes.com/industry/indl-goods/svs/steel/jspl-angul-steel-plant-to-start-production-by-march/articleshow/18167491.cms
        # opened in 2013 with 1/3 of capacity the last 2/3 within 3 years
        steel_df.loc[steel_df['Plant name (English)'] == "JSPL Odisha steel plant","Start year"] = 2013
        
        # 'Minyuan Iron and Steel Group Co., Ltd. equipment upgrade'
        # Source: https://www.gem.wiki/Minyuan_Iron_and_Steel_Group_Co.,_Ltd.
        steel_df.loc[steel_df["Plant ID"] == "SCN00307-1", "Start year"] = 2025
        
        # Some plants have an operating status without a start year,
        # add that start year in order to include them in analysis
        missing_years = pd.read_excel(self.missing_years_path, engine="calamine")
        
        # The 2022 and 2023 share the same plant IDS so this should work
        steel_df = pd.merge(steel_df, 
                            missing_years[["Plant ID", "Start year"]], 
                            on="Plant ID", 
                            how='left')
        # fill start years
        steel_df.loc[steel_df['Start year_x'].isnull(), "Start year_x"] = steel_df["Start year_y"]
        steel_df.drop(columns=['Start year_y'], inplace=True)
        steel_df.rename(columns={"Start year_x": "Start year"}, inplace=True)
        
        ## Closed/idled year completion

        
        ## Technology / Main production process completion
        
        # Operating plants between 2019 and 2021: impute with 'integrated (BF)'
        m1 = steel_df['Main production process'].isna() 
        m2 = steel_df['Start year'].notna()
        m3 = steel_df['Start year'] <= 2021
        steel_df.loc[m1 & m2 & m3, "Main production process"] = steel_df.loc[m1 & m2 & m3, "Main production process"].fillna("integrated (BF)")
        
        # Other plants
        
        # Ch'ollima Steel Complex steel plant
        # Source: https://en.wikipedia.org/wiki/Chollima_Steel_Complex
        steel_df.loc[steel_df["Plant ID"] == "SKP00005", "Main production process"] = "electric"
        # Aceros Arequipa Pisco steel plant
        steel_df.loc[steel_df["Plant ID"] == "SPE00001", "Main production process"] = "electric"
        
        steel_df = steel_df.loc[steel_df['Main production process'].notna()]
        # drop ironmaking plants
        steel_df = steel_df.loc[steel_df['Main production process'] != "ironmaking (other)"]
        return steel_df.reset_index(drop=True)


    
