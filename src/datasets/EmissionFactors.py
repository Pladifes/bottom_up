import pandas as pd
from pathlib import Path

try:
    from .GSPTDataset import GSPTDataset
except:
    from GSPTDataset import GSPTDataset
class EmissionFactors:
    def __init__(self, 
                 wsa_path: Path, 
                 jrc_22_path: Path,
                 sci_path: Path,
                 EU_27_path: Path
                 ) -> None:
        """Initalise constructor.

        Args:
            wsa_path (_type_): emission factors from the World Steel Association (global averages)
            jrc_22_path (_type_): emission factors from the European Union (national)
            sci_path (_type_): emission factors from the Global Intelligence Summit (national)
        """
        self.wsa_path = wsa_path
        self.jrc_22_path = jrc_22_path
        self.sci_path = sci_path
        self.EU_27_path = EU_27_path

        
    def match_emissions_factors(self, plants: pd.DataFrame, level: str,  source: str, techno_col: str = None):
        """Given a plants dataset with Country and Technology columns, match available emissions factors.

        Args:
            plants (pd.DataFrame): plant-level dataset (e.g. GSPT)
            techno_col (str): "Main production process" or "Techno_map" in general
            level (str): emission factors granularity (global, region, national, state ...)
            source (str): source for national emission factors ("jrc" or "sci")
            drop (bool): drop unnecessary columns
            
        Returns:
            plants (pd.DataFrame): with additional EF column with country level and global emissions factors
        """
        wsa_ef = self.read_wsa()
        jrc_ef = self.read_jrc_22()
        sci_ef = self.read_sci()
        
        efs = {"wsa": wsa_ef, "jrc": jrc_ef, "sci": sci_ef}
        if level == "global":
            plants = pd.merge(plants,
                                wsa_ef, 
                                how='left', 
                                on=techno_col)
            plants.rename(columns={"WSA_EF": "EF"}, inplace=True)
        elif level == "country":
            plants = self.map_national_ef(plants=plants, efs=efs, source=source, techno_col=techno_col)
            # DRI intensities differ significantly from BF-BOF intensities
            # so we assign global DRI emission factor for this particular technology
            # Source: World Steel Association
            plants.loc[plants["Main production process"] == "integrated (DRI)", "EF"] = 1.65
        else:
            raise Exception
        
        assert plants['EF'].notna().all()
        return plants
    
    def map_national_ef(self, plants: pd.DataFrame, efs: pd.DataFrame, source: str, techno_col: str = None) -> pd.DataFrame:
        """Map national emission factors/CO2 intensities to each plant based on technology.
        
        Country mappings:
        - for BF-BOF: maps EU countries to single "EU" emission factor (merge on "sci_country_map" column) and fills NA with mean
        - for EAF: maps to available countries (merge on "Country" column) and fills NA with mean

        Args:
            plants (pd.DataFrame): GSPT steel plants
            ef (pd.DataFrame): national emission factors
            source (str): "jrc" or "sci"
        """
        wsa_ef = efs["wsa"]
        ef = efs[source]
        
        if source == "sci":
            # create a country mapping between available countries for emission factors
            # and plant locations
            plants["sci_country_map_BF_BOF"] = plants['Country'].copy()
            
            # EU countries
            EU_27 = pd.read_excel(self.EU_27_path)
            EU_27_countries = set(EU_27['Country'])
            # These countries tend to have smaller economies and are not major players in global steel production.
            plants['sci_country_map_BF_BOF'] = plants['sci_country_map_BF_BOF'].replace(to_replace=EU_27_countries,
                                                                          value="EU")
            # map gspt technology to SCI technology
            plants['sci_techno_map'] = plants['Main production process'].apply(lambda x: "BF-BOF" if x == "integrated (BF)" else("EAF" if x == "electric" else "other"))
                                                                                   
            # map bf bof emission factors to each plant
            n0 = len(plants)
            plants = pd.merge(plants,
                                ef,
                                left_on=["sci_country_map_BF_BOF", "sci_techno_map"],
                                right_on=["Country", "Technology"],
                                how="left")
            plants = plants.rename(columns={"Country_x": "Country"})
            n1 = len(plants)
            assert n0 == n1
            
            # map eaf emission factors to each plant
            n2 = len(plants)
            plants = pd.merge(plants,
                    ef,
                    left_on=["Country", "sci_techno_map"],
                    right_on=["Country", "Technology"],
                    how="left",
                    suffixes=("_BF_BOF", "_EAF"))
            plants = plants.rename(columns={"Country_x": "Country"})
            n3 = len(plants)
            assert n2 == n3
            
            plants['EF'] = plants["EF_BF_BOF"].fillna(plants["EF_EAF"])

            # Missing values imputation
            # For BF-BOF impute mean
            mean_bof_ef = ef.loc[ef['Technology'] == "BF-BOF", "EF"].mean()
            mask_bf_bof = plants["sci_techno_map"] == "BF-BOF"
            plants.loc[mask_bf_bof, "EF"] = plants.loc[mask_bf_bof, "EF"].fillna(mean_bof_ef)

            # For EAF impute mean
            mean_eaf_ef = ef.loc[ef['Technology'] == "EAF", "EF"].mean()
            mask_eaf = plants["sci_techno_map"] == "EAF"
            plants.loc[mask_eaf, "EF"] = plants.loc[mask_eaf, "EF"].fillna(mean_eaf_ef)

            # For 'other' technology impute mean BF-BOF
            mask_other = plants["sci_techno_map"] == "other"
            plants.loc[mask_other, "EF"] = plants.loc[mask_other, "EF"].fillna(mean_bof_ef)

            # Clean
            to_drop = ["Country_y", 
                       "EF (kg CO2e / t steel)_BF_BOF",
                       "EF_BF_BOF",
                       "Technology_BF_BOF",
                        "EF (kg CO2e / t steel)_EAF",
                       "EF_EAF",
                       "Technology_EAF"]
            plants = plants.drop(columns=to_drop)
        elif source == "jrc":
            # mapping from JRC countries to plants datasets countries
            # Question: how many countries does the JRC report cover in
            # the GSPT database ?
            jrc_countries = set(ef.Country)
            plants_countries = set(plants['Country'])


            
            # Attribute the 'EU' emission factor to the 27 members of the EU
            EU_27 = ["Austria", 
                    "Belgium", 
                    "Bulgaria", 
                    "Croatia", 
                    "Republic of Cyprus", 
                    "Czech Republic", 
                    "Denmark", 
                    "Estonia", 
                    "Finland", 
                    "France", 
                    "Germany", 
                    "Greece", 
                    "Hungary", 
                    "Ireland", 
                    "Italy", 
                    "Latvia", 
                    "Lithuania", 
                    "Luxembourg", 
                    "Malta", 
                    "Netherlands", 
                    "Poland", 
                    "Portugal", 
                    "Romania", 
                    "Slovakia", 
                    "Slovenia", 
                    "Spain", 
                    "Sweden"]
            # remove countries that have their own emission factor
            EU_27_minus_JRC = set(EU_27).difference(jrc_countries)
            
            # test if all countries are present in plants: 'Croatia', 'Malta', 'Ireland', 'Estonia', 'Denmark', 'Lithuania', 'Republic of Cyprus'
            # maybe there are no large enough steel plants in theses countries
            diff_EU = set(EU_27).difference(plants['Country'])
            # country map
            plants["jrc_country_map"] = plants['Country'].copy()

            # countries that are not in the JRC report will be assigned global emissions factors
            plants.loc[~plants['jrc_country_map'].isin(jrc_countries.union(EU_27_minus_JRC)), 'jrc_country_map'] = "Global"
            plants.loc[plants['jrc_country_map'].isin(EU_27_minus_JRC), 'jrc_country_map'] = "EU"
            # techno map: technos from GSPT are converted into technos from JRC
            # there are only two technos in JRC (integrated BF-BOF and EAF/electric)
            plants['jrc_techno_map'] = plants[techno_col].copy()
            techno_mapping = {'integrated (DRI)': 'Integrated BF-BOF', 
                            'electric': "EAF", 
                            'integrated (BF)': 'Integrated BF-BOF', 
                            'integrated (unknown)': 'Integrated BF-BOF',
                            'integrated (BF and DRI)': 'Integrated BF-BOF', 
                            'oxygen': 'Integrated BF-BOF', 
                            'electric, oxygen': "EAF", 
                            'unknown': 'Integrated BF-BOF', 
                            "other": "Integrated BF-BOF" 
                            }
            plants['jrc_techno_map'] = plants[techno_col].replace(techno_mapping)
            # 1201 plants: only 397 do not have a country level intensity value
            plants = pd.merge(plants, 
                    ef, 
                    left_on=['jrc_country_map', "jrc_techno_map"], 
                    right_on=['Country', "Technology"],
                    how="left")
            
            # fill the values that are left with global emissions factors
            wsa_ef = wsa_ef.rename(columns={"Main production process": techno_col})
            plants = pd.merge(plants,
                            wsa_ef, 
                            how='left', 
                            on=techno_col)
            
            plants["EF"] = plants["JRC_EF"].fillna(plants["WSA_EF"])
            
            plants.rename(columns={"Country_x": "Country"}, inplace=True)
            plants.drop(columns=["jrc_country_map",
                                "jrc_techno_map",
                                "Country_y",
                                "Scope 1",
                                "Scope 2",
                                "JRC_EF",
                                "Technology",
                                "WSA_EF"],
                        inplace=True)
        else:
            raise Exception
        return plants
            
        
    
    def read_sci(self):
        """Read emission factors from Hasanbeigi & Springer."""
        sci = pd.read_excel(self.sci_path)
        sci["Country"] = sci["Country"].replace({"Russian Federation": "Russia"})
        return sci
    
    def read_wsa(self):
        """Read emissions factors from World Steel Association. Scope 1 + 2 (some scope 3).
        These are global averages for different technologies.

        Returns:
            _type_: _description_
        """
        df = pd.read_excel(self.wsa_path)
        to_keep = ["Technology", "EF"]
        EF = df.loc[:, to_keep]
        EF.rename(columns={"EF": "WSA_EF",
                           "Technology": "Main production process"}, inplace=True) #if techno_col is None else techno_col}, inplace=True)
        return EF
        
    
    def read_jrc_22(self):
        """Read emissions factors by country and technology (BF-BOF and EAF) from JRC report 2022.
        https://publications.jrc.ec.europa.eu/repository/handle/JRC129297
        Returns:
            pd.DataFrame: emissions factors (country, scope 1, scope 2, technology)
        """
        dfs = pd.read_excel(self.jrc_22_path, sheet_name=None)
        ef_techno = dfs['table A4']
        country_col = ef_techno.columns[0]
        bf_cols = ef_techno.columns[1:5]
        eaf_cols = ef_techno.columns[5:]
        countries = ef_techno[[country_col]]
        countries.columns = countries.iloc[0]
        countries = countries.iloc[1:].reset_index(drop=True)
        # bf
        bf = ef_techno.loc[:, bf_cols]
        bf.columns = bf.iloc[0]
        bf = bf.iloc[1:].reset_index(drop=True)
        bf = pd.concat([countries, bf], axis=1)
        bf['Country'] = bf['Country'].replace({'United\nKingdom': 'United Kingdom'})
        bf['Technology'] = "Integrated BF-BOF"
        # eaf
        eaf = ef_techno.loc[:, eaf_cols]
        eaf.columns = eaf.iloc[0]
        eaf = eaf.iloc[1:].reset_index(drop=True)
        eaf = pd.concat([countries, eaf], axis=1)
        eaf['Country'] = eaf['Country'].replace({'United\nKingdom': 'United Kingdom'})
        eaf['Technology'] = "EAF"
        
        # all efs
        EF = pd.concat([bf, eaf], axis=0, ignore_index=True)
        EF.loc[:, ["Scope 1", "Scope 2"]] = EF.loc[:, ["Scope 1", "Scope 2"]].astype(float)
        EF['JRC_EF'] = EF['Scope 1'] + EF['Scope 2']
        to_keep = ["Country", "Scope 1", "Scope 2", "JRC_EF", "Technology"]
        final_EF = EF.loc[:, to_keep]
        
        final_EF.dropna(subset=['JRC_EF'], axis=0, inplace=True)
        #final_EF.rename(columns={"Technology": "Technology" if techno_col is None else techno_col}, inplace=True)
        country_mapping = {"Korea": "South Korea"}
        final_EF['Country'] = final_EF['Country'].replace(country_mapping)
        return final_EF.reset_index(drop=True)
        
