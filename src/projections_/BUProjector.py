from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import itertools
import sqlite3

from pathlib import Path
project_dir = Path.cwd()

while project_dir.stem != "bottom_up_alignment":
    project_dir = project_dir.parent

import sys
sys.path.append(str(project_dir / "src"))
from datasets.GSPTDataset import GSPTDataset
from datasets.RefinitivDataset import RefinitivDataset
from datasets.EmissionFactors import EmissionFactors
from datasets.utils import get_parents_ids, convert_plant2parent



from feature_eng.company_feature_engineering import get_X_y_groups, get_features

class BUProjector:
    """Class for projecting top-down emissions from bottom-up emissions.
    """
    
    def __init__(self, refinitiv, gspt, gspt_panel, X_y, logged_model_path, parent_group_map_path):
        self.refinitiv = refinitiv
        self.gspt = gspt
        self.gspt_panel = gspt_panel
        self.X_y = X_y
        self.X_y = self.X_y.loc[self.X_y.index.get_level_values("GICSName") == "Steel"]
        self.X_y = self.X_y.loc[~self.X_y.index.get_level_values("GSPT Name").isin(['Vale SA'])]
        self.logged_model_path = logged_model_path
        self.parent_group_map_path = parent_group_map_path

    def get_projections_iea(op_plants_ts: pd.DataFrame, 
                            glob_prod_iea: pd.DataFrame, 
                            market_share: pd.DataFrame, 
                            gspt2refi_map: pd.DataFrame,
                            ):
        """Project bottom-up emissions using different hypotheses on the evolution of company utilsation rates.

        Args:
            op_plants_ts (pd.DataFrame): operating plants for a given year (use GSPT method "get_operating_plants" 
            and indicate relevant years for projection)
            glob_prod_iea (pd.DataFrame): global steel production projections from IEA
            market_share (pd.DataFrame): market share based on Refinitiv (company production) and WSA (global produciton) data
            method (str): different flavours for projecting bottom-up production
        """
        
    
    
    def get_proj_plot(self, wsa_ef_path, save_dir):
        """Compare projections from top-down method and bottom-up method and their respective
        prediction errors.
        """
        save_dir = Path(save_dir)
        
        companies = ["ArcelorMittal SA",
                     "JFE Holdings Inc",
                     "Cleveland-Cliffs",
                     "Nucor Corp",
                     "United States Steel Corp",
                     "EVRAZ plc",
                     "JSW Steel Ltd",
                     "Nippon Steel Corp",
                     "Posco Holdings Inc",
                     "Voestalpine AG",
                     "Hyundai Steel Co",
                     "Tata Steel Ltd",
                     "Kobe Steel Ltd",
                     "Jindal Steel And Power Ltd",
                     "Steel Authority of India Ltd"]
        emissions = self.get_merged_proj(wsa_ef_path=wsa_ef_path)
        to_plot = emissions.loc[(emissions.year == 2030) & emissions.Group.isin(companies)]
        to_plot = to_plot.dropna(axis=0)

        # Sample data (replace with your own data)
        companies = to_plot['Group'].unique()
        m1_name = 'Projected log CF12' 
        m2_name = 'Projected log BU emissions final' 
        method1 = to_plot.loc[:,m1_name].values
        method2 = to_plot.loc[:,m2_name].values
        error1 = to_plot["error_TD"] #"error_top_down"
        error2 = to_plot['error_BU']
        
        # Set the positions of the bars on the x-axis
        x = np.arange(len(companies))

        # Set the width of the bars
        width = 0.35

        # Plotting the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, method1, width, label='top-down')
        rects2 = ax.bar(x + width/2, method2, width, label='bottom-up')

        # Adding error bars for confidence
        ax.errorbar(x - width/2, method1, yerr=error1, fmt='none', capsize=4, color='black')
        ax.errorbar(x + width/2, method2, yerr=error2, fmt='none', capsize=4, color='black')

        # Adding labels, title, and ticks
        ax.set_xlabel('Companies')
        title = 'log emissions'
        ax.set_ylabel(title)
        ax.set_title('Projected top-down and bottom-up in 2030')
        ax.set_xticks(x)
        ax.set_xticklabels(companies, rotation=45, ha='right')
        ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()
        fig.savefig(save_dir / "projected_30_barplot_log.png", dpi=600, bbox_inches="tight")

    def get_merged_proj(self, wsa_ef_path):
        """Return base projection (from historical top down emissions) and from bottom up projection (using asset-level data)
        with prediction intervals.

        Args:
            logged_model_path: path to mlflow model
            parent_group_map_path: path to parent group mapping dict (parent_group_map.json)
        Returns:
            _type_: _description_
        """
        # model
        loaded_model = mlflow.sklearn.load_model(self.logged_model_path)
        
        final_proj = self.gspt.get_projected_data(parent_group_map_path=self.parent_group_map_path,
                                                  wsa_ef_path=wsa_ef_path)
        # add electricity intensity 2019
        elec = self.X_y.copy().reset_index()
        elec = elec.loc[elec.year == 2019, ["GSPT Name", "elec_int_prod"]]
        elec.rename(columns={"GSPT Name": "Group"}, inplace=True)
        
        final_proj = pd.merge(final_proj, elec, on="Group", how='left')
        final_proj = final_proj.dropna(subset=["elec_int_prod"], axis=0).reset_index(drop=True)
        
        proj_base = self.get_top_down_projections()
        emissions = self.get_bottom_up_projections(proj_features=final_proj, model=loaded_model)
        merge_proj = pd.merge(proj_base, emissions, 
                              how="left", 
                              on=["Group", "year"], 
                              suffixes=('_TD', '_BU'))
        return merge_proj
    
    def get_top_down_projections(self):
        """Top-down projections from historical top-down emissions.

        Returns:
            _type_: _description_
        """
        refi_steel = self.refinitiv.loc[self.refinitiv['GICSName'] == "Steel", ["Name", "GICSName", "FiscalYear", "CF12"]]
        refi_steel = refi_steel.dropna(subset=["CF12"], axis=0)
        refi_steel["log_CF12"] = np.log(refi_steel['CF12'])
        companies = list(refi_steel['Name'].unique())
        
        # boundary
        # company counts: keep companies with at least five data points
        company_counts = refi_steel.groupby("Name").size()
        valid_companies = company_counts[company_counts >= 5].index
        refi_steel = refi_steel.loc[refi_steel.Name.isin(valid_companies)]
        res = {}
        refi_steel_grouped = refi_steel.groupby("Name")
        for company, company_df in refi_steel_grouped:
            company_dict = {}
            X = company_df[['FiscalYear']]
            y = company_df['log_CF12']
            model = LinearRegression()
            model.fit(X, y)
            company_dict["model"] = model
            company_dict['X_train'] = X
            company_dict['y_train'] = y
            
            res[company] = company_dict            
        
        
        # prediction interval
        proj_list = []
        for company, company_dict in res.items():
            model = company_dict["model"]
            X_train = company_dict["X_train"]
            y_train = company_dict['y_train']
            yt = y_train.values
            t = X_train.values
            y_hat = model.predict(t)
            t0 = np.arange(2022, 2031).reshape(-1,1)
            yt0_hat = model.predict(t0)  
            # Prediction interval
            # standard error
            n = len(yt)
            se = np.sqrt((1/(n - 1)) * np.sum((yt - y_hat) ** 2))     
            se_yt0 = se * (np.sqrt(1 + 1/n + ((t0 - t.mean()) ** 2) / np.sum((t - t.mean()) ** 2)))
            # t value
            t_alpha = 1.96
            error = t_alpha * se_yt0
            proj_df = pd.DataFrame(index=range(len(t0)))
            proj_df["year"] = t0
            proj_df["Projected log CF12"] = yt0_hat
            proj_df["error"] = error
            proj_df["Group"] = company
            proj_list.append(proj_df)
        proj_res = pd.concat(proj_list, axis=0, ignore_index=True)
        return proj_res
        

    def get_bottom_up_projections(self, proj_features: pd.DataFrame, model: Pipeline):
        """Bottom-up projections based on projected capacity that depends on plant openings and closures.

        Args:
            bu_emissions (pd.DataFrame): company level bottom-up emissions. Should be in raw units. 
            model (sklearn.pipeline.Pipeline): fitted model log(TD_emissions) = f(log(BU_emissions))
        """
        proj_features = proj_features.loc[proj_features['Projected BU emissions (ttpa)'] > 0]
        proj_features.set_index(["Group", "year"], inplace=True)
        # two dimensional model
        proj_features["Projected BU emissions"] = proj_features["Projected BU emissions (ttpa)"] * 1000
        proj_features["Projected log BU emissions"] = np.log(proj_features["Projected BU emissions"])
        proj_features["Projected log BU emissions final"] = model.predict(proj_features[["Projected log BU emissions",
                                                                                         "elec_int_prod"]].values)
        
        # prediction intervaal
        # in sample
        n = len(self.X_y)
        X = self.X_y[["log_Attributed emissions", "elec_int_prod"]]
        yt = self.X_y[['log_CF12']].values
        y_hat = model.predict(X)
        se = np.sqrt((1/(len(yt) - 1)) * np.sum((yt - y_hat) ** 2))

        X_arr = X.values[:, :, None]
        X_arr = X_arr - X_arr.mean(axis=0,keepdims=True)
        S = (X_arr @ X_arr.transpose(0,2,1)).sum(axis=0)
        X0 = proj_features[['Projected log BU emissions', "elec_int_prod"]].values[:,:,None]
        proj_features["error"] = 1.96 * se * np.sqrt(((X0.transpose(0,2,1) @ np.linalg.inv(S)) @ X0).squeeze() + 1/n + 1)         
 
        return proj_features

def adjust_bottom_up(proj_company: pd.DataFrame, model):
    """Apply statistical model to raw bottom-up emissions.

    Args:
        proj_company (pd.DataFrame): _description_
        model (_type_): _description_
    """
    # Adjust bottom-up emissions with statistical model
    proj_company['log_Attributed emissions (ttpa)'] = np.log(proj_company['Attributed emissions (ttpa)'] * 1e3)
    proj_company['BU_emissions'] = model.predict(proj_company['log_Attributed emissions (ttpa)'])
    
    # calculate prediction interval
    
    # plot
    return proj_company

def get_projections_iea(gspt: pd.DataFrame, 
                        glob_prod_iea: pd.DataFrame, 
                        glob_capa, 
                        market_share: pd.DataFrame, 
                        parent_group_map_path: Path,
                        gspt2refi_map: dict,
                        EF: "EmissionFactor",
                        method: str,
                        start_year: int,
                        end_year: int):
    """Project bottom-up emissions using different hypotheses on the evolution of company utilsation rates.

    Args:
        op_plants_ts (pd.DataFrame): operating plants for a given year (use GSPT method "get_operating_plants" 
        and indicate relevant years for projection)
        glob_prod_iea (pd.DataFrame): global steel production projections from IEA
        market_share (pd.DataFrame): market share based on Refinitiv (company production) and WSA (global production) data
        method (str): different flavours for projecting bottom-up production
        
    Returns:
        proj_company (pd.DataFrame): projected production and emissions at company level
        proj_plants_prod (pd.DataFrame): projected production and emissions at plant level
    """
    assert start_year >= 2023
    assert start_year < end_year
    
    with open(parent_group_map_path, "r") as f:
        parent_group_map = json.load(f)
        
    interpolated_glob_prod = interpolate_glob_prod(glob_prod_iea=glob_prod_iea,
                                                   proj_start_year=start_year,
                                                   end_year=end_year)
    
    # create market share time series for relevant projection years
    market_share_ts = get_market_share_ts(market_share=market_share, start_year=start_year, end_year=end_year)
    

    # merge global projected production and market share to compute company level projected production
    proj_feats = pd.merge(market_share_ts, interpolated_glob_prod, on="Year", how='left')
    proj_feats["Company production (Mt)"] = proj_feats['Global production (Mt)'] * proj_feats['market_share']
        # TODO: temp
    proj_feats['year'] = proj_feats['Year'].copy()
    
    # Aggregate plant capacity based on group company mapping
    op_plants_ts = gspt.get_operating_plants(start_year=start_year, end_year=end_year, melt=True)
    op_plants_ts['Group'] = op_plants_ts['Parent'].copy()
    op_plants_ts = op_plants_ts.replace({'Group': parent_group_map})
    op_plants_ts['Attributed crude steel capacity (ttpa)'] = op_plants_ts['Nominal crude steel capacity (ttpa)'] * op_plants_ts['Share']
    
    # Baseline projected group company capacity
    proj_group_capa = op_plants_ts.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
    
    if method == "method 1":
        proj_plants_UR, _ = get_projected_UR_1(proj_feats, proj_group_capa, op_plants_ts)
        
    elif method == "method 2":

        # Get all steel plants and map parent companies to respective group company
        melted_steel_plants = gspt.get_steel_dataset(melt=True)
        melted_steel_plants['Group'] = melted_steel_plants['Parent'].replace(parent_group_map)
        # Get additions in electric capacity
        start_year, end_year = 2022, 2030
        delta_elec_capa = get_electric_capa_delta(melted_steel_plants=melted_steel_plants,
                                                 start_year=start_year,
                                                 end_year=end_year)
        # Incorporate assumptions on evolution of technology mix
        proj_group_capa = update_company_capacity(company_capa=proj_group_capa, 
                                                  delta_elec_capa=delta_elec_capa, 
                                                  method="substitute_elec")

        # Add projected capacity to projected features
        # match on group column: capacity for a corporate group is aggregated over available subsidiaries
        
        # Aggregate projected capacity at company level
        # This value includes substitution of coal for electric
        # Keep separate technology capacities
        proj_group_capa = proj_group_capa.groupby(["Group", "year"])["Attributed crude steel capacity (ttpa)"].sum().reset_index()
        
        proj_group = pd.merge( 
                    left=proj_group_capa, 
                    right=proj_feats[["Group", "year", "Company production (Mt)"]], 
                    on=["Group", "year"],
                    how='left')
        
        
        # compute projected utilisation rate
        proj_group['UR'] = (proj_group["Company production (Mt)"] * 1e6) / (proj_group["Attributed crude steel capacity (ttpa)"] * 1e3)
        
        # Impute missing UR with average for given year
        mean_UR = proj_group.groupby("year")["UR"].mean().to_frame("mean_UR").reset_index()
        
        proj_group = pd.merge(proj_group,
                                mean_UR,
                                on=["year"],
                                how='left')
        proj_group["UR"] = proj_group["UR"].fillna(proj_group['mean_UR'])
                
        
        
        # Impute UR estimated at the company level to each individual plant
        
        proj_plants = pd.merge(op_plants_ts, 
                               proj_group.loc[:, ["Group", "year", "UR"]], 
                               on=["Group", "year"], 
                               how="left"
                               )
        
        proj_plants['UR'] = proj_plants['UR'].fillna(1.0)
                
        # a plant's utilisation rate is defined as the weighted average of company utilisation rates
        # weighted by ownership share in the plant
        proj_plants['UR_share'] = proj_plants['UR'] * proj_plants['Share']
        # projected utilisation rate for each plant
        proj_plants_UR = proj_plants.groupby(["Plant ID", "year"])["UR_share"].sum().to_frame('UR').reset_index()
                
    elif method == "method 3": 
        proj_plants_UR, _, _ = get_projected_UR_3(proj_feats, gspt, op_plants_ts, parent_group_map, proj_group_capa)     
        
    elif method == "method 4":
        pass
    else:
        raise Exception
    
    # project plant production: merge the plant level UR with projected capacity
    proj_plants_prod = pd.merge(op_plants_ts, 
                                  proj_plants_UR, 
                                  on=['Plant ID', 'year'],
                                  how='left')
    assert proj_plants_prod['UR'].notna().all()
    
    proj_plants_prod['plant_prod (ttpa)'] = proj_plants_prod['UR'] * proj_plants_prod["Attributed crude steel capacity (ttpa)"]
    # Add emission factors
    if "Techno_map" in proj_plants_prod.columns:
        techno_col = "Techno_map"
    else:
        techno_col = "Main production process"
    proj_plants_prod = EF.match_emissions_factors(plants=proj_plants_prod, 
                                                  techno_col=techno_col,
                                                  level="country",
                                                  source='sci')
    # Share of plant-level emissions attributed to owner with respect to their stake in the plant
    proj_plants_prod['Attributed production (ttpa)'] = proj_plants_prod["plant_prod (ttpa)"] * proj_plants_prod['Share']
    proj_plants_prod['Attributed emissions (ttpa)'] = proj_plants_prod["Attributed production (ttpa)"] * proj_plants_prod['EF'] 
    # Company bottom-up emissions
    proj_company = proj_plants_prod.groupby(["Group", "year"]).agg({'Attributed emissions (ttpa)': "sum",
                                                                    "Attributed production (ttpa)": 'sum'}).reset_index()
    return proj_company, proj_plants_prod



def get_proj_market_feats(glob_prod_iea, market_share, start_year, end_year):
    interpolated_glob_prod = interpolate_glob_prod(glob_prod_iea=glob_prod_iea,
                                                   proj_start_year=start_year,
                                                   end_year=end_year)
    
    # create market share time series for relevant projection years
    market_share_ts = get_market_share_ts(market_share=market_share, start_year=start_year, end_year=end_year)
    

    # merge global projected production and market share to compute company level projected production
    proj_feats = pd.merge(market_share_ts, interpolated_glob_prod, on="Year", how='left')
    proj_feats["Company production (Mt)"] = proj_feats['Global production (Mt)'] * proj_feats['market_share']
    proj_feats['year'] = proj_feats['Year'].copy()
    return proj_feats


def get_projected_UR_1(proj_feats, proj_group_capa, op_plants_ts):
    """_summary_

    Args:
        proj_feats (_type_): company projected production, market share (and global production)
        proj_group_capa (_type_): company projected capacity
        op_plants_ts (_type_): _description_

    Returns:
        _type_: _description_
    """
    # default evolution of capacity: plants open and close based on start and close years
    # this is already taken into account in the above operating plants
    proj_group_capa = proj_group_capa.groupby(["Group", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
    
    # match on group column: capacity for a corporate group is aggregated over available subsidiaries
    proj_feats = pd.merge(proj_feats, 
                            proj_group_capa.loc[:, ["Group", "year", "Attributed crude steel capacity (ttpa)"]], 
                            on=["Group", "year"],
                            how="left")
    
    # compute projected utilisation rate
    proj_feats['UR'] = (proj_feats["Company production (Mt)"] * 1e6) / (proj_feats["Attributed crude steel capacity (ttpa)"] * 1e3)
    proj_feats['UR'] = proj_feats['UR'].clip(upper=1.0)
    
    
    proj_plants = pd.merge(op_plants_ts, 
                                proj_feats.loc[:, ["Group", "year", "UR"]], 
                                on=["Group", "year"],
                                how='left')
    

    # Impute new companies UR with yearly median UR
    proj_plants['UR'] = proj_plants['UR'].replace(0, np.nan)
    yearly_median_UR = proj_plants.groupby('year')['UR'].transform("median")
    proj_plants['UR'] = proj_plants['UR'].fillna(yearly_median_UR)
    
    # a plant's utilisation rate is defined as the weighted average of company utilisation rates
    # weighted by ownership share in the plant
    proj_plants['UR_share'] = proj_plants['UR'] * proj_plants['Share']
    # projected utilisation rate for each plant
    proj_plants_UR = proj_plants.groupby(["Plant ID", "year"])["UR_share"].sum().to_frame('UR').reset_index()
    return proj_plants_UR, proj_plants

def get_projected_UR_3(proj_feats, gspt, op_plants_ts, parent_group_map, proj_group_capa):
    """_summary_

    Args:
        proj_feats (_type_): _description_
        gspt (_type_): _description_
        op_plants_ts (_type_): _description_
        parent_group_map (_type_): _description_
        proj_group_capa (_type_): _description_

    Returns:
        proj_plants_UR (pd.DataFrame): (Plant ID, year, UR) -> for each year, a plant receive a UR from the company level
        proj_plants (pd.DataFrame): (Plant ID, Group, year, UR, Share, ...) -> company level dataset from which plant UR is derived based on ownership share in plant [testing purposes]
        proj_group (pd.DataFrame): (Group, Techno_map, year, UR) ->  UR is calculated at the company level for each technology and year [testing purposes]
    """
    # Get all steel plants and map parent companies to respective group company
    melted_steel_plants = gspt.get_steel_dataset(melt=True)
    melted_steel_plants['Group'] = melted_steel_plants['Parent'].replace(parent_group_map)
    
    # Evolution of total group capacity based on openings and closures
    group_total_capa = op_plants_ts.groupby(["Group", "year"])["Attributed crude steel capacity (ttpa)"].sum().reset_index()
    group_total_capa = group_total_capa.rename(columns={"Attributed crude steel capacity (ttpa)": "Total crude steel capacity (ttpa)"})
    # Share of technology per the company's total production
    techno_share = pd.merge(proj_group_capa,
                            group_total_capa,
                            on=["Group", "year"],
                            how='left')
    techno_share["Techno_map"] = techno_share['Main production process'].apply(lambda x: x if x in ['electric', "integrated (BF)", "integrated (DRI)"] else "other")
    techno_share = techno_share.groupby(["Group", "Techno_map", "year"]).agg({"Attributed crude steel capacity (ttpa)": "sum",
                                                               "Total crude steel capacity (ttpa)": 'sum'})
    
    techno_share["techno_share"] = techno_share["Attributed crude steel capacity (ttpa)"] / techno_share['Total crude steel capacity (ttpa)']
    techno_share = techno_share.reset_index()
    assert techno_share['techno_share'].between(0,1).all()
    
    # Allocate share of company production to each technology
    # based on techno_share

    proj_prod = pd.merge( 
        left=techno_share, 
        right=proj_feats[["Group", "year", "Company production (Mt)"]], # add market share
        on=["Group", "year"],
        how='left')
    proj_prod["Techno production (Mt)"] = proj_prod["Company production (Mt)"] * proj_prod["techno_share"]
    proj_prod = proj_prod.dropna(subset="Techno production (Mt)")
    proj_prod = proj_prod[["Group", "Techno_map", "year", "Techno production (Mt)"]]
    proj_prod = proj_prod.rename(columns={"Techno production (Mt)": "Company production (Mt)"}
                                    )
    # Get additions in electric capacity
    start_year, end_year = 2022, 2030
    delta_elec_capa = get_electric_capa_delta(melted_steel_plants=melted_steel_plants,
                                                start_year=start_year,
                                                end_year=end_year)
    # Incorporate assumptions on evolution of technology mix
    proj_group_capa = update_company_capacity(company_capa=proj_group_capa, 
                                                delta_elec_capa=delta_elec_capa, 
                                                method="substitute_elec")

    
    
    # Add projected capacity to projected features
    # match on group column: capacity for a corporate group is aggregated over available subsidiaries
    
    # Aggregate projected capacity at company level
    # This value includes substitution of coal for electric
    proj_group_capa["Techno_map"] = proj_group_capa['Main production process'].apply(lambda x: x if x in ['electric', "integrated (BF)", "integrated (DRI)"] else "other")
    proj_group_capa = proj_group_capa.groupby(["Group", "Techno_map", "year"])["Attributed crude steel capacity (ttpa)"].sum().reset_index()

    proj_group = pd.merge( 
                left=proj_group_capa, 
                right=proj_prod, 
                on=["Group", "Techno_map", "year"],
                how='left')

    
    # compute projected utilisation rate
    proj_group['UR'] = (proj_group["Company production (Mt)"] * 1e6) / (proj_group["Attributed crude steel capacity (ttpa)"] * 1e3)
    # if projected capacity is zero assume 0 utilisation rate
    proj_group['UR'] = proj_group['UR'].replace({np.inf: 0})

    # Impute missing UR with average for given year
    mean_UR = proj_group.groupby("year")["UR"].mean().to_frame("mean_UR").reset_index()
    
    proj_group = pd.merge(proj_group,
                            mean_UR,
                            on=["year"],
                            how='left')
    proj_group["UR"] = proj_group["UR"].fillna(proj_group['mean_UR'])
    # if technology production is overestimated relative to projected capacity
    # set UR to 1
    proj_group.loc[proj_group['UR'] > 1, "UR"] = 1.0 
    # Impute UR estimated at the company level to each individual plant
    op_plants_ts["Techno_map"] = op_plants_ts['Main production process'].apply(lambda x: x if x in ['electric', "integrated (BF)", "integrated (DRI)"] else "other")

    # Company level to plant level
    proj_plants = pd.merge(op_plants_ts, 
                        proj_group.loc[:, ["Group", "Techno_map", "year", "UR"]], 
                        on=["Group", "Techno_map", "year"], 
                        how='left'
                        )
    
    proj_plants['UR'] = proj_plants['UR'].fillna(1.0)
    
    # a plant's utilisation rate is defined as the weighted average of company utilisation rates
    # weighted by ownership share in the plant
    proj_plants['UR_share'] = proj_plants['UR'] * proj_plants['Share']
    # projected utilisation rate for each plant
    proj_plants_UR = proj_plants.groupby(["Plant ID", "year"])["UR_share"].sum().to_frame('UR').reset_index()
    return proj_plants_UR, proj_plants, proj_group
        
        
def get_projected_capacity(gspt: "GSPTDataset", 
                           parent_group_map: dict,
                           method: str,
                           level: str = "plant",
                           ):
    # Aggregate plant capacity based on group company mapping
    op_plants_ts = gspt.get_operating_plants(start_year=2023, end_year=2030, melt=True)
    op_plants_ts['Group'] = op_plants_ts['Parent'].copy()
    op_plants_ts = op_plants_ts.replace({'Group': parent_group_map})
    op_plants_ts['Attributed crude steel capacity (ttpa)'] = op_plants_ts['Nominal crude steel capacity (ttpa)'] * op_plants_ts['Share']

    # Baseline projected group company capacity
    proj_group_base_capa = op_plants_ts.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()  
    
    if method == "base":
        proj_group_capa = proj_group_base_capa
    elif method == "substitute_elec":
        # Get total additions in electric capacity
        melted_plants = gspt.get_steel_dataset(melt=True)
        melted_plants['Group'] = melted_plants['Parent'].replace(parent_group_map)

        delta_elec_capa = get_electric_capa_delta(melted_steel_plants=melted_plants, start_year=2023, end_year=2030)
        # Get 'integrated (BF)' capacity for each group company
        BF_mask = proj_group_base_capa['Main production process'] == "integrated (BF)"
        BF_capa = proj_group_base_capa.loc[BF_mask].copy()
        # Technology interaction
        new_BF_capa = pd.merge(BF_capa, 
                            delta_elec_capa.loc[:,['Group', 'year', 'delta_elec_capa_cumsum (ttpa)']], 
                            how="left", 
                            on=["Group", "year"])
        
        # Subtract additional electric capacity from BF capacity
        new_BF_capa['new Attributed crude steel capacity (ttpa)'] = new_BF_capa['Attributed crude steel capacity (ttpa)'] - new_BF_capa['delta_elec_capa_cumsum (ttpa)']
        
        # Update 'integrated (BF)' capacity values
        new_company_capa = pd.merge(proj_group_base_capa, 
                                    new_BF_capa[["Group", "year", "Main production process", "new Attributed crude steel capacity (ttpa)"]], 
                                    on=["Group", "Main production process", "year"], 
                                    how='left')

        new_company_capa['new Attributed crude steel capacity (ttpa)'] = new_company_capa['new Attributed crude steel capacity (ttpa)'].fillna(new_company_capa['Attributed crude steel capacity (ttpa)'])
        new_company_capa = new_company_capa.drop(columns=["Attributed crude steel capacity (ttpa)"])
        new_company_capa = new_company_capa.rename(columns={"new Attributed crude steel capacity (ttpa)": "Attributed crude steel capacity (ttpa)"})
        
        # Clean
        to_keep = ["Group", "Main production process", "year", "Attributed crude steel capacity (ttpa)"]
        new_company_capa = new_company_capa.loc[:, to_keep]
        proj_group_capa = new_company_capa
    else:
        raise ValueError('Method not implemented')
    
    # Table format
    if level == "techno":
        pass
    elif level == "group":
        proj_group_capa = proj_group_capa.groupby(["Group", "year"])["Attributed crude steel capacity (ttpa)"].sum().reset_index()
    
    return proj_group_capa

def interpolate_glob_prod(glob_prod_iea: pd.DataFrame, 
                          proj_start_year:int, 
                          end_year:int) -> pd.DataFrame:
    """Calculates global production values for each year during a given period using linear interpolation and
    based on projections from the IEA.

    Args:
        glob_prod_iea (pd.DataFrame): "steel_prod_IEA.xlsx"
        proj_start_year (int): lower bound on interpolated years
        proj_end_year (int): upper bound on interpolated years

    Returns:
        pd.DataFrame: yearly projected global production values
    """
    
    start_year = proj_start_year - 1
    glob_prod_iea = glob_prod_iea.loc[glob_prod_iea['Year'].between(start_year, end_year)]
    
    # linear interpolation of global production 
    interpolation_years = list(range(start_year, end_year + 1))
    interpolated_glob_prod = pd.DataFrame({"Year": interpolation_years, 
                                           "Global production (Mt)": [np.nan] * len(interpolation_years)})
    interpolated_glob_prod.loc[interpolated_glob_prod['Year'] == start_year, "Global production (Mt)"] = float(glob_prod_iea.loc[glob_prod_iea['Year'] == start_year, "Industrial production (Mt)"])
    interpolated_glob_prod.loc[interpolated_glob_prod['Year'] == end_year, "Global production (Mt)"] = float(glob_prod_iea.loc[glob_prod_iea['Year'] == end_year, "Industrial production (Mt)"])
    interpolated_glob_prod = interpolated_glob_prod.interpolate(method="linear", limit_direction="forward", axis=0)
    # subset to projection years
    interpolated_glob_prod = interpolated_glob_prod.loc[interpolated_glob_prod['Year'].between(start_year, end_year)]
    return interpolated_glob_prod

def update_company_capacity(company_capa, delta_elec_capa, method: str):
    """_summary_

    Args:
        company_capa (_type_): company capacity based on plant openings and closures
        delta_elec_capa (_type_): capacity to add or remove from baseline company capacity,
        based on assumptions of the evolution of the technology mix
        method (str): method for updating company capacity based on a particular assumption
            - "substitute_elec": subtract additional electric capacity (green) to operating BF capacity (brown)
    """
    if method == "substitute_elec":
        # Get 'integrated (BF)' capacity for each group company
        BF_mask = company_capa['Main production process'] == "integrated (BF)"
        BF_capa = company_capa.loc[BF_mask].copy()
        # Technology interaction
        new_BF_capa = pd.merge(BF_capa, 
                            delta_elec_capa.loc[:,['Group', 'year', 'delta_elec_capa_cumsum (ttpa)']], 
                            how="left", 
                            on=["Group", "year"])
        
        # Subtract additional electric capacity from BF capacity
        new_BF_capa['new Attributed crude steel capacity (ttpa)'] = new_BF_capa['Attributed crude steel capacity (ttpa)'] - new_BF_capa['delta_elec_capa_cumsum (ttpa)']
        # if additions in electric capacity are greater than current BF BOF
        # set BF BOF capacity to zero
        new_BF_capa['new Attributed crude steel capacity (ttpa)'] = new_BF_capa['new Attributed crude steel capacity (ttpa)'].apply(lambda x: max(x,0))
        
        # Update 'integrated (BF)' capacity values
        new_company_capa = pd.merge(company_capa, 
                                    new_BF_capa[["Group", "year", "Main production process", "new Attributed crude steel capacity (ttpa)"]], 
                                    on=["Group", "Main production process", "year"], 
                                    how='left')

        new_company_capa['new Attributed crude steel capacity (ttpa)'] = new_company_capa['new Attributed crude steel capacity (ttpa)'].fillna(new_company_capa['Attributed crude steel capacity (ttpa)'])
        new_company_capa = new_company_capa.drop(columns=["Attributed crude steel capacity (ttpa)"])
        new_company_capa = new_company_capa.rename(columns={"new Attributed crude steel capacity (ttpa)": "Attributed crude steel capacity (ttpa)"})
        
        # Clean
        to_keep = ["Group", "Main production process", "year", "Attributed crude steel capacity (ttpa)"]
        new_company_capa = new_company_capa.loc[:, to_keep]
        assert (new_company_capa['Attributed crude steel capacity (ttpa)'] >= 0).all()
    else:
        raise ValueError("Method not implemented")
    return new_company_capa

def get_electric_capa_delta(melted_steel_plants: pd.DataFrame,
                            start_year: int,
                            end_year: int) -> pd.DataFrame: 
    """For each year, calculates the cumulative addition in electric capacity for each Group company 
    since the beginning of the specified period [start_year, end_year].
    
    This delta in capacity may interact in different ways in the future with capacities from older technologies (e.g. coal).
    One approach consists in assuming the replacement of older technologies by more carbon efficient technologies (e.g. eletric).
    

    Args:
        melted_steel_plants (pd.DataFrame): all steel plants melted over group companies (gspt.get_steel_dataset(melt=True))
        start_year (int): start of observation period 
        end_year (int): end of observation period

    Returns:
        pd.DataFrame: cumulative additional electric capacity per group company and year
    """
    # 1. Find electric plants that will open in the future
    start_year_mask = melted_steel_plants['Start year'] >= start_year
    is_elec_mask = melted_steel_plants['Main production process'] == "electric"
    new_elec_plants = melted_steel_plants.loc[start_year_mask & is_elec_mask]
    
    # Aggregate capacity at group level
    new_elec_group = new_elec_plants.groupby(["Group", "Main production process", "Start year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()

    # 2. Calculate the cumulative sum of electric capacity additions
    # in order to have a delta value for each year (i.e. electric plants will not necessarily open every year)
    
    # Fill years over projection period (e.g. [start_year, end_year])
    names, years = list(new_elec_group['Group'].unique()), list(range(start_year, end_year+1))
    # Generate all combinations of years and names
    combinations = list(itertools.product(names, years))
    # Create a dataframe from the combinations list
    group_years = pd.DataFrame(combinations, columns=["Group", "Start year"])
    delta_elec = pd.merge(group_years, new_elec_group, on=["Group", "Start year"], how='left')
    
    # For a given company and year, the given electric capacity delta
    # is the cumulative sum of additions over past years
    delta_elec['delta_elec_capa_cumsum (ttpa)'] = delta_elec.groupby('Group')['Attributed crude steel capacity (ttpa)'].transform(lambda x: x.fillna(0).cumsum())
    
    # Cleaning
    delta_elec['Main production process'] = delta_elec['Main production process'].fillna("electric")
    delta_elec = delta_elec.rename(columns={"Start year": "year"})
    return delta_elec
    

def get_market_share_ts(market_share, start_year, end_year):
    """_summary_

    Args:
        market_share (_type_): market share
        start_year (_type_): projection start
        end_year (_type_): projection end

    Returns:
        _type_: _description_
    """
    proj_years = list(range(start_year, end_year+1))
    market_share_ts = [] 
    for year in proj_years:
        sub_market_share = market_share.loc[:, ["Group", "market_share"]].copy()
        sub_market_share['Year'] = year
        market_share_ts.append(sub_market_share)
    market_share_ts = pd.concat(market_share_ts, axis=0, ignore_index=True)
    return market_share_ts

        
def get_market_share(gspt,
                     db_path: Path,
                     histo_global_prod: pd.DataFrame, 
                     company_steel_prod: pd.DataFrame, 
                     mapping: pd.DataFrame,
                     gspt2gspt_path,
                     parent_group_map,
                     method: str, 
                     year: int = None) -> pd.DataFrame:
    """Compute steel company market shares from historical company production and global production.

    Args:
        gspt (GSPTDataset):
        histo_global_prod (pd.DataFrame): historical global production values 
        company_steel_prod (pd.DataFrame): Refinitiv extract containing company crude steel output
        method (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    assert year == 2022
    # 0. Import top 50 steelmakers production
    # Connect to the SQLite database (replace 'your_database.db' with the actual database file name)
    conn = sqlite3.connect(db_path)

    # Specify the table name you want to read
    table_name = 'steel_producers'

    # Use pandas to read the table into a DataFrame
    top50_prod = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    # Close the connection
    conn.close()
    top50_prod.columns = top50_prod.columns.str.lower()
    
    top50_prod = pd.merge(top50_prod,
                          histo_global_prod,
                          on="year",
                          how='left')
    top50_prod["market_share"] = top50_prod["tonnage"] / top50_prod["Crude steel production (Mt)"]


    assert top50_prod.groupby("year")["market_share"].sum().between(0,1).all()
    
    # 1. Import bottom up production
    bu_plant_prod = gspt.get_estimated_prod(year=year, entity="country", impute_prod="Country", average="micro")
    
    bu_group_prod = convert_plant2parent(bu_plant_prod, 
                                         gspt2gspt_path=gspt2gspt_path,
                                         parent_group_map=parent_group_map)
    
    bu_group_prod["Attributed crude steel production"] = bu_group_prod["Estimated crude steel production (ttpa)"] * 1e3 * bu_group_prod["Share"]
    
    # Aggregated bottom-up production at group level
    bu_group_prod = bu_group_prod.groupby(["year", "Group"])["Attributed crude steel production"].sum().reset_index()
    
    #bu_group_prod = bu_group_prod.rename(columns={"year": "prod_year"})
    
    bu_group_prod = pd.merge(bu_group_prod,
                        histo_global_prod,
                        on="year",
                        how='left')
    bu_group_prod["market_share"] = bu_group_prod["Attributed crude steel production"] / (bu_group_prod["Crude steel production (Mt)"]*1E6)
    
    
    # merge top 50 market share with BU market share
    # TOOD: reprendre ici
    wsa_bu_ms = pd.merge(bu_group_prod,
                        top50_prod[["year", "gspt_group_name", "market_share"]],
                        left_on=["year", "Group"],
                        right_on=["year", "gspt_group_name"],
                        how='left',
                        suffixes=("_bu", "_wsa"))
    # correct bottom-up market share to sum to 1 with wsa
    # we assume wsa values to be true and correct for bottom-up overestimation
    wsa_ms = wsa_bu_ms["market_share_wsa"].sum()
    bu_ms = wsa_bu_ms.loc[wsa_bu_ms["market_share_wsa"].isna(), "market_share_bu"].sum()
    adj_factor = (1. - wsa_ms) / bu_ms
    wsa_bu_ms["adj_market_share_bu"] = adj_factor * wsa_bu_ms['market_share_bu']
    wsa_bu_ms['market_share'] = wsa_bu_ms["market_share_wsa"].fillna(wsa_bu_ms['adj_market_share_bu'])
    
    return wsa_bu_ms.loc[:, ["year", "Group", "market_share"]]
    
    
    
