import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter
import pickle
from typing import Tuple
import copy

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from collections import defaultdict
import joblib

project_dir = Path().cwd()
while project_dir.stem != "bottom_up_alignment":
    project_dir = project_dir.parent
import sys
sys.path.append(str(project_dir / "src"))

from datasets.EmissionFactors import EmissionFactors
from datasets.GSPTDataset import GSPTDataset
from datasets.RefinitivDataset import RefinitivDataset
from datasets.utils import convert_plant2parent, get_group_col, get_cagr

from projections_.BUProjector import get_market_share
from projections_.BUProjector import interpolate_glob_prod, get_market_share_ts, get_projected_UR_1
from projections_.BUProjector import get_electric_capa_delta, update_company_capacity, get_projected_UR_3
from projections_.project_costs import get_proj_costs_emissions, shutdown_plants
from projections_.production import get_global_prod, get_steel_ghg
from projections_.proj_company import get_proj_elec_int, get_prediction_interval

from feature_eng.company_feature_engineering import get_X_y, get_electricity_intensity
from feature_eng.plant_features import get_all_country_ur


# BU historical emissions
# top-down historical emissions
# Historical IEA

def get_ghg_trajectory_plot(gspt: GSPTDataset,
                            EF: EmissionFactors,
                            market_share: pd.DataFrame,
                            proj_costs_iea: dict,
                            glob_prod: dict,
                            glob_capa: dict,
                            base_nze_emissions,
                            base_aps_emissions, 
                            base_steps_emissions,
                            elec_int_nze,
                            elec_int_aps,
                            elec_int_steps,
                            energy_mix_path,
                            parent_group_map_path,
                            gspt2gspt_path,
                            gspt2refi_map,
                            agg_BU: pd.DataFrame,
                            world_fpath,
                            regions_fpath,
                            gspt2iea_countries_path,
                            iea_level,
                            model,
                            X_train,
                            y_train,
                            err_style: str,
                            EAF_decarb: bool=False):
    """
    Generate a greenhouse gas emissions trajectory plot for the Iron & Steel sector.

    This function creates a line plot showing historical and projected emissions for the Iron & Steel sector,
    with annotations and a legend.

    Args:
        bu_histo_emissions (pandas.DataFrame): Historical emissions data for BU.
        sectoral_emissions (pandas.DataFrame): Sectoral emissions data.
        base_nze_emissions (pandas.DataFrame): baseline emissions trajectory for Iron & Steel sector under NZE assumptions.
        base_aps_emissions (pandas.DataFrame): Baseline emissions trajectory for Iron & Steel sector under APS assumptions.
        agg_BU (pd.DataFrame): historical bottom-up emissions aggregated at the sectoral level.
        iea_level (str): granularity of IEA features (i.e. global or region)
        X_train (pd.DataFrame): independent variables. Index = ["GSPT_Name", "year"]. Used for prediction error.
        y_train (pd.DataFrame): dependent variable. Index = ["GSPT_Name", "year"]. Used for prediction error.
        err_style (str): "line" or "band"

    Returns: 
        fig_vert (matplotlib.figure.Figure): sectoral pathways stacked vertically
        fig_hor (matplotlib.figure.Figure): sectoral pathways in a 2x2 frame
    """
    # Load bottom-up model
    check_is_fitted(model)
   
    agg_BU["Emissions (Gt)"] = agg_BU["BU emissions"] / 1e9
    glob_bu_prod = agg_BU.loc[agg_BU["year"] == 2022, 'Attributed production'].iloc[0]
    
    # 1. Production figure
    fig_prod, ax_prod = plt.subplots(figsize=(12,20))
    # BU historical emissions
    sns.lineplot(data=agg_BU,
                x='year',
                y="Attributed production",
                label="Historical BU prod",
                marker="o",
                ax=ax_prod)

    colors = ['#6FAA96', '#5AC2CC', '#D1E4F2', '#2E73A9', '#003f5c','#6FAA96', '#5AC2CC', '#D1E4F2', '#2E73A9', '#CCCCFF']

    # 2. Main figure with emissions trajectories (vertical)
    nrow = 3
    fig_vert, axs_vert = plt.subplots(nrow, 1, figsize=(12, 20))
    axs2_vert = np.ndarray(shape=len(axs_vert), dtype='object')
    for i, ax_vert in enumerate(axs_vert):
        # vertical chart
        ax_vert.set_xlim(2017, 2031)
        ax_vert.set_ylim(3.25,4.75)
        ax_vert.grid("both")


    # 3. Main figure with emissions trajectories (horizontal)
    fig_hor, axs_hor = plt.subplots(2, 2, figsize=(12, 12),
                                    constrained_layout=True) #, subplot_kw=dict(box_aspect=1))
    axs_hor.flatten()[-1].set_visible(False)
    axs_hor = axs_hor.flatten()[:3]
    axs2_hor = np.ndarray(shape=len(axs_hor), dtype='object')
    for i, ax_hor in enumerate(axs_hor):
        # horizontal chart
        ax_hor.set_xlim(2017, 2031)
        ax_hor.set_ylim(3.25,4.75)
        ax_hor.set_box_aspect(1)
        ax_hor.grid("both")
    
    base_nze_emissions.loc[base_nze_emissions.index[0], "Emissions (Gt)"] = float(agg_BU.loc[agg_BU['year'] == 2022, 'Emissions (Gt)'])
    base_steps_emissions.loc[base_steps_emissions.index[0], "Emissions (Gt)"] = float(agg_BU.loc[agg_BU['year'] == 2022, 'Emissions (Gt)'])
    base_aps_emissions.loc[base_aps_emissions.index[0], "Emissions (Gt)"] = float(agg_BU.loc[agg_BU['year'] == 2022, 'Emissions (Gt)'])
    
    base_emissions = {"NZE": base_nze_emissions,
                      "APS": base_aps_emissions,
                      "STEPS": base_steps_emissions}
    base_colors = {"NZE": "green",
                   "APS": "brown",
                   "STEPS": "black"}
    
    with open(parent_group_map_path, "r") as f:
        parent_group_map = json.load(f)
    
    scenarios = ["NZE", "APS", "STEPS"]
    # save projected values
    data = {scn: {} for scn in scenarios}
    
    for scenario, ax_vert, ax_hor, ax2_vert, ax2_hor in zip(scenarios, axs_vert, axs_hor, axs2_vert, axs2_hor):
        if scenario == "STEPS":
            title = "Stated Policies Scenario"
        elif scenario == "NZE":
            title = "Net Zero Scenario"
        elif scenario == "APS":
            title = "Announced Pledges Scenario"
        ax_vert.set_title(title)
        ax_hor.set_title(title, 
                         y=1.00625,
                         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
        
        # TODO:
        cbudget = pd.read_excel(Path("./src/projections_") /f"proj_country_prod_{scenario.lower()}.xlsx")
              
        # carbon price changes (~/2 for emergent countries)
        proj_costs = proj_costs_iea[scenario]

        for ax_ in [ax_vert, ax_hor]:
            # BU historical emissions
            sns.lineplot(data=agg_BU,
                        x='year',
                        y="Emissions (Gt)",
                        label="Historical BU emissions",
                        ax=ax_)   
            
            # Reference emissions trajectories from scenarios
            # Add a marker for the first value (we use the same emissions starting point. Scenarios
            # are only used for deriving decarbonisation slopes)
            sns.scatterplot(x=[base_nze_emissions.loc[base_nze_emissions.index[0], "year"]], 
                            y=[base_nze_emissions.loc[base_nze_emissions.index[0], "Emissions (Gt)"]],  
                            marker='o', 
                            color="grey",
                            s=200,
                            zorder=2,
                            ax=ax_)
                
            # ETP IEA + NZE
            base_nze_emissions_22 = float(base_nze_emissions.loc[base_nze_emissions['year'] == 2022, 'Emissions (Gt)'])
            ax_.annotate(f"{base_nze_emissions_22:.2f}", (2022, base_nze_emissions_22 + 0.055), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=12, color='grey', weight='bold')
        
            # IEA reference trajectories (NZE, APS, STEPS)
            for ref in ["NZE", "APS", "STEPS"]:
                base_ref_emissions = base_emissions[ref]
                # Projected emissions from IEA value in 2019 and projected annualized decarbonization rate from NZE scenario
                sns.lineplot(data=base_ref_emissions,
                            x='year',
                            y="Emissions (Gt)",
                            linewidth=2,
                            label=f"Projected IEA emissions ({ref} slope)",
                            color=base_colors[ref],
                            linestyle="dotted",
                            zorder=2,
                            ax=ax_)
                base_ref_emissions_30 = float(base_ref_emissions.loc[base_ref_emissions['year'] == 2030, 'Emissions (Gt)'])
                ax_.annotate(f"{base_ref_emissions_30:.2f}", (2030.5, base_ref_emissions_30), textcoords='offset points', xytext=(0, 10),
                            ha='center', fontsize=10, color=base_colors[ref], weight='bold')
            ax_.set_ylabel("Emissions (GtCO₂ e)")
        
        
        glob_prod_iea = glob_prod[scenario]
        # CAAGR for electricity generation intensity
        r_nze = get_elec_gen_caagr(elec_int=elec_int_nze, start_year=2022, end_year=2030)
        r_steps = get_elec_gen_caagr(elec_int=elec_int_steps, start_year=2022, end_year=2030)
        r_aps = get_elec_gen_caagr(elec_int=elec_int_aps, start_year=2022, end_year=2030)
        
        # colors for drivers
        drivers_colors = {"constant_UR": "orange",
                          "company_UR": "green",
                          "cost_efficiency": "red",
                          "carbon_efficiency": "blue",
                          "carbon_intensity": "brown"}
        methods = ["carbon_efficiency", "carbon_intensity", "constant_UR", "company_UR"]
        for i_method, method in enumerate(methods): # "cost_efficiency"
            ## Projected emissions
            start_year, end_year = 2023, 2030
            proj_company, proj_plants = get_bu_proj_emissions(gspt=gspt,
                                EF=EF,
                                glob_prod_iea=glob_prod[scenario], 
                                glob_bu_prod=glob_bu_prod,
                                market_share=market_share,
                                parent_group_map_path=parent_group_map_path,
                                gspt2refi_map=gspt2refi_map,
                                gspt2gspt_path=gspt2gspt_path,
                                proj_costs=proj_costs,
                                start_year=start_year,
                                end_year=end_year,
                                method=method,
                                glob_capa=glob_capa[scenario],
                                cbudget=cbudget) 
            r = r_nze if scenario == "NZE" else (r_aps if scenario == "APS" else r_steps)         
            proj_plants = get_proj_elec_int(proj_plants, 
                                            r=r, 
                                            energy_mix_path=energy_mix_path, 
                                            level=iea_level,
                                            scenario=scenario,
                                            world_fpath=world_fpath,
                                            regions_fpath=regions_fpath,
                                            gspt2iea_countries_path=gspt2iea_countries_path)
            # EAF country electrictiy generation intensity
            proj_plants["EAF_country_elec_int"] = (proj_plants["Main production process"] == "electric") * proj_plants[f"{scenario}_elec_int"]


            if EAF_decarb:
                # using global decarbonisation rates
                if (iea_level == "global") | (scenario == "NZE"):
                    base_all = 460 # 2022
                    end_steps = 303 # 2030
                    end_aps = 255 # 2030
                    end_nze = 186 # 2030
                    if scenario == "STEPS":
                        caagr = get_cagr(base_value=base_all, end_value=end_steps, n=8)
                    elif scenario == "APS":
                        caagr = get_cagr(base_value=base_all, end_value=end_aps, n=8)
                    elif scenario == "NZE":
                        caagr = get_cagr(base_value=base_all, end_value=end_nze, n=8)
                    
                    # rates to multiply directly with Emission factors
                    years = list(range(2023, 2031))
                    elec_cagr = pd.DataFrame({"year": years,
                        "CAGR": [(1+caagr)**(i+1) for i in range(len(years))],
                        })
                    
                    proj_plants = pd.merge(proj_plants,
                                        elec_cagr,
                                        on="year",
                                        how='left')
                # using regional decarbonisation slopes for
                # APS and STEPS
                elif iea_level == "region":
                    # base value and target value for all regions
                    
                    # calculate cagr for each region
                    
                    # create cagr df (cf above)
                    
                    # merge elec_cagr with proj_plants
                    pass
                
                proj_plants["CAGR"] = (proj_plants["Main production process"] == "electric") * proj_plants["CAGR"]
                proj_plants["CAGR"] = proj_plants["CAGR"].replace(0, 1)
                proj_plants["EF"] = proj_plants["EF"] * proj_plants["CAGR"]
                # update emissions based on adjusted EF
                proj_plants["Emissions (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF"] / 1E6
            #
            proj_group = convert_plant2parent(proj_plants, gspt2gspt_path=gspt2gspt_path, parent_group_map=parent_group_map)

            proj_group["Attributed crude steel capacity (ttpa)"] = proj_group['Nominal crude steel capacity (ttpa)'] * proj_group["Share"]
            proj_group["Attributed capacity"] = proj_group["Attributed crude steel capacity (ttpa)"] * 1E3
            proj_group["Attributed emissions"] = proj_group['Emissions (Gt)'] * 1e9 * proj_group["Share"]
            proj_group["Attributed production"] = proj_group['Estimated crude steel production (ttpa)'] * 1e3 * proj_group["Share"]
            
            wm_capa = lambda x: np.average(x, weights=proj_group.loc[x.index, "Attributed crude steel capacity (ttpa)"])
            proj_feats = proj_group.groupby(['Group', 'year']).agg(
                                                    emissions=("Attributed emissions", "sum"),
                                                    elec_int_capa=(f"{scenario}_elec_int", wm_capa),
                                                    production=("Attributed production", "sum"),
                                                    capacity=("Attributed capacity", "sum"),
                                                    EAF_elec_int_capa=("EAF_country_elec_int", wm_capa),)
            proj_feats["log_Attributed emissions"] = np.log(proj_feats["emissions"]) 
            
            # features on which model was fit
            try:
                features = model[0].get_feature_names_out()
            except:
                features = model[0].feature_names_in_
                
            # Add bottom-up indicators to projected features set
            # e.g. BU emissions, BU intensity
            assert (proj_feats["emissions"] > 0).all()
            proj_feats["log_Attributed emissions"] = np.log(proj_feats["emissions"])
            # add interaction term
            proj_feats['interaction'] = proj_feats["log_Attributed emissions"] * proj_feats["elec_int_capa"]
            proj_feats['ratio'] = proj_feats["log_Attributed emissions"] / proj_feats["elec_int_capa"]
            proj_feats[f"log BU emissions ({method})"] = model.predict(proj_feats[features])
            proj_feats[f"log BU err ({method})"] = get_prediction_interval(X0=proj_feats[features], 
                                          X_train=X_train, 
                                          y_train=y_train, 
                                          model=model, 
                                          center=True)
            proj_feats[f"BU emissions ({method})"] = np.exp(proj_feats[f"log BU emissions ({method})"])
            proj_feats[f"BU intensity ({method})"] = proj_feats[f"BU emissions ({method})"] / proj_feats["production"]
            proj_feats[f"BU intensity ({method})"] = proj_feats[f"BU intensity ({method})"].replace(np.inf, np.nan)
            proj_feats = proj_feats.reset_index()
            bu_sectoral_proj = proj_feats.groupby("year").agg({f"BU emissions ({method})": "sum",
                                                               "emissions": 'sum',
                                                               "production": "sum",
                                                               "capacity": "sum"
                                                               }).reset_index()
            # emissions = raw bu emissions
            bu_sectoral_proj['Emissions (Gt)'] = bu_sectoral_proj[f"BU emissions ({method})"] / 1e9
            bu_sectoral_proj['Raw Emissions (Gt)'] = bu_sectoral_proj["emissions"] / 1e9
            bu_sectoral_proj[f'UR ({method})'] = bu_sectoral_proj["production"] / bu_sectoral_proj["capacity"]
            bu_sectoral_proj[f"Intensity ({method})"] = bu_sectoral_proj[f"BU emissions ({method})"] / bu_sectoral_proj["production"]

            
            
            if method in ["constant_UR", "company_UR", "company_UR_fuel_switch", "techno_UR"]:
                # for hypotheses based on 2022 constant market share,
                # GSPT only accounts for ~93% of global production
                # which results in underestimating future emissions
                # we hence adjust sectoral emissions by the inverse of 
                pass
            else:
                # costs minimisation hypothesis directly meets global demand
                pass
            # make lines continuous between 2022 and 2023 projection
            bu_emissions_22 = float(agg_BU.loc[agg_BU["year"] == 2022, "Emissions (Gt)"])
            new_row = {"year": [2022], 
                    "Emissions (Gt)": [bu_emissions_22]}
            #bu_sectoral_proj = bu_sectoral_proj.append(new_row, ignore_index=True).sort_values(by="year", ascending=True)
            bu_sectoral_proj = pd.concat([pd.DataFrame(new_row), bu_sectoral_proj], axis=0).sort_values(by="year", ascending=True)
            data[scenario][method] = {"sector": bu_sectoral_proj,
                                      "company": proj_feats,
                                      "plants": proj_plants}
            
            # add projected production to fig_prod
            sns.lineplot(data=bu_sectoral_proj,
                x='year',
                y="production",
                label=f"{scenario}_{method} production",
                linestyle="dashed",
                ax=ax_prod)
            
            if method == "company_UR":
                emissions_label = f"Projected BU emissions (constant market share)"
            elif method == "company_UR_fuel_switch":
                emissions_label = f"Projected BU emissions (company UR + fuel switching)"
            elif method == "techno_UR":
                emissions_label = f"Projected BU emissions (company & techno UR + fuel switching)"
            elif method == "constant_UR":
                emissions_label = f"Projected BU emissions (country UR)"
            elif method == "cost_efficiency":
                emissions_label = f"Projected BU emissions (cost efficiency)"
            elif method == "carbon_efficiency":
                emissions_label = f"Projected BU emissions (carbon efficiency)"
            elif method == "carbon_intensity":
                emissions_label = f"Projected BU emissions (carbon intensity)"
            else:
                emissions_label = f"Projected BU emissions ({method})"
                
            if (err_style != "line") & (method in ["carbon_efficiency", "carbon_intensity"]):
                pass
            else:
                # vertical chart
                sns.lineplot(data=bu_sectoral_proj,
                            x='year',
                            y="Emissions (Gt)",
                            label=emissions_label,
                            linestyle="dashed",
                            ax=ax_vert)
                # horizontal chart
                sns.lineplot(data=bu_sectoral_proj,
                x='year',
                y="Emissions (Gt)",
                label=emissions_label,
                linestyle="dashed",
                ax=ax_hor)


        
        # Add uncertainty bands
        if err_style == "band":
            uct_index = list(range(2022, end_year+1))
            lower_uct_data = data[scenario]["carbon_efficiency"]["sector"]
            lower_uct_data = lower_uct_data.loc[lower_uct_data["year"].isin(uct_index), "Emissions (Gt)"]
            upper_uct_data = data[scenario]["carbon_intensity"]["sector"]
            upper_uct_data = upper_uct_data.loc[upper_uct_data["year"].isin(uct_index), "Emissions (Gt)"]
            uct_label = "Bounds on global emissions of the steel sector \n"\
                        "obtained by choosing the most polluting (upper \n bound) "\
                        "and least polluting (lower bound) plants"
            ax_vert.fill_between(uct_index, lower_uct_data, upper_uct_data, alpha=0.2, color="red", zorder=-1, label=uct_label)
            ax_hor.fill_between(uct_index, lower_uct_data, upper_uct_data, alpha=0.2, color="red", zorder=-1, label=uct_label)
        
        # Set labels and title
        for ax_ in [ax_vert, ax_hor]:
            ax_vert.set_ylabel("Emissions (GtCO₂ eq)")
            ax_vert.set_xlabel("Year")
    
    # Define a custom formatting function for the x-axis ticks
    def format_ticks(x, pos):
        return int(x)  # Convert to integer

    # Place the legend outside the plot (vertical chart)
    handles, labels = get_unique_handles_labels(fig_vert.axes)
    fig_vert.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.07), ncol=nrow)
    # horizontal chart
    fig_hor_axes = fig_hor.axes #+ [axs2_hor[0]]
    handles, labels = get_unique_handles_labels(fig_hor_axes) #fig_hor.axes)
    fig_hor.legend(handles=handles, 
                   labels=labels, 
                   loc="lower right", 
                   bbox_to_anchor=(0.9, 0.07), 
                   ncol=1,
                   frameon=False,
                   fontsize=12)
    
    # Remove legends for all but the last axis for both figures
    for axes in [fig_vert.get_axes(), fig_hor.get_axes()]:
        for ax in axes:
            ax.legend().remove()
            # Create a FuncFormatter with the custom formatting function
            formatter = FuncFormatter(format_ticks)
            # Set the x-axis major formatter to use the custom formatter
            ax.xaxis.set_major_formatter(formatter)        

    # Adjust the layout
    fig_vert.suptitle("Evolution of Iron & Steel sector emissions", fontdict={"fontsize": 14}, y=0.9)
    fig_hor.suptitle("Evolution of Iron & Steel sector emissions", fontdict={"fontsize": 14}, weight="bold", y=1.05)

    # Produce same graphs but rescaled relative to 2022 emissions
    fig_hor_rescaled = copy.deepcopy(fig_hor)
    axes = fig_hor_rescaled.get_axes()
    for ax in axes:
        for line in ax.get_lines():
            # Get the original data
            x_data, y_data = line.get_data()
            # Rescale the y-data
            y_data_rescaled = y_data / float(agg_BU.loc[agg_BU['year'] == 2022, 'Emissions (Gt)'].iloc[0])
            # Update the plot with the rescaled data
            line.set_ydata(y_data_rescaled)
        # Adjust the limits if needed
        ax.set_ylim(0.75,1.1)
        ax.set_ylabel("Emissions relative to 2022 levels")
        ax.annotate(f"{1.0:.2f}", (2022 + 0.055, 1.0), textcoords='offset points',
            xytext=(0, 10), ha='center', fontsize=12, color='grey', weight='bold')
        # ax.relim()
        # ax.autoscale_view()
    
    for ax in axes:
        for scenario in ["NZE", "APS", "STEPS"]:
            base_ref_emissions = base_emissions[scenario]
            # Normalized projected emissions for each reference trajectory
            base_ref_emissions_30 = float(base_ref_emissions.loc[base_ref_emissions['year'] == 2030, 'Emissions (Gt)']) / float(agg_BU.loc[agg_BU['year'] == 2022, 'Emissions (Gt)'].iloc[0])
            ax.annotate(f"{base_ref_emissions_30:.2f}", (2030.5, base_ref_emissions_30), textcoords='offset points', xytext=(0, 1),
                        ha='center', fontsize=10, color=base_colors[ref], weight='bold')
    
    for ax in axes:
        ax_title = ax.get_title()
        if ax_title == "Net Zero Scenario":
            scenario = "NZE"
        elif ax_title == "Announced Pledges Scenario":
            scenario = "APS"
        elif ax_title == "Stated Policies Scenario":
            scenario = "STEPS"
        
        # Add uncertainty bands
        if err_style == "band":
            uct_index = list(range(2022, end_year+1))
            lower_uct_data = data[scenario]["carbon_efficiency"]["sector"]
            lower_uct_data = lower_uct_data.loc[lower_uct_data["year"].isin(uct_index), "Emissions (Gt)"]
            # Normalize by 2022 emissions
            lower_uct_data = lower_uct_data / float(agg_BU.loc[agg_BU['year'] == 2022, 'Emissions (Gt)'].iloc[0])
            upper_uct_data = data[scenario]["carbon_intensity"]["sector"]
            upper_uct_data = upper_uct_data.loc[upper_uct_data["year"].isin(uct_index), "Emissions (Gt)"]
            # Normalize by 2022 emissions
            upper_uct_data = upper_uct_data / float(agg_BU.loc[agg_BU['year'] == 2022, 'Emissions (Gt)'].iloc[0])

            uct_label = "Bounds on global emissions of the steel sector \n"\
                        "obtained by choosing the most polluting (upper \n bound) "\
                        "and least polluting (lower bound) plants"
            ax.fill_between(uct_index, lower_uct_data, upper_uct_data, alpha=0.2, color="red", zorder=-1, label=uct_label)

    return fig_vert, fig_hor, fig_prod, data, fig_hor_rescaled

def get_unique_handles_labels(axes):
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    added_to_legend = set()
    
    handles = []
    final_labels = []
    for handle, label in zip(lines, labels):
        if label not in added_to_legend:
            handles.append(handle)
            final_labels.append(label)
            added_to_legend.add(label)
        else:
            pass
    return handles, final_labels


def get_elec_gen_caagr(elec_int: pd.DataFrame, start_year: int, end_year: int):
    """Calculate compound average annual growth rate (CAAGR) 
    of CO2 electricity generation intensity from IEA projections.
    
    CAAGR is a measure that provides a smoothed annualized representation of 
    the growth of a value over multiple periods. It considers the compounding 
    effect on the average growth rate, offering a more accurate reflection of 
    how a quantity, such as an investment or business metric, has changed over time. 

    Args:
        elec_int (pd.DataFrame): IEA data for a given scenario (e.g. NZE, APS, STEPS)
        start_year (int): base year 
        end_year (int): 
        return r (float): CAAGR
    """
    base_value = float(elec_int.loc[elec_int["Year"] == start_year, "CO₂ intensity of electricity generation (g CO₂ per kWh)"])
    end_value = float(elec_int.loc[elec_int["Year"] == end_year, "CO₂ intensity of electricity generation (g CO₂ per kWh)"])
    caagr = ((end_value / base_value) ** (1 / (end_year - start_year))) - 1
    return caagr
    
def get_bu_proj_emissions(gspt,
                          EF,
                          glob_prod_iea,
                          glob_bu_prod,
                          market_share,
                          parent_group_map_path,
                          gspt2refi_map,
                          gspt2gspt_path,
                          glob_capa,
                          proj_costs,
                          start_year,
                          end_year,
                          method,
                          cbudget):

    with open(parent_group_map_path, "r") as f:
        parent_group_map = json.load(f)
        
    if method == "constant_UR":
        proj_company, proj_plants = get_proj_BU_constant_UR(gspt=gspt, EF=EF, 
                                                            start_year=start_year, 
                                                            end_year=end_year, 
                                                            gspt2gspt_path=gspt2gspt_path,
                                                            parent_group_map=parent_group_map,
                                                            glob_capa=glob_capa,
                                                            glob_prod=glob_prod_iea,
                                                            base_bu_prod=glob_bu_prod)
    elif method in ["company_UR", "company_UR_fuel_switch", "techno_UR"]: 
        proj_company, proj_plants = get_proj_BU_constant_ms(gspt=gspt, 
                            glob_prod_iea=glob_prod_iea,
                            glob_bu_prod=glob_bu_prod,
                            glob_capa=glob_capa,
                            market_share=market_share,
                            parent_group_map_path=parent_group_map_path,
                            method=method,
                            gspt2refi_map=gspt2refi_map,
                            gspt2gspt_path=gspt2gspt_path,
                            EF=EF,
                            start_year=start_year,
                            end_year=end_year
                            )
        
    elif method == "cost_efficiency":
        proj_company, proj_plants = get_proj_BU_cost_efficiency(proj_costs=proj_costs, 
                                                                glob_prod_iea=glob_prod_iea,
                                                                glob_bu_prod=glob_bu_prod, 
                                                                glob_capa=glob_capa,
                                                                start_year=start_year, 
                                                                end_year=end_year, 
                                                                gspt2gspt_path=gspt2gspt_path)
    elif method == "carbon_efficiency":
        proj_company, proj_plants = get_proj_BU_carbon_impact(gspt=gspt, 
                                                              EF=EF,
                                                        glob_prod=glob_prod_iea,
                                                        glob_capa=glob_capa,
                                                        start_year=start_year, 
                                                        end_year=end_year, 
                                                        gspt2gspt_path=gspt2gspt_path,
                                                        parent_group_map=parent_group_map,
                                                        plant_method="carbon_efficiency",
                                                        cbudget=cbudget)
        
    elif method == "carbon_intensity":
        proj_company, proj_plants = get_proj_BU_carbon_impact(gspt=gspt, 
                                                              EF=EF,
                                                        glob_prod=glob_prod_iea,
                                                        glob_capa=glob_capa,
                                                        start_year=start_year, 
                                                        end_year=end_year, 
                                                        gspt2gspt_path=gspt2gspt_path,
                                                        parent_group_map=parent_group_map,
                                                        plant_method="carbon_intensity",
                                                        cbudget=cbudget)
    else:
        raise Exception('Unknown projection method!')
    return proj_company, proj_plants


def get_proj_BU_carbon_impact(gspt: "GSPTDataset", 
                              EF: EmissionFactors,
                              start_year: int,
                              end_year: int,
                              glob_capa: pd.DataFrame,
                              glob_prod: pd.DataFrame,
                            gspt2gspt_path: Path,
                              parent_group_map: dict,
                              plant_method: str,
                              cbudget: pd.DataFrame) -> pd.DataFrame:
    # operating plants (open and closed plants based on available information)
    op_plants = gspt.get_operating_plants(start_year=start_year,
                        end_year=end_year)
    
    # 1. Capacity constraint
    final_plants, removed_plants = shutdown_plants_hyp01(plants=op_plants, glob_capa=glob_capa, start_year=start_year, end_year=end_year)
    # 2. Production constraint
    # historical adjustment factor not required as 
    # we operate selected plants at 100% UR

    # country-level emission factors
    final_plants = EF.match_emissions_factors(final_plants, 
                            level="country", 
                            source="huizhong", # TODO: old sci
                            techno_col="Main production process",
                            proj=True) 
    final_plants['Estimated crude steel production (ttpa)'] = final_plants["Nominal crude steel capacity (ttpa)"]
    final_plants['Emissions (Gt)'] = final_plants['Estimated crude steel production (ttpa)'] * final_plants['EF'] / 1e6
    
    # projected global production
    iea_prod = pd.DataFrame({"year": range(2022, 2031)})
    iea_prod.loc[iea_prod["year"] == 2022, "production"] = 1885738 # WSA
    iea_prod.loc[iea_prod["year"] == 2030, "production"] = glob_prod.loc[glob_prod["Year"] == 2030, "Industrial production (Mt)"].iloc[0] * 1E3
    iea_prod['production'] = iea_prod['production'].interpolate(method="linear")
    iea_prod["Global production (ttpa)"] = iea_prod['production']
    
    # 3. Plant selection based on carbon impact
    proj_plants, total_capa = carbon_impact_selection(plants=final_plants, glob_prod=iea_prod, method=plant_method, start_year=start_year, end_year=end_year, level="country", cbudget=cbudget)
    
    # company projection
    proj_company = convert_plant2parent(proj_plants, 
                                        gspt2gspt_path=gspt2gspt_path,
                                        parent_group_map=parent_group_map)
    proj_company["Attributed emissions (ttpa)"] = proj_company['Emissions (Gt)'] * proj_company['Share'] * 1e6
    proj_company["Attributed production (ttpa)"] = proj_company['Estimated crude steel production (ttpa)'] * proj_company['Share']
    proj_company = proj_company.groupby(["year", "Group"]).agg({"Attributed production (ttpa)":  "sum",
                                                                    "Attributed emissions (ttpa)": "sum"}).reset_index()
    return proj_company, proj_plants


def carbon_impact_selection(plants: pd.DataFrame, glob_prod: pd.DataFrame, method: str, start_year: int, end_year: int, cbudget, level: str = "global"):
    """_summary_

    Args:
        plants (pd.DataFrame): projected operating plants under global capacity constraint
        glob_prod (pd.DataFrame): projected global production based on scenario trend
        method (str): plant-level carbon impact can either be "intensive" or "efficient"

    Returns:
        _type_: _description_
    """
    # selected plants
    res = []
    # available plants before selection
    total_plants = []

    if level == "global":
        for year in range(start_year, end_year + 1):
            if year == start_year:
                op_plants_ids = plants.loc[
                    plants["year"] == start_year, "Plant ID"
                ].tolist()
            else:
                # new year's plant ID to look at
                old_plant_ids = bounded_op_plants["Plant ID"].tolist()
                new_plant_ids = plants.loc[
                    (plants["year"] == year) & (plants["Start year"] == year),
                    "Plant ID",
                ].tolist()
                # merge old plant ids (still operating) with new plant ids (potentially operating)
                op_plants_ids = list(set(old_plant_ids).union(set(new_plant_ids)))

            # save total capacity before carbon impact selection
            total_capacity = pd.DataFrame({"year": year, "Plant ID": op_plants_ids})
            total_plants.append(total_capacity)

            # all operating plants based on start/closed years in gspt
            op_plants = plants.loc[
                (plants["year"] == year) & plants["Plant ID"].isin(op_plants_ids)
            ]
            # get global production constraint to determine which plants to shutdown
            constraint = float(
                glob_prod.loc[glob_prod["year"] == year, "Global production (ttpa)"]
            )
            # get list of plant ids to shut down in current year
            to_shutdown = shutdown_plants(
                op_plants=op_plants, constraint=constraint, kind="prod", sortby=method
            )
            # new op plants that satisfy the global constraint on prod/capacity
            bounded_op_plants = op_plants.loc[~op_plants["Plant ID"].isin(to_shutdown)]
            res.append(bounded_op_plants)
    elif level == "country":

        res = []

        # available plants before selection
        total_plants = []

        for year in range(start_year, end_year + 1):

            # We assume each year to be independent
            op_plants_ids = plants.loc[
                plants["year"] == start_year, "Plant ID"
                ].tolist()
            # save total capacity before carbon impact selection
            total_capacity = pd.DataFrame({"year": year, "Plant ID": op_plants_ids})
            total_plants.append(total_capacity)

            # all operating plants based on start/closed years in gspt
            op_plants = plants.loc[
                (plants["year"] == year) & plants["Plant ID"].isin(op_plants_ids)
            ]
            op_plants["Estimated crude steel production (ttpa)"] = op_plants["Nominal crude steel capacity (ttpa)"].copy()
            
            countries_mshare = set(cbudget['GSPT_Country'].unique().tolist())
            countries = op_plants["Country"].unique().tolist()
            
            country_res = []
            
            for country in countries:
                country_plants = op_plants.loc[op_plants["Country"] == country]
                
                if country in countries_mshare:
                    
                    constraint = float(
                        cbudget.loc[(cbudget["Year"] == year) & (cbudget["GSPT_Country"] == country), "proj_prod"] 
                    ) * 1E3 # convert in ttpa
                    
                    if constraint > 0:
                        # get list of plant ids to shut down in current year
                        country_bounded_op_plants = shutdown_plants(
                            op_plants=country_plants, constraint=constraint, kind="prod", sortby=method, switch=True
                        )
                        # new op plants that satisfy the global constraint on prod/capacity
                        country_res.append(country_bounded_op_plants)
                        print(country)
                        print(f"constraint: {constraint}")
                        print(f"selected prod: {country_bounded_op_plants['Estimated crude steel production (ttpa)'].sum()}")
                else:
                    country_res.append(country_plants)
            bounded_op_plants = pd.concat(country_res, axis=0, ignore_index=True)
            res.append(bounded_op_plants)
        # set of plants that satisfy global capacity constraint
        # and maximize costs/environmental efficiency
        final_plants = pd.concat(res, axis=0, ignore_index=True)
        # all available plants before selection
        total_capa = pd.concat(total_plants, axis=0, ignore_index=True)
        
    # set of plants that satisfy global capacity constraint
    # and maximize costs/environmental efficiency
    final_plants = pd.concat(res, axis=0, ignore_index=True)
    # all available plants before selection
    total_capa = pd.concat(total_plants, axis=0, ignore_index=True)
    return final_plants, total_capa
    
    
def get_proj_BU_cost_efficiency(proj_costs, glob_prod_iea, glob_bu_prod, glob_capa, start_year, end_year, gspt2gspt_path):
    # Capacity constraint
    proj_costs_filter, remove_plants = shutdown_plants_hyp01(plants=proj_costs, glob_capa=glob_capa, start_year=start_year, end_year=end_year)
    
    # Plant production = Plant capacity under this hypothesis 
    proj_costs_filter["Estimated crude steel production (ttpa)"] = proj_costs_filter["Nominal crude steel capacity (ttpa)"].copy()
    proj_costs_filter['Emissions (Gt)'] = proj_costs_filter['Estimated crude steel production (ttpa)'] * proj_costs_filter['EF'] / 1e6
    # plant view
    proj_plants = get_proj_costs_emissions(proj_costs=proj_costs_filter,
                                                glob_prod_iea=glob_prod_iea,
                                                glob_bu_prod=glob_bu_prod,
                                                start_year=start_year,
                                                end_year=end_year)
    

    
    # company projection
    proj_company = convert_plant2parent(proj_plants, 
                                        gspt2gspt_path=gspt2gspt_path,
                                        parent_group_map=parent_group_map)
    proj_company["Attributed emissions (ttpa)"] = proj_company['Emissions (Gt)'] * proj_company['Share'] * 1e6
    proj_company["Attributed production (ttpa)"] = proj_company['Estimated crude steel production (ttpa)'] * proj_company['Share']
    proj_company = proj_company.groupby(["year", "Group"]).agg({"Attributed production (ttpa)":  "sum",
                                                                    "Attributed emissions (ttpa)": "sum"}).reset_index()
    return proj_company, proj_plants


def get_proj_BU_constant_ms(gspt: pd.DataFrame, 
                        glob_prod_iea: pd.DataFrame, 
                        glob_bu_prod: float,
                        glob_capa, 
                        market_share: pd.DataFrame, 
                        parent_group_map_path: Path,
                        gspt2refi_map: dict,
                        gspt2gspt_path,
                        EF: "EmissionFactor",
                        method: str,
                        start_year: int,
                        end_year: int):
    """Project bottom-up emissions using different hypotheses on the evolution of company utilsation rates.

    Args:
        op_plants_ts (pd.DataFrame): operating plants for a given year (use GSPT method "get_operating_plants" 
        and indicate relevant years for projection)
        glob_prod_iea (pd.DataFrame): global steel production projections from IEA scenario (e.g. NZE, STEPS, APS)
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
    
    glob_prod_iea.loc[glob_prod_iea["Year"] == 2022, "Industrial production (Mt)"] = 1885.738
    interpolated_glob_prod = interpolate_glob_prod(glob_prod_iea=glob_prod_iea,
                                                   proj_start_year=start_year,
                                                   end_year=end_year)
    
    
    # create market share time series for relevant projection years
    market_share_ts = get_market_share_ts(market_share=market_share, start_year=start_year, end_year=end_year)
    

    # merge global projected production and market share to compute company level projected production
    proj_feats = pd.merge(market_share_ts, interpolated_glob_prod, on="Year", how='left')
    proj_feats["Company production (Mt)"] = proj_feats['Global production (Mt)'] * proj_feats['market_share']
    proj_feats['year'] = proj_feats['Year'].copy()
    
    # Aggregate plant capacity based on group company mapping
    op_plants_ts = gspt.get_operating_plants(start_year=start_year, end_year=end_year, melt=True)
    op_plants_ts['Group'] = get_group_col(gspt_parent_col=op_plants_ts['Parent'].copy(),
                                          gspt2gspt_path=gspt2gspt_path,
                                          parent_group_map=parent_group_map)
    op_plants_ts['Attributed crude steel capacity (ttpa)'] = op_plants_ts['Nominal crude steel capacity (ttpa)'] * op_plants_ts['Share']
    
    # Baseline projected group company capacity
    proj_group_capa = op_plants_ts.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
    
    if method == "company_UR":        
        # plant view
        op_plants_ts = gspt.get_operating_plants(start_year=start_year, end_year=end_year, melt=False)
        # CAPACITY CONSTRAINT
        op_plants_ts, remove_plants = shutdown_plants_hyp01(plants=op_plants_ts, glob_capa=glob_capa, start_year=start_year, end_year=end_year)
        
        # company view
        op_parents_ts = convert_plant2parent(op_plants_ts, 
                                             gspt2gspt_path=gspt2gspt_path,
                                             parent_group_map=parent_group_map)
        op_parents_ts['Attributed crude steel capacity (ttpa)'] = op_parents_ts['Nominal crude steel capacity (ttpa)'] * op_parents_ts['Share']
    
        # COMPANY UR
        proj_group_capa = op_parents_ts.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
        proj_plants_UR, _ = get_projected_UR_1(proj_feats=proj_feats, proj_group_capa=proj_group_capa, op_plants_ts=op_parents_ts)
        
    elif method == "company_UR_fuel_switch":
        # CALCULATE ELECTRIC CAPACITY TO SUBTRACT
        # Get all steel plants and map parent companies to respective group company
        melted_steel_plants = gspt.get_steel_dataset(melt=True)
        melted_steel_plants['Group'] = melted_steel_plants['Parent'].replace(gspt2gspt)
        melted_steel_plants['Group'] = melted_steel_plants['Parent'].replace(parent_group_map)
        # Get additions in electric capacity
        start_year, end_year = 2022, 2030
        delta_elec_capa = get_electric_capa_delta(melted_steel_plants=melted_steel_plants,
                                                 start_year=start_year,
                                                 end_year=end_year)
        
        # UPDATE COMPANY CAPACITY FOR EACH TECHNOLOGY (e.g. fuel switching)
        # plant view: close bf plants to satifsy global constraint
        op_plants_ts = gspt.get_operating_plants(start_year=start_year, end_year=end_year, melt=False)
        op_plants_ts, remove_plants = shutdown_plants_hyp01(plants=op_plants_ts, glob_capa=glob_capa, start_year=start_year, end_year=end_year)
        
        # company view
        op_parents_ts = convert_plant2parent(op_plants_ts, 
                                             gspt2gspt_path=gspt2gspt_path,
                                             parent_group_map=parent_group_map)
        op_parents_ts['Attributed crude steel capacity (ttpa)'] = op_parents_ts['Nominal crude steel capacity (ttpa)'] * op_parents_ts['Share']
    
        proj_group_capa = op_parents_ts.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
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
        proj_group['UR'] = proj_group['UR'].clip(upper=1.0)
        # Impute missing UR with average for given year
        mean_UR = proj_group.groupby("year")["UR"].mean().to_frame("mean_UR").reset_index()
        
        proj_group = pd.merge(proj_group,
                                mean_UR,
                                on=["year"],
                                how='left')
        proj_group["UR"] = proj_group["UR"].fillna(proj_group['mean_UR'])
                        
        # Impute UR estimated at the company level to each individual plant
        op_parents_ts = convert_plant2parent(proj_plants=op_plants_ts, 
                                             gspt2gspt_path=gspt2gspt_path,
                                             parent_group_map=parent_group_map)
        
        # projected parents
        proj_parents = pd.merge(op_parents_ts, 
                               proj_group.loc[:, ["Group", "year", "UR"]], 
                               on=["Group", "year"], 
                               how="left"
                               )
        proj_parents['UR'] = proj_parents['UR'].fillna(1.0)
                
        # a plant's utilisation rate is defined as the weighted average of company utilisation rates
        # weighted by ownership share in the plant
        proj_parents['UR_share'] = proj_parents['UR'] * proj_parents['Share']
        # projected utilisation rate for each plant
        proj_plants_UR = proj_parents.groupby(["Plant ID", "year"])["UR_share"].sum().to_frame('UR').reset_index()
                        
    elif method == "techno_UR": 
        # plant view
        op_plants_ts = gspt.get_operating_plants(start_year=start_year, end_year=end_year, melt=False)
        op_plants_ts, remove_plants = shutdown_plants_hyp01(plants=op_plants_ts, glob_capa=glob_capa, start_year=start_year, end_year=end_year)
        
        # company view
        op_parents_ts = convert_plant2parent(op_plants_ts, gspt2gspt_path=gspt2gspt_path, parent_group_map=parent_group_map)
        op_parents_ts['Attributed crude steel capacity (ttpa)'] = op_parents_ts['Nominal crude steel capacity (ttpa)'] * op_parents_ts['Share']
    
        proj_group_capa = op_parents_ts.groupby(["Group", "Main production process", "year"])['Attributed crude steel capacity (ttpa)'].sum().reset_index()
        proj_plants_UR, _, _ = get_projected_UR_3(proj_feats=proj_feats, gspt=gspt, 
                                                  op_plants_ts=op_parents_ts, 
                                                  parent_group_map=parent_group_map, 
                                                  proj_group_capa=proj_group_capa)     
    else:
        raise Exception
    
    # project plant production: merge the plant level UR with projected capacity
    proj_plants_prod = pd.merge(op_plants_ts, 
                                proj_plants_UR, 
                                on=['Plant ID', 'year'],
                                how='left')
    assert proj_plants_prod['UR'].notna().all()
    proj_plants_prod['Estimated crude steel production (ttpa)'] = proj_plants_prod['UR'] * proj_plants_prod["Nominal crude steel capacity (ttpa)"]
    
    agg_prod = proj_plants_prod.groupby("year")['Estimated crude steel production (ttpa)'].sum().reset_index()
    
    # compare bu prod with scenario prod
    for year in agg_prod['year'].tolist():
        # aggregate BU production for a given year
        bu_year_prod = agg_prod.loc[agg_prod['year'] == year, "Estimated crude steel production (ttpa)"].iloc[0]
        # scenario production for a given year
        interpolated_glob_prod['production'] = interpolated_glob_prod['Global production (Mt)'] * 1E3
        iea_year_prod = interpolated_glob_prod.loc[interpolated_glob_prod['Year'] == year, "production"].iloc[0]

        # ADJUST UR in order for bottom-up production to match scenario production
        # calculate adjustment factor
        alpha = iea_year_prod / bu_year_prod
        # Discount old UR for each plant using above alpha
        proj_plants_prod.loc[proj_plants_prod['year'] == year, 'UR'] = alpha * proj_plants_prod.loc[proj_plants_prod['year'] == year,'UR']
        
    # end adjustment
    
    # recalculate production
    proj_plants_prod['Estimated crude steel production (ttpa)'] = proj_plants_prod['UR'] * proj_plants_prod["Nominal crude steel capacity (ttpa)"]
    
    # Add emission factors
    if "Techno_map" in proj_plants_prod.columns:
        techno_col = "Techno_map"
    else:
        techno_col = "Main production process"
    proj_plants_prod = EF.match_emissions_factors(plants=proj_plants_prod, 
                                                  techno_col=techno_col,
                                                  level="country",
                                                  source='huizhong',
                                                  proj=True) # TODO: old sci
    # Share of plant-level emissions attributed to owner with respect to their stake in the plant
    proj_plants_prod['Estimated emissions (ttpa)'] = proj_plants_prod["Estimated crude steel production (ttpa)"] * proj_plants_prod['EF'] 
    
    # shutdown plants based on global capacity
    # look at removed_plants for debug
    final_plants, removed_plants = shutdown_plants_hyp01(plants=proj_plants_prod, glob_capa=glob_capa, start_year=start_year, end_year=end_year)
    
    # Company bottom-up emissions
    proj_company = convert_plant2parent(proj_plants=final_plants, gspt2gspt_path=gspt2gspt_path, parent_group_map=parent_group_map)
    
    proj_company["Attributed crude steel capacity (ttpa)"] = proj_company['Nominal crude steel capacity (ttpa)'] * proj_company['Share']
    proj_company["Attributed production (ttpa)"] = proj_company["Estimated crude steel production (ttpa)"] * proj_company['Share']
    proj_company['Attributed emissions (ttpa)'] = proj_company['Attributed production (ttpa)'] * proj_company['EF']

        
    proj_company = proj_company.groupby(["Group", "year"]).agg({'Attributed emissions (ttpa)': "sum",
                                                                    "Attributed production (ttpa)": 'sum',
                                                                    "Attributed crude steel capacity (ttpa)": "sum"}).reset_index()
    
    if method == "company_UR":
        proj_plants = final_plants.copy()
    else:
        proj_plants = final_plants.groupby(['Plant ID', "Region", "Country", "year"]).agg({"Estimated crude steel production (ttpa)": "sum",
                                                                                            "Estimated emissions (ttpa)": "sum"}).reset_index()
    proj_plants['Emissions (Gt)'] = proj_plants["Estimated emissions (ttpa)"] / 1e6
    
    # 
    return proj_company, proj_plants


def get_proj_BU_constant_UR(gspt, EF, start_year, end_year, gspt2gspt_path, parent_group_map, glob_capa, glob_prod, base_bu_prod: float):
    """Return projected emissions where utilisation rates from 2021 are held constant
    for future plants. Granularity: (technology, country/region)

    Args:
        gspt (GSPTDataset): Global Steel Plant Tracker object
        EF (EmissionFactors): Emission  Factors object
        glob_capa (pd.DataFrame): historical and projected global capacity. Source: OECD
        base_bu_prod (float): 2022 bottom-up prod

    Returns:
        _type_: _description_
    """
    op_plants = gspt.get_operating_plants(start_year=start_year,
                            end_year=end_year)
    
    # IMPUTE UR
    raw_data_dir = gspt.data_path.parent
    prod_dir = raw_data_dir / "production"
    global_capa_path = raw_data_dir / "capacity" / "STI_STEEL_MAKINGCAPACITY_23112023172621215.csv"
    adj_factor_path = raw_data_dir / "adj_factor.xlsx"
    all_country_ur = get_all_country_ur(start_year=2022,
                                        end_year=2022,
                                        prod_dir=prod_dir,
                                        global_capa_path=global_capa_path)
    # Adjust utilisation rates to match global production
    adj_factor = pd.read_excel(adj_factor_path)
    all_country_ur = pd.merge(all_country_ur,
                              adj_factor,on="year",
                              how="left")
    all_country_ur["country UR"] = all_country_ur["country UR"] * all_country_ur['adj_factor']
    # Enforce constraint that utilisation rates <= 100%
    all_country_ur["country UR"] = all_country_ur["country UR"].clip(upper=1.0)
    
    # (future) plants located in countries with missing UR in 2022
    # are imputed the median UR
    median_country_ur = all_country_ur['country UR'].median()
    mergedf = pd.merge(op_plants,
                        all_country_ur[["country", "country UR"]],
                        left_on=["Country"],
                        right_on=["country"], 
                        how="left")
    # unavailable UR for Hong Kong and Zimbabwe
    # assign median UR for projections
    # assuming future plants will operate at nonzero UR
    mergedf = mergedf.rename(columns={"country UR": "UR crude steel"})
    mergedf["UR crude steel"] = mergedf["UR crude steel"].replace(0, np.nan)    
    mergedf["UR crude steel"] = mergedf["UR crude steel"].fillna(median_country_ur)
        
    assert mergedf["UR crude steel"].between(0,1).all()
    
    # impute emission factors
    mergedf = EF.match_emissions_factors(mergedf, 
                                level="country", 
                                source="huizhong", # TODO: change for huizhong?
                                techno_col="Main production process",
                                proj=True)  
    
    mergedf['Estimated crude steel production (ttpa)'] = mergedf["Nominal crude steel capacity (ttpa)"] * mergedf["UR crude steel"]

    
    # 1. CAPACITY CONSTRAINT
    # shutdown plants based on global capacity
    # look at removed_plants for debug
    final_plants, removed_plants = shutdown_plants_hyp01(plants=mergedf, glob_capa=glob_capa, start_year=start_year, end_year=end_year)
    
    # 2. PRODUCTION CONSTRAINT
    agg_prod = final_plants.groupby("year")['Estimated crude steel production (ttpa)'].sum().reset_index()
    
    # yearly production values based on 2022 and 2030 values
    iea_prod = pd.DataFrame({"year": range(2022, 2031)})
    #iea_prod.loc[iea_prod["year"] == 2022, "production"] = glob_prod.loc[glob_prod["Year"] == 2022, "Industrial production (Mt)"].iloc[0] * 1E3
    iea_prod.loc[iea_prod["year"] == 2022, "production"] = 1885738 # WSA
    iea_prod.loc[iea_prod["year"] == 2030, "production"] = glob_prod.loc[glob_prod["Year"] == 2030, "Industrial production (Mt)"].iloc[0] * 1E3
    iea_prod['production'] = iea_prod['production'].interpolate(method="linear")
    
    # compare bu prod with scenario prod
    for year in agg_prod['year'].tolist():
        # aggregate BU production for a given year
        bu_year_prod = agg_prod.loc[agg_prod['year'] == year, "Estimated crude steel production (ttpa)"].iloc[0]
        # scenario production for a given year
        iea_year_prod = iea_prod.loc[iea_prod['year'] == year, "production"].iloc[0] 
        if bu_year_prod <= iea_year_prod:
            # no adjustment required
            pass
        else:
            # ADJUST UR in order for bottom-up production to match scenario production
            # calculate adjustment factor
            alpha = iea_year_prod / bu_year_prod
            # Discount old UR for each plant using above alpha
            final_plants.loc[final_plants['year'] == year, 'UR crude steel'] = alpha * final_plants.loc[final_plants['year'] == year,'UR crude steel']
            
    # Recalculate plant production accordingly
    final_plants['Estimated crude steel production (ttpa)'] = final_plants["Nominal crude steel capacity (ttpa)"] * final_plants["UR crude steel"]
            
    # calculate emissions
    final_plants['Emissions (Gt)'] = final_plants['Estimated crude steel production (ttpa)'] * final_plants['EF'] / 1e6
    
    # company projection
    proj_company = convert_plant2parent(final_plants, gspt2gspt_path=gspt2gspt_path, parent_group_map=parent_group_map)
    proj_company['Attributed production (ttpa)'] = proj_company['Estimated crude steel production (ttpa)'] * proj_company['Share']
    proj_company['Attributed emissions (ttpa)'] = proj_company['Emissions (Gt)'] * proj_company['Share'] * 1e6
    proj_company = proj_company.groupby(["year", "Group"]).agg({"Attributed production (ttpa)":  "sum",
                                                                "Attributed emissions (ttpa)":  "sum"}).reset_index()
    return proj_company, final_plants

def shutdown_plants_hyp01(plants, glob_capa, start_year, end_year):
    """When global capacity constraint is violated, shutdown oldest BF plants.
    This function is used in hypothesis 0 and 1, where utilisation rates are held constant
    based on historical values.

    Args:
        plants (_type_): projected operating plants with no constraint on global capacity
        glob_capa (_type_): historical and projected global capacity
        
    Returns:
        final_plants (pd.DataFrame): set of operating plants that fulfill global capacity constraint.
        removed_plants (pd.DataFrame): plants that were removed to fulfill global capacity constraint.
    """
    # aggregate bottom up capacity (with no constraint)
    agg_capa = plants.groupby("year").agg({"Nominal crude steel capacity (ttpa)":"sum"}).reset_index()
    
    # set of projected operating plants which will satisfy global capacity constraint
    final_plants = pd.DataFrame()
    removed_plants = pd.DataFrame()
    
    for year in range(start_year, end_year+1):
        # capacity in thousand of tonnes
        bu_capa = float(agg_capa.loc[agg_capa['year'] == year, "Nominal crude steel capacity (ttpa)"])
        oecd_capa = float(glob_capa.loc[glob_capa['year'] == year, "Global capacity (Mt)"]) * 1e3
        # operating plants for current year
        op_plants = plants.loc[plants["year"] == year]
        
        if bu_capa <= oecd_capa:
            # bottom up capacity does not violate global capacity
            # so keep operating plants as is
            final_plants = pd.concat([final_plants, op_plants], ignore_index=True, axis=0)
        else:
            # get BF plants
            bf_plants = op_plants.loc[op_plants['Main production process'] == "integrated (BF)"]
            not_bf_plants = op_plants.loc[op_plants['Main production process'] != "integrated (BF)"]
            
            # sort by start year
            bf_plants = bf_plants.sort_values(by="Start year", ascending=True)
            
            # while bu capa >= oecd capa: remove old bf plants
            i = 1
            while bu_capa > oecd_capa:
                selected_bf = bf_plants.iloc[i:]
                # merge not BF with selected BF plants
                selected_plants = pd.concat([selected_bf, not_bf_plants], axis=0, ignore_index=True)
                # update bu capacity
                bu_capa = selected_plants['Nominal crude steel capacity (ttpa)'].sum()
                i+=1

            # remove plants that shutdown for future years
            plants_to_remove = bf_plants.loc[bf_plants.index[:i], "Plant ID"]
            plants = plants.loc[~plants["Plant ID"].isin(plants_to_remove)]
            removed_df = bf_plants.loc[bf_plants['Plant ID'].isin(plants_to_remove)]
            removed_plants = pd.concat([removed_plants, removed_df], ignore_index=True, axis=0)
            
            # concat final operating plants 
            final_plants = pd.concat([final_plants, selected_plants], ignore_index=True, axis=0)
    
    out_agg_capa = final_plants.groupby("year").agg({"Nominal crude steel capacity (ttpa)": "sum"}).reset_index()
    out_agg_capa = pd.merge(out_agg_capa,
                            glob_capa,
                            on="year",
                            how='left')
    # ensure updated future operating plants respect global capacity constraint
    assert (out_agg_capa['Nominal crude steel capacity (ttpa)'] <= out_agg_capa["Global capacity (ttpa)"]).all()
    return final_plants, removed_plants

def get_glob_capa(oecd_path: Path, 
                  start_sample_year: int, 
                  scenario: str,
                  method: str = "backward"):
    """Get historical and projected global capacity.
    Source: OECD

    Args:
        oecd_path (Path): "capacity/STI_STEEL_MAKINGCAPACITY_08092023132636861.csv"
        start_sample_year (int): minimum year of sampling period for linear model (e.g. 2015)
        scenario (str): scenario from which to pick projected production growth rate
        method (str): extrapolate using backward or forward looking data
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    oecd = pd.read_csv(oecd_path)
    wld = oecd.loc[oecd['COUNTRY'] == "WLD"]
    cols_to_keep = ["Year", "Economy", "Indicator", "Value"]
    wld = wld.loc[:, cols_to_keep]
    wld["Type"] = "historical"

    assert wld['Year'].nunique() == len(wld)
    assert wld['Indicator'].nunique() == 1
    # consider 2015-2022 as representative period for extrapolation
    mask = wld['Year'] >= start_sample_year
    X = wld.loc[mask, 'Year'].values.reshape(-1, 1)
    y = wld.loc[mask, 'Value']
    
    # Create a pipeline with StandardScaler and LinearRegression
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('regressor', LinearRegression())  # Linear Regression model
    ])
    pipe.fit(X, y)
    
    # Project global capacity until 2030
    max_histo_year = max(wld['Year'])
    X_pred = np.array(range(max_histo_year+1, 2031)).reshape(-1,1)
    y_pred=pipe.predict(X_pred)
    
    if method == "backward":
        proj_capa = pd.DataFrame({"Year": X_pred.squeeze(),
                            "Value": y_pred,
                            "Economy": "World"})
        
        # Merge historical and projected values
        proj_capa['Type'] = "linear_proj"
        glob_capa = pd.concat([wld, proj_capa], axis=0, ignore_index=True)
        
    elif method == "forward":
        # Forward looking projection
        # CAAGR from WEO 2023 between 2022 and 2030
        caagr_dict = {"NZE": 0.006, "STEPS": 0.013, "APS": 0.01} 
        caagr = caagr_dict[scenario]
        proj_capa_fwd = pd.DataFrame({"Year": X_pred.squeeze(),
                            "Value": float(wld.loc[wld['Year'] == 2022, "Value"]),
                            "CAAGR": [(1+caagr)**(i+1) for i in range(len(X_pred))],
                            "Economy": "World"})
        proj_capa_fwd["Value"] = proj_capa_fwd["Value"] * proj_capa_fwd["CAAGR"]
        proj_capa_fwd['Type'] = scenario.lower() + "_prod_caagr"
        glob_capa = pd.concat([wld, proj_capa_fwd], axis=0, ignore_index=True)
        
    glob_capa = glob_capa.drop(columns=["Indicator"])
    glob_capa = glob_capa.rename(columns={"Value": "Global capacity (Mt)",
                                          "Year": "year"})
    
    # add capacity in ttpa
    glob_capa["Global capacity (ttpa)"] = glob_capa["Global capacity (Mt)"] * 1e3
    return glob_capa
    
def get_benchmark_emissions(x0_value: float, x0_year: int, slope_data: pd.DataFrame):
    """Extrapolate sectoral emission level (past and future) based on decarbonisation trends
    calculated from different scenarios (e.g. NZE, APS, STEPS).
    
    In other words, given a base value of emissions (x0_value) for a given year (x0_year),
    and a rate of change over a given period (x0_year, proj_year=2030), this function returns an
    emissions trajectory over (x0_year, proj_year=2030).

    Args:
        x0_value (float): initial historical emissions (GtCO2) 
        x0_year (int): initial year
        slope_data (pd.DataFrame): emissions levels from scenarios, from which a trend is calculated

    Returns:
        _type_: _description_
    """
    # Scenario decarbonisation slope: get last value and 2030 value
    max_year = 2030
    min_year = slope_data["year"].min()
    emissions30 = int(slope_data.loc[slope_data["year"] == max_year, 'Emissions (Mt)'])
    emissions_histo = int(slope_data.loc[slope_data["year"] == min_year, 'Emissions (Mt)'])
    
    # annualised decarbonisation rate
    r = (emissions30/emissions_histo) ** (1./(max_year - min_year)) - 1
    # baseline emissions of the industry based on historical values and projected annualised decarbonisation rate from NZE
    base_emissions = pd.DataFrame({"year": list(range(x0_year,2031)),
                                "factor": [(1+r) ** (i-x0_year) for i in range(x0_year, 2031)]
                                })
    base_emissions["Emissions (Gt)"] = x0_value * base_emissions['factor']
    return base_emissions


def get_iea_emissions(scenario: str) -> pd.DataFrame:
    """Get annualised trajectory of sectoral emissions
    based on IEA projections.

    Args:
        scenario (str): scenario name from the World Energy Outlook report.

    Returns:
        pd.DataFrame: emissions trajectiory
    """
    # Historical and projected emission values per scenario
    # Base year is 2022, projected year is 2030
    scenario_CO2 = {"STEPS": (2623, 2685),
                    "APS": (2623, 2474),
                    "NZE": (2623, 2118)}
    
    base_emissions = scenario_CO2[scenario][0]
    proj_emissions = scenario_CO2[scenario][1]
    
    emissions = pd.DataFrame({"Emissions (Mt)": np.nan,
                                  "year": list(range(2022, 2031))})
    emissions.loc[emissions['year'] == 2022, "Emissions (Mt)"] = base_emissions
    emissions.loc[emissions['year'] == 2030, "Emissions (Mt)"] = proj_emissions
    
    # Linearly interpolate values between base and projected years
    emissions["Emissions (Mt)"] = emissions["Emissions (Mt)"].interpolate(method="linear")
    emissions["Emissions (Gt)"] = emissions["Emissions (Mt)"] / 1e3
    return emissions


def get_historical_bu_emissions(gspt: GSPTDataset,
                                EF: EmissionFactors,
                                prod_dir: Path,
                                global_capa_path: Path,
                                global_prod_wsa_path: Path,
                                start_year: int,
                                end_year: int,
                                energy_mix_path: Path, 
                                parent_group_map: dict,
                                model: "sklearn.Pipeline",
                                gspt2gspt_path: Path,
                                ):
    assert start_year >= 2018
    
    wsa_prod = pd.read_excel(global_prod_wsa_path)
    
    all_country_ur = get_all_country_ur(start_year=start_year,
                                    end_year=end_year,
                                    prod_dir=prod_dir,
                                    global_capa_path=global_capa_path)
    
    plants = gspt.get_operating_plants(start_year=start_year, end_year=end_year)
    prod = pd.merge(plants,
            all_country_ur[["year", "country", "country UR"]],
            left_on=['year', "Country"],
            right_on=["year", "country"],
            how='left')
    assert len(plants) == len(prod)
    
    # GSPT Capacity/Production is < 100%
    # Adjust UR to match true global production
    prod["Estimated crude steel production (ttpa)"] = prod['Nominal crude steel capacity (ttpa)'] * prod['country UR']
    agg_bu_prod = prod.groupby("year")['Estimated crude steel production (ttpa)'].sum().reset_index()
    agg_bu_prod = pd.merge(agg_bu_prod,
                           wsa_prod[["year", "Crude steel production (ttpa)"]],
                           on="year",
                           how='left')
    # BU_prod := BU_prod * adj_factor 
    agg_bu_prod['adj_factor'] = agg_bu_prod["Crude steel production (ttpa)"] / agg_bu_prod["Estimated crude steel production (ttpa)"]
    prod = pd.merge(prod,
                    agg_bu_prod[["year", "adj_factor"]],
                    on="year",
                    how="left")
    prod["country UR"] = prod["country UR"] * prod["adj_factor"]
    prod["country UR"] = prod["country UR"].clip(upper=1)
    prod["Estimated crude steel production (ttpa)"] = prod['Nominal crude steel capacity (ttpa)'] * prod['country UR']
    
    prod = EF.match_emissions_factors(prod, level="country", source="sci", techno_col="Main production process")

    prod["Emissions"] = prod["Estimated crude steel production (ttpa)"] * prod['EF']
    agg_raw_BU = prod.groupby("year").agg({"Emissions":"sum"}).reset_index()
    agg_raw_BU["Emissions"] *= 1e3

    # add electricity intensity
    prod = get_electricity_intensity(prod, energy_mix_path)
    
    prod["EAF_country_elec_int"] = (prod["Main production process"] == "electric") * prod["country_elec_int"]
    
    # group mapping
    group_prod = convert_plant2parent(prod, gspt2gspt_path, parent_group_map)
    # features
    group_prod["Attributed emissions (ttpa)"] = group_prod['Emissions'] * group_prod["Share"]
    group_prod["Attributed emissions (ttpa)"] = group_prod["Attributed emissions (ttpa)"].clip(lower=1E-15)
    group_prod["Attributed production (ttpa)"] = group_prod["Estimated crude steel production (ttpa)"] * group_prod["Share"]
    group_prod["Attributed capacity (ttpa)"] = group_prod['Nominal crude steel capacity (ttpa)'] * group_prod["Share"]
    wm_capa = lambda x: np.average(x, weights=group_prod.loc[x.index, "Attributed capacity (ttpa)"])
    group = group_prod.groupby(["year", "Group"]).agg({"Attributed emissions (ttpa)": "sum",
                                                       "Attributed production (ttpa)": "sum",
                                                    "country_elec_int": wm_capa,
                                                    "EF": wm_capa,
                                                    "EAF_country_elec_int": wm_capa,
                                                    "Attributed capacity (ttpa)": "sum"})
    group["Attributed emissions"] = group["Attributed emissions (ttpa)"] * 1e3
    group["Attributed production"] = group["Attributed production (ttpa)"] * 1e3
    group['log_Attributed emissions'] = np.log(group["Attributed emissions"])
    group['log_Attributed production'] = np.log(group["Attributed production"])
    group["elec_int_capa"] = group["country_elec_int"].copy()
    group["EF_capa"] = group["EF"].copy()
    group["EAF_elec_int_capa"] = group["EAF_country_elec_int"].copy()
    group['interaction'] = group['log_Attributed emissions'] * group["elec_int_capa"] 
    group['ratio'] = group['log_Attributed emissions'] / group["elec_int_capa"]
    
    # features seen during fit
    try:
        features = model[0].get_feature_names_out()
    except:
        features = model[0].feature_names_in_
    # log emissions model
    if "log_Attributed emissions" in features:
        group["log_BU emissions"] = model.predict(group[features])
        group["BU emissions"] = np.exp(group["log_BU emissions"])
    else:
        group["BU emissions"] = model.predict(group[features])
    agg_BU = group.groupby("year").agg({"BU emissions": "sum",
                                        "Attributed emissions": "sum",
                                        "Attributed production": "sum"}).reset_index()
    agg_BU["Intensity"] = agg_BU["BU emissions"] / agg_BU["Attributed production"]
    agg_BU = agg_BU.rename(columns={"Attributed emissions": "raw BU emissions"})
    return agg_BU, group.reset_index(), agg_bu_prod, prod


def get_historical_bu_emissions_old(gspt: GSPTDataset, EF: EmissionFactors, etp19: float, entity: str, start_year: int,
                                end_year: int = None):
    """
    Calculate historical BU emissions for the Iron & Steel sector based on production data and emission factors.

    This function retrieves historical production data between the specified start and end years, calculates emissions
    based on provided emission factors, and aggregates emissions by year.

    Args:
        gspt: The object or function for retrieving production data.
        EF: The object or function for matching emission factors.
        etp19:  true emi19 in GtCO2
        impute_prod (str): production imputation method (Global, Region, Country)

    Returns:
        bu_emissions (pandas.DataFrame): DataFrame containing historical emissions by year.
        all_prod (pandas.DataFrame): DataFrame containing all production and emissions data.
    """
    assert start_year >= 2019
    # get historical production values beween 2019-2022
    all_prod = []
    iter_years = range(start_year, end_year+1) if end_year is not None else [start_year] 
    for year in iter_years:
        prod = gspt.get_estimated_prod(year=year, entity=entity, impute_prod="Country", average="micro")
        prod['year'] = year
        all_prod.append(prod)

    all_prod = pd.concat(all_prod, ignore_index=True, axis=0)
    # impute emission factors
    all_prod = EF.match_emissions_factors(all_prod, 
                                level="country", 
                                source="sci", 
                                techno_col="Main production process")
    
    all_prod.loc[all_prod["Main production process"] == "integrated (DRI)", "EF"] = 1.65
    
    # calculate emissions
    all_prod['Emissions (Gt)'] = all_prod['Estimated crude steel production (ttpa)'] * all_prod['EF'] / 1e6

    bu_emissions = all_prod.groupby("year")['Emissions (Gt)'].sum().reset_index()
    
    # Calculate adjusted emissions based on WRI true emissions
    bu19 = float(bu_emissions.loc[bu_emissions['year'] == 2019, "Emissions (Gt)"])
    alpha = etp19 / bu19
    bu_emissions["Adjusted Emissions (Gt)"] = bu_emissions['Emissions (Gt)'] * alpha
    return bu_emissions, all_prod #for checks

def save_plots(fig_vert, fig_hor, fig_prod, plots_dir: Path, EAF_decarb: bool, iea_level: str, model_name: str):
    if EAF_decarb:
        fig_vert_name_pdf = plots_dir / f"sectoral_proj_{iea_level}_{model_name}_decarb_vertical.pdf"
        fig_vert_name_png = plots_dir / f"sectoral_proj_{iea_level}_{model_name}_decarb_vertical.png"
        fig_hor_name_pdf = plots_dir/ f"sectoral_proj_{iea_level}_{model_name}_decarb_horizontal.pdf"
        fig_hor_name_png = plots_dir/ f"sectoral_proj_{iea_level}_{model_name}_decarb_horizontal.png"
    else:
        fig_vert_name_pdf = plots_dir / f"sectoral_proj_{iea_level}_{model_name}_vertical.pdf"
        fig_vert_name_png = plots_dir / f"sectoral_proj_{iea_level}_{model_name}_vertical.png"
        fig_hor_name_pdf = plots_dir / f"sectoral_proj_{iea_level}_{model_name}_horizontal.pdf"
        fig_hor_name_png = plots_dir / f"sectoral_proj_{iea_level}_{model_name}_horizontal.png"
    for fig_vert_name in [fig_vert_name_pdf, fig_vert_name_png]:
        fig_vert.savefig(fig_vert_name, dpi=600, bbox_inches="tight")
    for fig_hor_name in [fig_hor_name_pdf, fig_hor_name_png]:
        fig_hor.savefig(fig_hor_name, dpi=600, bbox_inches="tight")
    fig_prod.savefig(plots_dir / f"sectoral_production.png", dpi=600, bbox_inches="tight")

def save_projections(data, save_dir, EAF_decarb, iea_level, model_name):
    if not EAF_decarb:
        with open(save_dir / f"bu_proj_{iea_level}_{model_name}.pkl", "wb") as pickle_file:
            pickle.dump(data, pickle_file)
    else:
        with open(save_dir / f"bu_proj_{iea_level}_{model_name}_EAF_decarb.pkl", "wb") as pickle_file:
            pickle.dump(data, pickle_file)

def get_glob_prod_capa(iea_prod_dir: Path, oecd_capa_fpath: Path) -> Tuple[dict, dict]:
    # Projected global production based on IEA scenarios
    glob_prod = {}
    for s in ["NZE", "STEPS", "APS"]:
        glob_prod[s] = pd.read_excel(iea_prod_dir / f"steel_prod_{s}.xlsx")
                    
    # historical and projected global capacity based on OECD data
    glob_capa = {scenario: get_glob_capa(oecd_path=oecd_capa_fpath,
                                start_sample_year=2015,
                                method="forward",
                                scenario=scenario)
                for scenario in ["NZE", "APS", "STEPS"]}
    return glob_prod, glob_capa
    
