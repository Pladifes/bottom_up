import typer
from pathlib import Path
import toml
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from config import load_config
import copy

from src.projections_.projections import get_historical_bu_emissions, get_benchmark_emissions, get_iea_emissions, save_plots
from src.projections_.targets import preprocess_targets, get_stated_traj, merge_all_traj, get_sub_traj_melt, get_bu_vs_stated_fig
from src.datasets.RefinitivDataset import RefinitivDataset
from src.datasets.GSPTDataset import GSPTDataset
from src.datasets.EmissionFactors import EmissionFactors
from src.datasets.utils import fill_missing_CF12, convert_plant2parent
from src.feature_eng.company_feature_engineering import get_X_y, get_reported_emissions
from src.projections_.projections import get_ghg_trajectory_plot, save_projections, get_glob_prod_capa
from src.projections_.BUProjector import get_market_share

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def main(config_path: Path = typer.Argument(Path("./config.toml"), help="Path to the TOML configuration file"),
         help="Path for saving results"):
    # Read config file
    config_file = load_config(config_path)
    typer.echo(config_file)
    
    params = config_file.params
    # Specify save path
    save_dir = params.save_dir
    save_file = save_dir / "BU_results"
    save_file.mkdir(parents=True, exist_ok=True) 
    typer.echo(f"Saving results to {save_file}")

    ## Inputs
    project_dir = params.project_dir
    historical_data = config_file.historical_data
    emission_factors = config_file.emission_factors
    mappings = config_file.mappings
    scenarios = config_file.scenarios
    projected_data = config_file.projected_data

    raw_data_dir = project_dir / "data" / "raw"
    db_path = params.steel_db
    world_fpath = raw_data_dir / scenarios.iea_world
    regions_fpath = raw_data_dir / scenarios.iea_regions
    gspt2gspt_path = raw_data_dir / mappings.gspt2gspt
    refi2gspt_path = raw_data_dir / mappings.refi2gspt
    gspt2iea_countries_path = raw_data_dir / mappings.gspt2iea_countries
    gspt2cdp_map_path = raw_data_dir / mappings.gspt2cdp
    parent_group_map_path = raw_data_dir / mappings.parent_group_map
    energy_mix_path = raw_data_dir / emission_factors.energy_mix
    iea_prod_dir = raw_data_dir / historical_data.macro.iea_prod
    wsa_prod_path = raw_data_dir / historical_data.macro.wsa_prod
    oecd_capa_fpath = raw_data_dir / historical_data.macro.oecd_capa
    prod_dir = raw_data_dir / "production"
    carbon_price_path = raw_data_dir / historical_data.macro.carbon_price
    refinitiv_path = raw_data_dir / historical_data.top_down.refinitiv
    cdp_path = project_dir / historical_data.top_down.cdp
    elec_int_nze_path = raw_data_dir / projected_data.electricity.elec_int_nze
    elec_int_aps_path = raw_data_dir / projected_data.electricity.elec_int_aps
    elec_int_steps_path = raw_data_dir / projected_data.electricity.elec_int_steps
    company_steel_prod_path = raw_data_dir / historical_data.micro.company_prod

    # Bottom-up models save path
    models_dir = save_file / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Plots save path 
    plots_dir = save_file / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)


    # [HISTORICAL DATA]
    # Asset-level dataset
    gspt = GSPTDataset(data_path=raw_data_dir / historical_data.asset_level_data,
            missing_years_path=raw_data_dir / historical_data.missing_years,
            gspt2gspt_path=gspt2gspt_path,
            parent_group_map_path=parent_group_map_path,
            version_year=2023) 
    ## Capacity column name
    ## Technology column name
    ## Country column name
    # Historical intensities
    EF = EmissionFactors(wsa_path=raw_data_dir / emission_factors.wsa,
                    jrc_22_path=raw_data_dir / emission_factors.jrc,
                    sci_path=raw_data_dir / emission_factors.sci,
                    EU_27_path=raw_data_dir / mappings.EU_27,
                    huizhong_path=raw_data_dir / emission_factors.huizhong,
                    )
    with open(gspt2gspt_path, "r") as f:
        gspt2gspt_map = json.load(f)
    with open(parent_group_map_path, "r") as f:
        parent_group_map = json.load(f)
    gspt2refi = pd.read_excel(refi2gspt_path)

    # Top-down emissions
    refi_class = RefinitivDataset(data_path=refinitiv_path)
    refinitiv = refi_class.get_preprocessed_data()
    refinitiv = fill_missing_CF12(refinitiv=refinitiv, refi2gspt_path=refi2gspt_path, gspt2gspt_map=gspt2gspt_map)
    cdp = pd.read_excel(cdp_path)
    gspt2cdp = pd.read_excel(gspt2cdp_map_path)

    # Bottom-up training set
    X_y = get_X_y(gspt=gspt,
            impute_prod="Global", # TODO: try region and global
            average="micro", #TODO replace with macro
            EF=EF,
            energy_mix_path=energy_mix_path,
            carbon_price_path=carbon_price_path,
            gspt2refi=gspt2refi,
            gspt2gspt_path=gspt2gspt_path,
            parent_group_map=parent_group_map,
            refinitiv=refinitiv,
            gspt2cdp=gspt2cdp,
            cdp=cdp)

    # Fit bottom-up model
    # TODO: add if else based on model_name in config
    X, y = X_y[["log_Attributed emissions"]], X_y[["log_max_CF12"]]
    model = Pipeline([("scaler", StandardScaler()),
                  ("regressor", LinearRegression())])
    model.fit(X,y)
    # Save bottom-up training data and model
    X.to_excel(models_dir / "X_train.xlsx")
    y.to_excel(models_dir / "y_train.xlsx")
    joblib.dump(model, models_dir / f"{params.model_name}.joblib")

    # [PROJECTED DATA]
    # Transition scenarios
    ## Production column
    company_steel_prod = pd.read_excel(company_steel_prod_path)
    # TODO: source: https://en.wikipedia.org/wiki/List_of_countries_by_steel_production
    histo_global_prod = pd.DataFrame({"year":[2019, 2020, 2021, 2022],
                                      "Crude steel production (Mt)": [1874.4, 1877.5, 1951.9, 1885]})

    # TODO: reprendre ici
    year = 2022
    market_share = get_market_share(gspt=gspt,
                                    db_path=db_path,
                          histo_global_prod=histo_global_prod,
                            company_steel_prod=company_steel_prod,
                            mapping=gspt2refi,
                            gspt2gspt_path=gspt2gspt_path,
                            parent_group_map=parent_group_map,
                            method="single_year",
                            year=year)
    
    ## Emissions column
    ## CO2 intensity of electricity generation column
    # CO₂ intensity of electricity generation (g CO₂ per kWh)
    elec_int_nze = pd.read_excel(elec_int_nze_path)
    elec_int_steps = pd.read_excel(elec_int_steps_path)
    elec_int_aps = pd.read_excel(elec_int_aps_path)
    # Stated targets data

    ## Outputs

    # Stated trajectories plots
    
    # Historical bottom-up 
    agg_bu_histo, group_bu_histo, adj_factor, prod = get_historical_bu_emissions(gspt=gspt,
                                EF=EF,
                                prod_dir=prod_dir,
                                global_capa_path=oecd_capa_fpath,
                                global_prod_wsa_path=wsa_prod_path,
                                start_year=2019,
                                end_year=2022,
                                energy_mix_path=energy_mix_path,
                                parent_group_map=parent_group_map,
                                model=model,
                                gspt2gspt_path=gspt2gspt_path)
    agg_bu_histo['Emissions (Gt)'] = agg_bu_histo['BU emissions'] / 1e9
    nze_emissions = get_iea_emissions(scenario="NZE")
    aps_emissions = get_iea_emissions(scenario="APS")
    steps_emissions = get_iea_emissions(scenario="STEPS")
    x0_year = 2022
    x0_value = float(agg_bu_histo.loc[agg_bu_histo["year"] == x0_year, "Emissions (Gt)"])
    base_nze_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=nze_emissions)
    base_aps_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=aps_emissions)
    base_steps_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=steps_emissions)

    # read bottom-up projections for all companies
    with open(raw_data_dir / "results_data" / f"bu_proj_{params.scenario_level}_{params.model_name}_EAF_decarb.pkl", "rb") as pickle_file:
        bu_proj = pd.read_pickle(pickle_file)
    # load stated targets
    targets_path = raw_data_dir / "targets/2030_unique_targets.xlsx"

    targets = preprocess_targets(targets_path=targets_path,
                                refinitiv=refinitiv,
                                gspt2gspt_path=gspt2gspt_path)
    targets = pd.merge(targets,
                   group_bu_histo.loc[group_bu_histo['year'] == 2022,
                                      ["Group", "BU emissions"]],
                   on="Group",
                   how='left')

    # Get stated trajectories
    stated_traj = get_stated_traj(targets=targets, emissions_col='BU emissions')
    scenarios =["NZE", "APS", "STEPS"]
    methods = ["constant_UR", "company_UR"]
    all_traj = merge_all_traj(stated_traj=stated_traj, 
                              bu_proj=bu_proj,
                              scenarios=scenarios, methods=methods)
    all_traj = all_traj.loc[all_traj["Group"] != "Ternium SA"]
    # PREPROCESS ALL TRAJECTORIES BEFORE PLOTTING
    # use 2022 value to fill projections to make plot continuous
    all_traj = all_traj.fillna(method="ffill", axis=1)
    # Define the data types for each column
    dtypes = {'year': int, 'Group': object}
    float_columns = all_traj.columns.difference(['year', 'Group'])
    dtypes.update({col: float for col in float_columns})
    # Convert columns to specified data types
    all_traj = all_traj.astype(dtypes)
    # create a stated emissions column for each scenario 
    # helps with data manipulation
    for scenario in scenarios:
        all_traj[f"{scenario}_Stated BU emissions"] = all_traj["Stated BU emissions"].copy()
    
    # three groups of companies
    companies_with_stated = list(stated_traj['Group'].unique())
    top5_companies = ["ArcelorMittal SA", "JSW Steel Ltd", "Nippon Steel Corp", "POSCO Holding Co.", "Shougang Group Co Ltd"]
    taxonomy_companies = ["Acerinox SA", "Aperam SA", "ArcelorMittal SA", "Danieli & C Officine Meccaniche SpA", "Outokumpu Oyj", "Salzgitter AG", "Tenaris SA", "thyssenkrupp AG", "voestalpine AG"]
    subsamples = {"stated": companies_with_stated, 
                  "top5": top5_companies, 
                  "taxo": taxonomy_companies}
    # Compute share of capacity in 2022
    plant_capa22 = gspt.get_operating_plants(start_year=2022)
    group_capa22 = convert_plant2parent(plant_capa22, gspt2gspt_path=gspt2gspt_path, parent_group_map=parent_group_map)
    group_capa22["Attributed capacity (ttpa)"] = group_capa22['Nominal crude steel capacity (ttpa)'] * group_capa22["Share"]
    group_capa22 = group_capa22.groupby(["Group", "year"])["Attributed capacity (ttpa)"].sum().reset_index()


    
    with open(plots_dir / 'capacity_share.txt', 'w') as file:
        pass
    for prefix, companies in subsamples.items():
        #total_capa = group_capa22["Attributed capacity (ttpa)"].sum()
        total_capa = 2459000
        # Capacity of the subsample
        sum_capa = group_capa22.loc[group_capa22["Group"].isin(companies), "Attributed capacity (ttpa)"].sum()
        capa_share = sum_capa / total_capa
        with open(plots_dir / 'capacity_share.txt', 'a') as file:
            file.write(f"{prefix} \n")
            file.write(f"share of capacity: {capa_share*100:.2f}% \n\n")
        
    # Compute share of emissions in 2022

    # plot
    subsamples_df = []
    sns.set_style("whitegrid")
    # Levels plot: bottom up vs stated
    for prefix, companies in subsamples.items():
        sub_traj_melt = get_sub_traj_melt(all_traj, companies)
        if prefix == "stated":
            title = "Stated emissions vs bottom-up emissions for all companies with stated targets in 2030"
        elif prefix == "top5":
            title = "Stated emissions vs bottom-up emissions for top 5 emitters with stated targets in 2030"
        elif prefix == "taxo":
            title = "Stated emissions vs bottom-up emissions for 9 companies with both taxonomy"\
            "and stated targets in 2030"
        
        fig = get_bu_vs_stated_fig(sub_traj_melt, group_bu_histo, companies)
        fig_path = plots_dir / f"{prefix}_single_stated_traj_scenarios_EAF_decarb.pdf"
        fig.savefig(fig_path, bbox_inches='tight', dpi=600)
    
        # Percent deviation plot: bottom-up vs stated
        bu_sub_traj = sub_traj_melt.loc[sub_traj_melt["Source"] != "Stated emissions"]
        stated_sub_traj = sub_traj_melt.loc[sub_traj_melt["Source"] == "Stated emissions"]
        pct_dev_traj = pd.merge(bu_sub_traj, 
                                stated_sub_traj, 
                                how="left",
                                on=["year", "Scenario"],
                                suffixes=("_bu", "_stated"))
        pct_dev_traj["pct_misalignment"] = 100 * (pct_dev_traj.Emissions_bu - pct_dev_traj.Emissions_stated) / pct_dev_traj.Emissions_stated
        pct_dev_traj["Subsample"] = prefix
        subsamples_df.append(pct_dev_traj)

    subsamples_df = pd.concat(subsamples_df, axis=0, ignore_index=True)
    legend_handles_labels = []
    fig, axs = plt.subplots(2,3, figsize=(12,6), sharey=True)
    for i, method in enumerate(list(subsamples_df['Source_bu'].unique())):
        for j, scenario in enumerate(scenarios):
            ax = axs[i, j]
            # TODO: plot for each scenario
            m1 = subsamples_df["Scenario"] == scenario
            m2 = subsamples_df["Source_bu"] == method
            sns.lineplot(subsamples_df.loc[m1 & m2],
                        x="year",
                        y="pct_misalignment",
                        hue="Subsample",
                        linestyle="--",
                        ax=ax)
            ax.set_title(scenario)
            # TODO: rename title based on method
            if "constant_UR" in method:
                mname = "country UR"
            elif "constant market share" in method:
                mname = "constant market share"
            ax.set_ylabel(f"% overshoot \n ({mname})")
                    # Collect legend handles and labels for the first subplot
            if (i == 0) & (j == 0):
                legend_handles_labels.extend(ax.get_legend_handles_labels())
            ax.get_legend().remove()

    title = "Relative deviation of bottom-up projections "\
    "from the stated emissions trajectory \n"\
    "for 3 subsets of companies with stated targets"
    fig.suptitle(title)
    # Create figure-level legend outside the subplots
    handles, labels = legend_handles_labels
    fig.legend(handles=handles, 
            labels=labels, 
            #bbox_to_anchor=(1.05, 0.5), 
            loc='outside lower center', 
            borderaxespad=0., 
            frameon=False,
            ncol=3,
            )
    fig.tight_layout()
    fig_path = plots_dir / "pct_misalignment.pdf"
    fig.savefig(fig_path, bbox_inches='tight', dpi=600)

    # Pct misalignment with only two subsets (removing taxonomy group)
    fig_two = copy.deepcopy(fig)

    for ax in fig_two.get_axes():
        for line in ax.get_lines():
            label = line.get_label()
            if label == "taxo":
                ax.get_lines().remove(line)
    
    fig_two_path = plots_dir / "pct_misalignment_2.pdf"
    fig_two.savefig(fig_two_path, bbox_inches='tight', dpi=600)
    # Sectoral trajectories plots

    # Data
    # projected costs depend on the evolution of carbon price (each scenario has different carbon prices)
    proj_costs_iea = {scenario: pd.read_excel(raw_data_dir / "costs" / f"costs_proxy_{scenario}.xlsx")
                  for scenario in ["NZE", "APS", "STEPS"]}
    glob_prod, glob_capa = get_glob_prod_capa(iea_prod_dir=iea_prod_dir,
                                              oecd_capa_fpath=oecd_capa_fpath)


    fig_vert, fig_hor, fig_prod, data, fig_hor_rescaled = get_ghg_trajectory_plot(
        gspt=gspt,
        EF=EF,
        market_share=market_share,
                                    proj_costs_iea=proj_costs_iea,
                                    glob_prod=glob_prod,
                                    glob_capa=glob_capa,
                                    base_nze_emissions=base_nze_emissions,
                                    base_aps_emissions=base_aps_emissions,
                                    base_steps_emissions=base_steps_emissions,
                                    elec_int_nze=elec_int_nze,
                                    elec_int_aps=elec_int_aps,
                                    elec_int_steps=elec_int_steps,
                                    energy_mix_path=energy_mix_path,
                                    parent_group_map_path=parent_group_map_path,
                                    agg_BU=agg_bu_histo,
                                    world_fpath=world_fpath,
                                    regions_fpath=regions_fpath,
                                    gspt2gspt_path=gspt2gspt_path,
                                    gspt2iea_countries_path=gspt2iea_countries_path,
                                    gspt2refi_map=gspt2refi,
                                    iea_level=params.scenario_level,
                                    model=model,
                                    X_train=X,
                                    y_train=y,
                                    err_style="band",
                                    EAF_decarb=params.EAF_decarb)
    
    fig_hor_resc_name_pdf = plots_dir/ f"rel_sectoral_proj_{params.scenario_level}_{params.model_name}_decarb_horizontal.pdf"
    fig_hor_rescaled.savefig(fig_hor_resc_name_pdf, dpi=600, bbox_inches="tight")
    # Save bottom-up projections plots
    save_plots(fig_vert=fig_vert, fig_hor=fig_hor, fig_prod=fig_prod, plots_dir=plots_dir, EAF_decarb=params.EAF_decarb, iea_level=params.scenario_level, model_name=params.model_name)

    # Save bottom-up projections
    save_projections(data=data, 
                     save_dir=save_file, 
                     EAF_decarb=params.EAF_decarb, 
                     iea_level=params.scenario_level, 
                     model_name=params.model_name)





if __name__ == "__main__":
    bu_proj = typer.run(main)
