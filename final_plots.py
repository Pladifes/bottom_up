"""
Electricity Intensity Sensitivity Analysis for NZE Scenario

This script generates a sensitivity analysis plot showing how the NZE scenario
responds to different electricity intensity decarbonisation assumptions (NZE, APS, STEPS).
"""

import typer
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import json
from config import load_config

from src.datasets.GSPTDataset import GSPTDataset
from src.datasets.EmissionFactors import EmissionFactors
from src.datasets.utils import convert_plant2parent, get_cagr
from src.projections_.projections import (
    get_historical_bu_emissions, 
    get_benchmark_emissions, 
    get_iea_emissions,
    get_glob_prod_capa,
    get_elec_gen_caagr,
    get_bu_proj_emissions
)
from src.projections_.proj_company import get_proj_elec_int
from src.projections_.BUProjector import get_market_share


def main(config_path: Path = typer.Argument(Path("./config.toml"), help="Path to the TOML configuration file")):
    """Generate electricity intensity sensitivity plots for NZE scenario."""
    
    # Read config file
    config_file = load_config(config_path)
    typer.echo("Loading configuration and data...")
    
    params = config_file.params
    project_dir = params.project_dir
    historical_data = config_file.historical_data
    emission_factors = config_file.emission_factors
    mappings = config_file.mappings
    scenarios = config_file.scenarios
    projected_data = config_file.projected_data
    
    # Define paths
    raw_data_dir = project_dir / "data" / "raw"
    db_path = params.steel_db
    world_fpath = raw_data_dir / scenarios.iea_world
    regions_fpath = raw_data_dir / scenarios.iea_regions
    gspt2gspt_path = raw_data_dir / mappings.gspt2gspt
    refi2gspt_path = raw_data_dir / mappings.refi2gspt
    gspt2iea_countries_path = raw_data_dir / mappings.gspt2iea_countries
    parent_group_map_path = raw_data_dir / mappings.parent_group_map
    energy_mix_path = raw_data_dir / emission_factors.energy_mix
    iea_prod_dir = raw_data_dir / historical_data.macro.iea_prod
    wsa_prod_path = raw_data_dir / historical_data.macro.wsa_prod
    oecd_capa_fpath = raw_data_dir / historical_data.macro.oecd_capa
    prod_dir = raw_data_dir / "production"
    company_steel_prod_path = raw_data_dir / historical_data.micro.company_prod
    
    # Load electricity intensity data
    elec_int_nze_path = raw_data_dir / projected_data.electricity.elec_int_nze
    elec_int_aps_path = raw_data_dir / projected_data.electricity.elec_int_aps
    elec_int_steps_path = raw_data_dir / projected_data.electricity.elec_int_steps
    
    elec_int_nze = pd.read_excel(elec_int_nze_path)
    elec_int_steps = pd.read_excel(elec_int_steps_path)
    elec_int_aps = pd.read_excel(elec_int_aps_path)
    
    # Output directory
    save_dir = params.save_dir
    save_file = save_dir / "BU_results"
    plots_dir = save_file / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_data_dir = save_file / "plots_data"
    plots_data_dir.mkdir(parents=True, exist_ok=True)
    models_dir = save_file / "models"
    
    typer.echo(f"Plots will be saved to {plots_dir}")
    typer.echo(f"Plot data will be saved to {plots_data_dir}")
    
    # Load datasets
    typer.echo("Loading GSPT dataset...")
    gspt = GSPTDataset(
        data_path=raw_data_dir / historical_data.asset_level_data,
        missing_years_path=raw_data_dir / historical_data.missing_years,
        gspt2gspt_path=gspt2gspt_path,
        parent_group_map_path=parent_group_map_path,
        version_year=2023
    )
    
    typer.echo("Loading emission factors...")
    EF = EmissionFactors(
        wsa_path=raw_data_dir / emission_factors.wsa,
        jrc_22_path=raw_data_dir / emission_factors.jrc,
        sci_path=raw_data_dir / emission_factors.sci,
        EU_27_path=raw_data_dir / mappings.EU_27,
        huizhong_path=raw_data_dir / emission_factors.huizhong,
    )
    
    # Load mappings
    with open(parent_group_map_path, "r") as f:
        parent_group_map = json.load(f)
    gspt2refi = pd.read_excel(refi2gspt_path)
    
    # Load the trained model
    typer.echo("Loading trained model...")
    model = joblib.load(models_dir / f"{params.model_name}.joblib")
    
    # Get historical BU emissions
    typer.echo("Computing historical BU emissions...")
    histo_global_prod = pd.DataFrame({
        "year": [2019, 2020, 2021, 2022],
        "Crude steel production (Mt)": [1874.4, 1877.5, 1951.9, 1885]
    })
    
    year = 2022
    company_steel_prod = pd.read_excel(company_steel_prod_path)
    market_share = get_market_share(
        gspt=gspt,
        db_path=db_path,
        histo_global_prod=histo_global_prod,
        company_steel_prod=company_steel_prod,
        mapping=gspt2refi,
        gspt2gspt_path=gspt2gspt_path,
        parent_group_map=parent_group_map,
        method="single_year",
        year=year
    )
    
    agg_bu_histo, group_bu_histo, adj_factor, prod = get_historical_bu_emissions(
        gspt=gspt,
        EF=EF,
        prod_dir=prod_dir,
        global_capa_path=oecd_capa_fpath,
        global_prod_wsa_path=wsa_prod_path,
        start_year=2019,
        end_year=2022,
        energy_mix_path=energy_mix_path,
        parent_group_map=parent_group_map,
        model=model,
        gspt2gspt_path=gspt2gspt_path
    )
    agg_bu_histo['Emissions (Gt)'] = agg_bu_histo['BU emissions'] / 1e9
    
    # Get benchmark emissions
    typer.echo("Computing benchmark emissions...")
    nze_emissions = get_iea_emissions(scenario="NZE")
    x0_year = 2022
    x0_value = float(agg_bu_histo.loc[agg_bu_histo["year"] == x0_year, "Emissions (Gt)"])
    base_nze_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=nze_emissions)
    
    # Prepare data for combined Excel file
    all_data_for_excel = []
    
    # Get global production and capacity
    typer.echo("Loading global production and capacity projections...")
    glob_prod, glob_capa = get_glob_prod_capa(iea_prod_dir=iea_prod_dir, oecd_capa_fpath=oecd_capa_fpath)
    glob_bu_prod = agg_bu_histo.loc[agg_bu_histo['year'] == 2022, 'Attributed production'].iloc[0]
    
    # Load projected costs for NZE
    proj_costs_iea = {
        scenario: pd.read_excel(raw_data_dir / "costs" / f"costs_proxy_{scenario}.xlsx")
        for scenario in ["NZE", "APS", "STEPS"]
    }
    
    # ====================================================================
    # CREATE ELECTRICITY INTENSITY SENSITIVITY PLOT
    # ====================================================================
    typer.echo("Generating electricity intensity sensitivity plots...")
    
    # Create a figure with 3 subplots in one row
    fig_elec_sensitivity, axs_elec = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Define the electricity intensities and their labels
    elec_int_sources = {
        "NZE": elec_int_nze,
        "APS": elec_int_aps,
        "STEPS": elec_int_steps
    }
    
    # Store data for each electricity intensity source
    data_elec_sensitivity = {}
    
    # Add historical data to combined dataset
    historical_data = agg_bu_histo[['year', 'Emissions (Gt)', 'Attributed production']].copy()
    historical_data['period'] = 'historical'
    historical_data['elec_scenario'] = ''
    historical_data['method'] = ''
    historical_data['Emissions_low (Gt)'] = np.nan
    historical_data['Emissions_high (Gt)'] = np.nan
    historical_data['production'] = historical_data['Attributed production']
    historical_data['Emission Factor (tCO2/t)'] = (historical_data['Emissions (Gt)'] * 1e9) / historical_data['production']
    historical_data['trajectory_type'] = 'Historical BU emissions'
    all_data_for_excel.append(historical_data[['year', 'period', 'elec_scenario', 'method', 'trajectory_type', 
                                                'Emissions (Gt)', 'Emissions_low (Gt)', 'Emissions_high (Gt)', 
                                                'production', 'Emission Factor (tCO2/t)']])
    
    # Add NZE reference trajectory to combined dataset
    nze_reference_data = base_nze_emissions[['year', 'Emissions (Gt)']].copy()
    nze_reference_data['period'] = 'projected'
    nze_reference_data['elec_scenario'] = 'NZE'
    nze_reference_data['method'] = 'reference'
    nze_reference_data['Emissions_low (Gt)'] = np.nan
    nze_reference_data['Emissions_high (Gt)'] = np.nan
    nze_reference_data['production'] = np.nan
    nze_reference_data['Emission Factor (tCO2/t)'] = np.nan
    nze_reference_data['trajectory_type'] = 'NZE reference trajectory'
    all_data_for_excel.append(nze_reference_data[['year', 'period', 'elec_scenario', 'method', 'trajectory_type',
                                                   'Emissions (Gt)', 'Emissions_low (Gt)', 'Emissions_high (Gt)',
                                                   'production', 'Emission Factor (tCO2/t)']])
    
    # Plot for each electricity intensity source
    for idx, (elec_source_name, elec_int_data) in enumerate(elec_int_sources.items()):
        typer.echo(f"  Processing {elec_source_name} electricity intensity...")
        
        ax = axs_elec[idx]
        ax.set_xlim(2017, 2031)
        ax.set_ylim(3.25, 4.75)
        ax.set_box_aspect(1)
        ax.grid("both")
        ax.set_title(f"NZE with {elec_source_name} electricity intensity", fontsize=12, weight='bold')
        ax.set_ylabel("Emissions (GtCO₂ eq)")
        ax.set_xlabel("Year")
        
        # Plot historical BU emissions
        sns.lineplot(data=agg_bu_histo,
                    x='year',
                    y="Emissions (Gt)",
                    label="Historical BU emissions",
                    ax=ax)
        
        # Add starting point marker
        sns.scatterplot(
            x=[base_nze_emissions.loc[base_nze_emissions.index[0], "year"]], 
            y=[base_nze_emissions.loc[base_nze_emissions.index[0], "Emissions (Gt)"]],  
            marker='o', 
            color="grey",
            s=200,
            zorder=2,
            ax=ax
        )
        
        # Plot NZE reference trajectory
        sns.lineplot(data=base_nze_emissions,
                    x='year',
                    y="Emissions (Gt)",
                    linewidth=2,
                    label="Projected IEA emissions (NZE slope)",
                    color="green",
                    linestyle="dotted",
                    zorder=2,
                    ax=ax)
        
        # Annotate 2022 value
        base_nze_emissions_22 = float(base_nze_emissions.loc[base_nze_emissions['year'] == 2022, 'Emissions (Gt)'])
        ax.annotate(f"{base_nze_emissions_22:.2f}", (2022, base_nze_emissions_22 + 0.055), 
                   textcoords='offset points', xytext=(0, 10), ha='center', 
                   fontsize=12, color='grey', weight='bold')
        
        # Annotate 2030 value
        base_nze_emissions_30 = float(base_nze_emissions.loc[base_nze_emissions['year'] == 2030, 'Emissions (Gt)'])
        ax.annotate(f"{base_nze_emissions_30:.2f}", (2030.5, base_nze_emissions_30), 
                   textcoords='offset points', xytext=(0, 10), ha='center', 
                   fontsize=10, color='green', weight='bold')
        
        # Calculate CAAGR for this electricity intensity source
        r_elec = get_elec_gen_caagr(elec_int=elec_int_data, start_year=2022, end_year=2030)
        
        # Get projected costs for NZE scenario
        proj_costs_nze = proj_costs_iea["NZE"]
        
        # Get carbon budget for NZE
        cbudget_nze = pd.read_excel(Path("./src/projections_") / "proj_country_prod_nze.xlsx")
        
        # Define colors for different methods
        drivers_colors = {"constant_UR": "orange", "company_UR": "green"}
        methods_to_plot = ["constant_UR", "company_UR"]
        
        data_elec_sensitivity[elec_source_name] = {}
        
        # Plot each method for NZE scenario with this electricity intensity
        for method in methods_to_plot:
            start_year, end_year = 2023, 2030
            
            # Get BU projected emissions
            proj_company, proj_plants = get_bu_proj_emissions(
                gspt=gspt,
                EF=EF,
                glob_prod_iea=glob_prod["NZE"], 
                glob_bu_prod=glob_bu_prod,
                market_share=market_share,
                parent_group_map_path=parent_group_map_path,
                gspt2refi_map=gspt2refi,
                gspt2gspt_path=gspt2gspt_path,
                proj_costs=proj_costs_nze,
                start_year=start_year,
                end_year=end_year,
                method=method,
                glob_capa=glob_capa["NZE"],
                cbudget=cbudget_nze
            )
            
            # Convert uncertainty bounds to Gt if they exist (from huizhong source)
            if "Estimated emissions_low (ttpa)" in proj_plants.columns:
                proj_plants["Emissions_low (Gt)"] = proj_plants["Estimated emissions_low (ttpa)"] / 1e6
                proj_plants["Emissions_high (Gt)"] = proj_plants["Estimated emissions_high (ttpa)"] / 1e6
            
            # Determine level: NZE is global, APS and STEPS are regional
            if elec_source_name == "NZE":
                elec_level = "global"
            else:  # APS or STEPS
                elec_level = "region"
            
            typer.echo(f"    Applying {elec_source_name} electricity intensity at {elec_level} level for {method}...")
            
            # Apply electricity intensity adjustments
            proj_plants = get_proj_elec_int(
                proj_plants, 
                r=r_elec, 
                energy_mix_path=energy_mix_path, 
                level=elec_level,
                scenario=elec_source_name,  # Use the actual electricity scenario, not "NZE"
                world_fpath=world_fpath,
                regions_fpath=regions_fpath,
                gspt2iea_countries_path=gspt2iea_countries_path
            )
            
            # Apply EAF decarbonization if enabled
            if params.EAF_decarb:
                # Use scenario-specific electricity intensity targets for 2030
                base_all = 460  # 2022 global electricity intensity (g CO2/kWh)
                # 2030 targets from IEA WEO (g CO2/kWh)
                end_values = {
                    "NZE": 186,    # Net Zero Emissions
                    "APS": 255,    # Announced Pledges Scenario
                    "STEPS": 303   # Stated Policies Scenario
                }
                end_value = end_values[elec_source_name]
                
                typer.echo(f"    EAF decarb: {base_all} -> {end_value} g CO2/kWh (CAGR will vary by scenario)")
                
                # Calculate CAGR for this specific scenario
                caagr = get_cagr(base_value=base_all, end_value=end_value, n=8)
                years = list(range(2023, 2031))
                elec_cagr_df = pd.DataFrame({
                    "year": years,
                    "EAF_CAGR": [(1+caagr)**(i+1) for i in range(len(years))],
                })
                
                # Drop existing CAGR column if it exists (from get_proj_elec_int for regional scenarios)
                if "CAGR" in proj_plants.columns:
                    proj_plants = proj_plants.drop(columns=["CAGR"])
                
                # Merge the EAF-specific CAGR
                proj_plants = pd.merge(proj_plants, elec_cagr_df, on="year", how='left')
                
                # Apply CAGR only to electric arc furnace (EAF) plants
                proj_plants["EAF_CAGR"] = (proj_plants["Main production process"] == "electric") * proj_plants["EAF_CAGR"]
                proj_plants["EAF_CAGR"] = proj_plants["EAF_CAGR"].replace(0, 1)
                
                # Adjust emission factors
                proj_plants["EF"] = proj_plants["EF"] * proj_plants["EAF_CAGR"]
                
                # Apply CAGR to uncertainty bounds if they exist
                if "EF_12_lower" in proj_plants.columns:
                    proj_plants["EF_12_lower"] = proj_plants["EF_12_lower"] * proj_plants["EAF_CAGR"]
                    proj_plants["EF_12_upper"] = proj_plants["EF_12_upper"] * proj_plants["EAF_CAGR"]
                    # Update emissions bounds with adjusted EF
                    proj_plants["Emissions_low (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_lower"] / 1E6
                    proj_plants["Emissions_high (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_upper"] / 1E6
                
                # Update emissions with adjusted EF
                proj_plants["Emissions (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF"] / 1E6
                
                # Clean up temporary column
                proj_plants = proj_plants.drop(columns=["EAF_CAGR"])
            
            # Convert to parent company level
            proj_group = convert_plant2parent(
                proj_plants, 
                gspt2gspt_path=gspt2gspt_path, 
                parent_group_map=parent_group_map
            )
            
            # Calculate attributed values
            proj_group["Attributed crude steel capacity (ttpa)"] = proj_group['Nominal crude steel capacity (ttpa)'] * proj_group["Share"]
            proj_group["Attributed capacity"] = proj_group["Attributed crude steel capacity (ttpa)"] * 1E3
            proj_group["Attributed emissions"] = proj_group['Emissions (Gt)'] * 1e9 * proj_group["Share"]
            proj_group["Attributed production"] = proj_group['Estimated crude steel production (ttpa)'] * 1e3 * proj_group["Share"]
            
            # Add uncertainty bounds if available (from huizhong EF source)
            # Check in proj_group after convert_plant2parent
            if "Emissions_low (Gt)" in proj_group.columns:
                proj_group["Attributed emissions_low"] = proj_group['Emissions_low (Gt)'] * 1e9 * proj_group["Share"]
                proj_group["Attributed emissions_high"] = proj_group['Emissions_high (Gt)'] * 1e9 * proj_group["Share"]
            
            # Aggregate features
            # Use simpler aggregation approach
            proj_feats = proj_group.groupby(['Group', 'year']).agg({
                "Attributed emissions": "sum",
                "Attributed production": "sum",
                "Attributed capacity": "sum",
            }).rename(columns={
                "Attributed emissions": "emissions",
                "Attributed production": "production", 
                "Attributed capacity": "capacity"
            })
            
            # Add uncertainty bounds to aggregation if available
            if "Attributed emissions_low" in proj_group.columns:
                proj_feats_uncert = proj_group.groupby(['Group', 'year']).agg({
                    "Attributed emissions_low": "sum",
                    "Attributed emissions_high": "sum"
                }).rename(columns={
                    "Attributed emissions_low": "emissions_low",
                    "Attributed emissions_high": "emissions_high"
                })
                proj_feats = pd.concat([proj_feats, proj_feats_uncert], axis=1)
            proj_feats["log_Attributed emissions"] = np.log(proj_feats["emissions"])
            
            # Add uncertainty for log emissions if available
            if "emissions_low" in proj_feats.columns:
                proj_feats["log_Attributed emissions_low"] = np.log(proj_feats["emissions_low"])
                proj_feats["log_Attributed emissions_high"] = np.log(proj_feats["emissions_high"])
            
            # Predict using the model
            proj_feats[f"log BU emissions ({method})"] = model.predict(proj_feats[["log_Attributed emissions"]])
            proj_feats[f"BU emissions ({method})"] = np.exp(proj_feats[f"log BU emissions ({method})"])
            
            # Predict with uncertainty bounds if available
            if "log_Attributed emissions_low" in proj_feats.columns:
                proj_feats[f"log BU emissions_low ({method})"] = model.predict(proj_feats[["log_Attributed emissions_low"]].rename(columns={"log_Attributed emissions_low": "log_Attributed emissions"}))
                proj_feats[f"log BU emissions_high ({method})"] = model.predict(proj_feats[["log_Attributed emissions_high"]].rename(columns={"log_Attributed emissions_high": "log_Attributed emissions"}))
                proj_feats[f"BU emissions_low ({method})"] = np.exp(proj_feats[f"log BU emissions_low ({method})"])
                proj_feats[f"BU emissions_high ({method})"] = np.exp(proj_feats[f"log BU emissions_high ({method})"])
            
            proj_feats[f"BU intensity ({method})"] = proj_feats[f"BU emissions ({method})"] / proj_feats["production"]
            proj_feats[f"BU intensity ({method})"] = proj_feats[f"BU intensity ({method})"].replace(np.inf, np.nan)
            proj_feats = proj_feats.reset_index()
            
            # Aggregate to sectoral level
            agg_cols = {
                f"BU emissions ({method})": "sum",
                "emissions": 'sum',
                "production": "sum",
                "capacity": "sum"
            }
            
            # Add uncertainty to sectoral aggregation if available
            if f"BU emissions_low ({method})" in proj_feats.columns:
                agg_cols[f"BU emissions_low ({method})"] = "sum"
                agg_cols[f"BU emissions_high ({method})"] = "sum"
            
            bu_sectoral_proj = proj_feats.groupby("year").agg(agg_cols).reset_index()
            bu_sectoral_proj['Emissions (Gt)'] = bu_sectoral_proj[f"BU emissions ({method})"] / 1e9
            
            # Add uncertainty bounds to sectoral projections if available
            if f"BU emissions_low ({method})" in bu_sectoral_proj.columns:
                bu_sectoral_proj['Emissions_low (Gt)'] = bu_sectoral_proj[f"BU emissions_low ({method})"] / 1e9
                bu_sectoral_proj['Emissions_high (Gt)'] = bu_sectoral_proj[f"BU emissions_high ({method})"] / 1e9
            
            bu_sectoral_proj['Raw Emissions (Gt)'] = bu_sectoral_proj["emissions"] / 1e9
            bu_sectoral_proj[f'UR ({method})'] = bu_sectoral_proj["production"] / bu_sectoral_proj["capacity"]
            bu_sectoral_proj[f"Intensity ({method})"] = bu_sectoral_proj[f"BU emissions ({method})"] / bu_sectoral_proj["production"]
            
            # Add 2022 value for continuity
            bu_emissions_22 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Emissions (Gt)"])
            new_row = {"year": [2022], "Emissions (Gt)": [bu_emissions_22]}
            bu_sectoral_proj = pd.concat([pd.DataFrame(new_row), bu_sectoral_proj], axis=0).sort_values(by="year", ascending=True)
            
            # Store data
            data_elec_sensitivity[elec_source_name][method] = bu_sectoral_proj
            
            # Add to combined dataset
            trajectory_data = bu_sectoral_proj.copy()
            trajectory_data['period'] = 'projected'
            trajectory_data['elec_scenario'] = elec_source_name
            trajectory_data['method'] = method
            
            # Calculate emission factor
            trajectory_data['Emission Factor (tCO2/t)'] = (trajectory_data['Emissions (Gt)'] * 1e9) / trajectory_data['production']
            
            # Report emission factor for 2030 to verify differences
            ef_2030 = trajectory_data.loc[trajectory_data['year'] == 2030, 'Emission Factor (tCO2/t)'].values
            if len(ef_2030) > 0:
                typer.echo(f"    2030 Emission Factor: {ef_2030[0]:.3f} tCO2/t")
            
            # Add trajectory type description
            if method == "company_UR":
                method_label = "constant market share"
            elif method == "constant_UR":
                method_label = "country UR"
            else:
                method_label = method
            trajectory_data['trajectory_type'] = f"NZE with {elec_source_name} elec intensity ({method_label})"
            
            # Ensure uncertainty columns exist
            if 'Emissions_low (Gt)' not in trajectory_data.columns:
                trajectory_data['Emissions_low (Gt)'] = np.nan
            if 'Emissions_high (Gt)' not in trajectory_data.columns:
                trajectory_data['Emissions_high (Gt)'] = np.nan
            
            # Add to list
            all_data_for_excel.append(trajectory_data[['year', 'period', 'elec_scenario', 'method', 'trajectory_type',
                                                       'Emissions (Gt)', 'Emissions_low (Gt)', 'Emissions_high (Gt)',
                                                       'production', 'Emission Factor (tCO2/t)']])
            
            # Plot the trajectory
            if method == "company_UR":
                emissions_label = "Projected BU emissions (constant market share)"
            elif method == "constant_UR":
                emissions_label = "Projected BU emissions (country UR)"
            else:
                emissions_label = f"Projected BU emissions ({method})"
            
            sns.lineplot(data=bu_sectoral_proj,
                        x='year',
                        y="Emissions (Gt)",
                        label=emissions_label,
                        linestyle="dashed",
                        color=drivers_colors[method],
                        ax=ax)
            
            # Add error bars if uncertainty bounds are available
            if "Emissions_low (Gt)" in bu_sectoral_proj.columns:
                years = bu_sectoral_proj['year']
                emissions = bu_sectoral_proj["Emissions (Gt)"]
                emissions_low = bu_sectoral_proj["Emissions_low (Gt)"]
                emissions_high = bu_sectoral_proj["Emissions_high (Gt)"]
                
                # Calculate error bar sizes (distance from center to bounds)
                yerr_lower = emissions - emissions_low
                yerr_upper = emissions_high - emissions
                
                # Plot error bars
                ax.errorbar(years, emissions, 
                           yerr=[yerr_lower, yerr_upper],
                           fmt='none', 
                           color=drivers_colors[method],
                           alpha=0.3,
                           capsize=3,
                           capthick=1)
    
    # Create a single shared legend below all plots
    # Collect handles and labels from all axes
    handles_list = []
    labels_list = []
    for ax in axs_elec:
        h, labels = ax.get_legend_handles_labels()
        for handle, label in zip(h, labels):
            if label not in labels_list:  # Avoid duplicates
                handles_list.append(handle)
                labels_list.append(label)
        # Remove individual legends from each subplot
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    # Add uncertainty bar representation to legend
    # Create a custom error bar handle for the legend
    uncertainty_handle = Line2D(
        [], [], 
        color='gray', 
        marker='|', 
        linestyle='None',
        markersize=10, 
        markeredgewidth=1.5,
        alpha=0.5,
        label='Uncertainty bounds (95% CI)'
    )
    handles_list.append(uncertainty_handle)
    labels_list.append('Uncertainty bounds (95% CI)')
    
    # Add the shared legend below the plots (all items on one line)
    fig_elec_sensitivity.legend(
        handles_list, 
        labels_list, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels_list),  # All items on one line
        frameon=True,
        fontsize=10
    )
    
    # Add overall title as suptitle above everything
    fig_elec_sensitivity.suptitle(
        "Sensitivity of NZE scenario to electricity intensity decarbonisation assumptions",
        fontsize=18, 
        weight="bold", 
        y=0.98
    )
    
    # Adjust layout to make room for suptitle and legend
    fig_elec_sensitivity.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Combine all data into a single DataFrame
    typer.echo("Combining all data...")
    combined_data = pd.concat(all_data_for_excel, axis=0, ignore_index=True)
    
    # Ensure Emissions_low and Emissions_high columns exist (fill with NaN if missing)
    if 'Emissions_low (Gt)' not in combined_data.columns:
        combined_data['Emissions_low (Gt)'] = np.nan
    if 'Emissions_high (Gt)' not in combined_data.columns:
        combined_data['Emissions_high (Gt)'] = np.nan
    
    # Reorder columns for clarity
    column_order = ['year', 'period', 'elec_scenario', 'method', 'trajectory_type',
                   'Emissions (Gt)', 'Emissions_low (Gt)', 'Emissions_high (Gt)',
                   'production', 'Emission Factor (tCO2/t)']
    combined_data = combined_data[column_order]
    
    # Sort by period (historical first), then elec_scenario, method, and year
    combined_data['period_sort'] = combined_data['period'].map({'historical': 0, 'projected': 1})
    combined_data = combined_data.sort_values(['period_sort', 'elec_scenario', 'method', 'year']).drop('period_sort', axis=1)
    
    # Save to single Excel file
    typer.echo("Saving combined data to Excel...")
    combined_excel_path = plots_data_dir / "nze_elec_intensity_sensitivity_all_data.xlsx"
    combined_data.to_excel(combined_excel_path, index=False, sheet_name='All Trajectories')
    
    # Save the emissions trajectory figure
    typer.echo("Saving emissions trajectory plots...")
    fig_elec_sensitivity_path_pdf = plots_dir / "nze_elec_intensity_sensitivity.pdf"
    fig_elec_sensitivity_path_png = plots_dir / "nze_elec_intensity_sensitivity.png"
    fig_elec_sensitivity.savefig(fig_elec_sensitivity_path_pdf, dpi=600, bbox_inches='tight')
    fig_elec_sensitivity.savefig(fig_elec_sensitivity_path_png, dpi=600, bbox_inches='tight')
    
    # ====================================================================
    # CREATE EMISSION INTENSITY EVOLUTION PLOT
    # ====================================================================
    typer.echo("Generating emission intensity evolution plot...")
    
    # Create intensity plot (using constant market share method)
    fig_intensity, ax_intensity = plt.subplots(figsize=(12, 8))
    
    # Define colors for different scenarios
    intensity_colors = {
        'Historical BU emissions': '#2E73A9',
        'NZE with NZE elec intensity (constant market share)': '#1f77b4',
        'NZE with APS elec intensity (constant market share)': '#ff7f0e',
        'NZE with STEPS elec intensity (constant market share)': '#2ca02c'
    }
    
    # Define line styles
    intensity_linestyles = {
        'Historical BU emissions': '-',
        'NZE with NZE elec intensity (constant market share)': '--',
        'NZE with APS elec intensity (constant market share)': '--',
        'NZE with STEPS elec intensity (constant market share)': '--'
    }
    
    # Plot each trajectory
    for trajectory_type in intensity_colors.keys():
        # Filter data for this trajectory
        trajectory_mask = combined_data['trajectory_type'] == trajectory_type
        trajectory_subset = combined_data[trajectory_mask].copy()
        
        if len(trajectory_subset) > 0:
            # Sort by year
            trajectory_subset = trajectory_subset.sort_values('year')
            
            # Plot the line
            ax_intensity.plot(
                trajectory_subset['year'],
                trajectory_subset['Emission Factor (tCO2/t)'],
                label=trajectory_type,
                color=intensity_colors[trajectory_type],
                linestyle=intensity_linestyles[trajectory_type],
                linewidth=2.5,
                marker='o',
                markersize=5
            )
    
    # Formatting
    ax_intensity.set_xlabel('Year', fontsize=13)
    ax_intensity.set_ylabel('Emission Intensity (tCO₂/t steel)', fontsize=13)
    ax_intensity.set_title(
        'Evolution of Emission Intensity: Historical and NZE Projections\n(Constant Market Share Method)',
        fontsize=14,
        weight='bold',
        pad=20
    )
    ax_intensity.grid(True, alpha=0.3, linestyle='--')
    ax_intensity.set_xlim(2018, 2031)
    
    # Legend
    ax_intensity.legend(
        loc='upper right',
        frameon=True,
        fontsize=11,
        edgecolor='gray'
    )
    
    # Tight layout
    fig_intensity.tight_layout()
    
    # Save intensity plot
    fig_intensity_path_pdf = plots_dir / "nze_emission_intensity_evolution.pdf"
    fig_intensity_path_png = plots_dir / "nze_emission_intensity_evolution.png"
    fig_intensity.savefig(fig_intensity_path_pdf, dpi=600, bbox_inches='tight')
    fig_intensity.savefig(fig_intensity_path_png, dpi=600, bbox_inches='tight')
    
    typer.echo(f"✓ Saved emissions trajectory PDF: {fig_elec_sensitivity_path_pdf}")
    typer.echo(f"✓ Saved emissions trajectory PNG: {fig_elec_sensitivity_path_png}")
    typer.echo(f"✓ Saved emission intensity PDF: {fig_intensity_path_pdf}")
    typer.echo(f"✓ Saved emission intensity PNG: {fig_intensity_path_png}")
    typer.echo(f"✓ Saved combined plot data to: {combined_excel_path}")
    typer.echo(f"  Total rows: {len(combined_data)}")
    typer.echo(f"  Columns: {', '.join(column_order)}")
    typer.echo("Done!")
    
    return data_elec_sensitivity


if __name__ == "__main__":
    typer.run(main)

