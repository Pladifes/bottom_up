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
    
    # Output directory structure
    save_dir = params.save_dir
    save_file = save_dir / "BU_results"
    plots_dir = save_file / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for better organization
    debug_dir = plots_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = plots_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    intensity_dir = plots_dir / "intensity_evolution"
    intensity_dir.mkdir(parents=True, exist_ok=True)
    methods_dir = plots_dir / "methods_comparison"
    methods_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for individual electricity sensitivity plots (by method)
    elec_sensitivity_dir = plots_dir / "elec_sensitivity"
    elec_sensitivity_dir.mkdir(parents=True, exist_ok=True)
    constant_ur_dir = elec_sensitivity_dir / "constant_UR"
    constant_ur_dir.mkdir(parents=True, exist_ok=True)
    company_ur_dir = elec_sensitivity_dir / "company_UR"
    company_ur_dir.mkdir(parents=True, exist_ok=True)
    
    plots_data_dir = save_file / "plots_data"
    plots_data_dir.mkdir(parents=True, exist_ok=True)
    plant_data_dir = save_file / "plant_level_data"
    plant_data_dir.mkdir(parents=True, exist_ok=True)
    models_dir = save_file / "models"
    
    typer.echo(f"Main plots will be saved to {plots_dir}")
    typer.echo(f"Debug plots will be saved to {debug_dir}")
    typer.echo(f"Plot data will be saved to {plots_data_dir}")
    typer.echo(f"Plant-level data will be saved to {plant_data_dir}")
    
    # Load datasets
    typer.echo("Loading GSPT dataset...")
    gspt = GSPTDataset(
        data_path=raw_data_dir / historical_data.asset_level_data,
        missing_years_path=raw_data_dir / historical_data.missing_years,
        gspt2gspt_path=gspt2gspt_path,
        parent_group_map_path=parent_group_map_path,
        version_year=2023
    )
    
    # Get UR configuration
    use_country_techno_ur = params.use_country_techno_ur
    typer.echo(f"Using {'country_techno UR' if use_country_techno_ur else 'country UR'} for historical and projection calculations")
    
    typer.echo("Loading emission factors...")
    EF = EmissionFactors(
        wsa_path=raw_data_dir / emission_factors.wsa,
        jrc_22_path=raw_data_dir / emission_factors.jrc,
        sci_path=raw_data_dir / emission_factors.sci,
        EU_27_path=raw_data_dir / mappings.EU_27,
        huizhong_path=raw_data_dir / emission_factors.huizhong,
    )
    
    # Load mappings
    with open(gspt2gspt_path, "r") as f:
        gspt2gspt_map = json.load(f)
    with open(parent_group_map_path, "r") as f:
        parent_group_map = json.load(f)
    gspt2refi = pd.read_excel(refi2gspt_path)
    
    # Train the model fresh (instead of loading pre-trained)
    typer.echo("Training bottom-up model...")
    
    # Load top-down emissions data
    from src.datasets.RefinitivDataset import RefinitivDataset
    from src.datasets.utils import fill_missing_CF12
    from src.feature_eng.company_feature_engineering import get_X_y
    
    refinitiv_path = raw_data_dir / historical_data.top_down.refinitiv
    cdp_path = project_dir / historical_data.top_down.cdp
    gspt2cdp_map_path = raw_data_dir / mappings.gspt2cdp
    carbon_price_path = raw_data_dir / historical_data.macro.carbon_price
    
    refi_class = RefinitivDataset(data_path=refinitiv_path)
    refinitiv = refi_class.get_preprocessed_data()
    refinitiv = fill_missing_CF12(refinitiv=refinitiv, refi2gspt_path=refi2gspt_path, gspt2gspt_map=gspt2gspt_map)
    cdp = pd.read_excel(cdp_path)
    gspt2cdp = pd.read_excel(gspt2cdp_map_path)
    
    # Create training dataset
    typer.echo("  - Creating training dataset...")
    X_y = get_X_y(gspt=gspt,
            impute_prod="Global",
            average="micro",
            EF=EF,
            energy_mix_path=energy_mix_path,
            carbon_price_path=carbon_price_path,
            gspt2refi=gspt2refi,
            gspt2gspt_path=gspt2gspt_path,
            parent_group_map=parent_group_map,
            refinitiv=refinitiv,
            gspt2cdp=gspt2cdp,
            cdp=cdp)
    
    typer.echo(f"  - Training dataset shape: {X_y.shape}")
    typer.echo(f"  - Training on {len(X_y)} samples")
    
    # Check training data quality
    log_attr_em = X_y["log_Attributed emissions"]
    log_cf12 = X_y["log_max_CF12"]
    typer.echo(f"  - log_Attributed emissions: min={log_attr_em.min():.2f}, max={log_attr_em.max():.2f}, NaN={log_attr_em.isna().sum()}, Inf={np.isinf(log_attr_em).sum()}")
    typer.echo(f"  - log_max_CF12: min={log_cf12.min():.2f}, max={log_cf12.max():.2f}, NaN={log_cf12.isna().sum()}, Inf={np.isinf(log_cf12).sum()}")
    
    # Fit the model
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    
    X, y = X_y[["log_Attributed emissions"]], X_y[["log_max_CF12"]]
    
    typer.echo(f"  - Fitting model on {len(X)} samples...")
    model = Pipeline([("scaler", StandardScaler()),
                  ("regressor", LinearRegression())])
    model.fit(X, y)
    typer.echo("  ✓ Model trained successfully")
    
    # Save training data and model
    X.to_excel(models_dir / "X_train.xlsx")
    y.to_excel(models_dir / "y_train.xlsx")
    joblib.dump(model, models_dir / f"{params.model_name}.joblib")
    typer.echo(f"  ✓ Model and training data saved to {models_dir}")
    
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
        gspt2gspt_path=gspt2gspt_path,
        use_country_techno_ur=use_country_techno_ur
    )
    agg_bu_histo['Emissions (Gt)'] = agg_bu_histo['BU emissions'] / 1e9
    
    # Get benchmark emissions
    typer.echo("Computing benchmark emissions...")
    nze_emissions = get_iea_emissions(scenario="NZE")
    x0_year = 2022
    x0_value = float(agg_bu_histo.loc[agg_bu_histo["year"] == x0_year, "Emissions (Gt)"].iloc[0])
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
        base_nze_emissions_22 = float(base_nze_emissions.loc[base_nze_emissions['year'] == 2022, 'Emissions (Gt)'].iloc[0])
        ax.annotate(f"{base_nze_emissions_22:.2f}", (2022, base_nze_emissions_22 + 0.055), 
                   textcoords='offset points', xytext=(0, 10), ha='center', 
                   fontsize=12, color='grey', weight='bold')
        
        # Annotate 2030 value
        base_nze_emissions_30 = float(base_nze_emissions.loc[base_nze_emissions['year'] == 2030, 'Emissions (Gt)'].iloc[0])
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
                cbudget=cbudget_nze,
                use_country_techno_ur=use_country_techno_ur
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
            # Take scenario-specific electricity intensity slope
            # and apply it to the EAF emission factor
            # Apply CAGR only to electric arc furnace (EAF) plant
            proj_plants["EAF_CAGR"] = (proj_plants["Main production process"] == "electric") * proj_plants["elec_caagr_pow"]
            proj_plants["EAF_CAGR"] = proj_plants["EAF_CAGR"].replace(0, 1)
            proj_plants["EF"] = proj_plants["EF"] * proj_plants["EAF_CAGR"]
            # Update uncertainty bounds for EAF emission factors
            # That is, EAF content for upper and lower bounds
            # decreases with the electricity intensity slope
            proj_plants["EF_12_lower"] = proj_plants["EF_12_lower"] * proj_plants["EAF_CAGR"]
            proj_plants["EF_12_upper"] = proj_plants["EF_12_upper"] * proj_plants["EAF_CAGR"]
            proj_plants["Emissions (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF"] / 1E6
            proj_plants["Emissions_low (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_lower"] / 1E6
            proj_plants["Emissions_high (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_upper"] / 1E6

            # Clean up temporary column
            # proj_plants = proj_plants.drop(columns=["EAF_CAGR"])
            
            # Save plant-level data before aggregation
            plant_filename = f"nze_{elec_source_name.lower()}_elec_{method}_plants.csv"
            plant_filepath = plant_data_dir / plant_filename
            proj_plants.to_csv(plant_filepath, index=False)
            typer.echo(f"      ✓ Saved plant-level data: {plant_filename}")
            
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
            # Quick check for problematic values before taking log
            emissions = proj_feats["emissions"]
            zero_count = (emissions == 0).sum()
            negative_count = (emissions < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                typer.echo(f"DEBUG: Found {zero_count} zero and {negative_count} negative emissions values")
                if zero_count > 0:
                    typer.echo(f"  - Dropping {zero_count} rows with zero emissions")
                    proj_feats = proj_feats[emissions > 0].reset_index(drop=True)
                if negative_count > 0:
                    typer.echo(f"  - Dropping {negative_count} rows with negative emissions")
                    proj_feats = proj_feats[emissions >= 0].reset_index(drop=True)
            
            proj_feats["log_Attributed emissions"] = np.log(proj_feats["emissions"])
            
            # Add uncertainty for log emissions if available
            if "emissions_low" in proj_feats.columns:
                proj_feats["log_Attributed emissions_low"] = np.log(proj_feats["emissions_low"])
                proj_feats["log_Attributed emissions_high"] = np.log(proj_feats["emissions_high"])
            
            # Quick validation before model prediction
            if "log_Attributed emissions" in proj_feats.columns:
                log_emissions = proj_feats["log_Attributed emissions"]
                inf_count = np.isinf(log_emissions).sum()
                nan_count = log_emissions.isna().sum()
                
                if inf_count > 0 or nan_count > 0:
                    typer.echo(f"DEBUG: Found {inf_count} inf and {nan_count} NaN values in log_Attributed emissions")
                    # Drop problematic rows
                    valid_mask = np.isfinite(log_emissions)
                    proj_feats = proj_feats[valid_mask].reset_index(drop=True)
                    typer.echo(f"  - Kept {len(proj_feats)} valid rows")
            
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
            
            # Save company-aggregated data (one row per company per year with BU predictions)
            company_agg_filename = f"nze_{elec_source_name.lower()}_elec_{method}_company_aggregated.csv"
            company_agg_filepath = plant_data_dir / company_agg_filename
            proj_feats.to_csv(company_agg_filepath, index=False)
            typer.echo(f"      ✓ Saved company-aggregated data: {company_agg_filename}")
            
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
            
            # TODO: debug from here
            bu_sectoral_proj['Raw Emissions (Gt)'] = bu_sectoral_proj["emissions"] / 1e9
            bu_sectoral_proj[f'UR ({method})'] = bu_sectoral_proj["production"] / bu_sectoral_proj["capacity"]
            bu_sectoral_proj[f"Intensity ({method})"] = bu_sectoral_proj[f"BU emissions ({method})"] / bu_sectoral_proj["production"]
            
            # Add 2022 value for continuity (both emissions AND production)
            bu_emissions_22 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Emissions (Gt)"].iloc[0])
            bu_production_22 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Attributed production"].iloc[0])
            new_row = {"year": [2022], "Emissions (Gt)": [bu_emissions_22], "production": [bu_production_22]}
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
                method_label = "country_techno UR" if use_country_techno_ur else "country UR"
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
                emissions_label = f"Projected BU emissions ({'country_techno UR' if use_country_techno_ur else 'country UR'})"
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
        label='Uncertainty bounds'
    )
    handles_list.append(uncertainty_handle)
    labels_list.append('Uncertainty bounds')
    
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
    
    # Save intensity plot to intensity_evolution subfolder
    fig_intensity_path_pdf = intensity_dir / "nze_emission_intensity_evolution.pdf"
    fig_intensity_path_png = intensity_dir / "nze_emission_intensity_evolution.png"
    fig_intensity.savefig(fig_intensity_path_pdf, dpi=600, bbox_inches='tight')
    fig_intensity.savefig(fig_intensity_path_png, dpi=600, bbox_inches='tight')
    
    # ====================================================================
    # CREATE ELECTRICITY INTENSITY SENSITIVITY PLOTS FOR APS AND STEPS
    # ====================================================================
    typer.echo("Generating electricity intensity sensitivity plots for APS and STEPS scenarios...")
    
    # Store production data for APS and STEPS for debug plots
    production_data_all_scenarios = {"NZE": data_elec_sensitivity}
    
    # Process APS and STEPS scenarios
    for scenario_name in ["APS", "STEPS"]:
        typer.echo(f"  Processing {scenario_name} scenario...")
        
        # Get benchmark emissions for this scenario
        scenario_emissions = get_iea_emissions(scenario=scenario_name)
        x0_year = 2022
        x0_value = float(agg_bu_histo.loc[agg_bu_histo["year"] == x0_year, "Emissions (Gt)"].iloc[0])
        base_scenario_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=scenario_emissions)
        
        # Load carbon budget for this scenario
        cbudget_scenario = pd.read_excel(Path("./src/projections_") / f"proj_country_prod_{scenario_name.lower()}.xlsx")
        
        # Get projected costs for this scenario
        proj_costs_scenario = proj_costs_iea[scenario_name]
        
        # Create a figure with 3 subplots in one row
        fig_scenario_sensitivity, axs_scenario = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        
        # Store data for each electricity intensity source (for this scenario)
        data_scenario_sensitivity = {}
        production_data_all_scenarios[scenario_name] = {}
        
        # Plot for each electricity intensity source
        for idx, (elec_source_name, elec_int_data) in enumerate(elec_int_sources.items()):
            typer.echo(f"    Processing {elec_source_name} electricity intensity for {scenario_name}...")
            
            ax = axs_scenario[idx]
            ax.set_xlim(2017, 2031)
            ax.set_ylim(3.25, 4.75)
            ax.set_box_aspect(1)
            ax.grid("both")
            ax.set_title(f"{scenario_name} with {elec_source_name} electricity intensity", fontsize=12, weight='bold')
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
                x=[base_scenario_emissions.loc[base_scenario_emissions.index[0], "year"]], 
                y=[base_scenario_emissions.loc[base_scenario_emissions.index[0], "Emissions (Gt)"]],  
                marker='o', 
                color="grey",
                s=200,
                zorder=2,
                ax=ax
            )
            
            # Plot scenario reference trajectory
            sns.lineplot(data=base_scenario_emissions,
                        x='year',
                        y="Emissions (Gt)",
                        linewidth=2,
                        label=f"Projected IEA emissions ({scenario_name} slope)",
                        color="green",
                        linestyle="dotted",
                        zorder=2,
                        ax=ax)
            
            # Annotate 2022 value
            base_scenario_emissions_22 = float(base_scenario_emissions.loc[base_scenario_emissions['year'] == 2022, 'Emissions (Gt)'].iloc[0])
            ax.annotate(f"{base_scenario_emissions_22:.2f}", (2022, base_scenario_emissions_22 + 0.055), 
                       textcoords='offset points', xytext=(0, 10), ha='center', 
                       fontsize=12, color='grey', weight='bold')
            
            # Annotate 2030 value
            base_scenario_emissions_30 = float(base_scenario_emissions.loc[base_scenario_emissions['year'] == 2030, 'Emissions (Gt)'].iloc[0])
            ax.annotate(f"{base_scenario_emissions_30:.2f}", (2030.5, base_scenario_emissions_30), 
                       textcoords='offset points', xytext=(0, 10), ha='center', 
                       fontsize=10, color='green', weight='bold')
            
            # Calculate CAAGR for this electricity intensity source
            r_elec = get_elec_gen_caagr(elec_int=elec_int_data, start_year=2022, end_year=2030)
            
            # Define colors for different methods
            drivers_colors = {"constant_UR": "orange", "company_UR": "green"}
            methods_to_plot = ["constant_UR", "company_UR"]
            
            data_scenario_sensitivity[elec_source_name] = {}
            
            # Plot each method for this scenario with this electricity intensity
            for method in methods_to_plot:
                start_year, end_year = 2023, 2030
                
                # Get BU projected emissions
                proj_company, proj_plants = get_bu_proj_emissions(
                    gspt=gspt,
                    EF=EF,
                    glob_prod_iea=glob_prod[scenario_name], 
                    glob_bu_prod=glob_bu_prod,
                    market_share=market_share,
                    parent_group_map_path=parent_group_map_path,
                    gspt2refi_map=gspt2refi,
                    gspt2gspt_path=gspt2gspt_path,
                    proj_costs=proj_costs_scenario,
                    start_year=start_year,
                    end_year=end_year,
                    method=method,
                    glob_capa=glob_capa[scenario_name],
                    cbudget=cbudget_scenario,
                    use_country_techno_ur=use_country_techno_ur
                )
                
                # Convert uncertainty bounds to Gt if they exist
                if "Estimated emissions_low (ttpa)" in proj_plants.columns:
                    proj_plants["Emissions_low (Gt)"] = proj_plants["Estimated emissions_low (ttpa)"] / 1e6
                    proj_plants["Emissions_high (Gt)"] = proj_plants["Estimated emissions_high (ttpa)"] / 1e6
                
                # Determine level: NZE is global, APS and STEPS are regional
                if elec_source_name == "NZE":
                    elec_level = "global"
                else:  # APS or STEPS
                    elec_level = "region"
                
                typer.echo(f"      Applying {elec_source_name} electricity intensity at {elec_level} level for {method}...")
                
                # Apply electricity intensity adjustments
                proj_plants = get_proj_elec_int(
                    proj_plants, 
                    r=r_elec, 
                    energy_mix_path=energy_mix_path, 
                    level=elec_level,
                    scenario=elec_source_name,
                    world_fpath=world_fpath,
                    regions_fpath=regions_fpath,
                    gspt2iea_countries_path=gspt2iea_countries_path
                )
                
                # Apply EAF decarbonization if enabled
                if params.EAF_decarb:
                    base_all = 460
                    end_values = {
                        "NZE": 186,
                        "APS": 255,
                        "STEPS": 303
                    }
                    end_value = end_values[elec_source_name]
                    
                    caagr = get_cagr(base_value=base_all, end_value=end_value, n=8)
                    years = list(range(2023, 2031))
                    elec_cagr_df = pd.DataFrame({
                        "year": years,
                        "EAF_CAGR": [(1+caagr)**(i+1) for i in range(len(years))],
                    })
                    
                    if "CAGR" in proj_plants.columns:
                        proj_plants = proj_plants.drop(columns=["CAGR"])
                    
                    proj_plants = pd.merge(proj_plants, elec_cagr_df, on="year", how='left')
                    proj_plants["EAF_CAGR"] = (proj_plants["Main production process"] == "electric") * proj_plants["EAF_CAGR"]
                    proj_plants["EAF_CAGR"] = proj_plants["EAF_CAGR"].replace(0, 1)
                    proj_plants["EF"] = proj_plants["EF"] * proj_plants["EAF_CAGR"]
                    
                    if "EF_12_lower" in proj_plants.columns:
                        proj_plants["EF_12_lower"] = proj_plants["EF_12_lower"] * proj_plants["EAF_CAGR"]
                        proj_plants["EF_12_upper"] = proj_plants["EF_12_upper"] * proj_plants["EAF_CAGR"]
                        proj_plants["Emissions_low (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_lower"] / 1E6
                        proj_plants["Emissions_high (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_upper"] / 1E6
                    
                    proj_plants["Emissions (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF"] / 1E6
                    proj_plants = proj_plants.drop(columns=["EAF_CAGR"])
                
                # Save plant-level data before aggregation
                plant_filename = f"{scenario_name.lower()}_{elec_source_name.lower()}_elec_{method}_plants.xlsx"
                plant_filepath = plant_data_dir / plant_filename
                proj_plants.to_excel(plant_filepath, index=False)
                typer.echo(f"        ✓ Saved plant-level data: {plant_filename}")
                
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
                
                if "Emissions_low (Gt)" in proj_group.columns:
                    proj_group["Attributed emissions_low"] = proj_group['Emissions_low (Gt)'] * 1e9 * proj_group["Share"]
                    proj_group["Attributed emissions_high"] = proj_group['Emissions_high (Gt)'] * 1e9 * proj_group["Share"]
                
                # Aggregate features
                proj_feats = proj_group.groupby(['Group', 'year']).agg({
                    "Attributed emissions": "sum",
                    "Attributed production": "sum",
                    "Attributed capacity": "sum",
                }).rename(columns={
                    "Attributed emissions": "emissions",
                    "Attributed production": "production", 
                    "Attributed capacity": "capacity"
                })
                
                if "Attributed emissions_low" in proj_group.columns:
                    proj_feats_uncert = proj_group.groupby(['Group', 'year']).agg({
                        "Attributed emissions_low": "sum",
                        "Attributed emissions_high": "sum"
                    }).rename(columns={
                        "Attributed emissions_low": "emissions_low",
                        "Attributed emissions_high": "emissions_high"
                    })
                    proj_feats = pd.concat([proj_feats, proj_feats_uncert], axis=1)
                
                # Clean zero/negative emissions
                emissions = proj_feats["emissions"]
                zero_count = (emissions == 0).sum()
                negative_count = (emissions < 0).sum()
                
                if zero_count > 0 or negative_count > 0:
                    proj_feats = proj_feats[emissions > 0].reset_index(drop=True)
                
                proj_feats["log_Attributed emissions"] = np.log(proj_feats["emissions"])
                
                if "emissions_low" in proj_feats.columns:
                    proj_feats["log_Attributed emissions_low"] = np.log(proj_feats["emissions_low"])
                    proj_feats["log_Attributed emissions_high"] = np.log(proj_feats["emissions_high"])
                
                # Validate before model prediction
                if "log_Attributed emissions" in proj_feats.columns:
                    log_emissions = proj_feats["log_Attributed emissions"]
                    inf_count = np.isinf(log_emissions).sum()
                    nan_count = log_emissions.isna().sum()
                    
                    if inf_count > 0 or nan_count > 0:
                        valid_mask = np.isfinite(log_emissions)
                        proj_feats = proj_feats[valid_mask].reset_index(drop=True)
                
                # Predict using the model
                proj_feats[f"log BU emissions ({method})"] = model.predict(proj_feats[["log_Attributed emissions"]])
                proj_feats[f"BU emissions ({method})"] = np.exp(proj_feats[f"log BU emissions ({method})"])
                
                if "log_Attributed emissions_low" in proj_feats.columns:
                    proj_feats[f"log BU emissions_low ({method})"] = model.predict(proj_feats[["log_Attributed emissions_low"]].rename(columns={"log_Attributed emissions_low": "log_Attributed emissions"}))
                    proj_feats[f"log BU emissions_high ({method})"] = model.predict(proj_feats[["log_Attributed emissions_high"]].rename(columns={"log_Attributed emissions_high": "log_Attributed emissions"}))
                    proj_feats[f"BU emissions_low ({method})"] = np.exp(proj_feats[f"log BU emissions_low ({method})"])
                    proj_feats[f"BU emissions_high ({method})"] = np.exp(proj_feats[f"log BU emissions_high ({method})"])
                
                proj_feats[f"BU intensity ({method})"] = proj_feats[f"BU emissions ({method})"] / proj_feats["production"]
                proj_feats[f"BU intensity ({method})"] = proj_feats[f"BU intensity ({method})"].replace(np.inf, np.nan)
                proj_feats = proj_feats.reset_index()
                
                # Save company-aggregated data (one row per company per year with BU predictions)
                company_agg_filename = f"{scenario_name.lower()}_{elec_source_name.lower()}_elec_{method}_company_aggregated.csv"
                company_agg_filepath = plant_data_dir / company_agg_filename
                proj_feats.to_csv(company_agg_filepath, index=False)
                typer.echo(f"        ✓ Saved company-aggregated data: {company_agg_filename}")
                
                # Aggregate to sectoral level
                agg_cols = {
                    f"BU emissions ({method})": "sum",
                    "emissions": 'sum',
                    "production": "sum",
                    "capacity": "sum"
                }
                
                if f"BU emissions_low ({method})" in proj_feats.columns:
                    agg_cols[f"BU emissions_low ({method})"] = "sum"
                    agg_cols[f"BU emissions_high ({method})"] = "sum"
                
                bu_sectoral_proj = proj_feats.groupby("year").agg(agg_cols).reset_index()
                bu_sectoral_proj['Emissions (Gt)'] = bu_sectoral_proj[f"BU emissions ({method})"] / 1e9
                
                if f"BU emissions_low ({method})" in bu_sectoral_proj.columns:
                    bu_sectoral_proj['Emissions_low (Gt)'] = bu_sectoral_proj[f"BU emissions_low ({method})"] / 1e9
                    bu_sectoral_proj['Emissions_high (Gt)'] = bu_sectoral_proj[f"BU emissions_high ({method})"] / 1e9
                
                bu_sectoral_proj['Raw Emissions (Gt)'] = bu_sectoral_proj["emissions"] / 1e9
                bu_sectoral_proj[f'UR ({method})'] = bu_sectoral_proj["production"] / bu_sectoral_proj["capacity"]
                bu_sectoral_proj[f"Intensity ({method})"] = bu_sectoral_proj[f"BU emissions ({method})"] / bu_sectoral_proj["production"]
                
                # Add 2022 value for continuity (both emissions AND production)
                bu_emissions_22 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Emissions (Gt)"].iloc[0])
                bu_production_22 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Attributed production"].iloc[0])
                new_row = {"year": [2022], "Emissions (Gt)": [bu_emissions_22], "production": [bu_production_22]}
                bu_sectoral_proj = pd.concat([pd.DataFrame(new_row), bu_sectoral_proj], axis=0).sort_values(by="year", ascending=True)
                
                # Store data
                data_scenario_sensitivity[elec_source_name][method] = bu_sectoral_proj
                
                # Store ALL combinations for individual plots (not just matching elec intensity)
                if scenario_name not in ["NZE"]:  # NZE already stored in data_elec_sensitivity
                    if elec_source_name not in production_data_all_scenarios[scenario_name]:
                        production_data_all_scenarios[scenario_name][elec_source_name] = {}
                    production_data_all_scenarios[scenario_name][elec_source_name][method] = bu_sectoral_proj
                
                # Plot the trajectory
                if method == "company_UR":
                    emissions_label = "Projected BU emissions (constant market share)"
                elif method == "constant_UR":
                    emissions_label = f"Projected BU emissions ({'country_techno UR' if use_country_techno_ur else 'country UR'})"
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
                    
                    # Calculate error bar sizes
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
        handles_list_scenario = []
        labels_list_scenario = []
        for ax in axs_scenario:
            h, labels = ax.get_legend_handles_labels()
            for handle, label in zip(h, labels):
                if label not in labels_list_scenario:
                    handles_list_scenario.append(handle)
                    labels_list_scenario.append(label)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        
        # Add uncertainty bar representation to legend
        uncertainty_handle_scenario = Line2D(
            [], [], 
            color='gray', 
            marker='|', 
            linestyle='None',
            markersize=10, 
            markeredgewidth=1.5,
            alpha=0.5,
            label='Uncertainty bounds'
        )
        handles_list_scenario.append(uncertainty_handle_scenario)
        labels_list_scenario.append('Uncertainty bounds')
        
        fig_scenario_sensitivity.legend(
            handles_list_scenario, 
            labels_list_scenario, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(labels_list_scenario),
            frameon=True,
            fontsize=10
        )
        
        # Add overall title
        fig_scenario_sensitivity.suptitle(
            f"Sensitivity of {scenario_name} scenario to electricity intensity decarbonisation assumptions",
            fontsize=18, 
            weight="bold", 
            y=0.98
        )
        
        fig_scenario_sensitivity.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save the sensitivity plot
        fig_scenario_sensitivity_path_pdf = plots_dir / f"{scenario_name.lower()}_elec_intensity_sensitivity.pdf"
        fig_scenario_sensitivity_path_png = plots_dir / f"{scenario_name.lower()}_elec_intensity_sensitivity.png"
        fig_scenario_sensitivity.savefig(fig_scenario_sensitivity_path_pdf, dpi=600, bbox_inches='tight')
        fig_scenario_sensitivity.savefig(fig_scenario_sensitivity_path_png, dpi=600, bbox_inches='tight')
        
        typer.echo(f"  ✓ Saved {scenario_name} sensitivity plot: {fig_scenario_sensitivity_path_pdf}")
        
        # Save data to Excel
        scenario_data_for_excel = []
        
        # Add historical data
        historical_data_scenario = agg_bu_histo[['year', 'Emissions (Gt)', 'Attributed production']].copy()
        historical_data_scenario['period'] = 'historical'
        historical_data_scenario['scenario'] = scenario_name
        historical_data_scenario['elec_scenario'] = ''
        historical_data_scenario['method'] = ''
        historical_data_scenario['Emissions_low (Gt)'] = np.nan
        historical_data_scenario['Emissions_high (Gt)'] = np.nan
        historical_data_scenario['Emission Factor (tCO2/t)'] = (historical_data_scenario['Emissions (Gt)'] * 1e9) / historical_data_scenario['Attributed production']
        scenario_data_for_excel.append(historical_data_scenario[['year', 'period', 'scenario', 'elec_scenario', 'method',
                                                                  'Emissions (Gt)', 'Emissions_low (Gt)', 'Emissions_high (Gt)',
                                                                  'Attributed production', 'Emission Factor (tCO2/t)']])
        
        # Add reference trajectory
        scenario_reference_data = base_scenario_emissions[['year', 'Emissions (Gt)']].copy()
        scenario_reference_data['period'] = 'projected'
        scenario_reference_data['scenario'] = scenario_name
        scenario_reference_data['elec_scenario'] = scenario_name
        scenario_reference_data['method'] = 'reference'
        scenario_reference_data['Emissions_low (Gt)'] = np.nan
        scenario_reference_data['Emissions_high (Gt)'] = np.nan
        scenario_reference_data['Attributed production'] = np.nan
        scenario_reference_data['Emission Factor (tCO2/t)'] = np.nan
        scenario_data_for_excel.append(scenario_reference_data[['year', 'period', 'scenario', 'elec_scenario', 'method',
                                                                 'Emissions (Gt)', 'Emissions_low (Gt)', 'Emissions_high (Gt)',
                                                                 'Attributed production', 'Emission Factor (tCO2/t)']])
        
        # Add projection data for each electricity intensity and method
        for elec_source_name in ["NZE", "APS", "STEPS"]:
            for method in ["constant_UR", "company_UR"]:
                if elec_source_name in data_scenario_sensitivity and method in data_scenario_sensitivity[elec_source_name]:
                    proj_data = data_scenario_sensitivity[elec_source_name][method].copy()
                    proj_data['period'] = 'projected'
                    proj_data['scenario'] = scenario_name
                    proj_data['elec_scenario'] = elec_source_name
                    proj_data['method'] = method
                    
                    # Calculate emission factor
                    if 'production' in proj_data.columns:
                        proj_data['Emission Factor (tCO2/t)'] = (proj_data['Emissions (Gt)'] * 1e9) / proj_data['production']
                        proj_data = proj_data.rename(columns={'production': 'Attributed production'})
                    else:
                        proj_data['Emission Factor (tCO2/t)'] = np.nan
                        proj_data['Attributed production'] = np.nan
                    
                    # Ensure uncertainty columns exist
                    if 'Emissions_low (Gt)' not in proj_data.columns:
                        proj_data['Emissions_low (Gt)'] = np.nan
                    if 'Emissions_high (Gt)' not in proj_data.columns:
                        proj_data['Emissions_high (Gt)'] = np.nan
                    
                    scenario_data_for_excel.append(proj_data[['year', 'period', 'scenario', 'elec_scenario', 'method',
                                                              'Emissions (Gt)', 'Emissions_low (Gt)', 'Emissions_high (Gt)',
                                                              'Attributed production', 'Emission Factor (tCO2/t)']])
        
        # Combine and save
        scenario_combined_data = pd.concat(scenario_data_for_excel, axis=0, ignore_index=True)
        scenario_excel_path = plots_data_dir / f"{scenario_name.lower()}_elec_intensity_sensitivity_data.xlsx"
        scenario_combined_data.to_excel(scenario_excel_path, index=False, sheet_name=f'{scenario_name} Sensitivity Data')
        typer.echo(f"  ✓ Saved {scenario_name} data: {scenario_excel_path}")
    
    # ====================================================================
    # CREATE INDIVIDUAL ELECTRICITY SENSITIVITY PLOTS BY METHOD (6 FIGURES)
    # ====================================================================
    typer.echo("\nGenerating individual electricity sensitivity plots by method...")
    
    # Define method directories
    method_dirs = {
        "constant_UR": constant_ur_dir,
        "company_UR": company_ur_dir
    }
    
    method_labels_short = {
        "constant_UR": f"{'country_techno UR' if use_country_techno_ur else 'country UR'}",
        "company_UR": "constant market share"
    }
    
    # For each method, create 3 figures (one per scenario)
    # Each figure has 3 subplots (one per electricity intensity)
    for method in ["constant_UR", "company_UR"]:
        typer.echo(f"\n  Creating plots for {method_labels_short[method]} method...")
        method_dir = method_dirs[method]
        
        # For each scenario
        for scenario_name in ["NZE", "APS", "STEPS"]:
            typer.echo(f"    - {scenario_name} scenario (3 electricity intensities)")
            
            # Get reference emissions for this scenario
            if scenario_name == "NZE":
                base_scenario_emissions = base_nze_emissions
            else:
                scenario_emissions = get_iea_emissions(scenario=scenario_name)
                x0_year = 2022
                x0_value = float(agg_bu_histo.loc[agg_bu_histo["year"] == x0_year, "Emissions (Gt)"].iloc[0])
                base_scenario_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=scenario_emissions)
            
            # Create figure with 3 subplots (one per electricity intensity)
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
            
            # For each electricity intensity
            for idx, elec_source_name in enumerate(["NZE", "APS", "STEPS"]):
                ax = axs[idx]
                ax.set_xlim(2017, 2031)
                ax.set_ylim(3.25, 4.75)
                ax.set_box_aspect(1)
                ax.grid("both", alpha=0.3)
                ax.set_title(f"{elec_source_name} electricity", fontsize=12, weight='bold')
                ax.set_ylabel("Emissions (GtCO₂ eq)")
                ax.set_xlabel("Year")
                
                # Plot historical BU emissions
                sns.lineplot(
                    data=agg_bu_histo,
                    x='year',
                    y="Emissions (Gt)",
                    label="Historical BU emissions",
                    ax=ax,
                    linewidth=2.5,
                    color='#1f77b4'
                )
                
                # Add starting point marker
                ax.scatter(
                    [base_scenario_emissions.loc[base_scenario_emissions.index[0], "year"]], 
                    [base_scenario_emissions.loc[base_scenario_emissions.index[0], "Emissions (Gt)"]],  
                    marker='o', 
                    color="grey",
                    s=200,
                    zorder=3,
                    edgecolors='black',
                    linewidth=2
                )
                
                # Plot scenario reference trajectory
                sns.lineplot(
                    data=base_scenario_emissions,
                    x='year',
                    y="Emissions (Gt)",
                    linewidth=2.5,
                    label=f"IEA {scenario_name} reference",
                    color="darkgreen",
                    linestyle="dotted",
                    zorder=2,
                    ax=ax
                )
                
                # Annotate 2022 value
                base_scenario_emissions_22 = float(base_scenario_emissions.loc[base_scenario_emissions['year'] == 2022, 'Emissions (Gt)'].iloc[0])
                ax.annotate(
                    f"{base_scenario_emissions_22:.2f}", 
                    (2022, base_scenario_emissions_22 + 0.055), 
                    textcoords='offset points', 
                    xytext=(0, 10), 
                    ha='center', 
                    fontsize=11, 
                    color='grey', 
                    weight='bold'
                )
                
                # Annotate 2030 value
                base_scenario_emissions_30 = float(base_scenario_emissions.loc[base_scenario_emissions['year'] == 2030, 'Emissions (Gt)'].iloc[0])
                ax.annotate(
                    f"{base_scenario_emissions_30:.2f}", 
                    (2030.5, base_scenario_emissions_30), 
                    textcoords='offset points', 
                    xytext=(0, 10), 
                    ha='center', 
                    fontsize=10, 
                    color='darkgreen', 
                    weight='bold'
                )
                
                # Get the projection data for this combination
                bu_data = None
                
                if scenario_name == "NZE":
                    if elec_source_name in data_elec_sensitivity and method in data_elec_sensitivity[elec_source_name]:
                        bu_data = data_elec_sensitivity[elec_source_name][method].copy()
                elif scenario_name in production_data_all_scenarios:
                    if elec_source_name in production_data_all_scenarios[scenario_name]:
                        if method in production_data_all_scenarios[scenario_name][elec_source_name]:
                            bu_data = production_data_all_scenarios[scenario_name][elec_source_name][method].copy()
                
                if bu_data is not None and not bu_data.empty:
                    # Plot the BU emissions trajectory (2022 onwards)
                    emissions_label = f"BU emissions ({method_labels_short[method]})"
                    
                    # Main line
                    sns.lineplot(
                        data=bu_data,
                        x='year',
                        y="Emissions (Gt)",
                        label=emissions_label,
                        linestyle="dashed",
                        linewidth=2.5,
                        color='#ff7f0e',
                        ax=ax
                    )
                    
                    # Add uncertainty bands starting from 2022
                    if "Emissions_low (Gt)" in bu_data.columns and "Emissions_high (Gt)" in bu_data.columns:
                        # Get 2022 emissions value (no uncertainty at 2022, it's historical)
                        emissions_2022 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Emissions (Gt)"].iloc[0])
                        
                        # Get projected data (2023 onwards) with uncertainty
                        bu_data_proj = bu_data[bu_data['year'] >= 2023].copy()
                        bu_data_proj = bu_data_proj.dropna(subset=['Emissions_low (Gt)', 'Emissions_high (Gt)'])
                        
                        if not bu_data_proj.empty:
                            # Prepend 2022 value (same for low, central, and high at 2022)
                            years_with_2022 = [2022] + bu_data_proj['year'].tolist()
                            emissions_low_with_2022 = [emissions_2022] + bu_data_proj['Emissions_low (Gt)'].tolist()
                            emissions_high_with_2022 = [emissions_2022] + bu_data_proj['Emissions_high (Gt)'].tolist()
                            
                            # Plot smooth uncertainty band from 2022
                            ax.fill_between(
                                years_with_2022,
                                emissions_low_with_2022,
                                emissions_high_with_2022,
                                alpha=0.2,
                                color='#ff7f0e',
                                label='Uncertainty bounds'
                            )
                else:
                    # Data not available - add text message
                    ax.text(
                        0.5, 0.5, 
                        f"Data not available\nfor this combination",
                        transform=ax.transAxes, 
                        ha='center', 
                        va='center',
                        fontsize=12, 
                        alpha=0.3, 
                        style='italic'
                    )
            
            # Create shared legend below all subplots
            handles_list_method = []
            labels_list_method = []
            for ax in axs:
                h, labels = ax.get_legend_handles_labels()
                for handle, label in zip(h, labels):
                    if label not in labels_list_method:
                        handles_list_method.append(handle)
                        labels_list_method.append(label)
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            
            fig.legend(
                handles_list_method,
                labels_list_method,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(labels_list_method),
                frameon=True,
                fontsize=10
            )
            
            # Add overall title
            fig.suptitle(
                f"{scenario_name} Scenario - Electricity Intensity Sensitivity ({method_labels_short[method]})",
                fontsize=18,
                weight="bold",
                y=0.98
            )
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Save the figure
            fig_filename = f"{scenario_name.lower()}_sensitivity_{method}"
            fig_path_pdf = method_dir / f"{fig_filename}.pdf"
            fig_path_png = method_dir / f"{fig_filename}.png"
            fig.savefig(fig_path_pdf, dpi=600, bbox_inches='tight')
            fig.savefig(fig_path_png, dpi=600, bbox_inches='tight')
            plt.close(fig)
    
    typer.echo(f"\n  ✓ Created 6 method-specific sensitivity figures (3 scenarios × 2 methods):")
    typer.echo(f"    - {constant_ur_dir}: 3 figures for {method_labels_short['constant_UR']}")
    typer.echo(f"    - {company_ur_dir}: 3 figures for {method_labels_short['company_UR']}")
    typer.echo(f"    - Each figure has 3 subplots (NZE, APS, STEPS electricity intensities)")
    
    # ====================================================================
    # CREATE NORMALIZED VERSIONS OF METHOD-SPECIFIC PLOTS (6 MORE FIGURES)
    # ====================================================================
    typer.echo("\nGenerating normalized versions of method-specific sensitivity plots...")
    
    # Get 2022 baseline for normalization
    emissions_2022_baseline = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Emissions (Gt)"].iloc[0])
    
    # For each method, create 3 normalized figures (one per scenario)
    for method in ["constant_UR", "company_UR"]:
        typer.echo(f"\n  Creating normalized plots for {method_labels_short[method]} method...")
        method_dir = method_dirs[method]
        
        # For each scenario
        for scenario_name in ["NZE", "APS", "STEPS"]:
            typer.echo(f"    - {scenario_name} scenario (normalized)")
            
            # Get reference emissions for this scenario
            if scenario_name == "NZE":
                base_scenario_emissions = base_nze_emissions
            else:
                scenario_emissions = get_iea_emissions(scenario=scenario_name)
                x0_year = 2022
                x0_value = float(agg_bu_histo.loc[agg_bu_histo["year"] == x0_year, "Emissions (Gt)"].iloc[0])
                base_scenario_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=scenario_emissions)
            
            # Create figure with 3 subplots (one per electricity intensity)
            fig_norm, axs_norm = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
            
            # Define scenario-specific y-axis bounds for normalized plots
            ylim_bounds = {
                "NZE": (0.8, 1.05),
                "APS": (0.9, 1.05),
                "STEPS": (0.9, 1.05)
            }
            ylim_min, ylim_max = ylim_bounds[scenario_name]
            
            # For each electricity intensity
            for idx, elec_source_name in enumerate(["NZE", "APS", "STEPS"]):
                ax = axs_norm[idx]
                ax.set_xlim(2017, 2031)
                ax.set_ylim(ylim_min, ylim_max)
                ax.set_box_aspect(1)
                ax.grid("both", alpha=0.3)
                ax.set_title(f"{elec_source_name} electricity", fontsize=12, weight='bold')
                ax.set_ylabel("Normalized Emissions (2022 = 1.0)")
                ax.set_xlabel("Year")
                
                # Plot historical BU emissions (normalized)
                historical_norm = agg_bu_histo[['year', 'Emissions (Gt)']].copy()
                historical_norm['Normalized Emissions'] = historical_norm['Emissions (Gt)'] / emissions_2022_baseline
                sns.lineplot(
                    data=historical_norm,
                    x='year',
                    y="Normalized Emissions",
                    label="Historical BU emissions",
                    ax=ax,
                    linewidth=2.5,
                    color='#1f77b4'
                )
                
                # Add starting point marker at 2022
                ax.scatter([2022], [1.0], marker='o', color="grey", s=200, zorder=3, edgecolors='black', linewidth=2)
                ax.annotate(
                    "1.00", 
                    (2022, 1.0 + 0.015), 
                    textcoords='offset points', 
                    xytext=(0, 10), 
                    ha='center', 
                    fontsize=11, 
                    color='grey', 
                    weight='bold'
                )
                
                # Plot scenario reference trajectory (normalized)
                base_scenario_norm = base_scenario_emissions[['year', 'Emissions (Gt)']].copy()
                base_scenario_norm['Normalized Emissions'] = base_scenario_norm['Emissions (Gt)'] / emissions_2022_baseline
                sns.lineplot(
                    data=base_scenario_norm,
                    x='year',
                    y="Normalized Emissions",
                    linewidth=2.5,
                    label=f"IEA {scenario_name} reference",
                    color="darkgreen",
                    linestyle="dotted",
                    zorder=2,
                    ax=ax
                )
                
                # Annotate 2030 normalized value
                base_scenario_norm_30 = float(base_scenario_norm.loc[base_scenario_norm['year'] == 2030, 'Normalized Emissions'].iloc[0])
                ax.annotate(
                    f"{base_scenario_norm_30:.2f}", 
                    (2030.5, base_scenario_norm_30), 
                    textcoords='offset points', 
                    xytext=(0, 10), 
                    ha='center', 
                    fontsize=10, 
                    color='darkgreen', 
                    weight='bold'
                )
                
                # Get the projection data for this combination
                bu_data = None
                
                if scenario_name == "NZE":
                    if elec_source_name in data_elec_sensitivity and method in data_elec_sensitivity[elec_source_name]:
                        bu_data = data_elec_sensitivity[elec_source_name][method].copy()
                elif scenario_name in production_data_all_scenarios:
                    if elec_source_name in production_data_all_scenarios[scenario_name]:
                        if method in production_data_all_scenarios[scenario_name][elec_source_name]:
                            bu_data = production_data_all_scenarios[scenario_name][elec_source_name][method].copy()
                
                if bu_data is not None and not bu_data.empty:
                    # Normalize the emissions
                    bu_data_norm = bu_data.copy()
                    bu_data_norm['Normalized Emissions'] = bu_data_norm['Emissions (Gt)'] / emissions_2022_baseline
                    
                    # Plot the BU emissions trajectory (normalized)
                    emissions_label = f"BU emissions ({method_labels_short[method]})"
                    
                    # Main line
                    sns.lineplot(
                        data=bu_data_norm,
                        x='year',
                        y="Normalized Emissions",
                        label=emissions_label,
                        linestyle="dashed",
                        linewidth=2.5,
                        color='#ff7f0e',
                        ax=ax
                    )
                    
                    # Add uncertainty bands starting from 2022 (normalized)
                    if "Emissions_low (Gt)" in bu_data.columns and "Emissions_high (Gt)" in bu_data.columns:
                        # Get projected data (2023 onwards) with uncertainty
                        bu_data_proj = bu_data[bu_data['year'] >= 2023].copy()
                        bu_data_proj = bu_data_proj.dropna(subset=['Emissions_low (Gt)', 'Emissions_high (Gt)'])
                        
                        if not bu_data_proj.empty:
                            # Normalize uncertainty bounds
                            emissions_low_norm = bu_data_proj['Emissions_low (Gt)'] / emissions_2022_baseline
                            emissions_high_norm = bu_data_proj['Emissions_high (Gt)'] / emissions_2022_baseline
                            
                            # Prepend 2022 value (1.0 for normalized)
                            years_with_2022 = [2022] + bu_data_proj['year'].tolist()
                            emissions_low_with_2022 = [1.0] + emissions_low_norm.tolist()
                            emissions_high_with_2022 = [1.0] + emissions_high_norm.tolist()
                            
                            # Plot smooth uncertainty band from 2022
                            ax.fill_between(
                                years_with_2022,
                                emissions_low_with_2022,
                                emissions_high_with_2022,
                                alpha=0.2,
                                color='#ff7f0e',
                                label='Uncertainty bounds'
                            )
            
            # Create shared legend below all subplots
            handles_list_norm = []
            labels_list_norm = []
            for ax in axs_norm:
                h, labels = ax.get_legend_handles_labels()
                for handle, label in zip(h, labels):
                    if label not in labels_list_norm:
                        handles_list_norm.append(handle)
                        labels_list_norm.append(label)
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            
            fig_norm.legend(
                handles_list_norm,
                labels_list_norm,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(labels_list_norm),
                frameon=True,
                fontsize=10
            )
            
            # Add overall title
            fig_norm.suptitle(
                f"{scenario_name} Scenario - Electricity Intensity Sensitivity ({method_labels_short[method]})\nNormalized to 2022 = 1.0",
                fontsize=16,
                weight="bold",
                y=0.98
            )
            
            fig_norm.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Save the normalized figure
            fig_filename_norm = f"{scenario_name.lower()}_sensitivity_{method}_normalized"
            fig_path_pdf_norm = method_dir / f"{fig_filename_norm}.pdf"
            fig_path_png_norm = method_dir / f"{fig_filename_norm}.png"
            fig_norm.savefig(fig_path_pdf_norm, dpi=600, bbox_inches='tight')
            fig_norm.savefig(fig_path_png_norm, dpi=600, bbox_inches='tight')
            plt.close(fig_norm)
    
    typer.echo(f"\n  ✓ Created 6 normalized method-specific sensitivity figures:")
    typer.echo(f"    - {constant_ur_dir}: 3 normalized figures")
    typer.echo(f"    - {company_ur_dir}: 3 normalized figures")
    typer.echo(f"    - Total: 12 method-specific figures (6 absolute + 6 normalized)")
    
    # ====================================================================
    # CREATE NORMALIZED EMISSIONS PLOT
    # ====================================================================
    typer.echo("Generating normalized emissions plot...")
    
    # Get 2022 historical emissions value as baseline
    historical_2022_emissions = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Emissions (Gt)"].iloc[0])
    typer.echo(f"  Using 2022 baseline emissions: {historical_2022_emissions:.3f} Gt CO2")
    
    # Create normalized emissions figure with same layout as original
    fig_emis_norm, axs_emis_norm = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Plot for each electricity intensity source
    for idx, (elec_source_name, elec_int_data) in enumerate(elec_int_sources.items()):
        ax = axs_emis_norm[idx]
        ax.set_xlim(2017, 2031)
        ax.set_ylim(0.8, 1.05)  # NZE scenario normalized bounds
        ax.set_box_aspect(1)
        ax.grid("both")
        ax.set_title(f"NZE with {elec_source_name} electricity intensity", fontsize=12, weight='bold')
        ax.set_ylabel("Normalized Emissions (2022 = 1.0)")
        ax.set_xlabel("Year")
        
        # Plot historical BU emissions (normalized)
        historical_norm = agg_bu_histo[['year', 'Emissions (Gt)']].copy()
        historical_norm['Normalized Emissions'] = historical_norm['Emissions (Gt)'] / historical_2022_emissions
        sns.lineplot(data=historical_norm,
                    x='year',
                    y="Normalized Emissions",
                    label="Historical BU emissions",
                    ax=ax)
        
        # Add starting point marker at 2022
        ax.scatter([2022], [1.0], marker='o', color="grey", s=200, zorder=2)
        ax.annotate(f"{1.0:.2f}", (2022, 1.0 + 0.015), textcoords='offset points',
                   xytext=(0, 10), ha='center', fontsize=12, color='grey', weight='bold')
        
        # Plot NZE reference trajectory (normalized)
        nze_ref_norm = base_nze_emissions[['year', 'Emissions (Gt)']].copy()
        nze_ref_norm['Normalized Emissions'] = nze_ref_norm['Emissions (Gt)'] / historical_2022_emissions
        sns.lineplot(data=nze_ref_norm,
                    x='year',
                    y="Normalized Emissions",
                    linewidth=2,
                    label="Projected IEA emissions (NZE slope)",
                    color="green",
                    linestyle="dotted",
                    zorder=2,
                    ax=ax)
        
        # Annotate 2030 NZE reference value
        nze_ref_2030 = float(nze_ref_norm.loc[nze_ref_norm['year'] == 2030, 'Normalized Emissions'].iloc[0])
        ax.annotate(f"{nze_ref_2030:.2f}", (2030.5, nze_ref_2030), textcoords='offset points', 
                   xytext=(0, 10), ha='center', fontsize=10, color='green', weight='bold')
        
        # Plot each method's trajectory (normalized)
        drivers_colors = {"constant_UR": "orange", "company_UR": "green"}
        for method in ["constant_UR", "company_UR"]:
            if elec_source_name in data_elec_sensitivity and method in data_elec_sensitivity[elec_source_name]:
                bu_data = data_elec_sensitivity[elec_source_name][method].copy()
                bu_data['Normalized Emissions'] = bu_data['Emissions (Gt)'] / historical_2022_emissions
                
                if method == "company_UR":
                    emissions_label = "Projected BU emissions (constant market share)"
                elif method == "constant_UR":
                    emissions_label = f"Projected BU emissions ({'country_techno UR' if use_country_techno_ur else 'country UR'})"
                else:
                    emissions_label = f"Projected BU emissions ({method})"
                
                sns.lineplot(data=bu_data,
                            x='year',
                            y="Normalized Emissions",
                            label=emissions_label,
                            linestyle="dashed",
                            color=drivers_colors[method],
                            ax=ax)
                
                # Add error bars if uncertainty bounds are available (also normalized)
                if "Emissions_low (Gt)" in bu_data.columns and "Emissions_high (Gt)" in bu_data.columns:
                    years = bu_data['year']
                    emissions_norm = bu_data["Normalized Emissions"]
                    emissions_low_norm = bu_data["Emissions_low (Gt)"] / historical_2022_emissions
                    emissions_high_norm = bu_data["Emissions_high (Gt)"] / historical_2022_emissions
                    
                    # Calculate error bar sizes (distance from center to bounds)
                    yerr_lower = emissions_norm - emissions_low_norm
                    yerr_upper = emissions_high_norm - emissions_norm
                    
                    # Plot error bars
                    ax.errorbar(years, emissions_norm, 
                               yerr=[yerr_lower, yerr_upper],
                               fmt='none', 
                               color=drivers_colors[method],
                               alpha=0.3,
                               capsize=3,
                               capthick=1)
    
    # Add overall title
    fig_emis_norm.suptitle(
        "Sensitivity of NZE scenario to electricity intensity decarbonisation assumptions\n(Normalized to 2022 emissions = 1.0)",
        fontsize=16, 
        weight="bold", 
        y=1.02
    )
    
    # Create a single shared legend
    handles_list = []
    labels_list = []
    for ax in axs_emis_norm:
        h, labels = ax.get_legend_handles_labels()
        for handle, label in zip(h, labels):
            if label not in labels_list:
                handles_list.append(handle)
                labels_list.append(label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    # Add uncertainty bar representation to legend
    uncertainty_handle = Line2D(
        [], [], 
        color='gray', 
        marker='|', 
        linestyle='None',
        markersize=10, 
        markeredgewidth=1.5,
        alpha=0.5,
        label='Uncertainty bounds'
    )
    handles_list.append(uncertainty_handle)
    labels_list.append('Uncertainty bounds')
    
    fig_emis_norm.legend(
        handles_list, 
        labels_list, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels_list),
        frameon=True,
        fontsize=10
    )
    
    fig_emis_norm.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save normalized emissions plot to normalized subfolder
    fig_emis_norm_path_pdf = normalized_dir / "nze_elec_intensity_sensitivity_normalized.pdf"
    fig_emis_norm_path_png = normalized_dir / "nze_elec_intensity_sensitivity_normalized.png"
    fig_emis_norm.savefig(fig_emis_norm_path_pdf, dpi=600, bbox_inches='tight')
    fig_emis_norm.savefig(fig_emis_norm_path_png, dpi=600, bbox_inches='tight')
    
    # ====================================================================
    # CREATE NORMALIZED EMISSIONS PLOT - METHODS COMPARISON (NZE ONLY)
    # ====================================================================
    typer.echo("Generating normalized emissions plot comparing methods...")
    
    # Methods to compare
    methods_to_compare = ["carbon_efficiency", "carbon_intensity", "company_UR"]
    method_labels = {
        "carbon_efficiency": "Carbon Efficiency",
        "carbon_intensity": "Carbon Intensity", 
        "company_UR": "Constant Market Share"
    }
    
    # Store data for methods comparison
    data_methods_comparison = {}
    
    # Create figure with 3 subplots
    fig_methods_norm, axs_methods_norm = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    typer.echo("  Computing projections for methods comparison...")
    
    # Run projections for each method using NZE scenario
    for method in methods_to_compare:
        typer.echo(f"    Processing {method}...")
        
        start_year, end_year = 2023, 2030
        
        # Get BU projected emissions using NZE scenario
        proj_company, proj_plants = get_bu_proj_emissions(
            gspt=gspt,
            EF=EF,
            glob_prod_iea=glob_prod["NZE"], 
            glob_bu_prod=glob_bu_prod,
            market_share=market_share,
            parent_group_map_path=parent_group_map_path,
            gspt2refi_map=gspt2refi,
            gspt2gspt_path=gspt2gspt_path,
            proj_costs=proj_costs_iea["NZE"],
            start_year=start_year,
            end_year=end_year,
            method=method,
            glob_capa=glob_capa["NZE"],
            cbudget=pd.read_excel(Path("./src/projections_") / "proj_country_prod_nze.xlsx"),
            use_country_techno_ur=use_country_techno_ur
        )
        
        # Convert uncertainty bounds to Gt if they exist
        if "Estimated emissions_low (ttpa)" in proj_plants.columns:
            proj_plants["Emissions_low (Gt)"] = proj_plants["Estimated emissions_low (ttpa)"] / 1e6
            proj_plants["Emissions_high (Gt)"] = proj_plants["Estimated emissions_high (ttpa)"] / 1e6
        
        # Apply NZE electricity intensity (global level)
        r_elec_nze = get_elec_gen_caagr(elec_int=elec_int_nze, start_year=2022, end_year=2030)
        proj_plants = get_proj_elec_int(
            proj_plants, 
            r=r_elec_nze, 
            energy_mix_path=energy_mix_path, 
            level="global",
            scenario="NZE",
            world_fpath=world_fpath,
            regions_fpath=regions_fpath,
            gspt2iea_countries_path=gspt2iea_countries_path
        )
        
        # Apply EAF decarbonization if enabled
        if params.EAF_decarb:
            base_all = 460
            end_nze = 186
            caagr = get_cagr(base_value=base_all, end_value=end_nze, n=8)
            years = list(range(2023, 2031))
            elec_cagr_df = pd.DataFrame({
                "year": years,
                "EAF_CAGR": [(1+caagr)**(i+1) for i in range(len(years))],
            })
            
            if "CAGR" in proj_plants.columns:
                proj_plants = proj_plants.drop(columns=["CAGR"])
            
            proj_plants = pd.merge(proj_plants, elec_cagr_df, on="year", how='left')
            proj_plants["EAF_CAGR"] = (proj_plants["Main production process"] == "electric") * proj_plants["EAF_CAGR"]
            proj_plants["EAF_CAGR"] = proj_plants["EAF_CAGR"].replace(0, 1)
            proj_plants["EF"] = proj_plants["EF"] * proj_plants["EAF_CAGR"]
            
            if "EF_12_lower" in proj_plants.columns:
                proj_plants["EF_12_lower"] = proj_plants["EF_12_lower"] * proj_plants["EAF_CAGR"]
                proj_plants["EF_12_upper"] = proj_plants["EF_12_upper"] * proj_plants["EAF_CAGR"]
                proj_plants["Emissions_low (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_lower"] / 1E6
                proj_plants["Emissions_high (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF_12_upper"] / 1E6
            
            proj_plants["Emissions (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF"] / 1E6
            proj_plants = proj_plants.drop(columns=["EAF_CAGR"])
        
        # Convert to parent company level
        proj_group = convert_plant2parent(proj_plants, gspt2gspt_path=gspt2gspt_path, parent_group_map=parent_group_map)
        
        # Calculate attributed values
        proj_group["Attributed crude steel capacity (ttpa)"] = proj_group['Nominal crude steel capacity (ttpa)'] * proj_group["Share"]
        proj_group["Attributed capacity"] = proj_group["Attributed crude steel capacity (ttpa)"] * 1E3
        proj_group["Attributed emissions"] = proj_group['Emissions (Gt)'] * 1e9 * proj_group["Share"]
        proj_group["Attributed production"] = proj_group['Estimated crude steel production (ttpa)'] * 1e3 * proj_group["Share"]
        
        if "Emissions_low (Gt)" in proj_group.columns:
            proj_group["Attributed emissions_low"] = proj_group['Emissions_low (Gt)'] * 1e9 * proj_group["Share"]
            proj_group["Attributed emissions_high"] = proj_group['Emissions_high (Gt)'] * 1e9 * proj_group["Share"]
        
        # Quick cleanup of zero attributed emissions
        if "Attributed emissions" in proj_group.columns:
            attr_emissions = proj_group["Attributed emissions"]
            zero_count = (attr_emissions == 0).sum()
            negative_count = (attr_emissions < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                total_rows = len(proj_group)
                fraction_discarded = (zero_count + negative_count) / total_rows
                total_capacity_discarded = proj_group[attr_emissions <= 0]["Attributed crude steel capacity (ttpa)"].sum()
                
                typer.echo(f"DEBUG: Dropping {zero_count + negative_count} rows with zero/negative attributed emissions")
                typer.echo(f"  - Fraction discarded: {fraction_discarded:.4f} ({fraction_discarded*100:.2f}%)")
                typer.echo(f"  - Total capacity discarded: {total_capacity_discarded:.2f} ttpa")
                
                # Drop problematic rows
                proj_group = proj_group[attr_emissions > 0].reset_index(drop=True)
                typer.echo(f"  - Remaining rows after cleanup: {len(proj_group)}")
        
        # Aggregate features
        proj_feats = proj_group.groupby(['Group', 'year']).agg({
            "Attributed emissions": "sum",
            "Attributed production": "sum",
            "Attributed capacity": "sum",
        }).rename(columns={
            "Attributed emissions": "emissions",
            "Attributed production": "production", 
            "Attributed capacity": "capacity"
        })
        
        if "Attributed emissions_low" in proj_group.columns:
            proj_feats_uncert = proj_group.groupby(['Group', 'year']).agg({
                "Attributed emissions_low": "sum",
                "Attributed emissions_high": "sum"
            }).rename(columns={
                "Attributed emissions_low": "emissions_low",
                "Attributed emissions_high": "emissions_high"
            })
            proj_feats = pd.concat([proj_feats, proj_feats_uncert], axis=1)
        
        # Quick check for problematic values before taking log (second call)
        emissions = proj_feats["emissions"]
        zero_count = (emissions == 0).sum()
        negative_count = (emissions < 0).sum()
        
        if zero_count > 0 or negative_count > 0:
            typer.echo(f"DEBUG: Found {zero_count} zero and {negative_count} negative emissions values (second call)")
            if zero_count > 0:
                typer.echo(f"  - Dropping {zero_count} rows with zero emissions")
                proj_feats = proj_feats[emissions > 0].reset_index(drop=True)
            if negative_count > 0:
                typer.echo(f"  - Dropping {negative_count} rows with negative emissions")
                proj_feats = proj_feats[emissions >= 0].reset_index(drop=True)
        
        proj_feats["log_Attributed emissions"] = np.log(proj_feats["emissions"])
        
        if "emissions_low" in proj_feats.columns:
            proj_feats["log_Attributed emissions_low"] = np.log(proj_feats["emissions_low"])
            proj_feats["log_Attributed emissions_high"] = np.log(proj_feats["emissions_high"])
        
        # Quick validation before model prediction (second call)
        if "log_Attributed emissions" in proj_feats.columns:
            log_emissions = proj_feats["log_Attributed emissions"]
            inf_count = np.isinf(log_emissions).sum()
            nan_count = log_emissions.isna().sum()
            
            if inf_count > 0 or nan_count > 0:
                typer.echo(f"DEBUG: Found {inf_count} inf and {nan_count} NaN values in log_Attributed emissions (second call)")
                # Drop problematic rows
                valid_mask = np.isfinite(log_emissions)
                proj_feats = proj_feats[valid_mask].reset_index(drop=True)
                typer.echo(f"  - Kept {len(proj_feats)} valid rows")
        
        # Predict using the model
        proj_feats[f"log BU emissions ({method})"] = model.predict(proj_feats[["log_Attributed emissions"]])
        proj_feats[f"BU emissions ({method})"] = np.exp(proj_feats[f"log BU emissions ({method})"])
        
        if "log_Attributed emissions_low" in proj_feats.columns:
            proj_feats[f"log BU emissions_low ({method})"] = model.predict(proj_feats[["log_Attributed emissions_low"]].rename(columns={"log_Attributed emissions_low": "log_Attributed emissions"}))
            proj_feats[f"log BU emissions_high ({method})"] = model.predict(proj_feats[["log_Attributed emissions_high"]].rename(columns={"log_Attributed emissions_high": "log_Attributed emissions"}))
            proj_feats[f"BU emissions_low ({method})"] = np.exp(proj_feats[f"log BU emissions_low ({method})"])
            proj_feats[f"BU emissions_high ({method})"] = np.exp(proj_feats[f"log BU emissions_high ({method})"])
        
        proj_feats = proj_feats.reset_index()
        
        # Aggregate to sectoral level
        agg_cols = {
            f"BU emissions ({method})": "sum",
            "emissions": 'sum',
            "production": "sum",
            "capacity": "sum"
        }
        
        if f"BU emissions_low ({method})" in proj_feats.columns:
            agg_cols[f"BU emissions_low ({method})"] = "sum"
            agg_cols[f"BU emissions_high ({method})"] = "sum"
        
        bu_sectoral_proj = proj_feats.groupby("year").agg(agg_cols).reset_index()
        bu_sectoral_proj['Emissions (Gt)'] = bu_sectoral_proj[f"BU emissions ({method})"] / 1e9
        
        if f"BU emissions_low ({method})" in bu_sectoral_proj.columns:
            bu_sectoral_proj['Emissions_low (Gt)'] = bu_sectoral_proj[f"BU emissions_low ({method})"] / 1e9
            bu_sectoral_proj['Emissions_high (Gt)'] = bu_sectoral_proj[f"BU emissions_high ({method})"] / 1e9
        
        # Add 2022 value for continuity (both emissions AND production)
        bu_emissions_22 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Emissions (Gt)"].iloc[0])
        bu_production_22 = float(agg_bu_histo.loc[agg_bu_histo["year"] == 2022, "Attributed production"].iloc[0])
        new_row = {"year": [2022], "Emissions (Gt)": [bu_emissions_22], "production": [bu_production_22]}
        bu_sectoral_proj = pd.concat([pd.DataFrame(new_row), bu_sectoral_proj], axis=0).sort_values(by="year", ascending=True)
        
        # Store data
        data_methods_comparison[method] = bu_sectoral_proj
    
    # Plot each method in its own subplot
    for idx, method in enumerate(methods_to_compare):
        ax = axs_methods_norm[idx]
        ax.set_xlim(2017, 2031)
        ax.set_ylim(0.8, 1.05)  # NZE scenario normalized bounds
        ax.set_box_aspect(1)
        ax.grid("both")
        ax.set_title(f"{method_labels[method]}", fontsize=12, weight='bold')
        ax.set_ylabel("Normalized Emissions (2022 = 1.0)")
        ax.set_xlabel("Year")
        
        # Plot historical BU emissions (normalized)
        historical_norm = agg_bu_histo[['year', 'Emissions (Gt)']].copy()
        historical_norm['Normalized Emissions'] = historical_norm['Emissions (Gt)'] / historical_2022_emissions
        sns.lineplot(data=historical_norm,
                    x='year',
                    y="Normalized Emissions",
                    label="Historical BU emissions",
                    ax=ax)
        
        # Add starting point marker at 2022
        ax.scatter([2022], [1.0], marker='o', color="grey", s=200, zorder=2)
        ax.annotate(f"{1.0:.2f}", (2022, 1.0 + 0.015), textcoords='offset points',
                   xytext=(0, 10), ha='center', fontsize=12, color='grey', weight='bold')
        
        # Plot NZE reference trajectory (normalized)
        nze_ref_norm = base_nze_emissions[['year', 'Emissions (Gt)']].copy()
        nze_ref_norm['Normalized Emissions'] = nze_ref_norm['Emissions (Gt)'] / historical_2022_emissions
        sns.lineplot(data=nze_ref_norm,
                    x='year',
                    y="Normalized Emissions",
                    linewidth=2,
                    label="Projected IEA emissions (NZE slope)",
                    color="green",
                    linestyle="dotted",
                    zorder=2,
                    ax=ax)
        
        # Annotate 2030 NZE reference value
        nze_ref_2030 = float(nze_ref_norm.loc[nze_ref_norm['year'] == 2030, 'Normalized Emissions'].iloc[0])
        ax.annotate(f"{nze_ref_2030:.2f}", (2030.5, nze_ref_2030), textcoords='offset points', 
                   xytext=(0, 10), ha='center', fontsize=10, color='green', weight='bold')
        
        # Plot this method's trajectory (normalized)
        if method in data_methods_comparison:
            bu_data = data_methods_comparison[method].copy()
            bu_data['Normalized Emissions'] = bu_data['Emissions (Gt)'] / historical_2022_emissions
            
            emissions_label = f"Projected BU emissions ({method_labels[method]})"
            method_color = "blue"
            
            sns.lineplot(data=bu_data,
                        x='year',
                        y="Normalized Emissions",
                        label=emissions_label,
                        linestyle="dashed",
                        color=method_color,
                        ax=ax)
            
            # Add error bars if uncertainty bounds are available (also normalized)
            if "Emissions_low (Gt)" in bu_data.columns and "Emissions_high (Gt)" in bu_data.columns:
                years = bu_data['year']
                emissions_norm = bu_data["Normalized Emissions"]
                emissions_low_norm = bu_data["Emissions_low (Gt)"] / historical_2022_emissions
                emissions_high_norm = bu_data["Emissions_high (Gt)"] / historical_2022_emissions
                
                # Calculate error bar sizes
                yerr_lower = emissions_norm - emissions_low_norm
                yerr_upper = emissions_high_norm - emissions_norm
                
                # Plot error bars
                ax.errorbar(years, emissions_norm, 
                           yerr=[yerr_lower, yerr_upper],
                           fmt='none', 
                           color=method_color,
                           alpha=0.3,
                           capsize=3,
                           capthick=1)
    
    # Add overall title
    fig_methods_norm.suptitle(
        "Comparison of Projection Methods under NZE Scenario\n(Normalized to 2022 emissions = 1.0)",
        fontsize=16, 
        weight="bold", 
        y=1.02
    )
    
    # Create a single shared legend
    handles_list_methods = []
    labels_list_methods = []
    for ax in axs_methods_norm:
        h, labels = ax.get_legend_handles_labels()
        for handle, label in zip(h, labels):
            if label not in labels_list_methods:
                handles_list_methods.append(handle)
                labels_list_methods.append(label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    # Add uncertainty bar representation to legend
    uncertainty_handle_methods = Line2D(
        [], [], 
        color='gray', 
        marker='|', 
        linestyle='None',
        markersize=10, 
        markeredgewidth=1.5,
        alpha=0.5,
        label='Uncertainty bounds'
    )
    handles_list_methods.append(uncertainty_handle_methods)
    labels_list_methods.append('Uncertainty bounds')
    
    fig_methods_norm.legend(
        handles_list_methods, 
        labels_list_methods, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels_list_methods),
        frameon=True,
        fontsize=10
    )
    
    fig_methods_norm.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save methods comparison plot to methods_comparison subfolder
    fig_methods_norm_path_pdf = methods_dir / "nze_methods_comparison_normalized.pdf"
    fig_methods_norm_path_png = methods_dir / "nze_methods_comparison_normalized.png"
    fig_methods_norm.savefig(fig_methods_norm_path_pdf, dpi=600, bbox_inches='tight')
    fig_methods_norm.savefig(fig_methods_norm_path_png, dpi=600, bbox_inches='tight')
    
    # ====================================================================
    # CREATE AGGREGATE PRODUCTION PLOTS (DEBUG)
    # ====================================================================
    typer.echo("Generating aggregate production plots for debugging...")
    
    # production_data_all_scenarios now contains data for all three scenarios
    # Structure: production_data_all_scenarios[scenario_name][elec_intensity][method]
    typer.echo("  Using stored production data from sensitivity analysis...")
    
    # Show IEA production targets for each scenario
    typer.echo("\n  IEA Production Targets (linearly interpolated):")
    for scenario_name in ["NZE", "APS", "STEPS"]:
        iea_data = glob_prod[scenario_name].copy()
        typer.echo(f"  {scenario_name}:")
        for year in [2022, 2023, 2024, 2025, 2030]:
            if year in iea_data['Year'].values:
                prod_val = float(iea_data.loc[iea_data['Year'] == year, 'Industrial production (Mt)'].iloc[0])
                typer.echo(f"    {year}: {prod_val:.1f} Mt")
    
    # Create a figure with 3 subplots for production
    fig_prod_debug, axs_prod = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Define colors for methods
    method_colors = {"constant_UR": "orange", "company_UR": "green"}
    
    for idx, scenario_name in enumerate(["NZE", "APS", "STEPS"]):
        typer.echo(f"\n  Processing {scenario_name} production plot...")
        
        ax = axs_prod[idx]
        ax.set_xlim(2017, 2031)
        ax.set_box_aspect(1)
        ax.grid("both")
        ax.set_title(f"{scenario_name} Production Trajectory", fontsize=12, weight='bold')
        ax.set_ylabel("Production (Mt)")
        ax.set_xlabel("Year")
        
        # Plot historical production (bottom-up from plant data)
        historical_prod = agg_bu_histo[['year', 'Attributed production']].copy()
        historical_prod['Production (Mt)'] = historical_prod['Attributed production'] / 1e6
        
        typer.echo(f"    Historical BU production:")
        for year in [2019, 2020, 2021, 2022]:
            if year in historical_prod['year'].values:
                prod_val = float(historical_prod.loc[historical_prod['year'] == year, 'Production (Mt)'].iloc[0])
                typer.echo(f"      {year}: {prod_val:.1f} Mt")
        
        sns.lineplot(data=historical_prod,
                    x='year',
                    y="Production (Mt)",
                    label="Historical BU production",
                    ax=ax,
                    linewidth=2.5,
                    color='#1f77b4',
                    marker='o',
                    markersize=6)
        
        # Add marker and annotation at 2022
        prod_2022 = float(historical_prod.loc[historical_prod['year'] == 2022, 'Production (Mt)'].iloc[0])
        ax.scatter([2022], [prod_2022], marker='o', color="grey", s=200, zorder=3, edgecolors='black', linewidth=2)
        ax.annotate(f"2022: {prod_2022:.0f} Mt", (2022, prod_2022), 
                   textcoords='offset points', xytext=(0, 15), ha='center', 
                   fontsize=10, color='black', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))
        
        # Plot global IEA production trajectory
        iea_prod_scenario = glob_prod[scenario_name].copy()
        iea_prod_scenario['Production (Mt)'] = iea_prod_scenario['Industrial production (Mt)']
        sns.lineplot(data=iea_prod_scenario,
                    x='Year',
                    y="Production (Mt)",
                    label=f"IEA {scenario_name} global target",
                    linestyle="dotted",
                    linewidth=2.5,
                    color='#d62728',
                    ax=ax)
        
        # Add 2030 IEA value annotation
        iea_prod_2030 = float(iea_prod_scenario.loc[iea_prod_scenario['Year'] == 2030, 'Production (Mt)'].iloc[0])
        ax.annotate(f"IEA: {iea_prod_2030:.0f} Mt", (2030.5, iea_prod_2030), 
                   textcoords='offset points', xytext=(0, 10), ha='center', 
                   fontsize=9, color='#d62728', weight='bold')
        
        # Plot bottom-up projected production for BOTH methods
        # Use the scenario's own electricity intensity (e.g., NZE uses NZE elec, APS uses APS elec)
        if scenario_name in production_data_all_scenarios:
            scenario_data = production_data_all_scenarios[scenario_name]
            
            # Get data with scenario's own electricity intensity
            elec_key = scenario_name  # Use scenario's own electricity intensity
            
            if elec_key in scenario_data:
                for method in ["constant_UR", "company_UR"]:
                    if method in scenario_data[elec_key]:
                        bu_prod_data = scenario_data[elec_key][method].copy()
                        if 'production' in bu_prod_data.columns:
                            bu_prod_data['Production (Mt)'] = bu_prod_data['production'] / 1e6
                            
                            # Create label
                            if method == "company_UR":
                                method_label = "BU proj. (constant market share)"
                            else:
                                method_label = f"BU proj. ({'country_techno UR' if use_country_techno_ur else 'country UR'})"
                            
                            sns.lineplot(data=bu_prod_data,
                                        x='year',
                                        y="Production (Mt)",
                                        label=method_label,
                                        linestyle="dashed",
                                        linewidth=2,
                                        color=method_colors[method],
                                        ax=ax)
                            
                            # Add 2030 BU value annotation
                            if not bu_prod_data.empty and 2030 in bu_prod_data['year'].values:
                                bu_prod_2030 = float(bu_prod_data.loc[bu_prod_data['year'] == 2030, 'Production (Mt)'].iloc[0])
                                y_offset = -20 if method == "company_UR" else -35
                                ax.annotate(f"BU: {bu_prod_2030:.0f} Mt", (2030, bu_prod_2030), 
                                           textcoords='offset points', xytext=(0, y_offset), ha='center', 
                                           fontsize=9, color=method_colors[method], weight='bold')
                                
                                # Print production values for debugging
                                typer.echo(f"    {scenario_name} {method} - Production trajectory:")
                                for check_year in [2022, 2023, 2024, 2025, 2030]:
                                    if check_year in bu_prod_data['year'].values:
                                        prod_val = float(bu_prod_data.loc[bu_prod_data['year'] == check_year, 'Production (Mt)'].iloc[0])
                                        typer.echo(f"      {check_year}: {prod_val:.1f} Mt")
                                
                                # Check for jump between 2022 and 2023
                                if 2022 in bu_prod_data['year'].values and 2023 in bu_prod_data['year'].values:
                                    prod_2022 = float(bu_prod_data.loc[bu_prod_data['year'] == 2022, 'Production (Mt)'].iloc[0])
                                    prod_2023 = float(bu_prod_data.loc[bu_prod_data['year'] == 2023, 'Production (Mt)'].iloc[0])
                                    jump = prod_2023 - prod_2022
                                    jump_pct = (jump / prod_2022) * 100
                                    if abs(jump_pct) > 1:  # More than 1% change
                                        typer.echo(f"      ⚠️  Jump 2022→2023: {jump:+.1f} Mt ({jump_pct:+.1f}%)")
                            else:
                                typer.echo(f"    WARNING: No 2030 data found for {scenario_name} {method}")
                    else:
                        typer.echo(f"    WARNING: Method {method} not found in {scenario_name} data")
            else:
                typer.echo(f"    WARNING: Electricity scenario {elec_key} not found in {scenario_name} data")
    
    # Add overall title
    fig_prod_debug.suptitle(
        "Aggregate Production Trajectories by Scenario",
        fontsize=18, 
        weight="bold", 
        y=0.98
    )
    
    # Create a single shared legend
    handles_list_prod = []
    labels_list_prod = []
    for ax in axs_prod:
        h, labels = ax.get_legend_handles_labels()
        for handle, label in zip(h, labels):
            if label not in labels_list_prod:
                handles_list_prod.append(handle)
                labels_list_prod.append(label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    fig_prod_debug.legend(
        handles_list_prod, 
        labels_list_prod, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels_list_prod),
        frameon=True,
        fontsize=10
    )
    
    fig_prod_debug.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save production debug plot
    fig_prod_debug_path_pdf = debug_dir / "aggregate_production_by_scenario.pdf"
    fig_prod_debug_path_png = debug_dir / "aggregate_production_by_scenario.png"
    fig_prod_debug.savefig(fig_prod_debug_path_pdf, dpi=600, bbox_inches='tight')
    fig_prod_debug.savefig(fig_prod_debug_path_png, dpi=600, bbox_inches='tight')
    
    typer.echo(f"  ✓ Saved production debug plot: {fig_prod_debug_path_pdf}")
    
    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    typer.echo("\n" + "="*70)
    typer.echo("SUMMARY OF OUTPUTS")
    typer.echo("="*70)
    typer.echo(f"\n✓ Main electricity sensitivity plots (in {plots_dir}):")
    typer.echo(f"  - NZE: {fig_elec_sensitivity_path_pdf}")
    typer.echo(f"  - APS: {plots_dir / 'aps_elec_intensity_sensitivity.pdf'}")
    typer.echo(f"  - STEPS: {plots_dir / 'steps_elec_intensity_sensitivity.pdf'}")
    
    typer.echo(f"\n✓ Method-specific electricity sensitivity plots (in {elec_sensitivity_dir}):")
    typer.echo(f"  - {constant_ur_dir}:")
    typer.echo(f"      • 3 absolute value figures (NZE, APS, STEPS)")
    typer.echo(f"      • 3 normalized figures (2022 = 1.0)")
    typer.echo(f"  - {company_ur_dir}:")
    typer.echo(f"      • 3 absolute value figures (NZE, APS, STEPS)")
    typer.echo(f"      • 3 normalized figures (2022 = 1.0)")
    typer.echo(f"  - Each figure: 3 subplots (NZE, APS, STEPS electricity intensities)")
    typer.echo(f"  - Total: 12 method-specific figures (6 absolute + 6 normalized)")
    typer.echo(f"  - All with smooth uncertainty bands starting from 2022")
    
    typer.echo(f"\n✓ Intensity evolution plots (in {intensity_dir}):")
    typer.echo(f"  - NZE: {fig_intensity_path_pdf}")
    
    typer.echo(f"\n✓ Normalized emissions plots (in {normalized_dir}):")
    typer.echo(f"  - {fig_emis_norm_path_pdf}")
    
    typer.echo(f"\n✓ Methods comparison plots (in {methods_dir}):")
    typer.echo(f"  - {fig_methods_norm_path_pdf}")
    
    typer.echo(f"\n✓ Debug plots (in {debug_dir}):")
    typer.echo(f"  - Aggregate production: {fig_prod_debug_path_pdf}")
    
    typer.echo(f"\n✓ Plot data (in {plots_data_dir}):")
    typer.echo(f"  - Combined NZE sensitivity data: {combined_excel_path}")
    typer.echo(f"    Total rows: {len(combined_data)}")
    typer.echo(f"    Columns: {', '.join(column_order)}")
    
    typer.echo(f"\n✓ Plant & Company-level data (in {plant_data_dir}):")
    # Count files
    plant_files = list(plant_data_dir.glob("*_plants.xlsx"))
    company_agg_files = list(plant_data_dir.glob("*_company_aggregated.csv"))
    total_files = len(plant_files) + len(company_agg_files)
    
    typer.echo(f"  - Plant-level files: {len(plant_files)}")
    typer.echo(f"    Format: {{scenario}}_{{elec_intensity}}_elec_{{method}}_plants.xlsx")
    typer.echo(f"    Expected: 18 (3 scenarios × 3 elec intensities × 2 methods)")
    
    typer.echo(f"  - Company-aggregated files (CSV): {len(company_agg_files)}")
    typer.echo(f"    Format: {{scenario}}_{{elec_intensity}}_elec_{{method}}_company_aggregated.csv")
    typer.echo(f"    Expected: 18 (one row per company per year with BU predictions)")
    
    typer.echo(f"  - Total data files: {total_files} (expected: 36)")
    
    typer.echo("\n" + "="*70)
    typer.echo("✓ All plots and data generated successfully!")
    typer.echo(f"✓ Total plot files: {3 + 12 + 1 + 1 + 1 + 1} = 19 plots")
    typer.echo(f"  (3 main + 12 method-specific + 5 supporting)")
    typer.echo(f"✓ Total data files: {total_files}")
    typer.echo(f"  ({len(plant_files)} plants + {len(company_agg_files)} company aggregated)")
    typer.echo("="*70)
    
    return data_elec_sensitivity


if __name__ == "__main__":
    typer.run(main)

