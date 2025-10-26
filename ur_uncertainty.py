"""
UR Uncertainty Analysis Script

Generates sectoral emissions projections using 4 methods:
- constant_UR and company_UR as middle scenarios (lines)
- carbon_efficiency and carbon_intensity as lower/upper bounds (fill_between)

Outputs: Single 3-subplot figure (NZE, APS, STEPS) + data files
Saves to: BU_results/UR_uncertainty/
"""

import typer
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from config import load_config

from src.projections_.projections import (
    get_historical_bu_emissions, 
    get_benchmark_emissions, 
    get_iea_emissions,
    get_bu_proj_emissions,
    get_elec_gen_caagr,
    get_proj_elec_int
)
from src.datasets.RefinitivDataset import RefinitivDataset
from src.datasets.GSPTDataset import GSPTDataset
from src.datasets.EmissionFactors import EmissionFactors
from src.datasets.utils import fill_missing_CF12, convert_plant2parent, get_cagr
from src.feature_eng.company_feature_engineering import get_X_y
from src.projections_.BUProjector import get_market_share
from src.projections_.projections import get_glob_prod_capa

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def main(config_path: Path = typer.Argument(
    Path("./config.toml"), 
    help="Path to the TOML configuration file"
)):
    """
    Generate UR uncertainty analysis with 4 methods across 3 scenarios.
    """
    
    typer.echo("="*60)
    typer.echo("UR UNCERTAINTY ANALYSIS")
    typer.echo("="*60)
    
    # ========================================================================
    # CONFIGURATION & PATHS
    # ========================================================================
    config_file = load_config(config_path)
    params = config_file.params
    
    # Create output directory
    save_dir = params.save_dir / "BU_results" / "UR_uncertainty"
    save_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Output directory: {save_dir}")
    
    project_dir = params.project_dir
    raw_data_dir = project_dir / "data" / "raw"
    
    # All paths from config
    historical_data = config_file.historical_data
    emission_factors = config_file.emission_factors
    mappings = config_file.mappings
    scenarios_config = config_file.scenarios
    projected_data = config_file.projected_data
    
    # Mapping paths
    gspt2gspt_path = raw_data_dir / mappings.gspt2gspt
    refi2gspt_path = raw_data_dir / mappings.refi2gspt
    parent_group_map_path = raw_data_dir / mappings.parent_group_map
    gspt2iea_countries_path = raw_data_dir / mappings.gspt2iea_countries
    gspt2cdp_map_path = raw_data_dir / mappings.gspt2cdp
    
    # Data paths
    energy_mix_path = raw_data_dir / emission_factors.energy_mix
    carbon_price_path = raw_data_dir / historical_data.macro.carbon_price
    refinitiv_path = raw_data_dir / historical_data.top_down.refinitiv
    cdp_path = project_dir / historical_data.top_down.cdp
    company_steel_prod_path = raw_data_dir / historical_data.micro.company_prod
    wsa_prod_path = raw_data_dir / historical_data.macro.wsa_prod
    oecd_capa_fpath = raw_data_dir / historical_data.macro.oecd_capa
    iea_prod_dir = raw_data_dir / historical_data.macro.iea_prod
    prod_dir = raw_data_dir / "production"
    world_fpath = raw_data_dir / scenarios_config.iea_world
    regions_fpath = raw_data_dir / scenarios_config.iea_regions
    
    # Electricity intensity paths
    elec_int_nze_path = raw_data_dir / projected_data.electricity.elec_int_nze
    elec_int_aps_path = raw_data_dir / projected_data.electricity.elec_int_aps
    elec_int_steps_path = raw_data_dir / projected_data.electricity.elec_int_steps
    
    typer.echo("\n✓ Configuration loaded")
    
    # ========================================================================
    # LOAD HISTORICAL DATA
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("LOADING HISTORICAL DATA")
    typer.echo("="*60)
    
    # Asset-level data
    typer.echo("  Loading GSPT dataset...")
    gspt = GSPTDataset(
        data_path=raw_data_dir / historical_data.asset_level_data,
        missing_years_path=raw_data_dir / historical_data.missing_years,
        gspt2gspt_path=gspt2gspt_path,
        parent_group_map_path=parent_group_map_path,
        version_year=2023
    )
    
    # Emission factors
    typer.echo("  Loading emission factors...")
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
    
    # Top-down emissions
    typer.echo("  Loading Refinitiv data...")
    refi_class = RefinitivDataset(data_path=refinitiv_path)
    refinitiv = refi_class.get_preprocessed_data()
    refinitiv = fill_missing_CF12(
        refinitiv=refinitiv, 
        refi2gspt_path=refi2gspt_path, 
        gspt2gspt_map=gspt2gspt_map
    )
    cdp = pd.read_excel(cdp_path)
    gspt2cdp = pd.read_excel(gspt2cdp_map_path)
    
    # Historical global production
    histo_global_prod = pd.DataFrame({
        "year": [2019, 2020, 2021, 2022],
        "Crude steel production (Mt)": [1874.4, 1877.5, 1951.9, 1885]
    })
    
    typer.echo("✓ Historical data loaded")
    
    # ========================================================================
    # FIT BOTTOM-UP MODEL
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("FITTING BOTTOM-UP MODEL")
    typer.echo("="*60)
    
    X_y = get_X_y(
        gspt=gspt,
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
        cdp=cdp
    )
    
    X, y = X_y[["log_Attributed emissions"]], X_y[["log_max_CF12"]]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    model.fit(X, y)
    
    typer.echo(f"✓ Model fitted (R²: {model.score(X, y):.4f})")
    
    # ========================================================================
    # HISTORICAL BU EMISSIONS
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("COMPUTING HISTORICAL BU EMISSIONS")
    typer.echo("="*60)
    
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
        use_country_techno_ur=params.use_country_techno_ur
    )
    agg_bu_histo['Emissions (Gt)'] = agg_bu_histo['BU emissions'] / 1e9
    
    # Get IEA benchmarks
    nze_emissions = get_iea_emissions(scenario="NZE")
    aps_emissions = get_iea_emissions(scenario="APS")
    steps_emissions = get_iea_emissions(scenario="STEPS")
    
    x0_year = 2022
    x0_value = float(agg_bu_histo.loc[agg_bu_histo["year"] == x0_year, "Emissions (Gt)"])
    
    base_nze_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=nze_emissions)
    base_aps_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=aps_emissions)
    base_steps_emissions = get_benchmark_emissions(x0_value=x0_value, x0_year=x0_year, slope_data=steps_emissions)
    
    typer.echo(f"✓ Historical emissions computed (2022 baseline: {x0_value:.3f} Gt)")
    
    # ========================================================================
    # MARKET SHARE
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("COMPUTING MARKET SHARE")
    typer.echo("="*60)
    
    company_steel_prod = pd.read_excel(company_steel_prod_path)
    market_share = get_market_share(
        gspt=gspt,
        db_path=params.steel_db,
        histo_global_prod=histo_global_prod,
        company_steel_prod=company_steel_prod,
        mapping=gspt2refi,
        gspt2gspt_path=gspt2gspt_path,
        parent_group_map=parent_group_map,
        method="single_year",
        year=2022
    )
    
    typer.echo("✓ Market share computed")
    
    # ========================================================================
    # LOAD PROJECTED DATA
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("LOADING PROJECTED DATA")
    typer.echo("="*60)
    
    # Electricity intensity
    elec_int_nze = pd.read_excel(elec_int_nze_path)
    elec_int_aps = pd.read_excel(elec_int_aps_path)
    elec_int_steps = pd.read_excel(elec_int_steps_path)
    
    # Global production and capacity
    glob_prod, glob_capa = get_glob_prod_capa(
        iea_prod_dir=iea_prod_dir,
        oecd_capa_fpath=oecd_capa_fpath
    )
    
    # Get bottom-up production for 2022 (scalar value in ttpa for production adjustment)
    # This is used to normalize projected production to match IEA scenarios
    glob_bu_prod_2022 = prod.loc[prod["year"] == 2022, "Estimated crude steel production (ttpa)"].sum()
    typer.echo(f"  2022 BU production: {glob_bu_prod_2022/1e3:.1f} Mt (for production scaling)")
    
    typer.echo("✓ Projected data loaded")
    
    # ========================================================================
    # GENERATE PROJECTIONS FOR 4 METHODS × 3 SCENARIOS
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("GENERATING PROJECTIONS")
    typer.echo("="*60)
    
    scenarios = ["NZE", "APS", "STEPS"]
    methods = ["constant_UR", "company_UR", "carbon_efficiency", "carbon_intensity"]
    
    # Storage for results
    all_projections = {scenario: {method: None for method in methods} for scenario in scenarios}
    
    start_year, end_year = 2023, 2030
    
    # Electricity intensity CAGR
    r_elec_nze = get_elec_gen_caagr(elec_int=elec_int_nze, start_year=2022, end_year=2030)
    r_elec_aps = get_elec_gen_caagr(elec_int=elec_int_aps, start_year=2022, end_year=2030)
    r_elec_steps = get_elec_gen_caagr(elec_int=elec_int_steps, start_year=2022, end_year=2030)
    
    r_elec_map = {"NZE": r_elec_nze, "APS": r_elec_aps, "STEPS": r_elec_steps}
    
    # Load carbon budgets for carbon_efficiency and carbon_intensity methods
    cbudgets = {
        "NZE": pd.read_excel(Path("./src/projections_") / "proj_country_prod_nze.xlsx"),
        "APS": pd.read_excel(Path("./src/projections_") / "proj_country_prod_aps.xlsx"),
        "STEPS": pd.read_excel(Path("./src/projections_") / "proj_country_prod_steps.xlsx")
    }
    
    # Costs (required for carbon methods, though not directly used)
    proj_costs_iea = {
        "NZE": pd.DataFrame(),  # Placeholder
        "APS": pd.DataFrame(),
        "STEPS": pd.DataFrame()
    }
    
    for scenario in scenarios:
        typer.echo(f"\n  Scenario: {scenario}")
        
        for method in methods:
            typer.echo(f"    Method: {method}...")
            
            # Get BU projected emissions
            proj_company, proj_plants = get_bu_proj_emissions(
                gspt=gspt,
                EF=EF,
                glob_prod_iea=glob_prod[scenario],
                glob_bu_prod=glob_bu_prod_2022,
                market_share=market_share,
                parent_group_map_path=parent_group_map_path,
                gspt2refi_map=gspt2refi,
                gspt2gspt_path=gspt2gspt_path,
                proj_costs=proj_costs_iea[scenario],
                start_year=start_year,
                end_year=end_year,
                method=method,
                glob_capa=glob_capa[scenario],
                cbudget=cbudgets[scenario],
                use_country_techno_ur=params.use_country_techno_ur
            )
            
            # Apply electricity intensity
            proj_plants = get_proj_elec_int(
                proj_plants,
                r=r_elec_map[scenario],
                energy_mix_path=energy_mix_path,
                level="global",
                scenario=scenario,
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
                proj_plants["Emissions (Gt)"] = proj_plants["Estimated crude steel production (ttpa)"] * proj_plants["EF"] / 1E6
                proj_plants = proj_plants.drop(columns=["EAF_CAGR"])
            
            # Convert to parent company level
            proj_group = convert_plant2parent(proj_plants, gspt2gspt_path=gspt2gspt_path, parent_group_map=parent_group_map)
            
            # Calculate attributed values at company level
            proj_group["Attributed crude steel capacity (ttpa)"] = proj_group['Nominal crude steel capacity (ttpa)'] * proj_group["Share"]
            proj_group["Attributed capacity"] = proj_group["Attributed crude steel capacity (ttpa)"] * 1E3
            proj_group["Attributed emissions"] = proj_group['Emissions (Gt)'] * 1e9 * proj_group["Share"]
            proj_group["Attributed production"] = proj_group['Estimated crude steel production (ttpa)'] * 1e3 * proj_group["Share"]
            
            # Aggregate to company level (Group + year)
            proj_feats = proj_group.groupby(['Group', 'year']).agg({
                "Attributed emissions": "sum",
                "Attributed production": "sum",
                "Attributed capacity": "sum",
            }).rename(columns={
                "Attributed emissions": "emissions",
                "Attributed production": "production", 
                "Attributed capacity": "capacity"
            }).reset_index()
            
            # Clean zero/negative emissions before taking log
            emissions = proj_feats["emissions"]
            zero_count = (emissions == 0).sum()
            negative_count = (emissions < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                typer.echo(f"      DEBUG: Dropping {zero_count} zero and {negative_count} negative emissions rows")
                proj_feats = proj_feats[emissions > 0].reset_index(drop=True)
            
            # Take log of emissions (required by the model)
            proj_feats["log_Attributed emissions"] = np.log(proj_feats["emissions"])
            
            # Validate before model prediction
            log_emissions = proj_feats["log_Attributed emissions"]
            inf_count = np.isinf(log_emissions).sum()
            nan_count = log_emissions.isna().sum()
            
            if inf_count > 0 or nan_count > 0:
                typer.echo(f"      DEBUG: Dropping {inf_count} inf and {nan_count} NaN log emissions rows")
                valid_mask = np.isfinite(log_emissions)
                proj_feats = proj_feats[valid_mask].reset_index(drop=True)
            
            # ✅ APPLY ML MODEL TO PROJECTIONS (key step!)
            proj_feats[f"log BU emissions ({method})"] = model.predict(proj_feats[["log_Attributed emissions"]])
            proj_feats[f"BU emissions ({method})"] = np.exp(proj_feats[f"log BU emissions ({method})"])
            
            # Aggregate to sectoral level (sum over all companies for each year)
            sectoral_proj = proj_feats.groupby("year").agg({
                f"BU emissions ({method})": "sum",
                "emissions": "sum",  # Keep raw emissions for comparison
                "production": "sum",
                "capacity": "sum"
            }).reset_index()
            
            # Convert to Gt
            sectoral_proj['Emissions (Gt)'] = sectoral_proj[f"BU emissions ({method})"] / 1e9
            sectoral_proj['Raw Emissions (Gt)'] = sectoral_proj["emissions"] / 1e9
            
            # Add 2022 baseline for continuity
            emissions_2022 = x0_value
            baseline_row = pd.DataFrame({"year": [2022], "Emissions (Gt)": [emissions_2022]})
            sectoral_proj = pd.concat([baseline_row, sectoral_proj], ignore_index=True)
            
            # Store
            all_projections[scenario][method] = sectoral_proj
            
            typer.echo("      ✓ Completed")
    
    typer.echo("\n✓ All projections generated")
    
    # ========================================================================
    # NORMALIZE ALL PROJECTIONS RELATIVE TO 2022
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("NORMALIZING PROJECTIONS (2022 = 1.0)")
    typer.echo("="*60)
    
    # Get 2022 baseline value for normalization
    baseline_2022 = x0_value
    typer.echo(f"  2022 baseline: {baseline_2022:.3f} Gt CO₂")
    
    # Normalize all projections
    for scenario in scenarios:
        for method in methods:
            df = all_projections[scenario][method].copy()
            # Add normalized column
            df['Normalized Emissions'] = df['Emissions (Gt)'] / baseline_2022
            all_projections[scenario][method] = df
    
    typer.echo("✓ All projections normalized")
    
    # ========================================================================
    # SAVE DATA
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("SAVING DATA")
    typer.echo("="*60)
    
    # Save as pickle
    import pickle
    with open(save_dir / "ur_uncertainty_projections.pkl", "wb") as f:
        pickle.dump(all_projections, f)
    typer.echo("  ✓ Saved pickle: ur_uncertainty_projections.pkl")
    
    # Save as Excel (one sheet per scenario with both absolute and normalized)
    with pd.ExcelWriter(save_dir / "ur_uncertainty_projections.xlsx") as writer:
        for scenario in scenarios:
            # Combine all methods for this scenario
            scenario_data = pd.DataFrame({"year": all_projections[scenario]["constant_UR"]["year"]})
            
            # Add absolute emissions
            for method in methods:
                scenario_data[f"{method}_Gt"] = all_projections[scenario][method]["Emissions (Gt)"]
            
            # Add normalized emissions
            for method in methods:
                scenario_data[f"{method}_normalized"] = all_projections[scenario][method]["Normalized Emissions"]
            
            scenario_data.to_excel(writer, sheet_name=scenario, index=False)
    typer.echo("  ✓ Saved Excel: ur_uncertainty_projections.xlsx")
    
    # ========================================================================
    # DEBUG: SAVE DIAGNOSTIC DATA FOR JUMP INVESTIGATION
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("SAVING DEBUG DIAGNOSTICS")
    typer.echo("="*60)
    
    # Create debug subfolder
    debug_dir = save_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    # 1. Historical production and emissions (2019-2022)
    historical_debug = agg_bu_histo[['year', 'BU emissions']].copy()
    historical_debug['Emissions (Gt)'] = historical_debug['BU emissions'] / 1e9
    if 'Attributed production' in agg_bu_histo.columns:
        historical_debug['Production (Mt)'] = agg_bu_histo['Attributed production'] / 1e3
    historical_debug.to_excel(debug_dir / "1_historical_data.xlsx", index=False)
    typer.echo("  ✓ Saved: debug/1_historical_data.xlsx")
    
    # 2. IEA scenario production trajectories (2023-2030)
    iea_prod_list = []
    for scenario in scenarios:
        scenario_prod = glob_prod[scenario].copy()
        # Simply use glob_prod as-is, adding scenario column for identification
        scenario_prod['Scenario_label'] = scenario
        iea_prod_list.append(scenario_prod)
    
    iea_prod_debug = pd.concat(iea_prod_list, ignore_index=True)
    # Add 2022 BU production as a reference column
    iea_prod_debug['BU_2022_Production_Mt'] = glob_bu_prod_2022 / 1e3
    iea_prod_debug.to_excel(debug_dir / "2_iea_production_scenarios.xlsx", index=False)
    typer.echo("  ✓ Saved: debug/2_iea_production_scenarios.xlsx")
    
    # 3. Alpha scaling factors by year and scenario
    # Alpha = IEA scenario production / BU 2022 production
    # Simply duplicate the IEA production data and add alpha calculations
    alpha_debug = iea_prod_debug.copy()
    if 'Industrial production (Mt)' in alpha_debug.columns:
        alpha_debug['Alpha (IEA/BU_2022)'] = (alpha_debug['Industrial production (Mt)'] * 1e3) / glob_bu_prod_2022
    alpha_debug.to_excel(debug_dir / "3_alpha_scaling_factors.xlsx", index=False)
    typer.echo("  ✓ Saved: debug/3_alpha_scaling_factors.xlsx")
    
    # 4. Emissions comparison (2022 baseline vs 2023 first projection)
    comparison_data = []
    for scenario in scenarios:
        for method in methods:
            df = all_projections[scenario][method]
            
            # Get 2022 and 2023 values
            val_2022 = df.loc[df['year'] == 2022, 'Emissions (Gt)'].values
            val_2023 = df.loc[df['year'] == 2023, 'Emissions (Gt)'].values
            
            if len(val_2022) > 0 and len(val_2023) > 0:
                comparison_data.append({
                    'Scenario': scenario,
                    'Method': method,
                    '2022_Emissions_Gt': val_2022[0],
                    '2023_Emissions_Gt': val_2023[0],
                    'Jump_Gt': val_2023[0] - val_2022[0],
                    'Jump_Percent': ((val_2023[0] - val_2022[0]) / val_2022[0]) * 100
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_excel(debug_dir / "4_2022_2023_jump_analysis.xlsx", index=False)
    typer.echo("  ✓ Saved: debug/4_2022_2023_jump_analysis.xlsx")
    
    # 5. Summary statistics
    summary_data = {
        'Metric': [
            '2022 BU Production (Mt)',
            '2022 BU Emissions (Gt)',
            'Number of scenarios',
            'Number of methods',
            'Projection years'
        ],
        'Value': [
            f"{glob_bu_prod_2022/1e3:.2f}",
            f"{baseline_2022:.3f}",
            len(scenarios),
            len(methods),
            "2023-2030"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(debug_dir / "0_summary_info.xlsx", index=False)
    typer.echo("  ✓ Saved: debug/0_summary_info.xlsx")
    
    # 6. COMPREHENSIVE: Full time series 2019-2030 with emissions, production, and alpha
    # Create one sheet per scenario showing everything
    with pd.ExcelWriter(debug_dir / "5_comprehensive_time_series.xlsx") as writer:
        for scenario in scenarios:
            # Start with historical data (2019-2022)
            years_all = list(range(2019, 2031))
            comprehensive_data = pd.DataFrame({'year': years_all})
            
            # Add historical BU emissions (2019-2022)
            hist_emissions = agg_bu_histo[['year', 'BU emissions']].copy()
            hist_emissions['Historical_BU_Emissions_Gt'] = hist_emissions['BU emissions'] / 1e9
            comprehensive_data = pd.merge(
                comprehensive_data, 
                hist_emissions[['year', 'Historical_BU_Emissions_Gt']], 
                on='year', 
                how='left'
            )
            
            # Add historical BU production (2019-2022) if available
            if 'Attributed production' in agg_bu_histo.columns:
                hist_prod = agg_bu_histo[['year', 'Attributed production']].copy()
                hist_prod['Historical_BU_Production_Mt'] = hist_prod['Attributed production'] / 1e3
                comprehensive_data = pd.merge(
                    comprehensive_data,
                    hist_prod[['year', 'Historical_BU_Production_Mt']],
                    on='year',
                    how='left'
                )
            
            # Add 2022 BU production as reference (constant column)
            comprehensive_data['BU_2022_Production_Mt'] = glob_bu_prod_2022 / 1e3
            
            # Add IEA scenario production
            scenario_prod = glob_prod[scenario].copy()
            if 'Year' in scenario_prod.columns:
                scenario_prod = scenario_prod.rename(columns={'Year': 'year'})
            if 'Industrial production (Mt)' in scenario_prod.columns:
                scenario_prod[f'IEA_{scenario}_Production_Mt'] = scenario_prod['Industrial production (Mt)']
                comprehensive_data = pd.merge(
                    comprehensive_data,
                    scenario_prod[['year', f'IEA_{scenario}_Production_Mt']],
                    on='year',
                    how='left'
                )
                
                # Calculate alpha scaling factor
                comprehensive_data[f'Alpha_{scenario}'] = (
                    comprehensive_data[f'IEA_{scenario}_Production_Mt'] * 1e3
                ) / glob_bu_prod_2022
            
            # Add projected emissions for each method (2022-2030)
            for method in methods:
                proj_data = all_projections[scenario][method][['year', 'Emissions (Gt)', 'Normalized Emissions']].copy()
                proj_data = proj_data.rename(columns={
                    'Emissions (Gt)': f'{method}_Emissions_Gt',
                    'Normalized Emissions': f'{method}_Normalized'
                })
                comprehensive_data = pd.merge(
                    comprehensive_data,
                    proj_data,
                    on='year',
                    how='left'
                )
            
            # Add IEA reference emissions for this scenario
            if scenario == "NZE":
                benchmark_emissions = base_nze_emissions[['year', 'Emissions (Gt)']].copy()
            elif scenario == "APS":
                benchmark_emissions = base_aps_emissions[['year', 'Emissions (Gt)']].copy()
            else:  # STEPS
                benchmark_emissions = base_steps_emissions[['year', 'Emissions (Gt)']].copy()
            
            benchmark_emissions = benchmark_emissions.rename(columns={'Emissions (Gt)': f'IEA_{scenario}_Reference_Gt'})
            comprehensive_data = pd.merge(
                comprehensive_data,
                benchmark_emissions,
                on='year',
                how='left'
            )
            
            # Sort by year and save
            comprehensive_data = comprehensive_data.sort_values('year')
            comprehensive_data.to_excel(writer, sheet_name=scenario, index=False)
    
    typer.echo("  ✓ Saved: debug/5_comprehensive_time_series.xlsx")
    typer.echo(f"\n  📁 All debug files saved to: {debug_dir}")
    
    # ========================================================================
    # CREATE PLOTS (BOTH VERSIONS)
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("GENERATING PLOTS")
    typer.echo("="*60)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    
    # Colors
    color_constant = "#2E86AB"  # Blue
    color_company = "#A23B72"   # Purple
    color_fill = "#F18F01"      # Orange
    color_historical = "#333333"  # Dark gray
    color_iea = "#06A77D"       # Green
    
    # Normalize historical and benchmark data
    agg_bu_histo_norm = agg_bu_histo.copy()
    agg_bu_histo_norm['Normalized Emissions'] = agg_bu_histo_norm['Emissions (Gt)'] / baseline_2022
    
    benchmark_map = {
        "NZE": base_nze_emissions.copy(),
        "APS": base_aps_emissions.copy(),
        "STEPS": base_steps_emissions.copy()
    }
    
    # Normalize benchmarks
    for scenario in scenarios:
        benchmark_map[scenario]['Normalized Emissions'] = benchmark_map[scenario]['Emissions (Gt)'] / baseline_2022
    
    # Define colors for IEA scenarios
    iea_colors = {
        "NZE": "#06A77D",    # Green
        "APS": "#F18F01",    # Orange
        "STEPS": "#D62828"   # Red
    }
    
    # Create both versions of the plot
    for plot_version in ["variable_scale", "fixed_scale"]:
        typer.echo(f"\n  Creating {plot_version} version...")
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, scenario in enumerate(scenarios):
            ax = axs[idx]
            
            # Get data for this scenario's projections
            data_constant = all_projections[scenario]["constant_UR"]
            data_company = all_projections[scenario]["company_UR"]
            data_efficiency = all_projections[scenario]["carbon_efficiency"]
            data_intensity = all_projections[scenario]["carbon_intensity"]
            
            # Historical BU emissions (2019-2022) - NORMALIZED
            ax.plot(
                agg_bu_histo_norm['year'],
                agg_bu_histo_norm['Normalized Emissions'],
                label='Historical BU' if idx == 0 else None,
                color=color_historical,
                linewidth=2.5,
                marker='o',
                markersize=6,
                zorder=5
            )
            
            # IEA reference trajectory for THIS scenario only - with scenario-specific color
            benchmark = benchmark_map[scenario]
            ax.plot(
                benchmark['year'],
                benchmark['Normalized Emissions'],
                label=f'IEA {scenario}',
                color=iea_colors[scenario],
                linewidth=2,
                linestyle=':',
                alpha=0.8,
                zorder=3
            )
            
            # Fill between carbon_efficiency (lower) and carbon_intensity (upper) - NORMALIZED
            ax.fill_between(
                data_efficiency['year'],
                data_efficiency['Normalized Emissions'],
                data_intensity['Normalized Emissions'],
                alpha=0.2,
                color=color_fill,
                label='Uncertainty bounds' if idx == 0 else None
            )
            
            # Constant UR (middle scenario) - NORMALIZED
            ax.plot(
                data_constant['year'],
                data_constant['Normalized Emissions'],
                label='Constant UR' if idx == 0 else None,
                color=color_constant,
                linewidth=2.5,
                linestyle='--'
            )
            
            # Company UR (middle scenario) - NORMALIZED
            ax.plot(
                data_company['year'],
                data_company['Normalized Emissions'],
                label='Company UR' if idx == 0 else None,
                color=color_company,
                linewidth=2.5,
                linestyle='-.'
            )
            
            # Formatting
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Normalized Emissions (2022 = 1.0)', fontsize=12, fontweight='bold')
            ax.set_title(f'{scenario} Scenario', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlim(2018, 2031)
            
            # Set y-axis limits based on version
            if plot_version == "variable_scale":
                y_limits = {"NZE": (0.8, 1.10), "APS": (0.9, 1.10), "STEPS": (0.9, 1.10)}
                ax.set_ylim(y_limits[scenario])
            else:  # fixed_scale
                ax.set_ylim(0.8, 1.10)
            
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Overall title
        fig.suptitle(
            'UR Uncertainty Analysis: Normalized Sectoral Emissions Projections\n' +
            'Middle Scenarios (Constant/Company UR) with Uncertainty Bounds (2022 = 1.0)',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        
        # Collect legend handles from all subplots to get all IEA scenarios
        handle_dict = {}  # Store handles by label
        
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in handle_dict:
                    handle_dict[label] = handle
        
        # Define desired order: Historical -> Methods -> Scenarios -> Uncertainty
        desired_order = [
            'Historical BU',
            'Constant UR',
            'Company UR',
            'IEA NZE',
            'IEA APS',
            'IEA STEPS',
            'Uncertainty bounds'
        ]
        
        # Build ordered lists
        ordered_handles = []
        ordered_labels = []
        for label in desired_order:
            if label in handle_dict:
                ordered_handles.append(handle_dict[label])
                ordered_labels.append(label)
        
        # Create a single legend below all subplots with ordered items
        fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=len(ordered_labels), frameon=True, 
                   fontsize=10, edgecolor='gray', bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        
        # Save figure with appropriate naming
        fig_path_pdf = save_dir / f"ur_uncertainty_analysis_{plot_version}.pdf"
        fig_path_png = save_dir / f"ur_uncertainty_analysis_{plot_version}.png"
        fig.savefig(fig_path_pdf, dpi=600, bbox_inches='tight')
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        
        typer.echo(f"    ✓ Saved: ur_uncertainty_analysis_{plot_version}.pdf/.png")
        
        plt.close(fig)
    
    typer.echo("\n  ✓ Both plot versions generated")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    typer.echo("\n" + "="*60)
    typer.echo("SUMMARY")
    typer.echo("="*60)
    
    typer.echo(f"\n2022 Baseline: {baseline_2022:.3f} Gt CO₂ (normalized to 1.0)")
    
    for scenario in scenarios:
        typer.echo(f"\n{scenario} Scenario (2030):")
        for method in methods:
            val_2030_abs = all_projections[scenario][method].loc[
                all_projections[scenario][method]['year'] == 2030, 
                'Emissions (Gt)'
            ].values[0]
            val_2030_norm = all_projections[scenario][method].loc[
                all_projections[scenario][method]['year'] == 2030, 
                'Normalized Emissions'
            ].values[0]
            change_pct = (val_2030_norm - 1.0) * 100
            typer.echo(f"  {method:20s}: {val_2030_abs:.3f} Gt CO₂ | {val_2030_norm:.3f} ({change_pct:+.1f}% vs 2022)")
    
    typer.echo("\n" + "="*60)
    typer.echo("✓ UR UNCERTAINTY ANALYSIS COMPLETE")
    typer.echo(f"✓ Output saved to: {save_dir}")
    typer.echo("="*60 + "\n")


if __name__ == "__main__":
    typer.run(main)

