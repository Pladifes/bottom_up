import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

def correct_uncertainty_bounds(ef12_proj: pd.DataFrame, max_abat_path: Path) -> pd.DataFrame:
    """
    Correct uncertainty bounds for emission factors.
    """
    max_abat = pd.read_excel(max_abat_path)
    max_abat = max_abat.assign(**{"Abatement potential": max_abat["Abatement potential"]/1000})
    ef12_abat = ef12_proj.merge(max_abat[["country", "Abatement potential"]], 
                       on="country", 
                       how="left")
    # Compute difference between lower bound and abatement potential
    ef12_abat["diff_sci_abat"] = ef12_abat["EF_sci"] - ef12_abat["Abatement potential"]

    # Adjust bounds on projected emission factors
    # 1. Upper bound: If ef_upper_bound > EF_Hasanbeigi, then set ef_upper_bound = EF_Hasanbeigi
    ef12_abat["adj_ef_12_upper"] = ef12_abat["ef_12_upper"].clip(upper=ef12_abat["EF_sci"])

    # 2. Lower bound: If ef_lower_bound < EF_sci, then set ef_lower_bound = EF_sci
    ef12_abat["adj_ef_12_lower"] = ef12_abat["ef_12_lower"].clip(lower=ef12_abat["diff_sci_abat"])

    # 3. Cap and floor predicted emission factors
    ef12_abat["adj_ef_12"] = ef12_abat["ef_12"].clip(lower=ef12_abat["adj_ef_12_lower"], upper=ef12_abat["adj_ef_12_upper"])

    return ef12_abat

# Return historical and projected emission factors
def get_projected_efs(emissions_data_dir: Path, activity_data_dir: Path, sci_ef_path: Path):
    # Read huizhong emission factors (scope 1)
    efs = get_histo_efs(emissions_data_dir, activity_data_dir)
    efs["bof_ohf_ef"] = efs["bof_ohf_ef"].replace(np.inf, np.nan)
    # Drop values above 95% percentile
    quantile_95 = efs["bof_ohf_ef"].quantile(0.95)
    # Extrapolate trend based on most recent years
    efs = efs.loc[efs["year"].between(2000, 2020) & (efs["bof_ohf_ef"] < quantile_95)]
    sci_ef = pd.read_excel(sci_ef_path, engine="calamine").rename(columns={"Country":"country"})
    ef19_merge = get_efs_19(efs, sci_ef)
    
    # Apply to each country
    projections = (
        efs.loc[efs["year"].between(2000, 2020)].groupby("country", group_keys=False)
        .apply(extrapolate_group, target_col="bof_ohf_ef", years_ahead=11)
        .reset_index(drop=True)
    )
    ef_col = "bof_ohf_ef_logreg"
    projections = projections.dropna(subset=ef_col).reset_index(drop=True)
    # Keep logreg emission factors
    projections = projections[["country", "year", ef_col, f"{ef_col}_obs_lower_2", f"{ef_col}_obs_upper_2"]]
    projections = projections.rename(columns={ef_col: "bof_ohf_ef", f"{ef_col}_obs_lower_2": "bof_ohf_ef_lower", f"{ef_col}_obs_upper_2": "bof_ohf_ef_upper"})
    cf12_hist = pd.merge(efs, ef19_merge.loc[ef19_merge["Technology"]=="BF-BOF", ["country", "EF_delta"]], how="left", on="country")
    cf12 = pd.merge(projections, ef19_merge.loc[ef19_merge["Technology"]=="BF-BOF", ["country", "EF_sci", "EF_delta"]], how="left", on="country")

    cf12 = cf12.assign(ef_12=cf12["bof_ohf_ef"]+cf12["EF_delta"])
    cf12_hist = cf12_hist.assign(bof_12 = cf12_hist["bof_ohf_ef"] + cf12_hist["EF_delta"])
    # Add fixed ef part to ef1 uncertainty bounds
    cf12 = cf12.assign(ef_12_upper = cf12["bof_ohf_ef_upper"] + cf12["EF_delta"],
                ef_12_lower = cf12["bof_ohf_ef_lower"] + cf12["EF_delta"])

    return cf12_hist, cf12



def get_histo_efs(emissions_data_dir: Path, activity_data_dir: Path) -> pd.DataFrame:
    emissions_techno = concat_dfs(data_dir=emissions_data_dir, index_col=0)
    activity_techno = read_activity_xlsx(activity_data_dir).rename(columns={"name": "country"})
    # add aggregate values
    emissions_techno = emissions_techno.assign(inter_CO2_t=emissions_techno.loc[:, ~emissions_techno.columns.isin(["country", "year", "bof_CO2_t", "ohf_CO2_t", "eaf_CO2_t"])].sum(axis=1))
    emissions_techno = emissions_techno.assign(bof_ohf_CO2_t=emissions_techno.loc[:, emissions_techno.columns.isin(["inter_CO2_t", "bof_CO2_t", "ohf_CO2_t"])].sum(axis=1))
    # TODO: rename to bof separate from ohf
    activity_techno = activity_techno.assign(bof_ohf_t=activity_techno.loc[:, ["bof"]].sum(axis=1)*1000.,
                                            eaf_t=activity_techno["eaf"]*1000.,
                                            year=activity_techno["year"].astype(int))
    efs = pd.merge(activity_techno[["country", "year", "eaf_t", "bof_ohf_t"]], emissions_techno[["country", "year", "bof_ohf_CO2_t", "eaf_CO2_t"]], on=["year", "country"], how="left")
    efs = efs.assign(bof_ohf_ef=efs["bof_ohf_CO2_t"] / efs["bof_ohf_t"],
                    eaf_ef=efs["eaf_CO2_t"] / efs["eaf_t"])
    return efs


def extrapolate_group(group, target_col="ef", years_ahead=10, alpha=0.1):
    """
    Extrapolate EF series for a country using:
      - Linear regression (raw scale)
      - Log-linear regression (with normalized year and proper CI/PI transforms)
      - Simple rate-of-change extrapolation

    Returns point predictions and 95% confidence/prediction intervals.
    """
    group = group.sort_values("year")

    # --- Normalize (center) years ---
    year_mean = group["year"].mean()
    group["year_norm"] = group["year"] - year_mean

    # Future years
    last_year = group["year"].max()
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
    future_years_norm = future_years - year_mean

    # --- Linear regression (untransformed EF) ---
    if group["country"].iloc[0] ==  "Russia":
        X = sm.add_constant(group.loc[group["year"] != 2019, "year_norm"])
        y = group.loc[group["year"] != 2019, target_col]
    else:
        X = sm.add_constant(group["year_norm"])
        y = group[target_col]
    model = sm.OLS(y, X).fit()
    X_future = sm.add_constant(future_years_norm)
    preds = model.get_prediction(X_future)
    summary = preds.summary_frame(alpha=alpha)

    out_linear = pd.DataFrame({
        "country": group["country"].iloc[0],
        "year": future_years,
        f"{target_col}": summary["mean"],
        f"{target_col}_lower": summary["obs_ci_lower"],
        f"{target_col}_upper": summary["obs_ci_upper"]
    })

    # --- Log-linear regression (normalized years) ---
    if np.all(group[target_col] > 0):
        y_log = np.log(y)
        model_log = sm.OLS(y_log, X).fit()
        preds_log = model_log.get_prediction(X_future)
        summary_log = preds_log.summary_frame(alpha=alpha)

        # Residual variance for bias correction
        sigma2 = model_log.mse_resid

        # Mean predictions
        mean_log = np.exp(summary_log["mean"])  # median prediction
        mean_log_bc = np.exp(summary_log["mean"] + 0.5 * sigma2)  # bias-corrected mean

        # Confidence intervals for mean (no bias correction)
        mean_ci_lower = np.exp(summary_log["mean_ci_lower"])
        mean_ci_upper = np.exp(summary_log["mean_ci_upper"])

        # Prediction intervals for individual obs (no bias correction)
        obs_ci_lower = np.exp(summary_log["obs_ci_lower"])
        obs_ci_upper = np.exp(summary_log["obs_ci_upper"])
        
        # Manual calculation of prediction intervals
        # Using formula: predicted +/- t * sqrt(MSE * (1 + x_new'(X'X)^-1x_new))
        x_new = X_future
        x_old = X
        mse = model_log.mse_resid
        t_val = stats.t.ppf(1-alpha/2, model_log.df_resid)
        
        # Calculate (X'X)^-1
        xtx_inv = np.linalg.inv(x_old.T @ x_old)
        
        # Calculate variance for each prediction point
        pred_var = np.array([mse * (1 + x_i @ xtx_inv @ x_i.T) for x_i in x_new])
        
        # Calculate prediction intervals
        obs_ci_lower_2 = np.exp(summary_log["mean"] - t_val * np.sqrt(pred_var))
        obs_ci_upper_2 = np.exp(summary_log["mean"] + t_val * np.sqrt(pred_var))
        
    else:
        mean_log = mean_log_bc = mean_ci_lower = mean_ci_upper = obs_ci_lower = obs_ci_upper = obs_ci_lower_2 = obs_ci_upper_2 = [np.nan] * len(future_years)

    out_log = pd.DataFrame({
        "country": group["country"].iloc[0],
        "year": future_years,
        f"{target_col}_logreg": mean_log,
        f"{target_col}_logreg_bc": mean_log_bc,
        f"{target_col}_logreg_mean_lower": mean_ci_lower,
        f"{target_col}_logreg_mean_upper": mean_ci_upper,
        f"{target_col}_logreg_obs_lower": obs_ci_lower,
        f"{target_col}_logreg_obs_upper": obs_ci_upper,
        f"{target_col}_logreg_obs_lower_2": obs_ci_lower_2,
        f"{target_col}_logreg_obs_upper_2": obs_ci_upper_2,
    })

    # Combine all results
    return pd.concat([
        out_linear,
        out_log.drop(columns=["country", "year"]),
    ], axis=1)

def get_efs_19(efs: pd.DataFrame, sci_ef: pd.DataFrame) -> pd.DataFrame:
    efs_19 = efs.loc[efs["year"] == 2019]
    hui_19 = efs_19[["country", "bof_ohf_ef", "eaf_ef"]].melt(id_vars=["country"], value_vars=["bof_ohf_ef", "eaf_ef"], value_name="EF", var_name="Technology")
    hui_19 = hui_19.assign(Technology=hui_19["Technology"].replace({"bof_ohf_ef": "BF-BOF", "eaf_ef": "EAF"}))
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
    # EU_27_minus_huizhong = set(EU_27).difference(sci_ef.loc[sci_ef["Technology"] == "BF-BOF", "country"].unique())
    # country2eu = {c: "EU" for c in EU_27_minus_huizhong}
    EU_27_minus_huizhong = set(EU_27).difference(sci_ef.loc[sci_ef["Technology"] == "BF-BOF", "country"].unique())
    country2eu = {c: "EU" for c in EU_27_minus_huizhong}
    hui_19["country_map"] = hui_19["country"].replace(country2eu)
    ef19_merge = pd.merge(hui_19, sci_ef, how="left", left_on=["country_map", "Technology"], right_on=["country", "Technology"], suffixes=("_hui", "_sci"))
    ef19_merge["EF_sci"] = ef19_merge["EF_sci"].fillna(2.0)
    ef19_merge = ef19_merge.drop(columns=["country_sci"]).rename(columns={"country_hui": "country"})
    ef19_merge = ef19_merge.assign(EF_delta=ef19_merge["EF_sci"] - ef19_merge["EF_hui"])
    return ef19_merge


def concat_dfs(data_dir: Path, **kwargs) -> pd.DataFrame:
    dfs = []
    for f in data_dir.glob("*.csv"):
        year = f.stem  # filename without extension, e.g. "2018" from "2018.csv"
        df = pd.read_csv(f, index_col=kwargs["index_col"])
        df["year"] = int(year)
        dfs.append(df)

    # Combine into one DataFrame
    df_all = pd.concat(dfs, ignore_index=True).sort_values(["year", "country"], ascending=True).reset_index(drop=True)
    return df_all



def read_activity_xlsx(file_path, skip_first=True, add_year_column=True):
    """
    Reads all sheets from an Excel file, optionally skips the first sheet,
    optionally adds a Year column from sheet names, and concatenates into one DataFrame.

    Parameters:
        file_path (str): Path to the Excel file.
        skip_first (bool): If True, skips the first sheet (default: True).
        add_year_column (bool): If True, adds a column 'Year' with the sheet name (default: True).

    Returns:
        pd.DataFrame: Concatenated DataFrame from all sheets.
    """
    # Get all sheet names
    all_sheets = pd.ExcelFile(file_path).sheet_names
    
    # Decide which sheets to read
    sheets_to_read = all_sheets[1:] if skip_first else all_sheets
    
    # Read all sheets into a list
    df_list = []
    for sheet in sheets_to_read:
        df = pd.read_excel(file_path, sheet_name=sheet, engine="calamine")
        if add_year_column:
            df['year'] = sheet
        df_list.append(df)
    
    # Concatenate all DataFrames
    df_all = pd.concat(df_list, ignore_index=True)
    
    return df_all


if __name__ == "__main__":
    histo_efs_path = Path(__file__).parent.parent.parent / "data" / "raw" / "for_review" / "ef12_hist_huizhong.csv"
    sci_ef_path = Path(__file__).parent.parent.parent / "data" / "raw" / "emission_factors" / "EF_SCI_22.xlsx"
    emissions_data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "efs" / "steelmaking_country_year" / "steelmaking_country_year"
    activity_data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "efs" / "process_activity_data.xlsx"
    max_abat_path = Path(__file__).parent.parent.parent / "data" / "techno_abatement_bounds.xlsx"
    cf12_hist, cf12_proj = get_projected_efs(emissions_data_dir, activity_data_dir, sci_ef_path)
    print(cf12_hist)
    print(cf12_proj)

    # Correct emission factors based on adjusted uncertainty bounds
    cf12_proj_adj = correct_uncertainty_bounds(cf12_proj, max_abat_path=max_abat_path)
    cf12_proj_adj.to_excel(Path(__file__).parent.parent.parent / "data" / "raw" / "for_review" / "ef12_proj_adj.xlsx", index=False)

    print(cf12_proj_adj)
    # --- Plot for China (Adjusted EF_12) ---
    # Filter data for China from the adjusted projections
    china_proj_adj = cf12_proj_adj[cf12_proj_adj["country"] == "China"]

    # Create a new figure for China's adjusted EF_12
    plt.figure()

    # Plot adjusted projected values
    plt.plot(china_proj_adj["year"], china_proj_adj["adj_ef_12"], label="Adjusted Projected EF_12", marker='x', linestyle='--')
    
    # Plot adjusted uncertainty bounds
    if "adj_ef_12_lower" in china_proj_adj.columns and "adj_ef_12_upper" in china_proj_adj.columns:
        plt.fill_between(china_proj_adj["year"], china_proj_adj["adj_ef_12_lower"], china_proj_adj["adj_ef_12_upper"], color='red', alpha=0.1, label="Adjusted Projected 95% CI")

    # Add labels and title for China adjusted EF_12 plot
    plt.xlabel("Year")
    plt.ylabel("Adjusted EF_12 Emission Factor (tCO2/t)")
    plt.title("Adjusted Projected EF_12 Emission Factors for China")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot for India (Adjusted EF_12) ---
    # Filter data for India from the adjusted projections
    india_proj_adj = cf12_proj_adj[cf12_proj_adj["country"] == "India"]

    # Create a new figure for India's adjusted EF_12
    plt.figure()

    # Plot adjusted projected values for India
    plt.plot(india_proj_adj["year"], india_proj_adj["adj_ef_12"], label="Adjusted Projected EF_12", marker='x', linestyle='--')
    
    # Plot adjusted uncertainty bounds for India
    if "adj_ef_12_lower" in india_proj_adj.columns and "adj_ef_12_upper" in india_proj_adj.columns:
        plt.fill_between(india_proj_adj["year"], india_proj_adj["adj_ef_12_lower"], india_proj_adj["adj_ef_12_upper"], color='red', alpha=0.1, label="Adjusted Projected 95% CI")

    # Add labels and title for India adjusted EF_12 plot
    plt.xlabel("Year")
    plt.ylabel("Adjusted EF_12 Emission Factor (tCO2/t)")
    plt.title("Adjusted Projected EF_12 Emission Factors for India")
    plt.legend()
    plt.grid(True)
    plt.show()
    import matplotlib.pyplot as plt

    # Filter data for China
    china_hist = cf12_hist[cf12_hist["country"] == "China"]
    china_proj = cf12_proj[cf12_proj["country"] == "China"]

    # Plot historical values
    plt.plot(china_hist["year"], china_hist["bof_ohf_ef"], label="Historical", marker='o', linestyle='-')

    # Plot projected values
    plt.plot(china_proj["year"], china_proj["bof_ohf_ef"], label="Projected", marker='x', linestyle='--')
    
    # Plot projected uncertainty bounds if available
    if "bof_ohf_ef_lower" in china_proj.columns and "bof_ohf_ef_upper" in china_proj.columns:
        plt.fill_between(china_proj["year"], china_proj["bof_ohf_ef_lower"], china_proj["bof_ohf_ef_upper"], color='blue', alpha=0.1, label="Projected 95% CI")

    # Add labels and title for China plot
    plt.xlabel("Year")
    plt.ylabel("BOF/OHF Emission Factor (tCO2/t)")
    plt.title("Historical and Projected BOF/OHF Emission Factors for China")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot for India ---
    # Filter data for India
    india_hist = cf12_hist[cf12_hist["country"] == "India"]
    india_proj = cf12_proj[cf12_proj["country"] == "India"]

    # Create a new figure for India
    plt.figure()

    # Plot historical values for India
    plt.plot(india_hist["year"], india_hist["bof_ohf_ef"], label="Historical", marker='o', linestyle='-')

    # Plot projected values for India
    plt.plot(india_proj["year"], india_proj["bof_ohf_ef"], label="Projected", marker='x', linestyle='--')
    
    # Plot projected uncertainty bounds for India if available
    if "bof_ohf_ef_lower" in india_proj.columns and "bof_ohf_ef_upper" in india_proj.columns:
        plt.fill_between(india_proj["year"], india_proj["bof_ohf_ef_lower"], india_proj["bof_ohf_ef_upper"], color='blue', alpha=0.1, label="Projected 95% CI")

    # Add labels and title for India plot
    plt.xlabel("Year")
    plt.ylabel("BOF/OHF Emission Factor (tCO2/t)")
    plt.title("Historical and Projected BOF/OHF Emission Factors for India")
    plt.legend()
    plt.grid(True)
    plt.show()