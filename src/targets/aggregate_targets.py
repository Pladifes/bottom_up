import pandas as pd
import numpy as np
from pathlib import Path
import json

def get_stated_targets(gst_path, 
                       gspt2gst_path,
                       tpi_path,
                       tpi2gspt_path,
                       refi_path,
                       nzt_path,
                       nzt2gspt_path,
                       gspt2gspt_path):
    # 1. Green Steel Tracker
    gspt2gst = pd.read_excel(gspt2gst_path)

    gst = pd.read_excel(gst_path, sheet_name="2. Company targets", header=3)

    # companies with 2030 targets
    gst30 = gst.loc[gst['2030 - Summary'] == "Has a 2030 goal"]

    # map gspt companies
    gst30 = pd.merge(gst30, 
                gspt2gst,
                left_on=["Company"],
                right_on=["GST_Name"],
                how='left')

    gst30['2021 company production (Mtpa, million tonnes steel p.a.) (source: World Steel in Figures 2021)'] = gst30['2021 company production (Mtpa, million tonnes steel p.a.) (source: World Steel in Figures 2021)'].replace("<8", 8)
    # drop companies with 2030 target that are not in GSPT
    gst30 = gst30.dropna(subset="GSPT_Name")
    gst30["Source"] = "GST"
    # extract targets: adds reduction percentage and baseline year
    gst30 = get_gst_percent_year(gst=gst30, col="2030 climate target")
    gst30["Baseline Year"] = gst30["Baseline Year"].replace({"undefined": np.nan})
    # columns to add to final df
    gst_to_add = gst30.loc[:,["GSPT_Name", "GST_Name", "Percentage", "Baseline Year", "Source"]]
    
    # 2. Transition Pathways Initiative
    tpi = pd.read_csv(tpi_path)
    tpi2gspt = pd.read_excel(tpi2gspt_path)
    tpi2gspt = tpi2gspt.rename(columns={"GSPT_Group_Name": "GSPT_Name"})
    
    # get tpi companies that have goal in 2030
    m1 = tpi["Years with targets"].apply(lambda x:False if str(x) == "nan" else ("2030" in x))
    tpi30 = tpi.loc[m1]
    # map gspt companies
    tpi30 = pd.merge(tpi30,
                   tpi2gspt, 
                   how="left", 
                   left_on=["Company Name"],
                   right_on=["TPI_Name"])
    # drop companies with 2030 target that are not in GSPT
    tpi30 = tpi30.dropna(subset="GSPT_Name")
    tpi30 = tpi30.rename(columns={"Assumptions": "TPI_Assumptions"})
    tpi30['Source'] = "TPI"
    
    # columns to add to final df
    tpi_to_add = tpi30.loc[:,["GSPT_Name", "TPI_Name", "TPI_Assumptions", "Source"]]
    
    # 3. Refinitiv
    refi30 = pd.read_excel(refi_path)
    refi30 = refi30.rename(columns={"Company Common Name": "Refinitiv_Name"})
    refi30["Source"] = "Refinitiv"
    refi30 = refi30.rename(columns={"Emission Reduction Target Percentage": "Percentage",
                                    "Emissions Target Type": "Target Type"})
    refi_to_add = refi30.loc[:, ["GSPT_Name", "Refinitiv_Name", "Percentage", "Target Type", "Attributed production", "Source"]]
    
    # 4. Nero Zero Tracker
    nzt2gspt = pd.read_excel(nzt2gspt_path)
    nzt = pd.read_excel(nzt_path)
    # masks: select only companies with an end or interim target year equal to 2030
    m1 = nzt['actor_type'] == "Company"
    m2 = (nzt['end_target_year'] == 2030) | (nzt['interim_target_year'] == 2030)
    nzt = nzt.loc[m1 & m2].reset_index(drop=True)
    
    nzt = pd.merge(nzt,
                   nzt2gspt,
                   how='left',
                   left_on=["name"],
                   right_on=["NZT_Name"])
    # clean
    nzt = nzt.dropna(subset="GSPT_Name").reset_index(drop=True)
    nzt = nzt.loc[nzt["Sector"] == "Steel"].reset_index(drop=True)
    nzt = nzt.drop(columns=["Sector"])
    
    nzt['Source'] = "Net Zero Tracker"
    nzt['Percentage'] = nzt["interim_target_percentage_reduction"].fillna(nzt['end_target_percentage_reduction'])
    nzt["Baseline Year"] = nzt["interim_target_baseline_year"].fillna(nzt["end_target_baseline_year"])
    nzt = nzt.dropna(subset=("Percentage", "Baseline Year"), axis=0)
    #nzt = nzt.rename(columns={"end_target_percentage_reduction": "Percentage"})
    nzt_to_add = nzt.loc[:, ["GSPT_Name", "NZT_Name", "Percentage", "Baseline Year", "Source"]]
    
    targets = pd.concat([gst_to_add, refi_to_add, nzt_to_add], axis=0, ignore_index=True)
    targets = targets.sort_values(by="GSPT_Name", ascending=True).reset_index(drop=True)
    
    # normalise GSPT names
    with open(gspt2gspt_path, "r") as f:
        gspt2gspt_map = json.load(f)
    targets["GSPT_Name"] = targets["GSPT_Name"].replace(gspt2gspt_map)
    return targets


def get_gst_percent_year(gst, col):
    """Extract reduction percentage and baseline year for climate targets
    from Green Steel Tracker data.

    Args:
        gst (pd.DataFrame): Green Steel Tracker targets
        col (str): Column containing str descriptions of climate targets (e.g. "2030 climate target")

    Returns:
        _type_: _description_
    """
    # Define the regex pattern
    pattern = r'(\d{1,3})%\s?(?:[\w\s-]+)?\s?\(baseline\s?(\d{4}|\w+)\)'

    # Use str.extract to extract the reduction percentage and baseline year
    gst[["Percentage", "Baseline Year"]] = gst[col].str.extract(pattern)
    
    to_drop = ["Emission intensity reduction <1.8 tCO2/tcs by 2030 (baseline unspecified)",
    "N/A - 2025 emission reduction to 1.8 t CO2e per tonne of steel (baseline undefined)"]
    gst = gst.loc[~gst[col].isin(to_drop)].reset_index(drop=True)
    
    # tests
    assert gst[["Percentage", "Baseline Year"]].notna().all().all()
    return gst