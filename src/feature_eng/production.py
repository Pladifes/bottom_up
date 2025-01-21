from pathlib import Path
project_dir = Path().cwd()
import sys
sys.path.append(str(project_dir / "src"))
raw_data_dir = project_dir / "data" / "raw"
    
import pandas as pd
from projections_.targets import get_relevant_stated_targets

def get_stated_production(refi_targets_prod_path,
                          wsa_prod_path,
                          gspt,
                          refi2gspt_path,
                          gspt2wsa_path,
                          parent_group_map
                          ):
    """Get multi-sourced production values for companies with
    stated targets.
    refi_targets_prod_path: stated targets + production values
    """
    # prod
    targets_prod_raw = pd.read_excel(refi_targets_prod_path)
    targets_prod = get_relevant_stated_targets(targets_prod_raw)
    targets_prod = targets_prod.rename(columns={"Steel Production (Metric Tons)": "Refi prod"})
    wsa_prod = pd.read_excel(wsa_prod_path)
    bu_parent_prod = get_bu_parent_prod(gspt)
    bu_group_prod = get_bu_group_prod(gspt, parent_group_map)
    
    # mappings
    refi2gspt = pd.read_excel(refi2gspt_path)
    refi2gspt_dict = refi2gspt.set_index('Refinitiv_Name')['GSPT_Name'].to_dict()
    gspt2wsa = pd.read_excel(gspt2wsa_path)
    wsa2gspt_dict = gspt2wsa.set_index('WSA_Name')['GSPT_Name'].to_dict()

    # map refinitiv names to GSPT names
    targets_prod['GSPT_Name'] = targets_prod["Company Common Name"].replace(refi2gspt_dict)
    targets_prod['Group'] = targets_prod['GSPT_Name'].replace(parent_group_map)
    
    wsa_prod = pd.merge(wsa_prod,
                        gspt2wsa,
                        right_on=["WSA_Name"],
                        left_on=["Company"],
                        how="left")
    wsa_prod_ = wsa_prod.loc[wsa_prod["GSPT_Name"].notna()]
    wsa_prod_['Production 2022 (Mt)'] = wsa_prod_['Production 2022 (Mt)'].replace({"(e) 12.50": 12.5}).astype(float)
    wsa_prod_["WSA prod"] = wsa_prod_['Production 2022 (Mt)'] * 1e6
    
    # merge on stated & production refi
    targets_prod = pd.merge(targets_prod,
                    wsa_prod_.loc[:,["WSA prod", "GSPT_Name"]],
                    on="GSPT_Name",
                    how="left")
    
    targets_prod = pd.merge(targets_prod,
                bu_parent_prod.loc[:,["Parent", "Parent crude steel production"]],
                left_on="GSPT_Name",
                right_on="Parent",
                how="left")
    targets_prod = targets_prod.rename(columns={"Parent crude steel production": "BU parent prod"})

    targets_prod = pd.merge(targets_prod,
            bu_group_prod.loc[:,["Group", "Group crude steel production"]],
            left_on="GSPT_Name",
            right_on="Group",
            how="left")
    targets_prod = targets_prod.rename(columns={"Group crude steel production": "BU group prod"})

    targets_prod = targets_prod.dropna(subset="Emission Reduction Target Percentage")
    # check Essar, Vale is Outlier
    targets_prod = targets_prod.loc[~targets_prod['Company Common Name'].isin(["Essar Global Fund Ltd", "Vale SA"])]
    
    # tests
    targets_prod[["Company Common Name", "Refi prod", "WSA prod", "BU parent prod", "BU group prod"]]
    return targets_prod.reset_index(drop=True)

def get_bu_group_prod(gspt, parent_group_map):
    # 1. Import bottom up production
    bu_plant_prod = gspt.get_estimated_prod(year=2022, impute_prod="Country")

    # map Parent companies to Group
    bu_plant_prod['Group'] = bu_plant_prod['Parent'].replace(parent_group_map)

    bu_plant_prod["Group capacity"] = bu_plant_prod["Nominal crude steel capacity (ttpa)"] * bu_plant_prod["Share"] * 1e3
    # Aggregated bottom-up production at group level
    bu_group_prod = bu_plant_prod.groupby(["year", "Group"]).agg({"Attributed crude steel production": "sum",
                                                                  "Group capacity": "sum"}).reset_index()

    bu_group_prod = bu_group_prod.rename(columns={"year": "prod_year",
                                                "Attributed crude steel production": "Group crude steel production"})
    return bu_group_prod

def get_bu_parent_prod(gspt):
    # 1. Import bottom up production
    bu_plant_prod = gspt.get_estimated_prod(year=2022, impute_prod="Country")

    # Aggregated bottom-up production at group level
    bu_parent_prod = bu_plant_prod.groupby(["year", "Parent"])["Attributed crude steel production"].sum().reset_index()

    bu_parent_prod = bu_parent_prod.rename(columns={"year": "prod_year",
                                                "Attributed crude steel production": "Parent crude steel production"})
    return bu_parent_prod


