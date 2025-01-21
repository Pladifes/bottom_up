import pandas as pd
import re
from typing import List
import json
import numpy as np

def get_cagr(base_value: float, end_value: float, n: int) -> float:
    """Calculate compound average growth rate.

    Args:
        base_value (float): _description_
        end_value (float): _description_
        n (int): number of years between base and end years (including base year)

    Raises:
        ZeroDivisionError: _description_

    Returns:
        float: _description_
    """
    if isinstance(base_value, float):
        if base_value == 0.:
            raise ValueError
    
    cagr = (end_value/base_value) ** (1./n) - 1
    return cagr


def fill_missing_CF12(refinitiv, refi2gspt_path, gspt2gspt_map):
    # add missing CF12 based on reports
    refi2gspt = pd.read_excel(refi2gspt_path)
    refi2gspt_dict = refi2gspt.set_index('Refinitiv_Name')['GSPT_Name'].to_dict()
    refinitiv["Group"] = refinitiv["Name"].replace(refi2gspt_dict)
    refinitiv["Group"] = refinitiv["Group"].replace(gspt2gspt_map)
    refinitiv.loc[(refinitiv["Group"] == "Aichi Steel Corp") & (refinitiv["FiscalYear"] == 2019), "CF12"] = 687000
    refinitiv = pd.concat([refinitiv, pd.DataFrame({"Group": ["Benteler International AG", 
                                                            "SHS Stahl Holding Saar GmbH & Co KgaA",
                                                            "Hbis Group Co Ltd",
                                                            "Shougang Group Co Ltd"],
                "FiscalYear": [2019, 
                                2019,
                                2022,
                                2022],
                "CF12": [366705, 
                            8406140,
                            41 * (1.41 + 0.458) * 1e6, # (EF1_2022 + EF2_2019)
                            33.82 * (1.41 + 0.458) * 1e6]})], ignore_index=True, axis=0)
    return refinitiv


def convert_plant2parent(proj_plants, gspt2gspt_path, parent_group_map):
    """Convert frame from plant ID to Parent ID view and add group companies column.

    Args:
        proj_plants (_type_): _description_

    Returns:
        _type_: _description_
    """
    id_cols = proj_plants[['Parent', "Parent PermID", "Plant ID"]].drop_duplicates()
    parent_ids = get_parents_ids(id_cols, gspt2gspt_path=gspt2gspt_path)
    final_cols = [col for col in proj_plants.columns if col not in ["Parent", "Parent PermID"]]
    proj_parents = pd.merge(proj_plants.loc[:, final_cols],
                            parent_ids, 
                            on="Plant ID", 
                            how="left")
    proj_parents['Group'] = get_group_col(gspt_parent_col=proj_parents['Parent'],
                                          gspt2gspt_path=gspt2gspt_path,
                                          parent_group_map=parent_group_map)
    return proj_parents

def get_group_col(gspt_parent_col: pd.Series, gspt2gspt_path, parent_group_map):
    """Map Global Steel Plant Tracker parent companies to group companies
    when such exist.

    Args:
        gspt_parent_col (_type_): _description_
        gspt2gspt (dict): _description_
        parent_group_map (dict): _description_

    Returns:
        _type_: _description_
    """
    with open(gspt2gspt_path, "r") as f:
        gspt2gspt = json.load(f)
    # normalise GSPT parent company names
    group_col = gspt_parent_col.replace(gspt2gspt)
    # map GSPT parent companies to group companies
    group_col = group_col.replace(parent_group_map)
    return group_col


def get_parents_ids(steel_df, gspt2gspt_path):
    """Melt ownership information from each plant.
    Ex: 
    Company_A 50%; Company_B 30%, Company_C 20% ->
    
    [Parent, Share]
    [Company_A, 0.5]
    [Company_B, 0.3] 
    [Company_C, 0.2]

    Args:
        steel_df (pd.DataFrame): raw steel data
        gspt2gspt_path (str): path to json gspt2gspt dict

    Returns:
        pd.DataFrame: 
    """
    assert steel_df.columns.isin(["Plant ID", "Parent PermID", "Parent"]).sum() == 3
    melt_df2 = split_parent_shares(
        plant_ids_list=steel_df["Plant ID"].tolist(), parents_col=steel_df["Parent"]
    )

    melt_df = split_parent_shares(
        plant_ids_list=steel_df["Plant ID"].tolist(),
        parents_col=steel_df["Parent PermID"],
    )

    mergedf = pd.concat([melt_df2, melt_df], axis=1)

    mergedf = mergedf.loc[:, ~mergedf.columns.duplicated()].copy()
    # reorder cols
    mergedf = mergedf.loc[:, ["Parent PermID", "Parent", "Plant ID", "Share"]]
    
    # standardise parent names (avoid multiple entities for same company)
    with open(gspt2gspt_path) as f:
        gspt2gspt = json.load(f)
    
    mergedf["Parent"] = mergedf['Parent'].replace(gspt2gspt)
    
    return mergedf


def split_parent_shares(plant_ids_list: List, parents_col: pd.Series) -> pd.DataFrame:
    """
    Steel plants may be owned by multiple parent companies, with non equally distributed shares.
    Emissions should be attributed to individual companies according to those weights.

    Args:
        parents_list (List[dict]): parents_list[i] = {"comp_A": ownership share for plant i}
        emissions_list (_type_): [emissions_1 -> plant_1 emissions, emissions_2 -> plant_2 emissions]

    Returns:
        melted_parents_df (pd.DataFrame): ["Plant ID", "Parent", "Share"]
    """
    assert len(plant_ids_list) == len(parents_col)
    n_plants = len(set(plant_ids_list))
    
    parents_list = []
    for plant_id, parents in zip(plant_ids_list, parents_col.str.split("; ").tolist()):
        for name_share in parents:
            plantid_parent_share = clean_col(plant_id, name_share)
            parents_list.extend(plantid_parent_share)
            # Some exceptions are handled by hand
            # ["SCN00175", 'SCN00175-1', "SCN00096"]
            # break loop to avoid double counting share
            if plant_id in ["SCN00175", 'SCN00175-1', "SCN00096"]:
                break


    melted_parents_df = pd.DataFrame(
        parents_list, columns=["Plant ID", parents_col.name, "Share"]
    )
    assert len(melted_parents_df['Plant ID'].unique()) == n_plants
    assert melted_parents_df.notna().all().all()

    # Normalise company names
    to_rename = {
        "other": "Other",
        "Ansteel Group Corporation": "Ansteel Group Corp Ltd",
        "Baoshan Iron & Steel Co.,Ltd.": "Baoshan Iron & Steel Co Ltd",
        "Baoshan Iron and Steel Co., Ltd.": "Baoshan Iron & Steel Co Ltd",
        "Hunan Valin Steel Co.,Ltd.": "Hunan Valin Steel Co Ltd"
    }
    melted_parents_df.replace({"Parent": to_rename}, inplace=True)
    return melted_parents_df.reset_index(drop=True)


def clean_col(plant_id: str, name_share: str):
    if plant_id in ["SCN00175", 'SCN00175-1']:
        # 'Ansteel Group Corporation 42.67%, China Minmetals Corporation 6.96%, 
        # Benxi Beifang Investment Co., Ltd. 7.54%, other  42.83%
        id_parent_share = [(plant_id, "Ansteel Group Corporation", 0.4267),
                           (plant_id, "China Minmetals Corporation", 0.0696),
                           (plant_id, "Benxi Beifang Investment Co., Ltd.", 0.0754),
                           (plant_id, "Other", 0.4283)]   
    elif plant_id == "SCN00096":
        id_parent_share = [(plant_id, "Beijing Jianlong Investment Co., Ltd.", 0.7184),
            (plant_id, "Fosun Holdings", 0.249),
            (plant_id, "Other", 0.0326)]   
    else:
        id_parent_share = [(plant_id, None, None)]
        # retrieve string before first square bracket
        names = re.findall("(.*?)\s*\[", name_share)[0]

        # retrieve number between square brackets
        shares = re.findall(r"[-+]?(?:\d*\.*\d+)%", name_share)
        shares = [float(s[:-1]) / 100 for s in shares]

        # some strings before square brackets contain
        # all the information. Second split required
        names = re.split("\s(\d+%,\s|\d+%)", names)
        if len(names) > 1:
            names = [
                name for name in names if bool(name) if not bool(re.search("\d+%", name))
            ]
            shares = shares[:-1]
            assert len(names) == len(shares)

        if len(names) == len(shares) == 1:
            id_parent_share = [(plant_id, names[0], shares[0])]
        elif (len(names) > 1) & (len(shares) > 1):
            repeat_plant_id = [plant_id] * len(names)
            id_parent_share = list(zip(repeat_plant_id, names, shares))

    return id_parent_share

