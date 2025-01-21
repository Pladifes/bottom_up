from pathlib import Path
project_dir = Path().cwd()
while not "bottom_up_alignment" == str(project_dir.stem):
    project_dir = project_dir.parent
    
import sys
sys.path.append(str(project_dir / "src"))

from BUProjector import get_projections_iea, get_market_share
from projections import get_bu_proj_emissions, get_histo_plants_21

from datasets.utils import get_parents_ids, convert_plant2parent

import numpy as np
import pandas as pd

def plot_hhi():
    pass

def get_all_hhi(proj_comp_prod, proj_plants_prod, glob_prod, histo_plants_21, gspt2gspt_path):
    """Compute HHI for all levels of granularity.
    """
    all_hhi = []
    for level in ['Plant', "Country", "Region", "Company"]:
        if level == "Company":
            histo_plants_21_melt = convert_plant2parent(histo_plants_21, gspt2gspt_path=gspt2gspt_path)
            histo_plants_21_melt["Attributed production (ttpa)"] = histo_plants_21_melt['Estimated crude steel production (ttpa)'] * histo_plants_21_melt['Share']
            histo_parents_21 = histo_plants_21_melt.groupby(["year", "Parent"]).agg({"Attributed production (ttpa)": "sum"}).reset_index()
            histo_parents_21 = histo_parents_21.rename(columns={"Attributed production (ttpa)": "Estimated crude steel production (ttpa)"})
            
            proj_parents = proj_comp_prod.copy()
            proj_parents = proj_parents.rename(columns={"Attributed production (ttpa)": "Estimated crude steel production (ttpa)"})
            proj_parents = proj_parents.rename(columns={"Group": "Parent"})
            proj_parents = pd.concat([histo_parents_21, proj_parents], ignore_index=True, axis=0)        
            proj_parents = pd.merge(proj_parents,
                    glob_prod,
                    on="year",
                    how="left")
            proj_parents["market_share"] = proj_parents["Estimated crude steel production (ttpa)"] / proj_parents["Global production (ttpa)"]
            hhi = proj_parents.groupby("year").agg(company_hhi=("market_share", compute_hhi))
        else:
            proj_plants = pd.concat([histo_plants_21, proj_plants_prod], axis=0, ignore_index=True)
            hhi = get_agg_hhi(proj_plants=proj_plants, glob_prod=glob_prod, level=level, gspt2gspt_path=gspt2gspt_path)
        
        all_hhi.append(hhi)
    
    all_hhi = pd.concat(all_hhi, axis=1)
    return all_hhi

def get_agg_hhi(proj_plants, glob_prod, level, gspt2gspt_path=None):
    """From projected plants and their estimated future production, 
    this function aggregates production, in order to calculate HHI 
    at a coarser level (country, region, company).

    Args:
        proj_plants (_type_): operating plants over projection period. Selected plants vary 
        from hypothesis to hypothesis.
        glob_prod (_type_): historical and projected global production used for estimating market share
    """
    if level in ["Country", "Region"]:
        agg_prod = proj_plants.groupby(['year', level]).agg({"Estimated crude steel production (ttpa)": "sum"}).reset_index()
        agg_prod = pd.merge(agg_prod,
                            glob_prod,
                            on=["year"],
                            how='left')
        # compute market share
        agg_prod['market_share'] = agg_prod["Estimated crude steel production (ttpa)"] / agg_prod["Global production (ttpa)"]
        
        agg_hhi = agg_prod.groupby("year").agg({"market_share": compute_hhi})
        agg_hhi = agg_hhi.rename(columns={"market_share": f"{level}_hhi"})
        return agg_hhi
    elif level == "Plant":
        proj_plants = pd.merge(proj_plants,
                    glob_prod,
                    on=["year"],
                    how='left')
        # compute market share
        proj_plants['market_share'] = proj_plants["Estimated crude steel production (ttpa)"] / proj_plants["Global production (ttpa)"]
        plant_hhi = proj_plants.groupby("year").agg(plant_hhi=("market_share", compute_hhi))
        return plant_hhi

    elif level == "Company":
        parent_ids = get_parents_ids(proj_plants, gspt2gspt_path=gspt2gspt_path)
        
        final_cols = [col for col in proj_plants.columns if col not in ["Parent", "Parent PermID"]]
        
        proj_parents = pd.merge(parent_ids, proj_plants.loc[:, final_cols], on="Plant ID", how="left")
        
        # Company emissions / plant are proportional to ownership share
        proj_parents['Attributed production (ttpa)'] = proj_parents['Estimated crude steel production (ttpa)'] * proj_parents['Share']
        
        agg_prod = proj_parents.groupby(['year', "Parent"]).agg({"Attributed production (ttpa)": "sum"}).reset_index()
        # merge global production
        agg_prod = pd.merge(agg_prod,
                            glob_prod,
                            on=["year"],
                            how='left')
        
        agg_prod['market_share'] = agg_prod["Attributed production (ttpa)"] / agg_prod["Global production (ttpa)"]
        company_hhi = agg_prod.groupby("year").agg(parent_hhi=("market_share", compute_hhi))
        return company_hhi
    else:
        raise ValueError
        
    
def compute_hhi(s, weighted=False):
    """Herfindahl-Hirschmann index for measuring concentration of production
    in a given population (plants, country, region, company).

    Args:
        s (pd.Series): array of market shares
        weighted (boolean): if True, returns average squared market share
    """
    if not weighted:
        return np.sum(s ** 2)
    else:
        return np.mean(s ** 2)



    
