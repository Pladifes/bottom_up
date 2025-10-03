from pydantic import BaseModel
from pathlib import Path
import toml


class ElectricityConfig(BaseModel):
    elec_int_nze: Path
    elec_int_steps: Path
    elec_int_aps: Path
    
class ProjectedDataConfig(BaseModel):
    electricity: ElectricityConfig

    
class ScenariosConfig(BaseModel):
    iea_world: Path
    iea_regions: Path

class MappingsConfig(BaseModel):
    gspt2gspt: str
    gspt2iea_countries: Path
    refi2gspt: Path
    gspt2cdp: Path
    parent_group_map: str
    EU_27: Path

class MacroConfig(BaseModel):
    iea_prod: Path
    wsa_prod: Path
    oecd_capa: Path
    carbon_price: Path

class TopDownConfig(BaseModel):
    refinitiv: Path
    cdp: Path
    
class MicroConfig(BaseModel):
    company_prod: Path

class EmissionFactorsConfig(BaseModel):
    wsa: str
    jrc: str
    sci: str
    huizhong: Path
    energy_mix: Path

class HistoricalDataConfig(BaseModel):
    version_year: int
    asset_level_data: str
    missing_years: str
    macro: MacroConfig
    micro: MicroConfig
    top_down: TopDownConfig

class ParamsConfig(BaseModel):
    project_dir: Path 
    save_dir: Path
    scenario: str
    steel_db: Path
    scenario_level: str
    model_name: str
    EAF_decarb: bool

class Config(BaseModel):
    params: ParamsConfig
    historical_data: HistoricalDataConfig
    emission_factors: EmissionFactorsConfig
    mappings: MappingsConfig
    scenarios: ScenariosConfig
    projected_data: ProjectedDataConfig

def load_config(config_file: Path) -> Config:
    config_data = toml.load(config_file)
    return Config(**config_data)