from dataclasses import dataclass
import pandas as pd
import math

from pathlib import Path


@dataclass
class RefinitivDataset:
    def __init__(self, data_path: Path):
        """Initialise constructor with path to Refinitiv extract.

        Args:
            data_path (Path): Refinitiv extract path (old -> "total_available_companies.csv"; new -> "full_extract_refinitiv_2307.parquet")
        """
        super().__init__()
        self.data_path = data_path
        if "total_available_companies_Refinitiv.csv" in self.data_path.as_posix():
            self.extract = 'old'
        elif "full_extract_refinitiv_2307.parquet" in self.data_path.as_posix():
            self.extract = "new"
        else: 
            raise Exception("Unknown Refinitiv extract")
        self.raw_data = self.get_raw_data()

    def get_preprocessed_data(self):
        """Get Refinitiv dataset, where some scope 1+2 values are corrected, based on reported values
        in sustainability reports or updates from Refinitiv.

        Returns:
            pd.DataFrame: company level (financial and extra-financial) information
        """
        new_data = self.raw_data.copy()
        new_data = self.add_features(new_data)
        # Correct GHG emissions from Refinitiv with values from other sources (sustainability reports...)
        new_data = self.correct_company_emissions(new_data)
        return new_data
    
    def correct_company_emissions(self, new_data: pd.DataFrame):
        """Add/replace inaccurate top-down emissions from Refinitiv with
        values gathered elsewhere (e.g. sustainability reports).
        Apply correction to copy of raw data (new_data) in order to compare
        changes afterwards.
        """
        new_data = new_data.set_index(["Name", "FiscalYear"])

        if self.extract == "new":
            # Correct scope 1+2 values
            
            # Source: 2022 esg report (p 107)
            # https://www.tunghosteel.com/EN/HomeEg/csr/report
            CF12_corrections = [
                ("Tung Ho Steel Enterprise Corp", 2019, 796076),
                ("Tung Ho Steel Enterprise Corp", 2020, 810994),
                ("Tung Ho Steel Enterprise Corp", 2021, 882400),
                ("Tung Ho Steel Enterprise Corp", 2022, 810098),
                ("Cleveland-Cliffs Inc", 2019, 39.8 * 1E6)] # see report
            
            correct_CF12 = pd.DataFrame(CF12_corrections, columns=("Name", "FiscalYear", "CF12"))
            correct_CF12 = correct_CF12.set_index(["Name", "FiscalYear"])
            
            new_data.loc[correct_CF12.index, "CF12"] = correct_CF12
        else:
            raise NotImplementedError
        return new_data.reset_index()
    
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Any vanilla features can be directly calculated as dataset is first read.

        Args:
            data (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: Refinitiv extract with additional features
        """
        data["CF12"] = data["CF1"] + data["CF2"]
        return data
    
    
    def get_raw_data(self):
        if self.extract == "old":
            return pd.read_csv(self.data_path)
        elif self.extract == "new":
            return pd.read_parquet(self.data_path, engine='pyarrow')


    
