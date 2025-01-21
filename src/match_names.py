import pandas as pd
from name_matching.name_matcher import NameMatcher
from pathlib import Path

def get_mapping(ground_truth: pd.DataFrame,
                candidates: pd.DataFrame,
                distances: list,
                left_col: str, 
                right_col: str, 
                threshold=90):
    matches = match_names(ground_truth=ground_truth, candidates=candidates, distances=distances)
    raw_mapping = select_matches(ground_truth=ground_truth,
                                 candidates=candidates, 
                                 matches=matches,
                                 left_col=left_col,
                                 right_col=right_col,
                                 threshold=threshold)
    return raw_mapping


def select_matches(ground_truth, candidates, matches, left_col, right_col, threshold=90):
    """Return temporary company mapping based on distance index.
    Mapping should be hand inspected and edited.

    Args:
        ground_truth (pd.DataFrame): pivot names
        candidates (pd.DataFrame): prospective names
        matches (pd.DataFrame): matches with scores for different metrics
        left_col (str): pivot column, name column from the ground truth /reference dataset (e.g. GSPT_Name)
        right_col (str): new column, name column from the candidates / prospective dataset (e.g. CDP_Name)
        threshold (float): similarity score between 0 and 100
    """
    # number of metrics applied to string names
    n_metrics = matches.columns.str.contains("match_index").sum()
    
    top_matches = {}
    for i in range(n_metrics):
        # combine the datasets based on the matches and threshold
        combined = pd.merge(ground_truth, matches, how='left', left_index=True, right_on=f'match_index_{i}')
        combined = pd.merge(combined, candidates, how='left', left_index=True, right_index=True)
        top_matches[i] = combined.loc[combined[f"score_{i}"] > threshold,  [left_col, right_col, f"score_{i}"]]
        
    # aggregate top matches based on threshold for all metrics
    agg_matches = pd.concat([df for df in top_matches.values()], axis=0, ignore_index=True)
    agg_matches.sort_values(by=left_col, ascending=True, inplace=True)
    return agg_matches
    
    
def match_names(ground_truth: pd.DataFrame,
                candidates: pd.DataFrame,
                distances: list):
    """Match candidates to ground truth names.
    

    Args:
        ground_truth (pd.DataFrame): pivot names / names we are interested in matching
        candidates (pd.DataFrame): new names for which we have no reference / that should be aligned with ground truth
        distances (list, optional): choose among ['discounted_levenshtein', 'tichy', 'bag', 'SSK', 'fuzzy_wuzzy_token_sort', 'refined_soundex', 'typo']
        for more exhaustive list of distances, see documentation (https://name-matching.readthedocs.io/en/latest/index.html). There are a lot !

    Example:
            # define a dataset with bank names
    df_companies_a = pd.DataFrame({'Company name': [
            'Industrial and Commercial Bank of China Limited',
            'China Construction Bank',
            'Agricultural Bank of China',
            'Bank of China',
            'JPMorgan Chase',
            'Mitsubishi UFJ Financial Group',
            'Bank of America',
            'HSBC',
            'BNP Paribas',
            'CrÃ©dit Agricole']})

    # alter each of the bank names a bit to test the matching
    df_companies_b = pd.DataFrame({'name': [
            'Bank of China Limited',
            'Mitsubishi Financial Group',
            'Construction Bank China',
            'Agricultural Bank',
            'Bank of Amerika',
            'BNP Parisbas',
            'JP Morgan Chase',
            'HSCB',
            'Industrial and Commercial Bank of China',
            'Credite Agricole']})
    
    distances = ['discounted_levenshtein', 'tichy', 'bag', 'SSK', 'fuzzy_wuzzy_token_sort', 'refined_soundex', 'typo']
    matches = match_names(ground_truth=df_companies_a, candidates=df_companies_b,distances=distances)
    matches.to_excel('./test_matches.xlsx', index=False)
    
    Returns:
        _type_: _description_
    """
    # otherwise, data is modified inplace
    ground_truth = ground_truth.copy()
    candidates = candidates.copy()
    
    # initialise the name matcher
    matcher = NameMatcher(number_of_matches=len(distances), 
                        legal_suffixes=True, 
                        common_words=False, 
                        top_n=50, 
                        verbose=True)

    # adjust the distance metrics to use
    matcher.set_distance_metrics(distances)

    # load the data to which the names should be matched
    matcher.load_and_process_master_data(column=ground_truth.columns[0],
                                        df_matching_data=ground_truth, 
                                        transform=True)

    # perform the name matching on the data you want matched
    matches = matcher.match_names(to_be_matched=candidates, 
                                column_matching=candidates.columns[0])

    return matches

