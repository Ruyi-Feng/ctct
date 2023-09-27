import pandas as pd
import os


def add_total_rtg(data_path) -> None:
    """
    Calculates the total reward and adds a new column "total_rtg" to the given dataset.
    Args:
        data_path (str): The path to the CSV file containing the dataset.
    Returns:
        None
    """
    data = pd.read_csv(data_path)
    total_rtg = data["Reward"].sum()
    cum = data["Reward"].cumsum().values
    data["total_rtg"] = total_rtg - cum
    order = ["Step", "Observation_dim_1", "Observation_dim_2", "Observation_dim_3", "Observation_dim_4", "Observation_dim_5", "Observation_dim_6", "Action", "Reward", "total_rtg",
             "Next_Observation_dim_1", "Next_Observation_dim_2", "Next_Observation_dim_3", "Next_Observation_dim_4", "Next_Observation_dim_5", "Next_Observation_dim_6"]
    data = data[order]
    data.to_csv(data_path, index=False)

def deal_data(data_path) -> None:
    """
    生成一个index的字典来查询
    """
    dir_list = os.listdir(data_path)
    for sub_dir in dir_list:
        sub_dir_path = os.path.join(data_path, sub_dir)
        file_list = os.listdir(sub_dir_path)
        for file_name in file_list:
            file_path = os.path.join(sub_dir_path, file_name)
            add_total_rtg(file_path)

deal_data('./data/upper16w/')
