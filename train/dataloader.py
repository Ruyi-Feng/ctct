from torch.utils.data import Dataset
import numpy as np
import typing
import os


class STAR_Dataset(Dataset):
    def __init__(self,
                 data_path: str,
                 block_size: int,
                 if_total_rtg: bool=False) -> None:
        self.data_path = data_path
        self.block_size = block_size
        self.index = []
        self.f_dict = {}
        self.if_total_rtg = if_total_rtg
        self._gene_data_index()

    def _add_index(self, file_path: str, fl_line: int) -> None:
        for i in range(1, fl_line - self.block_size):
            self.index.append([file_path, i, i+self.block_size])

    def _gene_data_index(self) -> None:
        """
        生成一个index的字典来查询
        """
        dir_list = os.listdir(self.data_path)
        for sub_dir in dir_list:
            sub_dir_path = os.path.join(self.data_path, sub_dir)
            file_list = os.listdir(sub_dir_path)
            for file_name in file_list:
                file_path = os.path.join(sub_dir_path, file_name)
                fl_line = len(open(file_path).readlines())
                self.f_dict.setdefault(file_path, open(file_path, 'r'))
                self._add_index(file_path, fl_line)

    def _rwd2rtg(self, rwd: np.array) -> np.array:
        rtg = []
        [rtg.append(sum(rwd[i:])) for i in range(len(rwd))]
        return np.expand_dims(np.array(rtg), axis=1).astype(np.float32)

    def _load_data(self, path: str, start: int=-1, end: int=-1) -> tuple:
        # Step,
        # Observation_dim_1-6
        # Action, Reward,
        # Next_Observation_dim_1-6
        """
        从index中选取数据
        """
        data = []
        f = self.f_dict[path]
        f.seek(0)
        now_line = 0
        while now_line < start:
            f.readline()
            now_line += 1
        for i in range(start, end):
            line = f.readline().split(',')
            data.append(line)
        data = np.array(data)
        states = data[:, 1:7].astype(np.float32)
        timesteps = np.expand_dims(data[:, 0], axis=1).astype(np.int64)
        actions = np.expand_dims(data[:, 7], axis=1).astype(np.int64)
        if self.if_total_rtg:
            rtg = np.expand_dims(data[:, 9], axis=1).astype(np.float32)
        else:
            rwd = data[:, 8].astype(np.float32)
            rtg = self._rwd2rtg(rwd)
        return states, actions, rtg, timesteps

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        path, start, end = self.index[idx]
        return self._load_data(path, start, end)

