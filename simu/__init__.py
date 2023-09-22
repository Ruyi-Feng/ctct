
import numpy as np
import torch
from torch.nn import functional as F
from win32com.client import Dispatch

class Simu:
    def __init__(self, args):
        self.args = args
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.model = self._build_model()

    def _open_xlsm(self) -> None:
        self.VBA_name_1 = self.args.VBA_name_1
        self.VBA_name_2 = self.args.VBA_name_2
        self.xlApp = Dispatch('Excel.Application')
        self.wb = self.xlApp.Workbooks.Open(self.args.xlsm_path)
        self.sheet = self.wb.Worksheets("Duanhui")
        self.SheetDen = self.wb.Worksheets("dtcdensity")
        self.xlApp.Visible = True

    def _init_env(self) -> list:
        self.xlApp.Run(self.VBA_name_1)
        self.xlApp.Run(self.VBA_name_2)
        self.xlApp.Run(self.VBA_name_2)
        demand_GP = (self.sheet.Cells(3, 1).Value - 1.8) / (4.5 - 1.8)
        demand_ranp2 = (self.sheet.Cells(3, 2).Value - 0.48)/(2.01 - 0.48)
        den_up = (self.sheet.Cells(3, 3).Value - 1.86) / (4.45 - 1.86)
        den_down =(self.sheet.Cells(3, 4).Value) / 40.0
        den_rm2 = (self.sheet.Cells(3, 5).Value) / 350
        last_meter = self.sheet.Cells(3, 8).Value / 2.0
        return [demand_GP, demand_ranp2, den_up, den_down, den_rm2, last_meter]

    def _sim(self, a: int) -> tuple(list, float):
        nk = self.sheet.Cells(2, 9).Value + 1
        self.sheet.Cells(nk + 1, 8).Value = a * 0.5
        self.xlApp.Run(VBA_name_2)
        demand_GP = (self.sheet.Cells(nk + 1, 1).Value -1.8) / (4.5 - 1.8)
        demand_ranp2 = (self.sheet.Cells(nk + 1, 2).Value - 0.48) / (2.01 - 0.48)
        den_up = (self.sheet.Cells(nk + 1, 3).Value - 1.86) / (4.45 - 1.86)
        den_down = self.sheet.Cells(nk + 1, 4).Value / 40.0
        den_rm2 = self.sheet.Cells(nk + 1, 5).Value / 350
        last_meter = self.sheet.Cells(nk + 1, 8).Value / 2.0
        s_= [demand_GP, demand_ranp2, den_up, den_down, den_rm2, last_meter]
        r = self.sheet.Cells(nk + 1, 7).Value / 1.04
        return s_ , r

    def _build_model(self):
        model = GPT(self.args).float().to(self.device)
        if os.path.exists(self.args.save_path + 'checkpoint_best.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_best.pth', map_location=torch.device(self.device)))
        elif os.path.exists(self.args.save_path + 'checkpoint_last.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_last.pth', map_location=torch.device(self.device)))
        return model

    def _get_action(self, logits, top_k=None, sample=True) -> int:
        logits = logits[:, -1, :] / 1.0
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix

    def _format(self, s: list, a: list, r: list, t: list) -> tuple:
        bl_sz = self.args.block_size
        if len(s) > self.args.block_size:
            lam = lambda l: l[-bl_sz:]
            i = lam(i) for i in [s, a, r, t]
        s_t = np.array(s)
        s_t = np.expand_dims(s_t, axis=0)
        a_t = np.array(a).reshape(1, -1, 1)
        r_t = np.array(r).reshape(1, -1, 1)
        t_t = np.array(t).reshape(1, -1, 1)
        return s_t, a_t, r_t, t_t

    def _decide_action(self, s: list, a: list, r: list, t: list) -> int:
        x_t, y_t, rtg_t, t_t = self._format(s, a, r, t)
        logits, _ = self.model(x_t, y_t, None, rtg_t, t_t)   # batch, block_size, c_in
        tk = self.args.top_k if self.args.top_k != 0 else None
        return self._get_action(logits, tk, sample=False)

    def _update_historical_seq(self, observation: list, i: int, s: list, a: list, t: list) -> tuple(list, list, list):
        s.append(observation)
        a.append([7])
        t.append([i])
        return s, a, t

    def _update_rtg(self, rtg: list, reward: float) -> list:
        rtg.append([rtg[-1] - reward])
        return rtg

    def _update_a(self, a: list, action: int) -> list:
        a[-1] = action
        return a

    def run(self, aim_rtg=484):
        self._open_xlsm()
        observation = self._init_env()
        rtg = np.array([[aim_rtg]])
        s, a, t = [], [], []
        for i in range(self.args.episode_num):
            s, a, t = self._update_historical_seq(observation, i, s, a, t)
            action = self._decide_action(s, a, rtg, t)
            observation, reward = self._sim(action)
            rtg = self._update_rtg(rtg, reward)
            a = self._update_a(a, action)
        DtcDenSum = self.SheetDen.Cells(1, 29).Value
        name = "TTS" + str(DtcDenSum) + "TTT" + str(self.args.episode_num) + "Episodes"
        excel_path = self.args.save_dir + name + ".xlsm"
        self.wb.SaveAs(excel_path)
        self.wb.Close(True)
        self.xlApp.Quit()
