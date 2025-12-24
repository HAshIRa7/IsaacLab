from isaaclab.envs import ManagerBasedRLEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

import torch

class ManagerBasedIKResetRLEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs) 
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

    def ik_reset(self, goal: torch.Tensor, env_ids: torch.Tensor): 
        '''
        goal - torch.Tensor shape (len(env_ids), 7) 
        env_ids - torch.Tensor shape  (len(env_ids),)
        '''
        