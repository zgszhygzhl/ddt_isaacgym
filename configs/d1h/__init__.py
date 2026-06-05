from utils.task_registry import task_registry
from .d1h_flat_config import *
task_registry.register("d1h_flat",D1HFlat,D1HFlatCfg(),D1HFlatCfgPPO())
task_registry.register("d1h_flat_play",D1HFlat,D1HFlatCfg_Play(),D1HFlatCfgPPO())

from .d1h_base_config import *
task_registry.register("d1h_moe_base", D1HMoEBase, D1HMoEBaseCfg(), D1HMoEBaseCfgPPO())
task_registry.register("d1h_moe_base_play", D1HMoEBase, D1HMoEBaseCfg_Play(), D1HMoEBaseCfgPPO())

from .d1h_disc_config import *
task_registry.register("d1h_moe_disc", D1HMoEBase, D1HMoEDiscCfg(), D1HMoEDiscCfgPPO())

from .d1h_surf_config import *
task_registry.register("d1h_moe_surf", D1HMoEBase, D1HMoESurfCfg(), D1HMoESurfCfgPPO())

from .d1h_prec_config import *
task_registry.register("d1h_moe_prec", D1HMoEBase, D1HMoEPrecCfg(), D1HMoEPrecCfgPPO())

from .d1h_rec_config import *
task_registry.register("d1h_moe_rec", D1HMoERecovery, D1HMoERecCfg(), D1HMoERecCfgPPO())
