import os
import pickle

model_exp_name_base = '[GR ours recon feat random mask 6_stage2]<2023-10-25_22-26-38>'
model_exp_name_target = '[VOL_GAFL_PAF_RP_BEV_mask 6 1024_stage2]<2025-01-29_15-27-43>'

cfg_pickle_path_base = os.path.join('result', model_exp_name_base, 'cfg.pickle')
with open(cfg_pickle_path_base, 'rb') as f:
    cfg_base = pickle.load(f)

cfg_pickle_path_target = os.path.join('result', model_exp_name_target, 'cfg.pickle')
with open(cfg_pickle_path_target, 'rb') as f:
    cfg_target = pickle.load(f)

print(model_exp_name_base)
print(model_exp_name_target)
for key in dir(cfg_base):
    if key.startswith('__'):
        continue

    if key not in dir(cfg_target):
        print(f'{key} not in target')
        continue

    att_base = getattr(cfg_base, key)
    att_target = getattr(cfg_target, key)
    if att_base != att_target:
        print(key, att_base, att_target)