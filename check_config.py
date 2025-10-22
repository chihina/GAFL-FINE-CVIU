import os
import pickle


model_exp_name_list = []
# model_exp_name = '[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>'
# model_exp_name = '[VOL_GAFL_PAF_JAE_mask 6 1024_stage2]<2025-02-26_11-00-34>'
model_exp_name = '[VOL_GAFL_PAF_JAE_mask 5_stage2]<2024-08-24_22-57-59>'
model_exp_name_list.append(model_exp_name)

for model_exp_name in model_exp_name_list:
    # cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
    cfg_pickle_path = os.path.join('result_base', model_exp_name, 'cfg.pickle')
    with open(cfg_pickle_path, 'rb') as f:
        cfg = pickle.load(f)
    print(model_exp_name)

    for key in dir(cfg):
        print(key, getattr(cfg, key))

    # save cfg
    # cfg.mode = 'PAF'
    # cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
    # with open(cfg_pickle_path, 'wb') as f:
        # pickle.dump(cfg, f)
    # print(f'Saved: {cfg_pickle_path}')