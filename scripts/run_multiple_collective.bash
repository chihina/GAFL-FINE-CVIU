IFS_BACKUP=$IFS
IFS=$'\n'

recon_loss_type_array=(
  'all'
  # 'key'
)

anchor_agg_mode_array=(
  # 'max'
  'mean'
)

use_individual_action_type_array=(
  'gt_action'
  # 'gt_grouping'
  # 'det'
)

feature_adapt_type_array=(
  'ft'
  # 'line'
  # 'mlp'
)

model_exp_name_stage2_array=(
  '[CAD GA ours recon rand mask 6_stage2]<2023-10-26_16-18-47>'
  # '[CAD GR ours recon rand mask 0_stage2]<2023-10-19_22-26-11>'
)

feature_adapt_dim_array=(
  2048
  # 1024
  # 512
  # 256
  # 128
)

freeze_backbone_stage4_array=(
  'False'
  # 'True'
)

use_proximity_loss_array=(
  'False'
  # 'True'
)

use_query_classfication_loss_array=(
  'False'
  # 'True'
)

use_ga_recog_loss_array=(
  'False'
  # 'True'
)

use_ia_recog_loss_array=(
  'False'
  # 'True'
)

use_random_mask_array=(
  'False'
  # 'True'
)

use_recon_loss_array=(
  'False'
  # 'True'
)

use_disentangle_loss_array=(
  'False'
  # 'True'
)

key_recog_feat_type_array=(
  'gaf'
  # 'iaf'
)

key_person_mode_array=(
  'mask_one'
  # 'mask_zero'
)

use_maintain_loss_array=(
  # 'False'
  'True'
)

use_metric_lr_loss_array=(
  # 'False'
  'True'
)

metric_lr_margin_array=(
  # '1'
  '10'
)

use_anchor_type_array=(
  'normal'
  # 'pruning_gt'
  # 'pruning_p2g'
  # 'pruning_p2g_cos'
  # 'pruning_p2g_inner_cos'
)

all_pruning_ratio_array=(
  # '0.30'
  # '0.66'
  # '0.68'
  '0.70'
  # '0.72'
)

use_key_person_type_array=(
  # 'gt'
  'det_semi'
  # 'det'
)

use_key_person_loss_func_array=(
  # 'wbce'
  # 'bce'
  'mse'
)

use_key_recog_loss_array=(
  'False'
  # 'True'
)

query_init_array=(
  # '1'
  # '2'
  '3'
  # '5'
)

non_query_init_array=(
  '5'
  # 'all'
)

seed_num_start=99
seed_num_end=99

anchor_thresh_type_array=(
  # 'ratio'
  'val'
)

sampling_mode_array=(
  # 'near'
  'rand'
  # 'coreset_gaf'
)

train_smp_type_array=(
  'each_4'
  # 'each_pert5e-0_mk2_2'
  # 'each_pert5e-0_mk2_4'
  # 'each_pert5e-0_mk2_6'
  # 'each_pert2e-0_mk2_4'
  # 'each_pert8e-0_mk2_4'
)

query_array=(
    # 'Moving'
    # 'Waiting'
    # 'Queueing'
    'Talking'
)

train_learning_rate_array=(
  '1e-4'
  # '1e-5'
  # '1e-6'
)

key_person_det_interval_array=(
  '1000'
  # '1'
)

load_backbone_stage4_array=(
  # 'False'
  'True'
)

gpu='7'
max_epoch='50'
# max_epoch='100'

for seed_num in $(seq $seed_num_start $seed_num_end); do
  for model_exp_name_stage2 in ${model_exp_name_stage2_array[@]}; do
    for query in ${query_array[@]}; do
      for use_key_recog_loss in ${use_key_recog_loss_array[@]}; do
        for use_key_person_loss_func in ${use_key_person_loss_func_array[@]}; do
          for use_key_person_type in ${use_key_person_type_array[@]}; do
            for use_individual_action_type in ${use_individual_action_type_array[@]}; do
              for all_pruning_ratio in ${all_pruning_ratio_array[@]}; do
                for query_init in ${query_init_array[@]}; do
                  for non_query_init in ${non_query_init_array[@]}; do
                    for use_recon_loss in ${use_recon_loss_array[@]}; do
                      for recon_loss_type in ${recon_loss_type_array[@]}; do
                        for feature_adapt_type in ${feature_adapt_type_array[@]}; do
                          for freeze_backbone_stage4 in ${freeze_backbone_stage4_array[@]}; do
                            for use_disentangle_loss in ${use_disentangle_loss_array[@]}; do
                              for train_smp_type in ${train_smp_type_array[@]}; do
                                for train_learning_rate in ${train_learning_rate_array[@]}; do
                                  for key_person_det_interval in ${key_person_det_interval_array[@]}; do
                                    for anchor_agg_mode in ${anchor_agg_mode_array[@]}; do
                                      for key_recog_feat_type in ${key_recog_feat_type_array[@]}; do
                                        for use_random_mask in ${use_random_mask_array[@]}; do
                                          for use_proximity_loss in ${use_proximity_loss_array[@]}; do
                                            for feature_adapt_dim in ${feature_adapt_dim_array[@]}; do
                                              for use_query_classfication_loss in ${use_query_classfication_loss_array[@]}; do
                                                for use_anchor_type in ${use_anchor_type_array[@]}; do
                                                  for use_ga_recog_loss in ${use_ga_recog_loss_array[@]}; do
                                                    for use_ia_recog_loss in ${use_ia_recog_loss_array[@]}; do
                                                      for use_maintain_loss in ${use_maintain_loss_array[@]}; do
                                                        for use_metric_lr_loss in ${use_metric_lr_loss_array[@]}; do
                                                          for key_person_mode in ${key_person_mode_array[@]}; do
                                                            for anchor_thresh_type in ${anchor_thresh_type_array[@]}; do
                                                              for sampling_mode in ${sampling_mode_array[@]}; do
                                                                for metric_lr_margin in ${metric_lr_margin_array[@]}; do
                                                                  for load_backbone_stage4 in ${load_backbone_stage4_array[@]}; do
                                                                    echo $query $use_key_person_type $all_pruning_ratio $model_exp_name_stage2 $use_key_person_loss_func $use_individual_action_type $query_init \
                                                                    $non_query_init $use_recon_loss $feature_adapt_type $freeze_backbone_stage4 $use_key_recog_loss $recon_loss_type $use_disentangle_loss $seed_num \
                                                                    $sampling_mode $train_learning_rate $key_person_det_interval $anchor_agg_mode $key_recog_feat_type $use_random_mask $use_proximity_loss $feature_adapt_dim \
                                                                    $use_query_classfication_loss $use_anchor_type $use_ga_recog_loss $use_ia_recog_loss $use_maintain_loss $use_metric_lr_loss \
                                                                    $key_person_mode $anchor_thresh_type $train_smp_type $metric_lr_margin $load_backbone_stage4
                                                                    python scripts/train_eval_collective_stage4.py -gpu $gpu -query $query -max_epoch $max_epoch \
                                                                    -use_key_person_type $use_key_person_type -all_pruning_ratio $all_pruning_ratio -model_exp_name_stage2 $model_exp_name_stage2 \
                                                                    -use_key_person_loss_func $use_key_person_loss_func -use_individual_action_type $use_individual_action_type \
                                                                    -query_init $query_init -non_query_init $non_query_init -use_recon_loss $use_recon_loss -feature_adapt_type $feature_adapt_type \
                                                                    -freeze_backbone_stage4 $freeze_backbone_stage4 -use_key_recog_loss $use_key_recog_loss -recon_loss_type $recon_loss_type \
                                                                    -use_disentangle_loss $use_disentangle_loss -seed_num $seed_num -sampling_mode $sampling_mode -train_learning_rate $train_learning_rate \
                                                                    -key_person_det_interval $key_person_det_interval -anchor_agg_mode $anchor_agg_mode -key_recog_feat_type $key_recog_feat_type \
                                                                    -use_random_mask $use_random_mask -use_proximity_loss $use_proximity_loss -feature_adapt_dim $feature_adapt_dim \
                                                                    -use_query_classfication_loss $use_query_classfication_loss -use_anchor_type $use_anchor_type -use_ga_recog_loss $use_ga_recog_loss \
                                                                    -use_ia_recog_loss $use_ia_recog_loss -use_maintain_loss $use_maintain_loss -use_metric_lr_loss $use_metric_lr_loss \
                                                                    -key_person_mode $key_person_mode -anchor_thresh_type $anchor_thresh_type -train_smp_type $train_smp_type \
                                                                    -metric_lr_margin $metric_lr_margin -load_backbone_stage4 $load_backbone_stage4
                                                                  done
                                                                done
                                                              done
                                                            done
                                                          done
                                                        done
                                                      done
                                                    done
                                                  done
                                                done
                                              done
                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done