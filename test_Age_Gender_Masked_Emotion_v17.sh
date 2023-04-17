#----------------v9---------------
python FA_model_Hessian_Axes.py \
  --data_path "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/wiki_retinafacecrop/" \
  --model_name "Resnet50_EAC_eval" \
  --age_head_name "NegMargin" \
  --gender_head_name "BinaryHeadV2" \
  --masked_head_name "BinaryHeadV2" \
  --emotion_head_name "EACHead_eval_hm" \
  --emotion_value_head_name "BinaryHeadV4" \
  --pretrain_model "../Face_Attribute/train_log/checkpoints_Age_Gender_Masked_Emotion_v17/Resnet50_EAC_best_test_value.pt" \
  --feat_dim 2048 \
  --att_size 4 \
  --mean 0.485 0.456 0.406 \
  --std 0.229 0.224 0.225 \
  --emotion_value_class 4 \
  --return_body
