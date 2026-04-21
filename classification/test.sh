# Define variables
trained_on="all"
trained_category="all"
test_dataset="all"
test_category="all"
perturbed=0
weights="model_epoch8_val0.9995.tar"


python test.py \
    --base_dir /sotossta/DecepTIV \
    --dataset "${test_dataset}"\
    --category "${test_category}" \
    --perturbed "${perturbed}" \
    --frames_sampled 10 \
    --detector_config '/sotossta/DecepTIV/classification/configs/efficientnetb4.yaml' \
    --ckpt_dir "${trained_on}/${trained_category}" \
    --ckpt_weights "${weights}" 
