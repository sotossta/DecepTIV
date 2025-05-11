# Define variables
training_data="all"
training_category="all"

python train.py \
        --base_dir /sotossta/DecepTIV \
        --dataset "${training_data}" \
        --category "${training_category}" \
        --detector_config '/sotossta/DecepTIV/classification/configs/detectors/f3net.yaml' \
        --frames_sampled_real 48 \
        --balanced 1 \
