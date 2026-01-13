accelerate launch \
    --config_file "configs/accelerate_config.yaml" \
    "scripts/train.py" \
    --config "configs/univid_intrinsic_train.yaml"     
