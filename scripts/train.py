import argparse,os,yaml
from datetime import datetime
from torch.utils.data import DataLoader 
from registry import (
    DATASET_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY, 
    SCHEDULER_REGISTRY, CALLBACK_REGISTRY,TRAINER_REGISTRY
)
from src.dataset.util import collate_fn, SingleDatasetBatchSampler, DistributedSingleDatasetBatchSampler
now = datetime.now().strftime("%Y%m%d_%H%M%S9+1320")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/wan2_1_fun_1_3b_control_material_lora_image_train_v0.yaml", help="Path to the main training config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if 'environment_variables' in config and config['environment_variables']:
        print("--- Setting environment variables according to config ---")
        for key, value in config['environment_variables'].items():
            str_value = str(value)
            os.environ[key] = str_value
            print(f"✅ Set: {key} = {str_value}")
        print("------------------------------------")
    print("--- Building training components according to config ---")

    if config['dataset']['name'] == 'ComposedDataset':
        import torch.distributed as dist
        from torch.utils.data import ConcatDataset
        datasets = []
        for dataset_name in config["dataset"]["params"]["datasets"]:
            dataset_class = DATASET_REGISTRY[dataset_name["name"]]
            datasets.append(dataset_class(**dataset_name["params"]))
            print(f"✅ Dataset '{dataset_name}' has been created")
        dataset = ConcatDataset(datasets)
        print(f"✅ Concatenated dataset '{config['dataset']['name']}' has been created")
        batch_sampler = SingleDatasetBatchSampler(
            dataset, 
            batch_size=config['dataloader']['batch_size'], 
            shuffle=config['dataloader']['shuffle'], 
            drop_last=config['dataloader']['drop_last']
        )
        print(f"✅ Using SingleDatasetBatchSampler")
        
        dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler, num_workers=config['dataloader']['num_workers'], pin_memory=config['dataloader']['pin_memory'], persistent_workers=config['dataloader']['persistent_workers'])
        print(f"✅ DataLoader has been created")
    else:
        dataset_class = DATASET_REGISTRY[config['dataset']['name']]
        dataset = dataset_class(**config['dataset']['params'])
        print(f"✅ Dataset '{config['dataset']['name']}' has been created")
        dataloader = DataLoader(dataset, collate_fn=collate_fn, **config['dataloader'])
        print(f"✅ DataLoader has been created")    
 
    val_dataset_class = DATASET_REGISTRY[config['val_dataset']['name']]
    val_dataset = val_dataset_class(**config['val_dataset']['params'])
    print(f"✅ Validation dataset '{config['val_dataset']['name']}' has been created")
    print(f"Validation dataset size: {len(val_dataset)}")
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **config['val_dataloader'])
    model_class = MODEL_REGISTRY[config['model']['name']]
    model = model_class(**config['model']['params'])
    print(f"✅ Model '{config['model']['name']}' has been created")

    optimizer_class = OPTIMIZER_REGISTRY[config['optimizer']['name']]
    
    lr_value = config['optimizer']['params']['lr']
    if isinstance(lr_value, str):
        config['optimizer']['params']['lr'] = float(lr_value)
  
    optimizer = optimizer_class(model.parameters(), **config['optimizer']['params'])
    print(f"✅ Optimizer '{config['optimizer']['name']}' has been created")

    scheduler_class = SCHEDULER_REGISTRY[config['scheduler']['name']]
    eta_min_value = config['scheduler']['params']['eta_min']
    if isinstance(eta_min_value, str):
        config['scheduler']['params']['eta_min'] = float(eta_min_value)
    scheduler_params = config['scheduler']['params']
    scheduler_params['T_max'] = config['trainer_config']['num_epochs'] * len(dataloader)
    scheduler = scheduler_class(optimizer, **scheduler_params)
    print(f"✅ Scheduler '{config['scheduler']['name']}' has been created")

    callbacks = []
    if 'callbacks' in config:
        for cb_config in config['callbacks']:
            cb_class = CALLBACK_REGISTRY[cb_config['name']]
            params = cb_config.get('params', {})
            if 'logging_dir' in params and isinstance(params['logging_dir'], str):
                params['logging_dir'] = params['logging_dir'] + f"_{now}"
            if 'output_path' in params and isinstance(params['output_path'], str):
                params['output_path'] = params['output_path'] + f"_{now}"
            callbacks.append(cb_class(**cb_config.get('params', {})))
    print(f"✅ Callbacks created: {[type(cb).__name__ for cb in callbacks]}")

    print("\n--- Components built, initializing Trainer ---")
    trainer_class = TRAINER_REGISTRY[config['trainer_name']]  
    trainer = trainer_class(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        scheduler=scheduler,
        callbacks=callbacks,
        val_dataloader=val_dataloader,
        **config['trainer_config']
    )

    print("\n--- Start training! ---")
    trainer.train()

if __name__ == "__main__":
    main()