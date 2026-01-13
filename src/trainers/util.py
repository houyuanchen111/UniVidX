"""
这里放极其通用且一般不变的模块和基类，以及Diffusesynth里面一些比较重要，但是目前还没有用到的功能
"""
import torch,argparse,json
from peft import LoraConfig, inject_adapter_in_model
from accelerate import Accelerator
from tqdm import tqdm
from ..pipelines.util import WanVideoPipeline, ModelConfig
from typing import Optional
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

class Callback:
    """Callback的抽象基类"""
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_step_begin(self, trainer): pass
    def on_step_end(self, trainer): pass

class DiffusionTrainingModule(torch.nn.Module):
    # 这个应该可以算一个基类了
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        """
        为模型添加单个LoRA适配器

        Args:
            model: 要添加LoRA的模型
            target_modules: 目标模块列表
            lora_rank: LoRA的秩
            lora_alpha: LoRA的alpha参数，如果为None则等于lora_rank

        Returns:
            添加了LoRA的模型
        """
        if lora_alpha is None:
            lora_alpha = lora_rank # lora_alpha是用于调整lora的效果的参数，这里默认lora_alpha=lora_rank，则训练的时候的归一化系数是1
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules
        )
        model = inject_adapter_in_model(lora_config, model)
        return model

    def add_multiple_loras_to_model(self, model, lora_configs):
        for config in lora_configs:
            lora_config = LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config.get("lora_alpha", config["lora_rank"]),
                target_modules=config["target_modules"].split(",")
            )
            model = inject_adapter_in_model(
                lora_config, 
                model, 
                adapter_name=config["adapter_name"]
                )
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


# class ModelLogger:
#     def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
#         self.output_path = output_path
#         self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
#         self.state_dict_converter = state_dict_converter
        
    
#     def on_step_end(self, loss):
#         pass
    
    
#     def on_epoch_end(self, accelerator, model, epoch_id):
#         accelerator.wait_for_everyone()
#         if accelerator.is_main_process:
#             state_dict = accelerator.get_state_dict(model)
#             state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
#             state_dict = self.state_dict_converter(state_dict)
#             os.makedirs(self.output_path, exist_ok=True)
#             path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
#             accelerator.save(state_dict, path, safe_serialization=True)


# # 启动training和dataprocess的函数，一般的变化是不大的
# def launch_training_task(
#     dataset: torch.utils.data.Dataset,
#     model: DiffusionTrainingModule,
#     model_logger: ModelLogger,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LRScheduler,
#     num_epochs: int = 1,
#     gradient_accumulation_steps: int = 1,
#     batch_size :int = 4,
#     num_workers: int = 8,
#     logger: str = "wandb"
# ):
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              shuffle=True,
#                                              batch_size=batch_size,
#                                              num_workers=num_workers,
#                                              pin_memory=True,
#                                              collate_fn=collate_fn) # collate_fn对于处理dict类的dataset，几乎是必须的
#     accelerator = Accelerator(
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         log_with=logger
#         )
#     model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler) # 14B的模型加载要两分钟左右，且不能加进度条，vram占用43G左右
    
#     for epoch_id in range(num_epochs):
#         for data in tqdm(dataloader):
#             with accelerator.accumulate(model):
#                 optimizer.zero_grad()
#                 loss = model(data)
#                 accelerator.backward(loss)
#                 optimizer.step()
#                 model_logger.on_step_end(loss)
#                 scheduler.step()
#         model_logger.on_epoch_end(accelerator, model, epoch_id) # callback to save model
# def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
#     dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
#     accelerator = Accelerator()
#     model, dataloader = accelerator.prepare(model, dataloader)
#     os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
#     for data_id, data in enumerate(tqdm(dataloader)):
#         with torch.no_grad():
#             inputs = model.forward_preprocess(data)
#             inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
#             torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))

# 下面自己造轮子，搞一个trainer出来

class Trainer:
    def __init__(
            self, 
            model, 
            optimizer, 
            dataloader, 
            scheduler, 
            num_epochs, 
            gradient_accumulation_steps, 
            logger_type=None, 
            callbacks: Optional[list[Callback]] = None,
            val_dataloader=None # 新增验证集dataloader参数
            ):
        # 1. 初始化 Accelerator，logger_type 从外部传入
        # self.accelerator = Accelerator(
        #         gradient_accumulation_steps=gradient_accumulation_steps,
        #         log_with=logger_type
        # )
        if logger_type == "tensorboard":
            for cb in callbacks: # type: ignore
                if hasattr(cb, "logging_dir"):
                    logging_dir = cb.logging_dir # type: ignore
                    break
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type,
                project_dir=logging_dir # 如果用tensorboard，则需要指定project_dir，否则会报错
            )
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type
            )
        # 2. 准备所有组件, 这一步耗时是最长的 1.3B 15s以上
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        self.val_dataloader = val_dataloader  # 保存验证集dataloader
        self.num_epochs = num_epochs
        self.callbacks = callbacks if callbacks is not None else []
        self.global_step = 0
        self.epoch_id = 0
        self.loss = None

    def _call_callbacks(self, event_name):
        """一个统一调用所有回调函数的辅助方法"""
        for callback in self.callbacks:
            getattr(callback, event_name)(self)

    def validate(self):
        if self.val_dataloader is None:
            print("[Warning] No val_dataloader provided, skip validation.")
            return None
        self.model.eval()
        total_depth_loss = 0
        total_normal_loss = 0
        count = 0
        inference_params = {
            "prompt": "Output a video that assigns each 3D location in the world a consistent color.",
            "negative_prompt": "",
            "seed": 42,
            "num_inference_steps": 50,
            "cfg_scale": 5.0,
            "cfg_merge": False,
            "denoising_strength": 1.0,
            "tiled": True,
            "tile_size": [30, 52],
            "tile_stride": [15, 26],
        }

        with torch.no_grad():
            for data in tqdm(self.val_dataloader, disable=not self.accelerator.is_main_process, desc=f"Epoch {self.epoch_id}: Validating"):
                if data is None:
                    continue
                inference_params["input_videos"] = [
                    data["disparities"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype),
                    data["normals"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype),
                ]
                inference_params["control_video"] = data["rgbs"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype) # dataloader的collate_fn会自动加上batch
                inference_params["height"] = data.get("height", 480)
                inference_params["width"] = data.get("width", 832)
                inference_params["num_frames"] = data.get("num_frames", 81)
                inference_params["num_input_videos"] = len(inference_params["input_videos"])
                # 执行推理
                try:
                    video = self.model.pipe(**inference_params)
                    disparities = video["video_1"]
                    normals = video["video_2"]
                    disparity_predict_list = [to_tensor(disparity) for disparity in disparities]
                    normal_predict_list = [to_tensor(normal) for normal in normals]
                    disparity_predict_tensor = torch.stack(disparity_predict_list,dim=1)[None, ...] # 1,3,81,H,W
                    normal_predict_tensor = torch.stack(normal_predict_list,dim=1)[None, ...]
                    disparity_predict_tensor = disparity_predict_tensor.to(self.model.pipe.device).to(self.model.pipe.torch_dtype)*2 -1
                    normal_predict_tensor = normal_predict_tensor.to(self.model.pipe.device).to(self.model.pipe.torch_dtype)*2 -1
                    loss_disparity = F.mse_loss(disparity_predict_tensor, data["disparities"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype))
                    loss_normal = F.mse_loss(normal_predict_tensor, data["normals"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype))
                    total_depth_loss += loss_disparity
                    total_normal_loss += loss_normal
                    if count == 0:
                        self.video = video
                    count += 1
                except Exception as e:
                    print(f"❌ 推理失败: {e}")
                
        avg_depth_loss = total_depth_loss / max(count, 1)
        avg_normal_loss = total_normal_loss / max(count, 1)
        print(f"[Validation] Epoch {self.epoch_id}: val_depth_loss={avg_depth_loss:.6f}, val_normal_loss={avg_normal_loss:.6f}")
        self.model.train()
        return avg_depth_loss, avg_normal_loss

    def train(self):
        self._call_callbacks("on_train_begin")

        for epoch_id in range(self.num_epochs):
            self.epoch_id = epoch_id
            self._call_callbacks("on_epoch_begin")
            
            # 使用accelerator.main_process_<y_bin_401>来包装tqdm，确保只在主进程显示进度条
            pbar = tqdm(self.dataloader, disable=not self.accelerator.is_main_process, desc=f"Epoch {epoch_id+1}/{self.num_epochs}")
            
            for data in pbar:
                self._call_callbacks("on_step_begin")
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    self.loss = self.model(data) # 将loss保存在self中，供callback使用
                    self.accelerator.backward(self.loss)
                    self.optimizer.step()
                    self.scheduler.step()
                
                self.global_step += 1
                if self.accelerator.is_main_process:
                    pbar.set_postfix({"loss": float(self.loss.detach().cpu().item())}) # 在进度条上显示loss
                self._call_callbacks("on_step_end")
            
            # 每个epoch结束后做一次验证
            self.val_depth_loss, self.val_normal_loss = self.validate() # type: ignore
            self._call_callbacks("on_epoch_end")

        self._call_callbacks("on_train_end")

class WanTrainingModule(DiffusionTrainingModule):

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        unit_names: Optional[list[str]]=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        # 检测可用的设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            # 支持单个LoRA（向后兼容）
            if isinstance(lora_base_model, str):
                model = self.add_lora_to_model(
                    getattr(self.pipe, lora_base_model),
                    target_modules=lora_target_modules.split(","),
                    lora_rank=lora_rank
                )
                setattr(self.pipe, lora_base_model, model)
            # 支持多个LoRA（新功能）
            elif isinstance(lora_base_model, list):
                # lora_base_model是模型名称列表，每个模型使用默认配置
                for model_name in lora_base_model:
                    model = self.add_lora_to_model(
                        getattr(self.pipe, model_name),
                        target_modules=lora_target_modules.split(","),
                        lora_rank=lora_rank,
                        adapter_name=f"{model_name}_lora"
                    )
                    setattr(self.pipe, model_name, model)
            elif isinstance(lora_base_model, dict):
                # lora_base_model是配置字典，格式：{"model_name": lora_config}
                # lora_config可以是单个配置字典或配置字典列表
                for model_name, lora_config in lora_base_model.items():
                    base_model = getattr(self.pipe, model_name)

                    if isinstance(lora_config, dict):
                        # 单个配置
                        model = self.add_lora_to_model(
                            base_model,
                            target_modules=lora_config.get("target_modules", lora_target_modules).split(","),
                            lora_rank=lora_config.get("lora_rank", lora_rank),
                            lora_alpha=lora_config.get("lora_alpha"),
                            adapter_name=lora_config.get("adapter_name", f"{model_name}_lora")
                        )
                    elif isinstance(lora_config, list):
                        # 多个配置
                        model = self.add_multiple_loras_to_model(base_model, lora_config)

                    setattr(self.pipe, model_name, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units: # type: ignore
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

# def wan_parser():
#     parser = argparse.ArgumentParser(description="Simple example of a training script.")
#     parser.add_argument("--config", type=str, default="config to control parser")
#     parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
#     # parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
#     parser.add_argument("--batch_size", type=int, default=4,help="batch_size")
#     parser.add_argument("--num_workers", type=int, default=8)
#     parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
#     parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
#     parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
#     parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
#     parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
#     parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
#     parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
#     parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
#     parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
#     parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
#     parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
#     parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
#     parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
#     parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
#     parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
#     parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
#     parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
#     parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
#     parser.add_argument("--video_num_per_scene",type=int, default=200, help="sample num of each scene" )
#     return parser