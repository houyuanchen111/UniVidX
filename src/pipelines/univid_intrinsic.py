from numpy import vdot
import torch, types, copy,json,random
from typing import Optional, Union
from einops import reduce
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from ..trainers.util import DiffusionTrainingModule
from .util import BasePipeline, PipelineUnit, PipelineUnitRunner, ModelConfig, TeaCache, TemporalTiler_BCTHW
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit_intrinsic import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample 
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
from PIL import Image

class WanVideoPipeline(BasePipeline):

    def __init__(self, 
                 device="cuda", 
                 torch_dtype=torch.bfloat16, 
                 tokenizer_path=None,
                #  unit_names: list[str] = None
                 ):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None # type: ignore
        self.image_encoder: WanImageEncoder = None # type: ignore
        self.dit: WanModel = None # type: ignore
        self.vae: WanVideoVAE = None # type: ignore
        self.motion_controller: WanMotionControllerModel = None # type: ignore
        self.vace: VaceWanModel = None # type: ignore
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]
       
        self.model_fn = model_fn_wan_video
        
    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device) # type: ignore
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device) # type: ignore
        loader.load(module, lora, alpha=alpha)

    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep_weight = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        training_mode = inputs["training_mode"]
        if training_mode == "t2RAIN":
                timestep = [
                    timestep_weight,
                    timestep_weight,
                    timestep_weight,
                    timestep_weight,
                ]
        elif training_mode == "R2AIN":
            timestep = [
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                timestep_weight,
                timestep_weight,
            ]
        elif training_mode == "A2RIN":
            timestep = [
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                timestep_weight,
            ]
        elif training_mode == "I2RAN":
            timestep = [
                timestep_weight,
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
            ]
        elif training_mode == "N2RAI":
            timestep = [
                timestep_weight,
                timestep_weight,
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
            ]
        elif training_mode == "RA2IN":
            timestep = [
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                timestep_weight,
            ]
        elif training_mode == "RI2AN":
            timestep = [
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
            ]
        elif training_mode == "RN2AI":
            timestep = [
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
            ]
        elif training_mode == "AI2RN":
            timestep = [
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
            ]
        elif training_mode == "AN2RI":
            timestep = [
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
            ]
        elif training_mode == "IN2RA":
            timestep = [
                timestep_weight,
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
            ]
        elif training_mode == "AIN2R":
            timestep = [
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
            ]
        elif training_mode == "RIN2A":
            timestep = [
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
            ]
        elif training_mode == "RAN2I":
            timestep = [
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
            ]
        elif training_mode == "RAI2N":
            timestep = [
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                timestep_weight,
            ]
        inputs["latents"], sigmas = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        noise_pred = self.model_fn(**inputs, timestep=timestep) # model_fn are used to predict noise
        if training_mode == "t2RAIN":
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float()) 
        elif training_mode == "R2AIN":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[1,2,3]], training_target.float()[[1,2,3]])
        elif training_mode == "A2RIN":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0,2,3]], training_target.float()[[0,2,3]])
        elif training_mode == "I2RAN":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0,1,3]], training_target.float()[[0,1,3]])
        elif training_mode == "N2RAI":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0,1,2]], training_target.float()[[0,1,2]])
        elif training_mode == "N2RAI":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0,1,2]], training_target.float()[[0,1,2]])
        elif training_mode == "RA2IN":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[2,3]], training_target.float()[[2,3]])
        elif training_mode == "RI2AN":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[1,3]], training_target.float()[[1,3]])
        elif training_mode == "RN2AI":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[1,2]], training_target.float()[[1,2]])
        elif training_mode == "AI2RN":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0,3]], training_target.float()[[0,3]])
        elif training_mode == "AN2RI":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0,2]], training_target.float()[[0,2]])
        elif training_mode == "IN2RA":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0,1]], training_target.float()[[0,1]])
        elif training_mode == "AIN2R":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[0]], training_target.float()[[0]])
        elif training_mode == "RIN2A":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[1]], training_target.float()[[1]])
        elif training_mode == "RAN2I":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[2]], training_target.float()[[2]])
        elif training_mode == "RAI2N":
            loss = torch.nn.functional.mse_loss(noise_pred.float()[[3]], training_target.float()[[3]])
        loss = loss * self.scheduler.training_weight(timestep_weight)
        return loss
    
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
                        
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn) # type: ignore
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    def vae_normal_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean") # H W C
        normal_unit = vae_output / (torch.norm(vae_output, dim=-1, keepdim=True) + 1e-6) # [-1,1] unit
        return normal_unit.permute(2,0,1) # C H W

    def vae_normal_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video_normal_unit = []
        for image in vae_output:
            normal_unit= self.vae_normal_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value)
            video_normal_unit.append(normal_unit)
        return torch.stack(video_normal_unit, dim=1)
    
    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean") # H W C
        return vae_output.permute(2,0,1) # C H W

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video_output = []
        for image in vae_output:
            image = self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value)
            video_output.append(image)
        return torch.stack(video_output, dim=1) # C T H W

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="google/*"),
        local_model_path: str = "./models",
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-14B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-14B",
                # "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype) # type: ignore
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download, use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder") # type: ignore
        config_path = "configs/wan2_1_14b_t2v_dit_config.json"
        with open(config_path, "r") as f:
            dit_kwargs = json.load(f)
        pipe.dit = WanModel(**dit_kwargs)
        pipe.dit.to(dtype=torch_dtype, device=device)
        state_dict = {}
        for i in range(1,7):
            ckpt_path = f"models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
            state_dict.update(load_file(ckpt_path))
        incompatible = pipe.dit.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
        loaded_keys = [k for k in state_dict.keys() if k not in unexpected_keys]
        print(f"Loaded keys ({len(loaded_keys)}):")
        for k in loaded_keys:
            print(f"- {k}")
        print(f"Missing keys ({len(missing_keys)}):")
        for k in missing_keys:
            print(f"- {k}")
        print(f"Unexpected keys ({len(unexpected_keys)}):")
        for k in unexpected_keys:
            print(f"- {k}")
        pipe.vae = model_manager.fetch_model("wan_video_vae") # type: ignore
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder") # type: ignore
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller") # type: ignore
        pipe.vace = model_manager.fetch_model("wan_video_vace") # type: ignore

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(local_model_path, skip_download=skip_download)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        **kwargs 
    ):
        inputs_shared = kwargs.copy()
        num_inference_steps = inputs_shared.pop("num_inference_steps", 50)
        denoising_strength = inputs_shared.get("denoising_strength", 1.0)
        sigma_shift = inputs_shared.get("sigma_shift", 5.0)
        prompt = inputs_shared.pop("prompt", "")
        tea_cache_l1_thresh = inputs_shared.pop("tea_cache_l1_thresh", None)
        tea_cache_model_id = inputs_shared.pop("tea_cache_model_id", "")
        negative_prompt = inputs_shared.pop("negative_prompt", "")
        vace_reference_image = inputs_shared.get("vace_reference_image", None)
        cfg_scale = inputs_shared.get("cfg_scale", 5.0)
        cfg_merge = inputs_shared.get("cfg_merge", False)
        tiled = inputs_shared.get("tiled", True)
        tile_size = inputs_shared.get("tile_size", (30,52))
        tile_stride = inputs_shared.get("tile_stride", (15,26))
        # Scheduler, here is the flow matching scheduler
        scheduler_copy = copy.deepcopy(self.scheduler)
        scheduler_copy.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        training_mode = inputs_shared.get("training_mode")
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        

        for unit in self.units: # type: ignore
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega) # type: ignore

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        B, C, T, H, W = inputs_shared["latents"].shape
        noise_0 = inputs_shared["latents"][ [0] ]
        if training_mode == "R2AIN":
            inputs_shared["latents"][0], sigmas = self.scheduler.add_noise(inputs_shared["inference_rgb_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "A2RIN":
            inputs_shared["latents"][1], sigmas = self.scheduler.add_noise(inputs_shared["inference_albedo_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "I2RAN":
            inputs_shared["latents"][2], sigmas = self.scheduler.add_noise(inputs_shared["inference_irradiance_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "N2RAI":
            inputs_shared["latents"][3], sigmas = self.scheduler.add_noise(inputs_shared["inference_normal_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "RA2IN":
            inputs_shared["latents"][0], sigmas = self.scheduler.add_noise(inputs_shared["inference_rgb_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][1], sigmas = self.scheduler.add_noise(inputs_shared["inference_albedo_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "RI2AN":
            inputs_shared["latents"][0], sigmas = self.scheduler.add_noise(inputs_shared["inference_rgb_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][2], sigmas = self.scheduler.add_noise(inputs_shared["inference_irradiance_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "RN2AI":
            inputs_shared["latents"][0], sigmas = self.scheduler.add_noise(inputs_shared["inference_rgb_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][3], sigmas = self.scheduler.add_noise(inputs_shared["inference_normal_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "AI2RN":
            inputs_shared["latents"][1], sigmas = self.scheduler.add_noise(inputs_shared["inference_albedo_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][2], sigmas = self.scheduler.add_noise(inputs_shared["inference_irradiance_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "AN2RI":
            inputs_shared["latents"][1], sigmas = self.scheduler.add_noise(inputs_shared["inference_albedo_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][3], sigmas = self.scheduler.add_noise(inputs_shared["inference_normal_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "IN2RA":
            inputs_shared["latents"][2], sigmas = self.scheduler.add_noise(inputs_shared["inference_irradiance_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][3], sigmas = self.scheduler.add_noise(inputs_shared["inference_normal_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "AIN2R":
            inputs_shared["latents"][1], sigmas = self.scheduler.add_noise(inputs_shared["inference_albedo_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][2], sigmas = self.scheduler.add_noise(inputs_shared["inference_irradiance_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][3], sigmas = self.scheduler.add_noise(inputs_shared["inference_normal_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "RIN2A":
            inputs_shared["latents"][0], sigmas = self.scheduler.add_noise(inputs_shared["inference_rgb_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][2], sigmas = self.scheduler.add_noise(inputs_shared["inference_irradiance_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][3], sigmas = self.scheduler.add_noise(inputs_shared["inference_normal_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "RAN2I":
            inputs_shared["latents"][0], sigmas = self.scheduler.add_noise(inputs_shared["inference_rgb_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][1], sigmas = self.scheduler.add_noise(inputs_shared["inference_albedo_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][3], sigmas = self.scheduler.add_noise(inputs_shared["inference_normal_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
        elif training_mode == "RAI2N":
            inputs_shared["latents"][0], sigmas = self.scheduler.add_noise(inputs_shared["inference_rgb_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][1], sigmas = self.scheduler.add_noise(inputs_shared["inference_albedo_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))
            inputs_shared["latents"][2], sigmas = self.scheduler.add_noise(inputs_shared["inference_irradiance_latents"], noise_0, self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device))

        for progress_id, timestep in enumerate(tqdm(scheduler_copy.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            if training_mode == "t2RAIN":
                timestep = [
                    timestep,
                    timestep,
                    timestep,
                    timestep,
                ]
            elif training_mode == "R2AIN":
                timestep = [
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    timestep,
                    timestep,
                ]
            elif training_mode == "A2RIN":
                timestep = [
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    timestep,
                ]
            elif training_mode == "I2RAN":
                timestep = [
                    timestep,
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                ]
            elif training_mode == "N2RAI":
                timestep = [
                    timestep,
                    timestep,
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                ]
            elif training_mode == "RA2IN":
                timestep = [
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    timestep,
                ]
            elif training_mode == "RI2AN":
                timestep = [
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                ]
            elif training_mode == "RN2AI":
                timestep = [
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                ]
            elif training_mode == "AI2RN":
                timestep = [
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                ]
            elif training_mode == "AN2RI":
                timestep = [
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                ]
            elif training_mode == "IN2RA":
                timestep = [
                    timestep,
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                ]
            elif training_mode == "AIN2R":
                timestep = [
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                ]
            elif training_mode == "RIN2A":
                timestep = [
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                ]
            elif training_mode == "RAN2I":
                timestep = [
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                ]
            elif training_mode == "RAI2N":
                timestep = [
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    self.scheduler.timesteps[torch.tensor([999])].to(dtype=self.torch_dtype, device=self.device),
                    timestep,
                ]
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep) 
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega) 
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            if training_mode == "t2RAIN":
                inputs_shared["latents"] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"]) 
            elif training_mode == "R2AIN":
                inputs_shared["latents"][[1,2,3]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[1,2,3]] 
            elif training_mode == "A2RIN":
                inputs_shared["latents"][[0,2,3]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[0,2,3]] 
            elif training_mode == "I2RAN":
                inputs_shared["latents"][[0,1,3]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[0,1,3]] 
            elif training_mode == "N2RAI":
                inputs_shared["latents"][[0,1,2]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[0,1,2]]
            elif training_mode == "RA2IN":
                inputs_shared["latents"][[2,3]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[2,3]] 
            elif training_mode == "RI2AN":
                inputs_shared["latents"][[1,3]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[1,3]] 
            elif training_mode == "RN2AI":
                inputs_shared["latents"][[1,2]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[1,2]] 
            elif training_mode == "AI2RN":
                inputs_shared["latents"][[0,3]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[0,3]] 
            elif training_mode == "AN2RI":
                inputs_shared["latents"][[0,2]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[0,2]] 
            elif training_mode == "IN2RA":
                inputs_shared["latents"][[0,1]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[0,1]] 
            elif training_mode == "AIN2R":
                inputs_shared["latents"][[0]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[0]]   
            elif training_mode == "RIN2A":
                inputs_shared["latents"][[1]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[1]] 
            elif training_mode == "RAN2I":
                inputs_shared["latents"][[2]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[2]] 
            elif training_mode == "RAI2N":
                inputs_shared["latents"][[3]] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"])[[3]] 

       
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        video_dict = {}
        if training_mode == "t2RAIN":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]

            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]

            video_irradiance = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_irradiance = self.vae_output_to_video(video_irradiance, pattern="B C T H W")
            video_dict["irradiance"] = video_irradiance # torch.tensor[c,t,h,w]

            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        elif training_mode == "R2AIN":
            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]

            video_irradiance = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_irradiance = self.vae_output_to_video(video_irradiance, pattern="B C T H W")
            video_dict["irradiance"] = video_irradiance # torch.tensor[c,t,h,w]

            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        elif training_mode == "A2RIN":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]

            video_albedo = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["irradiance"] = video_albedo # torch.tensor[c,t,h,w]

            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        elif training_mode == "I2RAN":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]

            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]

            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        elif training_mode == "N2RAI":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]

            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]

            video_irradiance = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_irradiance = self.vae_output_to_video(video_irradiance, pattern="B C T H W")
            video_dict["irradiance"] = video_irradiance # torch.tensor[c,t,h,w]
        elif training_mode == "RA2IN":
            video_irradiance = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_irradiance = self.vae_output_to_video(video_irradiance, pattern="B C T H W")
            video_dict["irradiance"] = video_irradiance # torch.tensor[c,t,h,w]

            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]

        elif training_mode == "RI2AN":
            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]

            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]

        elif training_mode == "RN2AI":
            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]

            video_irradiance = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_irradiance = self.vae_output_to_video(video_irradiance, pattern="B C T H W")
            video_dict["irradiance"] = video_irradiance # torch.tensor[c,t,h,w]

        
        elif training_mode == "AI2RN":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]

            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        
        elif training_mode == "AN2RI":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]

            video_irradiance = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_irradiance = self.vae_output_to_video(video_irradiance, pattern="B C T H W")
            video_dict["irradiance"] = video_irradiance # torch.tensor[c,t,h,w]
        
        elif training_mode == "IN2RA":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]

            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]
        
        elif training_mode == "AIN2R":
            video_rgb = self.vae.decode(inputs_shared["latents"][[0]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb, pattern="B C T H W")
            video_dict["rgb"] = video_rgb # torch.tensor[c,t,h,w]
        
        elif training_mode == "RIN2A":
            video_albedo = self.vae.decode(inputs_shared["latents"][[1]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_albedo = self.vae_output_to_video(video_albedo, pattern="B C T H W")
            video_dict["albedo"] = video_albedo # torch.tensor[c,t,h,w]
        
        elif training_mode == "RAN2I":
            video_irradiance = self.vae.decode(inputs_shared["latents"][[2]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_irradiance = self.vae_output_to_video(video_irradiance, pattern="B C T H W")
            video_dict["irradiance"] = video_irradiance # torch.tensor[c,t,h,w]
        
        elif training_mode == "RAI2N":
            video_normal = self.vae.decode(inputs_shared["latents"][[3]], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= self.vae_normal_output_to_video(video_normal)
            video_dict["normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        


 
        return video_dict

class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        if isinstance(height, torch.Tensor):
            height = height[0].item()
        if isinstance(width, torch.Tensor):
            width = width[0].item()
        if isinstance(num_frames, torch.Tensor):
            num_frames = num_frames[0].item()
        result = pipe.check_resize_height_width(height, width, num_frames)
        if len(result) == 2:
            height, width = result
        else:
            height, width, num_frames = result
        return {"height": height, "width": width, "num_frames": num_frames}

class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1 
        noise = pipe.generate_noise((4, 16, length, height//8, width//8), seed=seed, rand_device=rand_device) 
        return {"noise": noise}

class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
   
    def __init__(self):
        super().__init__(
            input_params=("input_videos", "noise", "tiled", "tile_size", "tile_stride", "is_inference", "modality_index", "inference_rgb", "inference_albedo", "inference_irradiance", "inference_normal", "height", "width", "training_mode"),
            onload_model_names=("vae",)
        )
    def process(self, pipe: WanVideoPipeline, input_videos, noise, tiled, tile_size, tile_stride, is_inference, modality_index, inference_rgb, inference_albedo, inference_irradiance, inference_normal, height, width, training_mode):
        if is_inference:
       
            if training_mode == "t2RAIN":
                return {"latents": noise}
            elif training_mode == "R2AIN":
                if inference_rgb is not None and isinstance(inference_rgb, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_image(inference_rgb.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_rgb_latents": inference_rgb_latents}
                elif inference_rgb is not None and isinstance(inference_rgb, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_video(inference_rgb)
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_rgb_latents": inference_rgb_latents}
            elif training_mode == "A2RIN":
                if inference_albedo is not None and isinstance(inference_albedo, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_image(inference_albedo.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_albedo_latents": inference_albedo_latents}
                elif inference_albedo is not None and isinstance(inference_albedo, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_video(inference_albedo)
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_albedo_latents": inference_albedo_latents}
            elif training_mode == "I2RAN":
                if inference_irradiance is not None and isinstance(inference_irradiance, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_image(inference_irradiance.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_irradiance_latents": inference_irradiance_latents}
                elif inference_irradiance is not None and isinstance(inference_irradiance, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_video(inference_irradiance)
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_irradiance_latents": inference_irradiance_latents}
            elif training_mode == "N2RAI":
                if inference_normal is not None and isinstance(inference_normal, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_image(inference_normal.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_normal_latents": inference_normal_latents}
                elif inference_normal is not None and isinstance(inference_normal, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_video(inference_normal)
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    return {"latents": noise, "inference_normal_latents": inference_normal_latents}
            elif training_mode == "RA2IN":
                if inference_rgb is not None and isinstance(inference_rgb, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_image(inference_rgb.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_rgb is not None and isinstance(inference_rgb, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_video(inference_rgb)
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_albedo is not None and isinstance(inference_albedo, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_image(inference_albedo.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_albedo is not None and isinstance(inference_albedo, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_video(inference_albedo)
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_rgb_latents": inference_rgb_latents, "inference_albedo_latents": inference_albedo_latents}
            elif training_mode == "RI2AN":
                if inference_rgb is not None and isinstance(inference_rgb, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_image(inference_rgb.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_rgb is not None and isinstance(inference_rgb, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_video(inference_rgb)
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_irradiance is not None and isinstance(inference_irradiance, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_image(inference_irradiance.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_irradiance is not None and isinstance(inference_irradiance, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_video(inference_irradiance)
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_rgb_latents": inference_rgb_latents, "inference_irradiance_latents": inference_irradiance_latents}
   
            elif training_mode == "RN2AI":
                if inference_rgb is not None and isinstance(inference_rgb, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_image(inference_rgb.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_rgb is not None and isinstance(inference_rgb, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_video(inference_rgb)
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_normal is not None and isinstance(inference_normal, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_image(inference_normal.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_normal is not None and isinstance(inference_normal, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_video(inference_normal)
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_rgb_latents": inference_rgb_latents, "inference_normal_latents": inference_normal_latents}

            elif training_mode == "AI2RN":
                if inference_albedo is not None and isinstance(inference_albedo, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_image(inference_albedo.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_albedo is not None and isinstance(inference_albedo, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_video(inference_albedo)
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_irradiance is not None and isinstance(inference_irradiance, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_image(inference_irradiance.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_irradiance is not None and isinstance(inference_irradiance, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_video(inference_irradiance)
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_albedo_latents": inference_albedo_latents, "inference_irradiance_latents": inference_irradiance_latents}
            # AN2RI: Albedo, Normal to RGB, Irradiance - 需要 Albedo 和 Normal 条件
            elif training_mode == "AN2RI":
                if inference_albedo is not None and isinstance(inference_albedo, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_image(inference_albedo.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_albedo is not None and isinstance(inference_albedo, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_video(inference_albedo)
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_normal is not None and isinstance(inference_normal, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_image(inference_normal.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_normal is not None and isinstance(inference_normal, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_video(inference_normal)
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_albedo_latents": inference_albedo_latents, "inference_normal_latents": inference_normal_latents}

            elif training_mode == "IN2RA":
                if inference_irradiance is not None and isinstance(inference_irradiance, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_image(inference_irradiance.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_irradiance is not None and isinstance(inference_irradiance, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_video(inference_irradiance)
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_normal is not None and isinstance(inference_normal, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_image(inference_normal.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_normal is not None and isinstance(inference_normal, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_video(inference_normal)
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_irradiance_latents": inference_irradiance_latents, "inference_normal_latents": inference_normal_latents}
   
            elif training_mode == "AIN2R":
                if inference_albedo is not None and isinstance(inference_albedo, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_image(inference_albedo.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_albedo is not None and isinstance(inference_albedo, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_video(inference_albedo)
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_irradiance is not None and isinstance(inference_irradiance, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_image(inference_irradiance.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_irradiance is not None and isinstance(inference_irradiance, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_video(inference_irradiance)
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_normal is not None and isinstance(inference_normal, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_image(inference_normal.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_normal is not None and isinstance(inference_normal, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_video(inference_normal)
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_albedo_latents": inference_albedo_latents, "inference_irradiance_latents": inference_irradiance_latents, "inference_normal_latents": inference_normal_latents}

            elif training_mode == "RIN2A":
                if inference_rgb is not None and isinstance(inference_rgb, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_image(inference_rgb.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_rgb is not None and isinstance(inference_rgb, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_video(inference_rgb)
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_irradiance is not None and isinstance(inference_irradiance, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_image(inference_irradiance.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_irradiance is not None and isinstance(inference_irradiance, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_video(inference_irradiance)
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_normal is not None and isinstance(inference_normal, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_image(inference_normal.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_normal is not None and isinstance(inference_normal, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_video(inference_normal)
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_rgb_latents": inference_rgb_latents, "inference_irradiance_latents": inference_irradiance_latents, "inference_normal_latents": inference_normal_latents}

            elif training_mode == "RAN2I":
                if inference_rgb is not None and isinstance(inference_rgb, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_image(inference_rgb.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_rgb is not None and isinstance(inference_rgb, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_video(inference_rgb)
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_albedo is not None and isinstance(inference_albedo, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_image(inference_albedo.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_albedo is not None and isinstance(inference_albedo, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_video(inference_albedo)
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_normal is not None and isinstance(inference_normal, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_image(inference_normal.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_normal is not None and isinstance(inference_normal, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_normal = pipe.preprocess_video(inference_normal)
                    inference_normal_latents = pipe.vae.encode(inference_normal, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_rgb_latents": inference_rgb_latents, "inference_albedo_latents": inference_albedo_latents, "inference_normal_latents": inference_normal_latents}

            elif training_mode == "RAI2N":
                if inference_rgb is not None and isinstance(inference_rgb, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_image(inference_rgb.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_rgb is not None and isinstance(inference_rgb, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_rgb = pipe.preprocess_video(inference_rgb)
                    inference_rgb_latents = pipe.vae.encode(inference_rgb, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_albedo is not None and isinstance(inference_albedo, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_image(inference_albedo.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_albedo is not None and isinstance(inference_albedo, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_albedo = pipe.preprocess_video(inference_albedo)
                    inference_albedo_latents = pipe.vae.encode(inference_albedo, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                if inference_irradiance is not None and isinstance(inference_irradiance, Image.Image):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_image(inference_irradiance.resize((width, height))).to(pipe.device).to(pipe.torch_dtype)[:,:,None]
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                elif inference_irradiance is not None and isinstance(inference_irradiance, torch.Tensor):
                    pipe.load_models_to_device(["vae"])
                    inference_irradiance = pipe.preprocess_video(inference_irradiance)
                    inference_irradiance_latents = pipe.vae.encode(inference_irradiance, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": noise, "inference_rgb_latents": inference_rgb_latents, "inference_albedo_latents": inference_albedo_latents, "inference_irradiance_latents": inference_irradiance_latents}
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
    
        video = input_videos[0]
        video = pipe.preprocess_video(video)
        input_latents = pipe.vae.encode(video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        assert input_latents.shape[1] == noise.shape[1], "input_latents.shape[1] != noise.shape[1]"
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}

class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True, # 
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=str(pipe.device)) 
        return {"context": prompt_emb}
        
def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None, # type: ignore
    vace: VaceWanModel = None, # type: ignore
    latents: torch.Tensor = None, # type: ignore
    timestep: torch.Tensor = None, # type: ignore
    context: torch.Tensor = None, # type: ignore
    clip_feature: Optional[torch.Tensor] = None, # type: ignore
    y: Optional[torch.Tensor] = None, # type: ignore
    reference_latents = None, # type: ignore
    vace_context = None, # type: ignore
    vace_scale = 1.0, # type: ignore
    tea_cache: TeaCache = None, # type: ignore
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None, # type: ignore
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = True,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None, # type: ignore
    training_mode: str = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = torch.cat([dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep[i])) for i in range(len(timestep))], dim=0) # 4 1536(d)
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)) 
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context) 

    x = latents # noise
    B = x.shape[0]
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)


    if dit.has_image_input:
        x = torch.cat([x, y], dim=1) 
        clip_embdding = dit.img_emb(clip_feature).repeat(B, 1, 1)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x, control_camera_latents_input) # type: ignore

    # Reference image 
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1) 
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device) 
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module, training_mode):
            def custom_forward(*inputs):
                return module(*inputs, training_mode=training_mode)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint( # type: ignore
                        create_custom_forward(block, training_mode),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint( # type: ignore
                    create_custom_forward(block, training_mode),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context.to(x.dtype), t_mod, freqs, training_mode=training_mode)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:] # type: ignore
        f -= 1
    x = dit.unpatchify(x, (f, h, w)) # type: ignore just a rearrangement
    return x

class UniVidIntrinsic(DiffusionTrainingModule): 
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_modalities:list[str]=None,
        use_gradient_checkpointing=True, 
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        resume_from_checkpoint: Optional[str] = None,
        albedo_resume_from_checkpoint: Optional[str] = None,
        material_resume_from_checkpoint: Optional[str] = None,
        normal_resume_from_checkpoint: Optional[str] = None,
        lora_ckpt_path: Optional[str] = None,
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
        tokenizer_config = ModelConfig(path="models/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Using device: {device}")
        self.torch_dtype = torch.bfloat16
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, tokenizer_config=tokenizer_config, model_configs=model_configs,redirect_common_files=False) # Load pretrained weights

        if lora_modalities is not None:
            lora_configs = []
            for modality in lora_modalities:
                lora_configs.append({
                    "target_modules": lora_target_modules,
                    "lora_rank": lora_rank,
                    "adapter_name": modality
                })
            self.pipe.dit = self.add_multiple_loras_to_model(
                model = getattr(self.pipe, lora_base_model),
                lora_configs = lora_configs
                )


        if lora_modalities is None:
            if lora_base_model is not None:
                model = self.add_lora_to_model(
                    getattr(self.pipe, lora_base_model), 
                    target_modules=lora_target_modules.split(","),
                    lora_rank=lora_rank
                )
                setattr(self.pipe, lora_base_model, model)
                print(f"[LoRA] has been added to: {lora_base_model}")
        

        if lora_base_model is not None:
            hit_names = []
            for name, module in self.pipe.dit.named_modules():
                if any(x in name for x in lora_target_modules.split(",")):
                    has_lora = any(subname.lower().startswith("lora") for subname, _ in module.named_modules())
                    has_lora_param = any("lora" in pname.lower() for pname, _ in module.named_parameters(recurse=True))
                    if has_lora or has_lora_param:
                        hit_names.append(name)

        if resume_from_checkpoint is not None:
            state_dict = load_file(resume_from_checkpoint) 
            prefix = "dit."
            state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            try:
                missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(state_dict, strict=False)
                assert unexpected_keys == [], "unexpected_keys not empty"
            except Exception as e:
                print(f"Failed to resume training from checkpoint {resume_from_checkpoint}: {e}")
            if missing_keys:
                print(f"⚠️ missing keys: {len(missing_keys)} ")
                for key in list(missing_keys)[:5]:
                    print(f"  - {key}")

        self.pipe.units = [ 
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]

        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        if lora_base_model is not None:
            lora_trainable_params = 0
            processed_lora_modules = set()  
            for name, param in self.pipe.dit.named_parameters():
                if "lora" in name.lower(): 
                    param.requires_grad = True
                    lora_trainable_params += param.numel()
                    module_name = name.rsplit('.', 2)[0] 
                    if module_name not in processed_lora_modules:
                        processed_lora_modules.add(module_name)
                        print(f"Trainable LoRA: {module_name} ({param.numel() /1024/1024:.2f} MB)")
            
            print(f"LoRA trainable params: {lora_trainable_params /1024/1024:.2f} MB")

        trainable_params = 0
        for name, p in self.pipe.dit.named_parameters():
            if p.requires_grad:
                trainable_params += p.numel()
                print(f"Trainable: {name} ({p.numel() /1024/1024:.2f} MB)")
        print(f"Total Trainable: {trainable_params /1024/1024:.2f} MB")
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
    
   

    def training_mode(self) -> str:
        import torch.distributed as dist
        modes = [
                "t2RAIN", "R2AIN", "A2RIN", "I2RAN", "N2RAI", 
                "RA2IN", "RI2AN", "RN2AI", "AI2RN", "AN2RI",
                "IN2RA", "AIN2R", "RIN2A", "RAN2I", "RAI2N"
                ]
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                flag = torch.rand(1).item()
                mode_idx = int(flag * 15)
            else:
                mode_idx = 0

            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

            mode_tensor = torch.tensor([mode_idx], dtype=torch.long, device=device)
            dist.broadcast(mode_tensor, src=0)
            mode_idx = mode_tensor.item()

            return modes[mode_idx]  # type: ignore[index]

        else:
            flag = torch.rand(1).item()
            mode_idx = int(flag * 15)
            return modes[mode_idx]  # type: ignore[index]


    def forward_preprocess(self, data):
        # CFG-sensitive parameters

        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            "height": data["height"],
            "width": data["width"],
            "num_frames": data["num_frames"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "clip_feature": None,
            "y": None,
            "training_mode": data["training_mode"],
        }
        if inputs_shared["training_mode"] in ["t2RAIN", "A2RIN", "I2RAN", "N2RAI", "AI2RN", "AN2RI", "IN2RA"]: 
            prompt_list = [
                data["prompt"][0],
                data["prompt"][0],
                data["prompt"][0],
                data["prompt"][0]
            ]
            inputs_posi = {"prompt": prompt_list}
        else:
            prompt_list = [
                "",
                "",
                "",
                ""
            ]
            inputs_posi = {"prompt": prompt_list}
        input_videos = torch.cat((data["rgb"], data["albedo"], data["irradiance"], data["normal"]), dim=0) # 4b c t h w
        inputs_shared["input_videos"] = [input_videos]
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "control_video" and "control_video" not in data and "rgb" in data:
                inputs_shared[extra_input] = torch.cat((data["rgb"], data["rgb"], data["rgb"], data["rgb"]), dim=0) # 4b c t h w
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        if self.pipe.units is not None:
            for unit in self.pipe.units:
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega) # type: ignore
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss