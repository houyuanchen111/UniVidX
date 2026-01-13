import torch
from pathlib import Path
import warnings
from tqdm import tqdm
from typing import Optional
from accelerate import Accelerator
from accelerate.state import AcceleratorState
import os
from ..trainers.util import Callback
from torchvision.io import write_video
class ModelCheckpointCallback(Callback):     
    """
    checkpoint callback
    """
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = Path(output_path)
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
     
    def _save_trainable_weights(self, trainer, checkpoint_dir: Path):
        """
        Helper function: extract, clean and save trainable weights.
        """
        print("--- Extracting and saving trainable-only weights... ---")
        
        # a. Get the original unwrapped model
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        
        # b. Filter out all trainable parameters
        trainable_state_dict = {
            name: param
            for name, param in unwrapped_model.named_parameters()
            if param.requires_grad
        }

        if not trainable_state_dict:
            warnings.warn("No trainable parameters found in the model, skipping saving trainable_only.safetensors.")
            return
        if self.remove_prefix_in_ckpt:
            clean_state_dict = {}
            prefix = self.remove_prefix_in_ckpt
            for k, v in trainable_state_dict.items():
                if k.startswith(prefix):
                    # Only keep the part after the prefix
                    clean_state_dict[k[len(prefix):]] = v
                else:
                    # If there is no such prefix, keep the key as is
                    clean_state_dict[k] = v
            trainable_state_dict = clean_state_dict

        # d. Define the save path and write weights
        save_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
        trainer.accelerator.save(trainable_state_dict, str(save_path),safe_serialization=True) # safe_serialization=True must be enabled
        print(f"✅ Trainable-only weights have been saved to -> {save_path}")

    def on_epoch_end(self, trainer):

        trainer.accelerator.wait_for_everyone()
        epoch_checkpoint_dir = self.output_path / f"epoch_{trainer.epoch_id}"
        if trainer.accelerator.is_main_process:
            print(f"\n--- End of Epoch {trainer.epoch_id} ---")
            print(f"Preparing to create directory on main process: {epoch_checkpoint_dir}")
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Epoch {trainer.epoch_id}: Saving inference weights...")
            self._save_trainable_weights(trainer, epoch_checkpoint_dir)
            print("Checkpoint directory is ready. Starting to save state...")

class TensorboardLoggingCallback(Callback):    
    """tensorboard callback"""
    def __init__(self, logging_dir=None, hps=None):
        self.logging_dir = logging_dir
        self.hps = hps

    def on_train_begin(self, trainer):
        trainer.accelerator.init_trackers(
            project_name="tensorboard", 
            config=self.hps, 
            )

    def on_step_end(self, trainer):
        if trainer.accelerator.is_main_process:
            current_lr = trainer.scheduler.get_last_lr()[0]
            trainer.accelerator.log(
                {"training_loss": trainer.loss.item(), "learning_rate": current_lr},
                step=trainer.global_step
            )
    
    def on_epoch_end(self, trainer):
        if trainer.accelerator.is_main_process:
            trainer.accelerator.log(
                {"val_loss": trainer.val_loss},
                step=trainer.global_step,
            )
    
    def on_train_end(self, trainer):
        pass

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
            resume_from_deepspeed: Optional[str] = None,
            val_dataloader=None 
            ):
        
        if logger_type == "tensorboard":
            for cb in callbacks: # type: ignore
                if hasattr(cb, "logging_dir"):
                    logging_dir = cb.logging_dir # type: ignore
                if hasattr(cb, "output_path"):
                    self.output_path = cb.output_path # type: ignore
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type,
                project_dir=logging_dir # When using tensorboard, project_dir must be specified
            )
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type,
                #deepspeed_plugin=plugin
            )
        if dataloader.batch_size == None:
            state = AcceleratorState()
            if state.deepspeed_plugin is not None:
                state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = dataloader.batch_sampler.batch_size
                # Optional: print confirmation
                print("DS micro batch size set to:",
                    state.deepspeed_plugin.deepspeed_config.get('train_micro_batch_size_per_gpu'))
        # 2. Prepare all components; this step is the most time-consuming (over 15s for 1.3B)
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, optimizer, dataloader, scheduler
        ) 
        if resume_from_deepspeed is not None:
            try:
                print(f"Resuming training from checkpoint {resume_from_deepspeed}")
                self.accelerator.load_state(resume_from_deepspeed)
            except Exception as e:
                raise ValueError(f"Failed to resume training from checkpoint {resume_from_deepspeed}: {e}")
            
        self.val_dataloader = val_dataloader  # Store validation dataloader
        self.num_epochs = num_epochs
        self.callbacks = callbacks if callbacks is not None else []
        self.global_step = 0
        self.epoch_id = 0
        self.loss = None
    
    def _tensor2video(self, tensor: torch.Tensor, file_path: str, fps: int = 15):
        assert tensor.ndim == 4, "tensor must be [c,t,h,w]"
        tensor = tensor.cpu()
        video_tensor = tensor.permute(1, 2, 3, 0) # c t h w -> t h w c       
        video_tensor = (video_tensor * 255).to(torch.uint8)
        print(f"Saving video to: {file_path}")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path),exist_ok = True)
        write_video(
            filename=file_path,
            video_array=video_tensor,
            fps=fps,
            video_codec='h264' # Use common h264 codec
        )
        print("Video saved successfully!")

    def validate(self):
        # You can implement your own validation logic here
        return 0.0
       
    def train(self):
        self._call_callbacks("on_train_begin")

        for epoch_id in range(self.num_epochs):
            self.epoch_id = epoch_id
            self._call_callbacks("on_epoch_begin")
            
            # Wrap tqdm with accelerator.main_process_* to ensure the progress bar is only shown on the main process
            pbar = tqdm(self.dataloader, disable=not self.accelerator.is_main_process, desc=f"Epoch {epoch_id+1}/{self.num_epochs}")
            
            for batch_idx, data in enumerate(pbar):
                try:
                    self._call_callbacks("on_step_begin")
                    with self.accelerator.accumulate(self.model):
                        self.optimizer.zero_grad()
                        self.loss = self.model(data) # Store loss on self for callbacks to use
                        
                        # Check loss validity; skip this batch if it is invalid
                        if not torch.isfinite(self.loss):
                            print(f"⚠️ Batch {batch_idx + 1}: nan or inf loss detected, skipping...")
                            continue
                        
                        self.accelerator.backward(self.loss)
                        self.optimizer.step()               
                        self.scheduler.step()
                    
                    self.global_step += 1
                    if self.accelerator.is_main_process:
                        pbar.set_postfix({"loss": float(self.loss.detach().cpu().item())}) # Show loss in the progress bar
                    self._call_callbacks("on_step_end")
                    
                except Exception as e:
                    print(f"Error occurred while processing batch {batch_idx + 1}: {e}")
                    if self.accelerator.is_main_process:
                        import traceback
                        traceback.print_exc()
                    # Continue with the next batch
                    continue
            self.val_loss = self.validate() # type: ignore
            self.accelerator.wait_for_everyone()
            self._call_callbacks("on_epoch_end")
    
        self._call_callbacks("on_train_end")


