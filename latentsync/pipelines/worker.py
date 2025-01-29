import torch
import torch.multiprocessing as mp
from diffusers import DDIMScheduler,AutoencoderKL
from latentsync.models.unet import UNet3DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from einops import rearrange
from ..utils.image_processor import ImageProcessor
import torchvision
import numpy as np

import time
import torch.multiprocessing as mp

# Set start method at the beginning
mp.set_start_method('spawn', force=True)

def initialize_worker_models(config, checkpoint_path, gpu_id):
    """
    Initialize UNet and scheduler for a worker process.
    
    Args:
        config: OmegaConf configuration object
        checkpoint_path: Path to the UNet checkpoint
        gpu_id: GPU device ID to use
    
    Returns:
        tuple: (unet, scheduler)
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    # Initialize scheduler
    scheduler = DDIMScheduler.from_pretrained("configs")
    
    # Initialize UNet
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        checkpoint_path,
        device="cpu",  # Initially load on CPU
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    
    

    unet = unet.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device)
    
    # Enable xformers if available
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    
    return unet, scheduler, vae


def initialize_workers(
    worker_config: dict,  # Format: {"denoise": {0: 2, 1: 1}, "restore": {0: 1, 1: 2}}
    config,
    checkpoint_path,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    final_queue: mp.Queue,
):
    workers = []
    
    # Initialize denoise workers
    if "denoise" in worker_config:
        for gpu_id, num_workers in worker_config["denoise"].items():
            for worker_idx in range(num_workers):
                # Initialize models for denoise worker
                unet, scheduler, vae = initialize_worker_models(
                    config, 
                    checkpoint_path, 
                    gpu_id,
                )
                print(f"------------- Finish creating models for denoise worker {worker_idx} on GPU {gpu_id} ------------------")
                
                # Start denoise worker
                denoise_p = mp.Process(
                    target=denoise_worker,
                    args=(gpu_id, input_queue, output_queue, unet, scheduler, vae)
                )
                print(f"############## Finish starting denoise worker {worker_idx} on GPU {gpu_id} ####################")
                denoise_p.start()
                workers.append(denoise_p)

    # Initialize restore workers
    if "restore" in worker_config:
        for gpu_id, num_workers in worker_config["restore"].items():
            for worker_idx in range(num_workers):
                # Start restore worker
                restore_p = mp.Process(
                    target=restore_video_worker,
                    args=(output_queue, final_queue, config, gpu_id)
                )
                print(f"############## Finish starting restore worker {worker_idx} on GPU {gpu_id} ####################")
                restore_p.start()
                workers.append(restore_p)

    return workers


@torch.no_grad()
def decode_latents( latents, vae):
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = vae.decode(latents).sample
        return decoded_latents
def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

def restore_video(faces, video_frames, boxes, affine_matrices, image_processor):
   out_frames = []
   for index, face in enumerate(faces):
       x1, y1, x2, y2 = boxes[index]
       height = int(y2 - y1)
       width = int(x2 - x1)
       face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
       face = rearrange(face, "c h w -> h w c")
       face = (face / 2 + 0.5).clamp(0, 1)
       face = (face * 255).to(torch.uint8).cpu().numpy()
       out_frame = image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
       out_frames.append(out_frame)
   return np.stack(out_frames, axis=0)
    
@torch.no_grad()
def restore_video_worker(input_queue: mp.Queue, output_queue: mp.Queue, config, gpu_id):

   # Initialize Image Processor
   image_processor = ImageProcessor(config.data.resolution, mask="fix_mask", device=f"cuda:{gpu_id}")
   while True:
       try:
           batch_data = input_queue.get()
           if batch_data is None:
               break

           
           start_time = time.time()
           batch_id, decoded_latents, video_frames, boxes, affine_matrices = batch_data
           
           # Since we're getting per-batch data, just use the indices directly
           # No need to calculate start_idx and end_idx
           restored = restore_video(
               decoded_latents, 
               video_frames[batch_id * len(decoded_latents):(batch_id + 1) * len(decoded_latents)],
               boxes[batch_id * len(decoded_latents):(batch_id + 1) * len(decoded_latents)],
               affine_matrices[batch_id * len(decoded_latents):(batch_id + 1) * len(decoded_latents)],
               image_processor
           )
           
           output_queue.put((batch_id, restored))

           print(f"total restore worker : ", time.time() - start_time)
       except Exception as e:
           print(f"Error in restore worker: {str(e)}")
           output_queue.put((None, e))
           break


@torch.no_grad()
def denoise_worker(gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue, unet, scheduler, vae):
# def denoise_worker(gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue, unet, scheduler, pipeline):

    # Set up device
    device = torch.device(f"cuda:{gpu_id}")
    unet = unet.to(device)
    
    while True:
        try:
            # Get next batch from queue
            batch_data = input_queue.get()
            
            # Check for termination signal
            if batch_data is None:
                break


            start_time = time.time()
            
            # Unpack batch data
            (batch_id, latents, timesteps_t, mask_latents, masked_image_latents, 
             image_latents, audio_embeds, do_classifier_free_guidance, 
             guidance_scale, extra_step_kwargs, num_inference_steps, 
             pixel_values, masks, weight_dtype, original_video_frames, boxes, affine_matrices) = batch_data

            # Set timesteps here
            scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = scheduler.timesteps
            
            # Move everything to GPU
            latents = latents.to(device)
            timesteps = timesteps.to(device)
            mask_latents = mask_latents.to(device)
            masked_image_latents = masked_image_latents.to(device)
            image_latents = image_latents.to(device)
            if audio_embeds is not None:
                audio_embeds = audio_embeds.to(device)
            
            # Process the batch
            num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

            fetch_move_time = time.time()
            # print(f"fetch move time {batch_id}: ", fetch_move_time - start_time )
            
            for j, t in enumerate(timesteps):
                
                
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat(
                    [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                )

                # predict the noise residual
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample


                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample



            denoise_time = time.time()
            # print(f"denoise loop time {batch_id}: ", denoise_time - fetch_move_time )
            
            decoded_latents = decode_latents(latents, vae)
            decoded_latents = paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
        
            paste_time = time.time()
            # print(f"paste back time {batch_id}: ", paste_time - denoise_time )


            # result = (batch_id, decoded_latents.cpu(), original_video_frames, boxes, affine_matrices)
            result = (batch_id, decoded_latents, original_video_frames, boxes, affine_matrices)
            output_queue.put(result)


            putback_time = time.time()

            print(f"total denoise worker {batch_id}: ", time.time() - start_time)
     
        except Exception as e:
            print(f"Error in worker on GPU {gpu_id}: {str(e)}")
            output_queue.put((None, e))
            break



def cleanup_workers(workers, worker_config, input_queue, output_queue, final_queue):
   
   n_denoise = sum(worker_config["denoise"].values())  
   n_restore = sum(worker_config["restore"].values())
   
   print(f"Terminating {n_denoise} denoise workers and {n_restore} restore workers")
   
   # Send termination to denoise workers
   for _ in range(n_denoise):
       input_queue.put(None)
   
   # Send termination to restore workers 
   for _ in range(n_restore):
       output_queue.put(None)
   
   # Wait for all workers to finish and terminate
   for p in workers:
       p.join()
       p.terminate()
       print("----- Terminate Worker ------")
   
   # Close all queues
   input_queue.close() 
   output_queue.close()
   final_queue.close()
   
   torch.cuda.empty_cache()
    
def launch_workers(num_gpus, unet, scheduler):
    """
    Launch worker processes on specified number of GPUs.
    
    Args:
        num_gpus: Number of GPU workers to launch
        unet: UNet model to use
        scheduler: Noise scheduler
    
    Returns:
        tuple: (list of input queues, list of output queues, list of processes)
    """
    workers = []
    input_queues = []
    output_queues = []
    
    for i in range(num_gpus):
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        
        # Start worker process
        p = mp.Process(
            target=denoise_worker,
            args=(i, input_queue, output_queue, unet, scheduler)
        )
        p.start()
        
        workers.append(p)
        input_queues.append(input_queue)
        output_queues.append(output_queue)
    
    return input_queues, output_queues, workers