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
import os


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
    shared_tensor_queues,
    restore_tensor_queues
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
                    args=(gpu_id, input_queue, output_queue, unet, scheduler, vae, config, shared_tensor_queues[gpu_id])
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
                    args=(output_queue, final_queue, config, gpu_id, worker_idx, restore_tensor_queues[gpu_id])
                )
                print(f"############## Finish starting restore worker {worker_idx} on GPU {gpu_id} ####################")
                restore_p.start()
                workers.append(restore_p)

    return workers


def prepare_image_latents(images, device, dtype, generator, do_classifier_free_guidance, vae):
    # vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    images = images.to(device=device, dtype=dtype)
    image_latents = vae.encode(images).latent_dist.sample(generator=generator)
    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
    image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
    image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
    return image_latents

def prepare_mask_latents(
    mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance, vae
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    mask = torch.nn.functional.interpolate(
        mask, size=(height // vae_scale_factor, width // vae_scale_factor)
    )
    masked_image = masked_image.to(device=device, dtype=dtype)

    # encode the mask image into latents space so we can concatenate it to the latents
    masked_image_latents = vae.encode(masked_image).latent_dist.sample(generator=generator)
    masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor

    # aligning device to prevent device errors when concating it with the latent model input
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    # assume batch size = 1
    mask = rearrange(mask, "f c h w -> 1 c f h w")
    masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

    mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = (
        torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
    )
    return mask, masked_image_latents

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
def restore_video_worker(input_queue: mp.Queue, output_queue: mp.Queue, config, gpu_id, worker_id, restore_tensor_queue):
    restore_tensors = restore_tensor_queue.get()  
    restore_original_frames, restore_boxes, restore_affine = restore_tensors
    print("############ GET SHARED RESTORE TENSOR #############")
    # Initialize Image Processor
    image_processor = ImageProcessor(config.data.resolution, mask="fix_mask", device=f"cuda:{gpu_id}")
    while True:
        try:
            q_fetch = time.time()
            batch_data = input_queue.get()
            if batch_data is None:
                break
            print(f"restore worker fetch {worker_id}: ", time.time() - q_fetch)
            
            start_time = time.time()
            batch_id, decoded_latents, start_idx, end_idx = batch_data
        
            restored = restore_video(
                decoded_latents, 
                restore_original_frames[start_idx:end_idx] ,
                restore_boxes[start_idx:end_idx] if restore_boxes is not None else None,
                restore_affine[start_idx:end_idx] if restore_affine is not None else None,
                image_processor
            )
            
            output_queue.put((batch_id, restored))
            print(f"XXX total restore worker {worker_id}: ", time.time() - start_time)
        except Exception as e:
            print(f"Error in restore worker: {str(e)}")
            output_queue.put((None, e))
            break




@torch.no_grad()
def denoise_worker(gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue, 
                  unet, scheduler, vae, config, shared_tensor_queue):
  

    shared_tensors = shared_tensor_queue.get()  
    shared_video_frames, shared_latents, shared_original_frames, shared_boxes, shared_affine = shared_tensors
    print("############ GET SHARED TENSOR #############")
    
    device = torch.device(f"cuda:{gpu_id}")
    unet = unet.to(device)
    image_processor = ImageProcessor(config.data.resolution, mask="fix_mask", device=f"cuda:{gpu_id}")
    while True:
        try:
            current_size = input_queue.qsize()
            
            q_fetch = time.time()
            # Get next batch from queue
            batch_data = input_queue.get()
            
            # Check for termination signal
            if batch_data is None:
                break
            print(f"denoise worker fetch: ", time.time() - q_fetch)

            start_time = time.time()
        
            (batch_id, start_idx, end_idx, whisper_chunks, height, width,
             generator, do_classifier_free_guidance, guidance_scale, 
             extra_step_kwargs, num_inference_steps, weight_dtype) = batch_data

            video_frames = shared_video_frames[start_idx:end_idx]
            all_latents = shared_latents[:, :, start_idx:end_idx]
            original_video_frames = shared_original_frames[start_idx:end_idx]  
            boxes = shared_boxes[start_idx:end_idx] if shared_boxes is not None else None  
            affine_matrices = shared_affine[start_idx:end_idx] if shared_affine is not None else None  
            

            # Audio embedding preprocessing
            if unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks)
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    empty_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([empty_audio_embeds, audio_embeds])
            else:
                audio_embeds = None

            # Get latents for this batch
            latents = all_latents.to(device)
            
            # Prepare masks and images
            pixel_values, masked_pixel_values, masks = image_processor.prepare_masks_and_masked_images(
                video_frames, affine_transform=False
            )

            # Prepare mask latents
            mask_latents, masked_image_latents = prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
                vae,
            )

            # Prepare image latents
            image_latents = prepare_image_latents(
                pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
                vae,
            )

            preprocess_time = time.time()

            # Set timesteps here
            scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = scheduler.timesteps
            
            # Process the batch
            num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
            
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
            decoded_latents = decode_latents(latents, vae)
            decoded_latents = paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
        
            paste_time = time.time()
            result = (batch_id, decoded_latents, start_idx, end_idx) 
            output_queue.put(result)

            print(f" ### total denoise worker {batch_id}: ", time.time() - start_time)
     
        except Exception as e:
            print(f"Error in worker on GPU {gpu_id}: {str(e)}")
            output_queue.put((None, e))
            break


def cleanup_workers(workers, worker_config, input_queue, output_queue, final_queue, shared_tensor_queues, restore_tensor_queues):
    n_denoise = sum(worker_config["denoise"].values())  
    n_restore = sum(worker_config["restore"].values())
    
    print(f"Force terminating {n_denoise} denoise workers and {n_restore} restore workers")
    
    # Force terminate all processes
    for p in workers:
        if p.is_alive():
            p.terminate()  # Force kill
            p.join(timeout=1)  
            if p.is_alive():  
                os.kill(p.pid, signal.SIGKILL)  # Force kill with SIGKILL
        print("----- Terminate Worker ------")
    
    # Force close all queues
    try:
        input_queue.close()
        input_queue.cancel_join_thread()
    except:
        pass
        
    try:
        output_queue.close()
        output_queue.cancel_join_thread()
    except:
        pass
        
    try:
        final_queue.close()
        final_queue.cancel_join_thread()
    except:
        pass
        
    for gpu_id in shared_tensor_queues:
        try:
            shared_tensor_queues[gpu_id].close()
            shared_tensor_queues[gpu_id].cancel_join_thread()
        except:
            pass

    for gpu_id in restore_tensor_queues:
        try:
            restore_tensor_queues[gpu_id].close()
            restore_tensor_queues[gpu_id].cancel_join_thread()
        except:
            pass
    
    torch.cuda.empty_cache()
    print("Cleanup complete")
    
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