import sys, datetime, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline

from config.train_config import AppConfig
from utils import log_util
from dataset.dataloaders.cad_sketch_dataset import NormalSketchControlNetDataset
from models.diffusion import Diffusion_Models

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




def main():
    args = AppConfig.from_cli()
    if not args.debug:
        log_dir, log_file, tsboard_writer, compare_log = log_util.setup_logdir(args.parent_log_dir, args.compare_log)  # 结果路径、tensorboard、日志文件
        AppConfig.write_config(config_obj=args, log_file=log_file, compare_log=compare_log)

    train_dataset = NormalSketchControlNetDataset(
        root_dir=args.file_path,
        mode='train',
        res=args.res
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True, # GPU加速
    )

    models = Diffusion_Models(args)

    train_pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=models.vae,
        text_encoder=None,
        tokenizer=None,
        unet=models.unet,
        controlnet=models.controlnet,
        scheduler=models.scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(args.device) 

    for param in train_pipe.unet.parameters():
        param.requires_grad = False

    loss_fn = nn.MSELoss()
    
    optimizer = optim.AdamW(
        train_pipe.controlnet.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    ) 


    # train
    for epoch in range(args.num_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        for step, batch in loop:
            
            base = batch["base"].to(args.device, dtype=models.td)            
            sketch = batch["sketch"].to(args.device, dtype=models.td)
            target = batch["target"].to(args.device, dtype=models.td)
            # mask = batch["mask"].to(args.device, dtype=td)

            with torch.no_grad():
                # 预处理图像,输入到pipe中
                clip_input = models.clip_processor(images=base, return_tensors="pt").pixel_values.to(args.device)
                image_embeds = models.clip_model(clip_input).last_hidden_state  
                # image_embeds = projector1(image_embeds.transpose(1, 2)).transpose(1, 2)  # [1, 77, 1024]
                # image_embeds = projector2(image_embeds)
                
            # Encode target image成latents
            latents = models.vae.encode(target).latent_dist.sample() * 0.18215
            
            # 采样随机噪声，加到latent
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, models.scheduler.config.num_train_timesteps, (bsz,), device=args.device).long()
            noisy_latents = models.scheduler.add_noise(latents, noise, timesteps)

            # controlnet 向前传播
            down_block_res_samples, mid_block_res_sample = train_pipe.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=image_embeds,  # 提示嵌入
                controlnet_cond=sketch,  # 控制， 该模型 提示使用3通道图像
                return_dict=False,
            )

            noise_pred = train_pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=image_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # 计算loss
            noise_loss = loss_fn(noise_pred, noise)
            optimizer.zero_grad()
            noise_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                # if step % 200 == 0:
                    # bp = '/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/train/base_img/000000.png'
                    #sp = '/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/train/sketch_img/000000.png'
                    #test_sketch_img = Image.open(sp).convert("RGB")
                    #test_sketch_img = train_dataset.normlize(train_dataset.transform(test_sketch_img)).to(args.device)
                #     output = train_pipe(
                #         prompt_embeds=image_embeds,
                #         pooled_prompt_embeds=image_embeds.mean(dim=1),
                #         negative_prompt_embeds=torch.zeros_like(image_embeds),  # 简单使用零作为负提示
                #         negative_pooled_prompt_embeds=torch.zeros_like(image_embeds.mean(dim=1)),
                #         image=sketch,
                #         num_inference_steps=20,
                #         guidance_scale=7.5,
                #     ).images[0]

                # output.save(os.path.join(log_dir, "vis", f"{epoch}_{step}.png"))
                tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
                log_util.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}", log_file)
            torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet_epoch{epoch}.pth"))
        torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet.pth"))

            

if __name__ == "__main__":
    main()


    # # 其他
    # # 将img编码进行投影，以符合stable diffusion的输入
    # projector1 = nn.Linear(257, 77).to(args.device)
    # projector2 = nn.Linear(1024, 768).to(args.device)
    # projector1.load_state_dict(projector_ckpt['projector_257to77'])
    # projector2.load_state_dict(projector_ckpt['projector_1024to768'])
    # # 用于重建loss，和target对齐
    # normlize_1 = transforms.Normalize([0.5], [0.5])
    raise Exception('over')


            


            # 非mask区域的重建loss
    if args.lam:
        pred_latents = noisy_latents - noise_pred
        pred_image = train_pipe.vae.decode(pred_latents / 0.18215).sample  #  预测图像

        valid_area = (mask < 0.1).float() 
        original_valid = normlize_1(input) * valid_area
        pred_valid = pred_image * valid_area

        re_loss = loss_fn(pred_valid, original_valid)
    
        total_loss = args.lam * re_loss  + noise_loss
    else: 
        total_loss = noise_loss







        if args.lam:
            tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
            tsboard_writer.add_scalar('re_loss', re_loss.item(), step)
            log_util.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}, re_loss: {re_loss.item():.4f}", log_file)
        else:
            tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
            log_util.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}", log_file)

        torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet_epoch{epoch}.pth"))

    torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet.pth"))

    