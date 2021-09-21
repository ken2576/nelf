import subprocess
import torch
from tqdm import tqdm
from .validation import validate
from .visualization import visualize_sample_images, visualize_loss

# Get git commit hash
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

def update_loss_dict(loss_hist, loss):
    for name, value in loss.items():
        if name not in loss_hist:
            loss_hist[name] = [value.item()]
        else:
            loss_hist[name].append(value.item())

def train(
    model, optimizer, train_data_loader, val_data_loader, val_dataset,
    train_step, prompt_interval,
    vis_interval, val_interval, ckpt_interval,
    vis_path, val_path, ckpt_path,
    vis_data_id, vis_source_image_id, vis_target_image_ids,
    use_pretrain=False, pretrain_step=0
):

    if use_pretrain:
        ckpt = torch.load(f'{ckpt_path}/{pretrain_step}.pth')
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        pretrain_step = ckpt['step']
        train_loss_hist = ckpt['train_loss_hist']
        val_loss_hist = ckpt['val_loss_hist']
    else:
        pretrain_step = 0
        train_loss_hist, val_loss_hist = {}, []


    with tqdm(train_data_loader, unit="it", initial=pretrain_step, total=train_step) as tstep:
        postfix = {}
        for step, data in enumerate(tstep, start=pretrain_step):
            ### Show train/val loss, visualization, and save model
            if step != pretrain_step and step % prompt_interval == 0:
                update_loss_dict(train_loss_hist, train_loss)
                postfix['train_loss'] = train_loss_total.item()
                tstep.set_postfix(**postfix)

            if step != pretrain_step and step % val_interval == 0:
                val_loss_total = validate(
                    model, val_data_loader, val_path, write_image=(step == train_step)
                ).item()
                val_loss_hist.append(val_loss_total)
                postfix['val_loss'] = val_loss_total
                tstep.set_postfix(**postfix)

            if step != pretrain_step and step % vis_interval == 0:
                visualize_sample_images(
                    model, val_dataset, vis_path, step,
                    vis_data_id, vis_source_image_id, vis_target_image_ids
                )
                visualize_loss(
                    vis_path, step, 
                    train_loss_hist, prompt_interval,
                    val_loss_hist, val_interval
                )

            if (step != pretrain_step and step % ckpt_interval == 0) or step == train_step:
                try:
                    model_state_dict = model.module.state_dict()
                except AttributeError:
                    model_state_dict = model.state_dict()
                checkpoint = {
                    'step': step,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss_hist': train_loss_hist,
                    'val_loss_hist': val_loss_hist,
                    'git_commit': get_git_revision_hash()
                }
                torch.save(checkpoint, f'{ckpt_path}/{step}.pth')

            if step == train_step:
                break


            ### Training step
            output = model.module.render(
                True, model,
                **{k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
            )
            train_loss = model.module.train_loss(output)
            train_loss_total = sum([v for _, v in train_loss.items()])
            train_loss['total'] = train_loss_total

            train_loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print(f"Coarse RGB: {output['coarse_rgb'].min().item()}\t{output['coarse_rgb'].mean().item()}\t{output['coarse_rgb'].max().item()}")
            # print(f"Fine RGB: {output['fine_rgb'].min().item()}\t{output['fine_rgb'].mean().item()}\t{output['fine_rgb'].max().item()}")
            # print(f"Target RGB: {output['target_rgb'].min().item()}\t{output['target_rgb'].mean().item()}\t{output['target_rgb'].max().item()}")
            # print()
