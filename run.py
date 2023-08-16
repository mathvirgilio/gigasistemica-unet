import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from config import *
from utils import dice_score_metric, get_metrics, save_checkpoint, get_otsu_threshold_from_tensor
from augmentations import apply_TTA, revert_TTA, get_final_bin_mask


writer = SummaryWriter(log_dir= RUN_DIR + "plot/")
def run_on_dataloader(model, loader, iter_counter, loss_fn, train=True, img_path=None, optimizer=None, scaler=None):
    run_type = 'training'
    context = torch.enable_grad
    metrics = {}
    if not train:
        model.eval()
        context = torch.no_grad
        run_type = "validation"
        dice_score = 0
        precision = 0
        recall = 0
        f1_score = 0

    with context():
        loss_sum = 0
        loop = tqdm(loader)
        
        if(USE_DATA_PARALLEL):
            model = DataParallel(model)
        
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # forward
            batch_size = len(data)
            
            if APPLY_TTA and not train:
                flip_data, filter_data = apply_TTA(data)
                flip_data = flip_data.to(device=DEVICE)
                filter_data = filter_data.to(device=DEVICE)
                data = torch.cat([data, flip_data, filter_data], axis=0)
            
            predictions = model(data)

            loss = []
            for i in range(0, len(predictions), batch_size):
                curr_slice = predictions[i:i + batch_size]
                loss += [loss_fn(curr_slice, targets, reduction='mean')]
            
            loss = torch.mean(torch.stack(loss))
            
            if train:
            # backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            loss_sum += loss.item()

            # update tqdm loop
            loop.set_postfix(loss = loss.item())
            
            if not train:
                if APPLY_TTA:
                    predictions[batch_size:2*batch_size], predictions[2*batch_size:] = revert_TTA(predictions[batch_size:2*batch_size], predictions[2*batch_size:])
                    
                    predictions_sigmoided = torch.sigmoid(predictions)
                    preds_thresholded = predictions_sigmoided.detach().clone()
                    for i in range(len(predictions)):
                        threshold = get_otsu_threshold_from_tensor(predictions_sigmoided[i])
                        preds_thresholded[i] = (preds_thresholded[i] > threshold).float()
                        
                    #preds_thresholded = get_final_bin_mask(preds_thresholded[:batch_size], 
                    #                                       preds_thresholded[batch_size:2*batch_size], 
                    #                                       preds_thresholded[2*batch_size:]).float()
                    
                    if img_path:
                        for i in range(batch_size):
                            aux_pred = torch.cat([data[i:i+1], targets[i:i+1]], axis=0) 
                            aux_pred = torch.cat([aux_pred, predictions_sigmoided[i:i+1]], axis=0)
                            aux_pred = torch.cat([aux_pred, preds_thresholded[i:i+1]], axis=0)
                            aux_pred = torch.cat([aux_pred, preds_thresholded[i+batch_size:i+batch_size+1]], axis=0)
                            aux_pred = torch.cat([aux_pred, preds_thresholded[i+2*batch_size:i+2*batch_size+1]], axis=0)
                            torchvision.utils.save_image(aux_pred, f"{img_path}{batch_idx}_{i}.png")
                else:
                    predictions_sigmoided = torch.sigmoid(predictions)
                    preds_thresholded = predictions_sigmoided.detach().clone()
                    
                    for i in range(batch_size):
                        threshold = get_otsu_threshold_from_tensor(predictions_sigmoided[i])
                        preds_thresholded[i] = (preds_thresholded[i] > threshold).float()
                        
                        #Save images
                        if img_path:
                            aux_pred = torch.cat([data[i:i+1], targets[i:i+1]], axis=0) 
                            aux_pred = torch.cat([aux_pred, predictions_sigmoided[i:i+1]], axis=0)
                            aux_pred = torch.cat([aux_pred, preds_thresholded[i:i+1]], axis=0)
                            torchvision.utils.save_image(aux_pred, f"{img_path}{batch_idx}_{i}.png")
                
                preds_thresholded = get_final_bin_mask(preds_thresholded[:batch_size], 
                                                           preds_thresholded[batch_size:2*batch_size], 
                                                           preds_thresholded[2*batch_size:]).float()
                dice_score += dice_score_metric(preds_thresholded, targets)
                p, r, f1 = get_metrics(preds_thresholded, targets, apply_sigmoid=False)
                precision += p
                recall += r
                f1_score += f1

            writer.add_scalar(run_type + "/loss_per_batch", loss.item(), iter_counter)
            iter_counter += 1
        
        len_loader = len(loader)
        if not train:
            metrics["dice score"] = dice_score/len_loader
            metrics["precision"] = precision/ len_loader
            metrics["recall"] = recall/len_loader
            metrics["f1_score"] = f1_score/len_loader
            
        mean_loss = loss_sum/len_loader
        return mean_loss, iter_counter, metrics

def train_fn(loader, model, optimizer, loss_fn, scaler, val_loader=None):
    iter_counter = 0
    iter_counter_val = 0
    for epoch in range(NUM_EPOCHS):
        print("Época", epoch)
        print("Learning rate:", optimizer.param_groups[0]['lr'])
        mean_loss, iter_counter, _ = run_on_dataloader(model, loader, iter_counter, loss_fn, train=True, img_path=None, optimizer=optimizer, scaler=scaler)

        # save model
        checkpoint = {"state_dict" : model.state_dict(), "optimizer" : optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=NAME_CHECKPOINT)

        # print some examples to a folder
        if(val_loader is not None):
            mean_loss, iter_counter_val, metrics = run_on_dataloader(model, val_loader, iter_counter_val, loss_fn, train=False, img_path=IMG_PATH + "val/", optimizer=None)
            
            print("validation/loss = ", mean_loss)
            print("validation/score = ", metrics["dice score"].item())
            print("validation/precision = ", metrics["precision"].item())
            print("validation/recall = ", metrics["recall"].item())
            print("validation/f1_score = ", metrics["f1_score"].item())
            
            writer.add_scalar('validation/loss', mean_loss, epoch)
            writer.add_scalar('validation/score', metrics["dice score"], epoch)
            writer.add_scalar('validation/precision', metrics["precision"], epoch)
            writer.add_scalar('validation/recall', metrics["recall"], epoch)
            writer.add_scalar('validation/f1_score', metrics["f1_score"], epoch)
            
            
def validate(loader, model, loss_fn):
    iter_counter_val = 0
    mean_loss, iter_counter_val, metrics = run_on_dataloader(model, loader, iter_counter_val, loss_fn, train=False, img_path=IMG_PATH + "val/", optimizer=None)
    print("validation/loss = ", mean_loss)
    print("validation/score = ", metrics["dice score"].item())
    print("validation/precision = ", metrics["precision"].item())
    print("validation/recall = ", metrics["recall"].item())
    print("validation/f1_score = ", metrics["f1_score"].item())
