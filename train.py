
import os
import sys
import numpy as np
import time
import random
import setproctitle
from functools import partial
import torch
import torch.nn.functional as F
from base_trainer import BaseTrainer
from opts import arguments
from core.config import cfg, cfg_from_file, cfg_from_list


class Trainer(base):

    def __init__(self, args, cfg):
        super(Trainer, self).__init__(args, cfg)
        self.load = dataload(args, cfg, 'train')       
        self.denorm = self.load.dataset.denorm     
        self.valloads = dataload(args, cfg, 'val')

        self.writer_val = {}
        for val_set in self.valloads.keys():
            logdir_val = os.path.join(args.logdir, val_set)

      
        self.net = model(cfg, remove_layers=cfg.MODEL.REMOVE_LAYERS)
        net_params = self.net.parameter_groups(cfg.MODEL.LR, cfg.MODEL.WEIGHT_DECAY)

        self.optim = self.optim(net_params, cfg.MODEL)


        
        if cfg.MODEL.LR_SCHEDULER == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, \
                                                             step_size=cfg.MODEL.LR_STEP, \
                                                             gamma=cfg.MODEL.LR_GAMMA)
        elif cfg.MODEL.LR_SCHEDULER == "linear": 

            def lr_lambda(epoch):
                mult = 1 - epoch / (float(self.cfg.TRAIN.NUM_EPOCHS) - self.start_epoch)
                mult = mult ** self.cfg.MODEL.LR_POWER
                
                return mult

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        else:
            self.scheduler = None

        self.vis_batch = None

        
        self.net.cuda()
        self.crw = CRW(cfg.TEST)

        
        self.checkpoint.create_model(self.net, self.optim)
        if not args.resume is None:
            self.start_epoch, self.best_score = self.checkpoint.load(args.resume, "cuda:0")


    def step(self, epoch, batch_src, key, temp=None, train=False, visualise=False, \
                 save_batch=False, writer=None, tag="train_src"):

        frames, masks_gt, n_obj, seq_name = batch_src

        
        frames = frames.flatten(0,1)
        masks_gt = masks_gt.flatten(0,1)
        masks_gt = masks_gt[:, :n_obj.item()]

        masks_ref = masks_gt.clone()
        masks_ref[1:] *= 0

        T = frames.shape[0]

        fetch = {"res3": lambda x: x[0], \
                 "res4": lambda x: x[1], \
                 "key": lambda x: x[2]}

        
        bs = self.cfg.TRAIN.BATCH_SIZE
        feats = []
        t0 = time.time()

        torch.cuda.empty_cache()

        for t in range(0, T, bs):

            
            frames_batch = frames[t:t+bs].cuda()

            
            feats_ = self.net(frames_batch, embd_only=True)
            feats.append(fetch[key](feats_).cpu())

        feats = torch.cat(feats, 0)
        sys.stdout.flush()
        t0 = time.time()
        outs = self.crw.forward(feats, masks_ref)
        sys.stdout.flush()
        outs["masks_gt"] = masks_gt.argmax(1)

        if visualise:
            outs["frames"] = frames
            self._visualise_seg(epoch, outs, writer, tag)

        if save_batch:
            
            
            self.save_vis_batch(tag, batch_src)

        return outs

    def step(self, epoch, batch_in, train=False, visualise=False, save_batch=False, writer=None, tag="train"):

        frames1, frames2, affine1, affine2 = batch_in
        assert frames1.size() == frames2.size(), "Frames shape mismatch"

        
        
        
        
        
        

        B,T,C,H,W = frames1.shape
        images1 = torch.cat((frames1, frames2[:, ::T]), 1)
        images1 = images1.flatten(0,1).cuda()
        images2 = frames2[:, 1:].flatten(0,1).cuda()

        affine1 = affine1.flatten(0,1).cuda()
        affine2 = affine2.flatten(0,1).cuda()

        
        losses, outs = self.net(images1, frames2=images2, T=T, \
                                affine=affine1, affine2=affine2, \
                                dbg=visualise)

        if train:
            self.optim.zero_grad()
            losses["main"].backward()
            self.optim.step()

        if visualise:
            self._visualise(epoch, outs, T, writer, tag)

        if save_batch:
            
            self.save_vis_batch(tag, batch_in)

        
        
        losses_ret = {}
        for key, val in losses.items():
            losses_ret[key] = val.mean().item()

        return losses_ret, outs

    def train_epoch(self, epoch):

        stat = StatManager()

        
        timer = Timer("Epoch {}".format(epoch))
        step = partial(self.step, train=True, visualise=False)

        
        self.net.train()

        for i, batch in enumerate(self.load):

            save_batch = i == 0

            
            
            
            losses, _ = step(epoch, batch, save_batch=save_batch, tag="train")

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            
            if i % 10 == 0:
                msg =  "Loss [{:04d}]: ".format(i)
                for loss_key, loss_val in losses.items():
                    msg += " {} {:.4f} | ".format(loss_key, loss_val)
                msg += " | Im/Sec: {:.1f}".format(i * self.cfg.TRAIN.BATCH_SIZE / timer.stage_elapsed())
                sys.stdout.flush()
        
        for name, val in stat.items():
            self.writer.add_scalar('all/{}'.format(name), val, epoch)

        
        for ii, l in enumerate(self.optim.param_groups):
            self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

        
        if stat.has_vals("lr_gamma"):
            self.writer.add_scalar('hyper/gamma', stat.summarize_key("lr_gamma"), epoch)

        if epoch % self.cfg.LOG.ITER_TRAIN == 0:
            self.visualise_results(epoch, self.writer, "train", self.step)

    def validation(self, epoch, writer, load, tag=None, max_iter=None):

        stat = StatManager()

        if max_iter is None:
            max_iter = len(load)

        
        def eval_batch(batch):

            loss, masks = self.step(epoch, batch, train=False, visualise=False)

            for loss_key, loss_val in loss.items():
                stat.update_stats(loss_key, loss_val)

            return masks

        self.net.eval()

        sys.stdout.flush()
        for n, batch in enumerate(load):

            with torch.no_grad():
                
                eval_batch(batch)

            if not tag is None and not self.has_vis_batch(tag):
                self.save_vis_batch(tag, batch)

        checkpoint_score = 0.0

        
        for stat_key, stat_val in stat.items():
            writer.add_scalar('all/{}'.format(stat_key), stat_val, epoch)

        if not tag is None and epoch % self.cfg.LOG.ITER_TRAIN == 0:
            self.visualise_results(epoch, writer, tag, self.step)

        return checkpoint_score

    def validation_seg(self, epoch, writer, load, key="all", temp=None, tag=None, max_iter=None):

        vis = key == "res4"
        stat = StatManager()

        if max_iter is None:
            max_iter = len(load)

        if temp is None:
            temp = self.cfg.TEST.TEMP

        step_fn = partial(self.step, key=key, temp=temp, train=False, visualise=vis, writer=writer)

        
        def eval_batch(n, batch):
            tag_n = tag + "_{:02d}".format(n)
            masks = step_fn(epoch, batch, tag=tag_n)
            return masks

        self.net.eval()

        def davis_mask(masks):
            masks = masks.cpu() 
            num_objects = int(masks.max())
            tmp = torch.ones(num_objects, *masks.shape)
            tmp = tmp * torch.arange(1, num_objects + 1)[:, None, None, None]
            return (tmp == masks[None, ...]).long().numpy()

        Js = {"M": [], "R": [], "D": []}
        Fs = {"M": [], "R": [], "D": []}

        timer = Timer("[Epoch {}] Validation-Seg".format(epoch))
        tag_key = "{}_{}_{:3.2f}".format(tag, key, temp)
        for n, batch in enumerate(load):
            seq_name = batch[-1][0]
            sys.stdout.flush()

            with torch.no_grad():
                masks_out = eval_batch(n, batch)
            masks_gt = davis_mask(masks_out["masks_gt"])
            masks_pred = davis_mask(masks_out["masks_pred_idx"])
            assert masks_gt.shape == masks_pred.shape

            if not tag_key is None and not self.has_vis_batch(tag_key):
                self.save_vis_batch(tag_key, batch)

            start_t = time.time()
            metrics_res = evaluate_semi((masks_gt, ), (masks_pred, ))
            J, F = metrics_res['J'], metrics_res['F']

            for l in ("M", "R", "D"):
                Js[l] += J[l]
                Fs[l] += F[l]

            msg = "{} | Im/Sec: {:.1f}".format(n, n * batch[0].shape[1] / timer.stage_elapsed())
            sys.stdout.flush()

        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']

        
        final_mean = (np.mean(Js["M"]) + np.mean(Fs["M"])) / 2.
        g_res = [final_mean, \
                 np.mean(Js["M"]), np.mean(Js["R"]), np.mean(Js["D"]), \
                 np.mean(Fs["M"]), np.mean(Fs["R"]), np.mean(Fs["D"])]

        for (name, val) in zip(g_measures, g_res):
            writer.add_scalar('{}_{:3.2f}/{}'.format(key, temp, name), val, epoch)

        return final_mean


def train(args, cfg):

    setproctitle.setproctitle("dense-ulearn | {}".format(args.run))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args, cfg)

    timer = Timer()
    def time_call(func, msg, *args, **kwargs):
        timer.reset_stage()
        val = func(*args, **kwargs)
        return val

    for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):

        
        time_call(trainer.train_epoch, "Train epoch: ", epoch)

        if epoch % cfg.LOG.ITER_VAL == 0:

            for val_set in ("val_video", ):
                time_call(trainer.validation, "Validation / {} /  Val: ".format(val_set), \
                          epoch, trainer.writer_val[val_set], trainer.valloads[val_set], tag=val_set)

            best_layer = None
            best_score = -1e10
            for val_set in ("val_video_seg", ):
                writer = trainer.writer_val[val_set]
                load = trainer.valloads[val_set]
                for layer in ("key", "res4"):
                    msg = ">>> Validation {} / {} <<<".format(layer, val_set)
                    score = time_call(trainer.validation_seg, msg, epoch, writer, load, key=layer, tag=val_set)
                    if score > best_score:
                        best_score = score
                        best_layer = layer
                
                if val_set =="val_video_seg":
                    trainer.checkpoint_best(best_score, epoch, best_layer)

        if not trainer.scheduler is None and cfg.MODEL.LR_SCHED_USE_EPOCH:
            trainer.scheduler.step()

def main():
    args = arguments(sys.argv[1:])

    
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    train(args, cfg)

if __name__ == "__main__":
    main()
