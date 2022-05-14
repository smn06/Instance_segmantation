import os
import torch
import math
import numpy as np
from PIL import Image
from matplotlib import cm

class base(object):

    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.start_epoch = 0
        self.best_score = -1e16
        self.checkpoint = Checkpoint(args.snapshot_dir, max_n = 3)

    def checkpoint_best(self, score, epoch, temp):

        if score > self.best_score:
            self.best_score = score
            self.checkpoint.checkpoint(score, epoch, temp)

            return True

        return False

    def optim(params, cfg):

        if not hasattr(torch.optim, cfg.OPT):
            raise NotImplementedError

        optim = getattr(torch.optim, cfg.OPT)

        if cfg.OPT == 'Adam':
            upd = torch.optim.Adam(params, lr=cfg.LR, betas=(cfg.BETA1, 0.999), weight_decay=cfg.WEIGHT_DECAY)

        else:
            upd = optim(params, lr=cfg.LR)

        upd.zero_grad()

        return upd
    def set_lr(optim, lr):
        for param_group in optim.param_groups:
            param_group['lr'] = lr

    def down(self, x, mode="bilinear"):
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)

        scale = min(*self.cfg.TB.IM_SIZE) / min(x.shape[-1], x.shape[-2])
        if mode == "nearest":
            x = F.interpolate(x, scale_factor=scale, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=scale, mode=mode, align_corners=True)

        return x.squeeze(1)

    def vis(self, epoch, outs, writer, tag, S = 5):

        def with_frame(image, m, alpha=0.3):
            return alpha * image + (1 - alpha) * m

        frames = outs["frames"][::S]
        frames_norm = self.denorm(frames.cpu().clone())
        frames_down = self.down(frames_norm)
        T,C,h,w = frames_down.shape

        vis = []
        vis.append(frames_down)

        if "ms_gt" in outs:
            mrgb = self._apply_cmap(outs["ms_gt"][::S].cpu(), palette_davis, rand=False)
            mrgb = self.down(mrgb)
            mrgb = with_frame(frames_down, mrgb)
            vis.append(mrgb)

        m_rgb_idx = self._apply_cmap(outs["ms_pred_idx"][::S].cpu(), palette_davis, rand=False)
        m_rgb_idx = self.down(m_rgb_idx)
        m_rgb_idx = with_frame(frames_down, m_rgb_idx)
        vis.append(m_rgb_idx)

        
        conf = self.down(outs["ms_pred_conf"][::S].cpu())
        conf_rgb = self._err_rgb(conf, cm.cmap("plasma"), frames_down, 0.3)
        vis.append(conf_rgb)

        vis = [x.float() for x in vis]
        vis = torch.cat(vis, -1)

        self._vis_grid(writer, vis, epoch, tag)

    def _vis(self, epoch, outs, T, writer, tag):
        vis = []

        def overlay(m, image, alpha=0.3):
            return alpha * image + (1 - alpha) * m

        frames_orig = outs["frames_orig"]
        frames_orig = self.denorm(frames_orig.cpu().clone())
        frames_orig = self.down(frames_orig)
        vis.append(frames_orig)

        frames = outs["frames"]
        frames_norm = self.denorm(frames.cpu().clone())
        frames_down = self.down(frames_norm)

        if "grid_m" in outs:
            val_m = outs["grid_m"]
            val_m = self.down(val_m)
            val_m = val_m.unsqueeze(1).expand(-1,3,-1,-1).cpu()

            val_m = overlay(val_m, frames_orig)
            vis.append(val_m)

        if "map_target" in outs:
            val = outs["map_target"]
            val = self._apply_cmap(val)
            val = self.down(val, "nearest")
            val = overlay(val, frames_down)
            vis.append(val)

        if "map_soft" in outs:
            val = outs["map_soft"]
            val = self._m_rgb(val)
            val = self.down(val)
            vis.append(val)

        vis.append(frames_down)

        frames2 = outs["frames2"]
        frames2_norm = self.denorm(frames2.cpu().clone())
        frames2_down = self.down(frames2_norm)
        vis.append(frames2_down)

        
        if "err_map" in outs:
            err_m = outs["err_map"]
            err_m = (err_m - err_m.min()) / (err_m.max() - err_m.min() + 1e-8)
            err_m_rgb = self._err_rgb(err_m, cmap=cm.cmap("plasma"), alpha=0.5)
            err_m_rgb = self.down(err_m_rgb)
            vis.append(err_m_rgb)

        if "aff_m1" in outs:
            aff_m = outs["aff_m1"].unsqueeze(1).expand(-1,3,-1,-1).cpu()
            aff_m = self.down(aff_m)

            aff_frames = frames_orig.clone()
            aff_frames[::T] = overlay(aff_m, aff_frames[::T], 0.5)
            vis.append(aff_frames)

            aff_m1 = self._err_rgb(outs["aff11"], cm.cmap("inferno"))
            aff_m1 = self.down(aff_m1)
            aff_m1 = overlay(aff_m1, frames_orig, 0.3)
            vis.append(aff_m1)

            aff_m2 = self._err_rgb(outs["aff12"], cm.cmap("inferno"))
            aff_m2 = self.down(aff_m2)
            aff_m2 = overlay(aff_m2, frames2_down, 0.3)
            vis.append(aff_m2)

        if "aff_m2" in outs:
            aff_m = outs["aff_m2"].unsqueeze(1).expand(-1,3,-1,-1).cpu()
            aff_m = self.down(aff_m)

            aff_frames = frames_down.clone()
            aff_frames[::T] = overlay(aff_m, aff_frames[::T], 0.5)
            vis.append(aff_frames)

            aff_m1 = self._err_rgb(outs["aff21"], cm.cmap("inferno"))
            aff_m1 = self.down(aff_m1)
            aff_m1 = overlay(aff_m1, frames_orig, 0.3)
            vis.append(aff_m1)

            aff_m2 = self._err_rgb(outs["aff22"], cm.cmap("inferno"))
            aff_m2 = self.down(aff_m2)
            aff_m2 = overlay(aff_m2, frames2_down, 0.3)
            vis.append(aff_m2)

        vis = [x.cpu().float() for x in vis]
        vis = torch.cat(vis, -1)

        self._vis_grid(writer, vis, epoch, tag, 4 * T)

    def save_vis_batch(self, key, batch):

        if self.vis_batch is None:
            self.vis_batch = {}

        if key in self.vis_batch:
            return

        batch_items = []
        for el in batch:
            el = el.clone().cpu() if torch.is_tensor(el) else el
            batch_items.append(el)

        self.vis_batch[key] = batch_items

    def has_vis_batch(self, key):
        return (not self.vis_batch is None and \
                    key in self.vis_batch)

    def _m_rgb(self, ms, image_norm=None, palette=None, alpha=0.3):

        if palette is None:
            palette = self.loader.dataset.palette

        
        ms_conf, ms_idx = torch.max(ms, 1)
        ms_conf = ms_conf - F.relu(ms_conf - 1, 0)

        ms_idx_rgb = self._apply_cmap(ms_idx.cpu(), palette, m_conf=ms_conf.cpu())
        if not image_norm is None:
            return alpha * image_norm + (1 - alpha) * ms_idx_rgb

        return ms_idx_rgb

    def _apply_cmap(self, m_idx, palette=None, m_conf=None, rand=True):

        if palette is None:
            palette = self.loader.dataset.palette

        ignore_m = (m_idx == -1).cpu()

        
        if rand:
            memsize = self.cfg.TRAIN.BATCH_SIZE * self.cfg.MODEL.GRID_SIZE**2
            m_idx = ((m_idx + 1) * 123) % memsize

        
        m = m_idx.cpu().numpy().astype(np.uint32)
        m_rgb = palette(m)
        m_rgb = torch.from_numpy(m_rgb[:,:,:,:3])
        m_rgb[ignore_m] *= 0
        m_rgb = m_rgb.permute(0,3,1,2)

        if not m_conf is None:
            
            m_rgb *= m_conf[:, None, :, :]

        return m_rgb

    def _err_rgb(self, err_m, cmap = cm.cmap('jet'), image=None, alpha=0.3):
        err_np = err_m.cpu().numpy()

        
        err_rgb = cmap(err_np)[:, :, :, :3]
        err_rgb = np.transpose(err_rgb, (0,3,1,2))
        err_rgb = torch.from_numpy(err_rgb)

        if not image is None:
            return alpha * image + (1 - alpha) * err_rgb

        return err_rgb

    def _vis_grid(self, writer, x_all, t, tag, T=1):
        
        
        bs, ch, h, w = x_all.size()
        x_all_new = torch.zeros(T, ch, h, w)
        for b in range(bs):

            x_all_new[b % T] = x_all[b]

            if (b + 1) % T == 0:
                summary_grid = vutils.make_grid(x_all_new, nrow=1, padding=8, pad_value=0.9).numpy()
                writer.add_image(tag + "_{:02d}".format(b // T), summary_grid, t)
                x_all_new.zero_()

    def vis_results(self, epoch, writer, tag, step_func):
        
        self.net.eval()

        with torch.no_grad():
            step_func(epoch, self.vis_batch[tag],train=False, vis=True,  writer=writer, tag=tag)
