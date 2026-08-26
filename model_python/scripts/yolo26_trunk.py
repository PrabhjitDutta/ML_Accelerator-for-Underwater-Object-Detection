                      
"""Trunk / decode split of the end2end YOLO26s.

The end2end Detect head's decode (`_inference`: anchor grid via torch.full, dist2bbox,
strides, top-k) is (a) untraceable by NNCF's torch quantizer and (b) not a conv op you
put on an FPGA fabric. This wrapper exposes the **trunk** = backbone + neck + the head's
pred-convs (one2many `cv2/cv3` and one2one `one2one_cv2/one2one_cv3`), returning the raw
per-scale prediction maps [B, 4*reg_max + nc, H, W]. That trunk is:
  * a clean conv-only graph NNCF CAN trace/quantize (enabling QAT), and
  * exactly what the HLS accelerator computes; the decode + top-k run on the host (ARM PS).

Helpers rebuild the ultralytics E2E loss inputs (for QAT) and run the original decode
(for eval), so the split is loss- and metric-faithful.
"""
import torch
import torch.nn as nn


class Yolo26Trunk(nn.Module):
    """Runs layers 0..22 + the Detect pred-convs; returns flat per-scale raw maps.

    forward(x) -> tuple of 2*nl tensors: (o2m_0..o2m_{nl-1}, o2o_0..o2o_{nl-1}),
    each [B, 4*reg_max + nc, H_i, W_i]. one2one (o2o_*) is the inference/deploy branch.
    """

    def __init__(self, detection_model):
        super().__init__()
        self.model = detection_model.model                                      
        self.save = detection_model.save
        self.det = self.model[-1]                           
        self.nl = self.det.nl
        self.reg_max = self.det.reg_max
        self.nc = self.det.nc
        self.no = self.reg_max * 4 + self.nc

    def feats(self, x):
        """Replicate DetectionModel._predict_once, return the nl maps fed into Detect."""
        y = []
        for m in self.model[:-1]:                                              
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        det = self.det
                                                                                     
        return [x if j == -1 else y[j] for j in det.f]

    def forward(self, x):
        f = self.feats(x)
        d = self.det
        o2m = [torch.cat((d.cv2[i](f[i]), d.cv3[i](f[i])), 1) for i in range(self.nl)]
        o2o = [torch.cat((d.one2one_cv2[i](f[i]), d.one2one_cv3[i](f[i])), 1) for i in range(self.nl)]
        return tuple(o2m) + tuple(o2o)

                                                              
    def split(self, out):
        """tuple(2*nl) -> (one2many_maps, one2one_maps) lists of per-scale [B,no,H,W]."""
        return list(out[: self.nl]), list(out[self.nl:])

    def loss_preds(self, out):
        """Rebuild the E2EDetectLoss input dict from the trunk's raw maps.
        E2EDetectLoss expects {'one2many': feats_list, 'one2one': feats_list}."""
        o2m, o2o = self.split(out)
        return {"one2many": o2m, "one2one": o2o}

    @torch.no_grad()
    def decode(self, out):
        """Host-side decode of the one2one branch -> final (B, max_det, 6) detections,
        reusing the ORIGINAL Detect._inference + end2end postprocess (FP, off-fabric)."""
        _, o2o = self.split(out)
        d = self.det
        d_was_training = d.training
        d.eval()
                                                                                         
                                                                                         
        bs = o2o[0].shape[0]
        boxes = torch.cat([m[:, : 4 * self.reg_max].view(bs, 4 * self.reg_max, -1) for m in o2o], -1)
        scores = torch.cat([m[:, 4 * self.reg_max:].view(bs, self.nc, -1) for m in o2o], -1)
                                                                                             
                                                                                           
        y = d._inference({"boxes": boxes, "scores": scores, "feats": o2o})
        out_dets = d.postprocess(y.permute(0, 2, 1))                                     
        if d_was_training:
            d.train()
        return out_dets
