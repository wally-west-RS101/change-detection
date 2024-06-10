import os

import torch

from misc.imutils import save_image
from models.networks import *


class CDEvaluator():

    def __init__(self, cfg):

        self.n_class = cfg.n_class
        # define G
        self.embed_dim = cfg.embeddim
        self.net_G = define_G(embed_dim = self.embed_dim, gpu_ids=0)

        self.device = torch.device("cuda:0")

        

        self.checkpoint_path = cfg.checkpoint_path

        self.pred_dir =cfg.output_folder
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self):

        if os.path.exists(self.checkpoint_path):
            # load the entire checkpoint
            checkpoint = torch.load(self.checkpoint_path,
                                    map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

        return self.net_G


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        self.G_pred = self.net_G(img_in1, img_in2)[-1]
        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)

