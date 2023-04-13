import torch
from torch.nn import functional as F
import numpy as np
import PIL.Image
from models.network_swinir import SwinIR as net
from utils import utils_calculate_psnr_ssim as util

class SwinIR_Merger:
    def __init__(self, pretrained_model, ttt_train_checkpoints, ttt_test_checkpoints, train_images, test_images, gt_train_images, gt_test_images):
        self.pretrained_model = pretrained_model
        self.ttt_train_checkpoints = ttt_train_checkpoints
        self.ttt_test_checkpoints = ttt_test_checkpoints
        self.train_images = train_images
        self.test_images = test_images
        self.gt_train_images = gt_train_images
        self.gt_test_images = gt_test_images
        self.alphas = [i for i in range(0, 1, 0.1)]
        self.psnrs = []
        self.ssims = []
    
    def prep_image(self, path):
        image = PIL.Image.open(path)
        image = np.array(image) / 255.
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis,...]
        return torch.from_numpy(image).float()

    def prep_gt(self, lr, gt):
        lr_image = self.prep_image(lr)
        gt_image = self.prep_image(gt)
        _, _, h_old, w_old = lr_image.size()
        gt_image = gt_image[..., :h_old * 4, :w_old * 4]
        return gt_image

    def merge_weights(self, model1, model2, alpha):
        merged_model = {}
        for key in model1.keys():
            merged_model[key] = (1 - alpha) * model1[key] + alpha * model2[key]
        return merged_model

    def infer(self, merged_model, image):
        input_image = self.prep_image(image)
        with torch.no_grad():
            output = merged_model(input_image)
        return output

    def merge_and_infer(self):
        train_psnr, train_ssim = [], []
        test_psnr, test_ssim = [], []

        for alpha in self.alphas:
            for idx, (ttt_checkpoint, train_image) in enumerate(zip(self.ttt_train_checkpoints, self.train_images)):
                ttt_model = torch.load(ttt_checkpoint)
                merged_weights = self.merge_weights(self.pretrained_model.state_dict(), ttt_model.state_dict(), alpha)

                merged_model = self.pretrained_model
                merged_model.load_state_dict(merged_weights)
                merged_model.eval()

                output_image = self.infer(merged_model, train_image)
                self.gt_train_images[idx] = self.prep_gt(train_image, self.gt_train_images[idx])

                psnr_y = util.calculate_psnr(output_image, self.gt_train_images[idx], crop_border=4, test_y_channel=True)
                ssim_y = util.calculate_ssim(output_image, self.gt_train_images[idx], crop_border=4, test_y_channel=True)
                train_psnr.append((alpha, psnr_y))
                train_ssim.append((alpha, ssim_y))

        # Validation
        for alpha in self.alphas:
            for idx, (ttt_checkpoint, test_image) in enumerate(zip(self.ttt_test_checkpoints, self.test_images)):
                ttt_model = torch.load(ttt_checkpoint)
                merged_weights = self.merge_weights(self.pretrained_model.state_dict(), ttt_model.state_dict(), alpha)

                merged_model = self.pretrained_model
                merged_model.load_state_dict(merged_weights)
                merged_model.eval()

                output_image = self.infer(merged_model, test_image)
                # output_images.append(output_image)
                self.gt_test_images[idx] = self.prep_gt(test_image, self.gt_test_images[idx])
                psnr_y = util.calculate_psnr(output_image, self.gt_test_images[idx], crop_border=4, test_y_channel=True)
                ssim_y = util.calculate_ssim(output_image, self.gt_test_images[idx], crop_border=4, test_y_channel=True)
                test_psnr.append((alpha, psnr_y))
                test_ssim.append((alpha, ssim_y))

        return train_psnr, train_ssim, test_psnr, test_ssim

def main(args):

    # Set relevant directories


    # Instantiate pretrained model 


    # Instantiate merger instance


    # Run merger


    # Plot PSNR/SSIM for all alphas


    pass

if __name__ == '__main__':
    # set up args & argparser

    # call main