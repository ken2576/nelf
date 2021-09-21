import numpy as np
from numpy.lib.utils import source
import torch

from network.u_net import PortraitRelightingNet


class SIPR(torch.nn.Module):
    def __init__(
        self, light_size
    ):
        super().__init__()
        self.light_size = light_size

        self.net = PortraitRelightingNet(light_size=light_size)

    
    ############ Training Loss  ############
    def train_loss(self, output):
        rgb_loss = torch.mean(torch.abs(output['rgb'] - output['target_rgb']))
        light_loss = torch.mean(torch.abs(
            torch.log(1 + output['source_light_pred'])
            - torch.log(1 + output['source_light'])
        ))
        return {
            'rgb_loss': rgb_loss,
            'light_loss': light_loss
        }
    
    def val_loss(self, output, **data):
        rgb_loss = torch.mean(torch.abs(output['rgb'] - data['target_image']))
        light_loss = torch.mean(torch.abs(
            torch.log(1 + output['light']) - torch.log(1 + data['source_light'])
        ))
        return {
            'rgb_loss': rgb_loss,
            'light_loss': light_loss
        }


    ############ Class Entrance ############
    def render(self, train, model, **data):
        '''
        Given the image from the source_camera,
        render the image from the target_camera.
        '''
        self.train(train)
        with torch.set_grad_enabled(train):
            mask = data['source_mask']
            target_image_pred, source_light_pred = model(
                data['source_image'].permute(2, 0, 1).unsqueeze(0),
                data['target_light'].unsqueeze(0)
            )
            target_image_pred = (target_image_pred[0] * mask).permute(1, 2, 0)

        if train:
            return {
                'rgb': target_image_pred,
                'target_rgb': data['target_image'],
                'source_light_pred': source_light_pred[0],
                'source_light': data['source_light']
            }
        else:
            return {
                'rgb': target_image_pred,
                'light': source_light_pred[0]
            }


    def forward(self, source_image, target_light):
        return self.net(source_image, target_light)
        
