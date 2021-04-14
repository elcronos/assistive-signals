#!/usr/bin/env python
# coding: utf-8

#######################################################
# IMPORTS
#######################################################
import os,sys
import torch
from torch import nn
import numpy as np
from torchvision import models as model
from advertorch.utils import NormalizeByChannelMeanStd
from tqdm.notebook import tqdm
from PIL import Image
from plot_image_grid import image_grid
from robustness import model_utils
from robustness.datasets import ImageNet
import matplotlib.pyplot as plt
from termcolor import colored

import pathlib
current_path = pathlib.Path().absolute()

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

#######################################################
# FUNCTIONS & HELPERS
#######################################################
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def create_folder(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

#######################################################
# Initialize Assistive-Based Optimization
#######################################################
class AssistiveTexturization():
    def __init__(self,  model_name, n_views, inter_cam, lights, image_size=224):
        super().__init__()

        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_views    = n_views
        self.image_size = image_size
        self.model_name = model_name
        self.inter_cam  = inter_cam
        self.lights     = lights.to(self.device)
        #########################################################
        #    MODELS: Pretrained
        #########################################################
        vgg        = model.vgg16(pretrained=True)
        resnet     = model.resnet50(pretrained=True)
        densenet   = model.densenet121(pretrained=True)
        squeezenet = model.squeezenet1_0(pretrained=True)
        shufflenet = model.shufflenet_v2_x1_0(pretrained=True)
        mobilenet  = model.mobilenet_v2(pretrained=True)


        self.networks = {
            'vgg': vgg,
            'resnet': resnet,
            'densenet': densenet,
            'squeezenet': squeezenet,
            'mobilenet': mobilenet,
            'shufflenet': shufflenet,
        }

        #########################################################
        #    NORMALIZE
        #########################################################
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = NormalizeByChannelMeanStd(mean, std)


    def get_model(self, model_name):
        return nn.Sequential(self.normalize, self.networks[model_name].eval()).to(self.device)

    def _get_renderers(self):
        # Cameras
        dist, elev_start, elev_end, azim_start, azim_end = self.inter_cam
        elev = torch.linspace(elev_start, elev_end, self.n_views).to(self.device)
        azim = torch.linspace(azim_start, azim_end, self.n_views).to(self.device)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R.to(self.device), T=T.to(self.device)).to(self.device)
        # Differentiable soft renderer using per vertex RGB colors for texture

        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(device=self.device,
                cameras=cameras,
                lights=self.lights)
        )

        return renderer, R,T

    def texturize(self, init_color, path_current, obj_filename, target, n_iter=500, lr=1e-2,show_every=100):

        #torch.autograd.set_detect_anomaly(True)
        target_batch =  torch.LongTensor([target]*self.n_views).to(self.device)

        # Load Object
        # Initialize Sphere (Source Mesh)
        src_mesh = load_objs_as_meshes([obj_filename], device=self.device)
        # We scale normalize and center the target mesh to fit in a sphere of radius 1
        # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
        # to its original center and scale.  Note that normalizing the target mesh,
        # speeds up the optimization but is not necessary!
        verts = src_mesh.verts_packed()
        N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        src_mesh.offset_verts_(-center.expand(N, 3))
        src_mesh.scale_verts_((1.0 / float(scale)));

        # We will learn to deform the source mesh by offsetting its vertices
        # The shape of the deform parameters is equal to the total number of vertices in
        # src_mesh
        verts_shape = src_mesh.verts_packed().shape
        deform_verts = torch.full(verts_shape, 0.0, device=self.device, requires_grad=True)

        # We will also learn per vertex colors for our sphere mesh that define texture
        # of the mesh
        rgb_color = torch.FloatTensor(init_color) / 255.
        verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=self.device, requires_grad=False)
        verts_rgb[...,:] =  torch.FloatTensor(rgb_color).to(self.device)
        verts_rgb.requires_grad = True

        opt = torch.optim.Adam([verts_rgb], lr=lr)
        # Loss Function
        criterion = torch.nn.CrossEntropyLoss()

        #########################################################
        #    PREPARE OPTIMIZATION LOOP
        #########################################################
        net = self.get_model(self.model_name)
        loop = tqdm(range(n_iter))
        renderer, R, T = self._get_renderers()
        target_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        # Show original model
        # Show Baseline
        with torch.no_grad():
            meshes = src_mesh.extend(self.n_views)
            images_predicted = renderer(meshes, cameras=target_cameras, lights=self.lights)
            # Fix Problem with NaN values
            images_predicted[torch.isnan(images_predicted)] = 1
            predicted_rgb = images_predicted[..., :3].permute(0,3,1,2)
            output = net(predicted_rgb)
            pred = nn.functional.softmax(output, dim=1).topk(1)
            cls = list(torch.flatten(pred[1]).cpu().numpy())
            scores = list(torch.flatten(pred[0]).cpu().numpy())
            res = {f'Image_{i}_Pred:{cls[i]}': scores[i] for i in range(len(cls))}
            print('Prediction Baseline')
            for i in range(len(cls)):
                color = "red" if int(cls[i]) != int(target) else "green"
                print(f'Prediction Image_{i}_Pred:{cls[i]}, Score:', colored(scores[i],color))
            image_grid(images_predicted.cpu().numpy(), rows=1, cols=self.n_views, rgb=True)
            plt.show()

        for i in loop:
            loss = torch.tensor(0.0, device=self.device)
            # Initialize optimizer
            opt.zero_grad()

            # Deform the mesh
            new_src_mesh = src_mesh.offset_verts(deform_verts)

            # Add per vertex colors to texture the mesh
            new_src_mesh.textures = TexturesVertex(verts_features=torch.clamp(verts_rgb,0.0,1.0))

            #########################################################
            #    MULTIVIEW OPTIMIZATION
            #########################################################
            images_predicted = renderer(new_src_mesh.extend(self.n_views), cameras=target_cameras, lights=self.lights)

            # image from our dataset
            predicted_rgb = images_predicted[..., :3].permute(0,3,1,2)
            pred = net(predicted_rgb)

            # Calculate Loss
            loss += criterion(pred, target_batch).mean()
            # Print the losses
            loop.set_description(f"total_loss = {loss}")

            # Optimization step
            loss.backward()
            opt.step()

            with torch.no_grad():
                if i == 0:
                    for i,image in enumerate(images_predicted.detach().cpu().numpy()[...,:3]):
                        im = Image.fromarray((image * 255).astype(np.uint8))
                        im.save(f"{path_current}/original_{i}.png")

        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        # Add per vertex colors to texture the mesh
        new_src_mesh.textures = TexturesVertex(verts_features=torch.clamp(verts_rgb,0.0,1.0))

        with torch.no_grad():
            meshes = new_src_mesh.extend(self.n_views)
            images_predicted = renderer(meshes, cameras=target_cameras, lights=self.lights)
            # Fix Problem with NaN values
            images_predicted[torch.isnan(images_predicted)] = 1
            predicted_rgb = images_predicted[..., :3].permute(0,3,1,2)
            output = net(predicted_rgb)
            pred = nn.functional.softmax(output, dim=1).topk(1)
            cls = list(torch.flatten(pred[1]).cpu().numpy())
            scores = list(torch.flatten(pred[0]).cpu().numpy())
            print('Prediction Full Assistive Texture - Robust Design')
            for i in range(len(cls)):
                color = "red" if int(cls[i]) != int(target) else "green"
                print(f'Prediction Image_{i}_Pred:{cls[i]}, Score:', colored(scores[i],color))

            image_grid(images_predicted.cpu().numpy(), rows=1, cols=self.n_views, rgb=True)

            plt.show()

        return new_src_mesh


def run_texturization(settings):
    lr = settings['lr']
    n_iter = settings['n_iter']
    device = settings['device']
    n_views = settings['n_views']
    image_size = settings['image_size']

    #Loop Meshes
    for mesh_name, params in settings['meshes'].items():
        # Loop Classifiers
        for model_name in settings['model_names']:
            # Interpolation Camera settings
            inter_cam = settings['meshes'][mesh_name]['inter_cam']
            target = settings['meshes'][mesh_name]['target']
            init_color = settings['meshes'][mesh_name]['init_color']

            # Camera settings
            dist, elev_start, elev_end, azim_start, azim_end = inter_cam
            elev = torch.linspace(elev_start, elev_end, n_views).to(device)
            azim = torch.linspace(azim_start, azim_end, n_views).to(device)
            R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
            cameras = FoVPerspectiveCameras(device=device, R=R.to(device), T=T.to(device)).to(device)
            # Differentiable soft renderer using per vertex RGB colors for texture
            sigma = 1e-4
            raster_settings_soft = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # Place a point light in front of the object. As mentioned above, the front of
            # the cow is facing the -z direction.
            lights = PointLights(device=device, location=[[10.0, 10.0,10.0]])

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings_soft
                ),
                shader=SoftPhongShader(device=device,
                    cameras=cameras,
                    lights=lights))


            print('*'*50)
            print(f'{mesh_name} - {model_name}')
            print('*'*50)
            print('Loading models...')
            with HiddenPrints():
                texture = AssistiveTexturization(model_name, n_views, inter_cam, lights)

            path = f'{current_path}/outputs/{mesh_name}/'
            # Create Folder
            create_folder(path)
            obj_filename = f'{os.getcwd()}/meshes/{mesh_name}/{mesh_name}.obj'
            output_mesh = texture.texturize(init_color, path, obj_filename, target, n_iter, lr)
            output_meshes = output_mesh.extend(n_views)
            images_predicted = renderer(output_meshes, cameras=cameras, lights=lights)

            # Save Images
            for i,image in enumerate(images_predicted.detach().cpu().numpy()[...,:3]):
                im = Image.fromarray((image * 255).astype(np.uint8))
                im.save(f"{path}/output_{i}_{model_name}.png")
