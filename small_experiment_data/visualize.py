#!/usr/bin/env python3
#written by Leon Zueger, Eth Zurich, April 2020

import os
import numpy as np
import pytorch3d as p3
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import sample_points_from_meshes #make point cloud from mesh
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pyntcloud import PyntCloud
import pandas as pd
import torch
from random import sample

#def load_obj_file(file_obj,load_textures=True):
#    obj= pytorch3d.io.load_obj(file_obj, load_textures)
#    return obj
'''
def load_obj_file_as_mesh(file_obj,load_textures=True):
    obj= pytorch3d.io.load_objs_as_meshes(file_obj, load_textures)
    return obj

def convert_mesh_to_pointcloud(mesh,point_number):


    return pointcloud


def convert_pointcloud_to_mesh(point_cloud,...):

    return mesh
'''
#taken from dolhpin example and adapted, Pytorch3D
def plot_pointcloud_from_mesh(mesh,num_points, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, num_points)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()
    #TODO: introduce system to sample points
def plot_pointcloud(pc,title=""):
    # Sample points uniformly from the surface of the mesh.
    #x, y, z = points.clone().detach().cpu().squeeze().unbind(1) 
    x=pc[:,0] 
    y=pc[:,1]   
    z=pc[:,2]    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

#TODO: fix directories such that models cn be opened and saved in another folder
if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Set the device
    device = torch.device("cpu") #this could be done by GPU to speed up
    #BASE_DIR=os.path.join(ROOT_DIR,'test_PointClouds')
    
    #import latent_code saved in .pth file
    latent_code = torch.load('d00391a3bec2a9456a44fdd49dec8069.pth')



    #lamp_vert, lamp_faces, lamp_aux = load_obj(BASE_DIR + '\model_normalized.obj',False) #False --> no texture file required
    #faces_idx= lamp_faces.verts_idx.to(device)
    #lamp_mesh=Meshes(verts=[lamp_vert],faces=[faces_idx])
    #lamp_point_cloud=sample_points_from_meshes(lamp_mesh,10) #samples 10 points from mesh

    #import point cloud in .txt format
    #airplane= np.loadtxt("airplane_0005_modelnet.txt", delimiter= ',')
    #airplane_pc=airplane[:,0:3]
    #plot_pointcloud(airplane_pc, "A wonderul airplane")
    
    #save_obj("nice_lamp_pc",lamp_vert,faces_idx) 

    #plot_pointcloud_from_mesh(lamp_mesh, 200, "A nice lamp")
    #import point cloud directly from .obj file
    #lamp_pc = PyntCloud.from_file("nice_lamp_pc.obj")
    #lamp_points is a pandas DataFrame
    #lamp_points=lamp_pc.points
    #print(lamp_points)
    print(latent_code.shape)
    print(latent_code[0,0,:])