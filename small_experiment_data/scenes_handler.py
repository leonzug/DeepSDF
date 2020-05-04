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
from visualize import plot_pointcloud

def get_statistics (pointcloud):
    '''
    inputs: 
    -pointclouds, dimension [N,3]:
    returns:
    -dictionary with information about centroid, 
    '''
    statistics= {}
    
    statistics["points_number"]=np.shape(pointcloud)[0]
    statistics["centroid"]=np.mean(pointcloud)
    statistics["maxima"]= np.amax(pointcloud,axis=0)
    statistics["minima"]=np.amin(pointcloud,axis=0)
    return statistics


#TODO: do this for the entire point cloud
def divide_and_conquer(pointcloud,number_of_cells,index):
    '''
    inputs: 
    -pointclouds, dimension [N,3]:
    -number_of_cells, array with number of desired cells for each dimension: [cells_x, cells_y, cells_z]
    -index: [i,j,k]: which array subgrid we want to return (which subscene)
    returns:
    -subgrid element [i,j,k]
    '''
    stats=get_statistics(pointcloud)
    pc_physicalsize=stats["maxima"]-stats["minima"]
    cell_length=np.divide(pc_physicalsize,number_of_cells)

    x_bins=np.linspace(stats["minima"][0],stats["maxima"][0],number_of_cells[0]+1)
    y_bins=np.linspace(stats["minima"][1],stats["maxima"][1],number_of_cells[1]+1)
    z_bins=np.linspace(stats["minima"][2],stats["maxima"][2],number_of_cells[2]+1)

    #<= and >= needed because some points are on the "minima" or "maxima" "limit"
    assignments_x=((pointcloud[:,0]>=x_bins[index[0]]) & (pointcloud[:,0]<=x_bins[index[0]+1]))
    assignments_y=((pointcloud[:,1]>=y_bins[index[1]]) & (pointcloud[:,1]<=y_bins[index[1]+1]))
    assignments_z=((pointcloud[:,2]>=z_bins[index[2]]) & (pointcloud[:,2]<=z_bins[index[2]+1]))

    assignments=assignments_x & assignments_y & assignments_z
    grid_element=pointcloud[assignments]
    return grid_element

def normalize_pc(pointcloud,a,b):
    if np.size(pointcloud)==0:
        print("Pointcloud to normalize is empty. Empty pointcloud is returned.")
        return pointcloud
    stats=get_statistics(pointcloud)
    stacked_min=np.vstack([stats["minima"] for _ in range(stats["points_number"])])
    pointcloud_normalized=a*np.ones((stats["points_number"],3))+np.divide((pointcloud-stacked_min)*(b-a),stats["maxima"]-stats["minima"])
    return pointcloud_normalized


        
if __name__ == "__main__":
    device = torch.device("cpu")
    #import point cloud directly from .obj file
    ind, faces,  aux = load_obj("unknown.obj", False)
    faces= faces.verts_idx.to(device)
    mesh=Meshes(verts=[ind],faces=[faces])
    pointcloud=sample_points_from_meshes(mesh,10000)
    pointcloud=pointcloud[0,:,:]

    #plot_pointcloud(pointcloud,"A beautiful unknown pointcloud")
    
    #TODO: this is with PyntCloud, leave out for now
    #save_obj("unknown_saved.obj",ind,faces) 
    #pointcloud2= PyntCloud.from_file("unknown_saved.obj")

    #parameters that dictate how to divide the scene
    cells=np.array([2,2,200])
    index=np.array([0,1,1])

    pointcloud=pointcloud.numpy()
    
    pointcloud=np.loadtxt("pointcloud_test.txt") #loas previously saved pc so that we don't resample every time
    plot_pointcloud(pointcloud,"A beautiful unknown pointcloud")

    division=divide_and_conquer(pointcloud,cells,index)

    division=np.array(division)
    norm_pc=normalize_pc(division,-1,1)
    plot_pointcloud(division,"A beautiful divided pointcloud")
    plot_pointcloud(norm_pc,"A beautiful divided pointcloud")



    #TODO: normalize PC!
    
