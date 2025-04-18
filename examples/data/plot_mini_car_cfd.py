"""
Visualizing computational fluid dynamics on a car
===================================================
In this example we visualize a mesh drawn from the CarCFDDataset. 
"""

# %%
# Import dependencies
# --------------------
# We first import our `neuralop` library and required dependencies.
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_mini_car

font = {'size' : 12}
matplotlib.rc('font', **font)

torch.manual_seed(0)
np.random.seed(0)

# %%
# Understanding the data 
# ----------------------
# The data in a ``MeshDataModule`` is structured as a dictionary of tensors and important scalar values encoding 
# a 3-d triangle mesh over the surface of a car. 
# Each sample includes the coordinates of all triangle vertices and the centroids of each triangle face.
# 
# In this case, the creators used OpenFOAM to simulate the surface air pressure on car geometries in a wind tunnel. 
# The 3-d Navier-Stokes equations were simulated for a variety of inlet velocities over each surface using the 
# **OpenFOAM** computational solver to predict pressure at every vertex on the mesh. 
# Each sample here also has an inlet velocity scalar and a pressure field that maps 1-to-1 with the vertices on the mesh.
# The actual CarCFDDataset is stored in triangle mesh files for downstream processing. 
# For the sake of simplicity, we've packaged a few examples of the data after processing in tensor form to visualize here:

data_list = load_mini_car()
sample = data_list[0]
print(f'{sample.keys()=}')

# %%
# Visualizing the car 
# -------------------
# Let's take a look at the vertices and pressure values.

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# By default the data is normalized into the unit cube. To get a 
# better look at it, we scale the z-direction up.
scatter = ax.scatter(sample['vertices'][:,0],sample['vertices'][:,1],
                     sample['vertices'][:,2]*2, s=2, c=sample['press']) 
ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=20, azim=150, roll=0, vertical_axis='y')
ax.set_title("Pressure over car mesh vertices")
fig.colorbar(scatter, pad=0.2, label="normalized pressure", ax=ax)
plt.draw()
# %%
# Query points  
# -------------
# Each sample in the ``CarCFDDataset`` also includes a set of latent query points on which we learn a function
# to enable learning with an FNO in the middle of our geometry-informed models. Let's visualize the queries
# on top of the car from before:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(sample['vertices'][:,0],sample['vertices'][:,1],
                     sample['vertices'][:,2]*2, s=2)
queries = sample['query_points'].view(-1, 3) # unroll our cube tensor into a point cloud
ax.scatter(queries[:,0],queries[:,1],queries[:,2]*2,s=0.4, alpha=0.5)

ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=20, azim=150, roll=0, vertical_axis='y')
ax.set_title("Query points and vertices")
fig.colorbar(scatter, pad=0.2, label="normalized pressure", ax=ax)
# %%
# Visualizing neighbor search
# In :ref:`examples/layers/plot_neighbor_search` (just a dummy ref for now) we demonstrate our neighbor search
# on a simple 2-d point cloud. Let's try that again with our points here:

from neuralop.layers.neighbor_search import native_neighbor_search
verts = sample['vertices']
query_point = queries[1000]
nbr_data = native_neighbor_search(data=verts, queries=query_point.unsqueeze(0), radius=0.15)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
neighbors = verts[nbr_data['neighbors_index']]
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2]*2, s=1, alpha=0.5)
ax.scatter(queries[:, 0], queries[:, 1], queries[:, 2]*2, s=1, alpha=0.5)
ax.scatter(neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]*2, label="neighbors of x")
ax.legend()
ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=20, azim=-20, roll=0, vertical_axis='y')
ax.set_title("Neighbor points from car for one query point")
plt.draw()

# %%
# Looking closer, without all the query points:

from neuralop.layers.neighbor_search import native_neighbor_search
verts = sample['vertices']
query_point = queries[1000]
nbr_data = native_neighbor_search(data=verts, queries=query_point.unsqueeze(0), radius=0.15)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
neighbors = verts[nbr_data['neighbors_index']]
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2]*2, s=1, alpha=0.5)
ax.scatter(neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]*2, label="neighbors of x")
ax.legend()
ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=20, azim=-120, roll=0, vertical_axis='y')
ax.set_title("Neighbor points from car for one query point")
plt.draw()