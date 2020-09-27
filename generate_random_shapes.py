#!/usr/bin/env python3
import scipy.io

import chart_studio.plotly as py
import plotly.graph_objs   as go
import numpy               as np
import scipy               as sp
import plotly.express      as px
import matplotlib.tri      as mtri

import argparse
import shutil
import glob
import os

from tqdm                  import tqdm
from sklearn.decomposition import PCA

import cufflinks
cufflinks.go_offline()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#default_model_data_dir = 'caesar'
#default_model_data_dir = 'caesar-norm-nh'
default_model_data_dir = 'caesar-norm-wsx'

#default_mesh_data_dir = 'caesar-fitted-meshes'
#default_mesh_data_dir = 'caesar-norm-nh-fitted-meshes'
default_mesh_data_dir = 'caesar-norm-wsx-fitted-meshes'

parser = argparse.ArgumentParser()
parser.add_argument( '--mesh_input_data_dir'  , type=str, default=default_mesh_data_dir   )
parser.add_argument( '--model_input_data_dir' , type=str, default=default_model_data_dir  )

parser.add_argument( '--faces_shape_model'    , type=str, default='./facesShapeModel.mat' )

parser.add_argument( '--mesh_output_data_dir' , type=str, default='./temporary/mesh_output_data'  )
parser.add_argument( '--model_output_data_dir', type=str, default='./temporary/model_output_data' )

parser.add_argument( '--n_components'         , type=int, default=20      )
parser.add_argument( '--number'               , type=int, default=1000    )
parser.add_argument( '--verbose'              , action='store_true'       )
parser.add_argument( '--output'               , type=str, default='output')

args = parser.parse_args()

model_input_data_dir  = args.model_input_data_dir
mesh_input_data_dir   = args.mesh_input_data_dir

faces_shape_model     = args.faces_shape_model

model_output_data_dir = args.model_output_data_dir
mesh_output_data_dir  = args.mesh_output_data_dir

n_components          = args.n_components
number                = args.number
verbose               = args.verbose
output                = args.output

def generateRandomShape( meanShape, eigenShape_s, eigenVariance_s ):
    randomParam_s = ( np.sqrt(eigenVariance_s) * np.random.standard_normal( len(eigenVariance_s) ) ).reshape(-1,1)
    randomShape   = meanShape + ( np.sum( randomParam_s*eigenShape_s, axis=0 ) ).reshape( meanShape.shape )
    
    return randomShape, randomParam_s

def generateRandomShapes( N, meanShape, eigenShape_s, eigenVariance_s ):
    randomParams_s   = ( np.sqrt(eigenVariance_s).reshape(1,-1) * np.random.standard_normal( N*len(eigenVariance_s) ).reshape(N, len(eigenVariance_s) ) )
    randomShape_s    = meanShape.reshape( (1,)+meanShape.shape ) + ( np.dot( randomParams_s, eigenShape_s ) ).reshape( (N,)+meanShape.shape )
    return randomShape_s, randomParams_s

def plotRandomShape( randomShape, facesShapeModel ):
    x = randomShape[:,0]; y = randomShape[:,1]; z = randomShape[:,2]
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(x, y, z, triangles=facesShapeModel, cmap=plt.cm.Spectral)
    plt.show()
    
    color_s = np.zeros(facesShapeModel.shape[0])
    
    def set_size(w,h, ax=None):
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)
    
    fig1, ax_s = plt.subplots(1,2,constrained_layout=True)
    
    ax_s[0].set_aspect('equal')
    ax_s[0].tripcolor( x, z, facesShapeModel, facecolors=color_s, edgecolors='k')
    ax_s[0].axis('off')
    set_size( 5,5,ax_s[0] )

    ax_s[1].set_aspect('equal')
    ax_s[1].tripcolor( y, z, facesShapeModel, facecolors=color_s, edgecolors='k')
    ax_s[1].axis('off')
    set_size( 5,5,ax_s[1] )
    
    plt.show()

def renderRandomShapes( randomShape, facesShapeModel ):
    x = randomShape[:,0]; y = randomShape[:,1]; z = randomShape[:,2]
    
    x_min = np.min( x ); x_max = np.max( x ); x_length = np.abs( x_max - x_min ); x_center = (x_max + x_min)/2.
    y_min = np.min( y ); y_max = np.max( y ); y_length = np.abs( y_max - y_min ); y_center = (y_max + y_min)/2.
    z_min = np.min( z ); z_max = np.max( z ); z_length = np.abs( z_max - z_min ); z_center = (z_max + z_min)/2.
    
    xy_length = max( x_length, y_length )
    
    x0_min = x_center - xy_length/2.; x0_max = x_center + xy_length/2.
    y0_min = y_center - xy_length/2.; y0_max = y_center + xy_length/2.
    z0_min = z_center -  z_length/2.; z0_max = z_center +  z_length/2.
    
    color_s = np.zeros(facesShapeModel.shape[0])
    
    fig0, ax0 = plt.subplots(1,1)
    
    ax0.set_aspect('equal'); ax0.set_xlim([x0_min, x0_max]); ax0.set_ylim([z0_min, z0_max])
    ax0.tripcolor( x, z, facesShapeModel, facecolors=color_s, edgecolors='k')
    ax0.axis('off')
    fig0.savefig('ax2_figure_0.png', bbox_inches='tight')
    
    fig1, ax1 = plt.subplots(1,1)
    ax1.set_aspect('equal'); ax1.set_xlim([y0_min, y0_max]); ax1.set_ylim([z0_min, z0_max])
    ax1.tripcolor( y, z, facesShapeModel, facecolors=color_s, edgecolors='k')
    ax1.axis('off')
    fig1.savefig('ax2_figure_1.png', bbox_inches='tight')

def saveShapesData( randomShape_s, facesShapeModel, randomParams_s, outputPath ):
    frontViewPath = os.path.join( outputPath, 'frontview' )
    sideViewPath  = os.path.join( outputPath, 'sideview'  )
    betasPath     = os.path.join( outputPath, 'betas'     )
    
    if not os.path.exists( frontViewPath ):
        os.makedirs( frontViewPath )
    else:
        filelist = glob.glob(os.path.join(frontViewPath, "*.png"))
        for f in filelist:
            os.remove(f)
    if not os.path.exists( sideViewPath ):
        os.makedirs( sideViewPath )
    else:
        filelist = glob.glob(os.path.join(sideViewPath, "*.png"))
        for f in filelist:
            os.remove(f)
    if not os.path.exists( betasPath ):
        os.makedirs( betasPath )
    else:
        filelist = glob.glob(os.path.join(betasPath, "*.txt"))
        for f in filelist:
            os.remove(f)
    
    x = randomShape_s[:,:,0]; y = randomShape_s[:,:,1]; z = randomShape_s[:,:,2]
    
    x_min = np.min( x ); x_max = np.max( x ); x_length = np.abs( x_max - x_min ); x_center = (x_max + x_min)/2.
    y_min = np.min( y ); y_max = np.max( y ); y_length = np.abs( y_max - y_min ); y_center = (y_max + y_min)/2.
    z_min = np.min( z ); z_max = np.max( z ); z_length = np.abs( z_max - z_min ); z_center = (z_max + z_min)/2.
    
    xy_length = max( x_length, y_length )
    
    x0_min = x_center - xy_length/2.; x0_max = x_center + xy_length/2.
    y0_min = y_center - xy_length/2.; y0_max = y_center + xy_length/2.
    z0_min = z_center -  z_length/2.; z0_max = z_center +  z_length/2.
    
    color_s = np.zeros(facesShapeModel.shape[0])
    
    fig, ax = plt.subplots(1,1);
    for index, (randomShape, randomParams) in tqdm( enumerate( zip( randomShape_s, randomParams_s ) ) ):
        baseFilename = '{:06d}'.format( index ) 
        
        fronViewFilename = '{0}.{1}'.format( baseFilename, 'png' )
        sideViewFilename = '{0}.{1}'.format( baseFilename, 'png' )
        betasFilename    = '{0}.{1}'.format( baseFilename, 'txt' )
        
        frontViewFilePath = os.path.join( frontViewPath, fronViewFilename )
        sideViewFilePath  = os.path.join( sideViewPath , sideViewFilename )
        betasFilePath     = os.path.join( betasPath    , betasFilename    )
        
        x = randomShape[:,0]; y = randomShape[:,1]; z = randomShape[:,2]
        
        ax.set_aspect('equal'); ax.set_xlim([x0_min, x0_max]); ax.set_ylim([z0_min, z0_max])
        ax.tripcolor( x, z, facesShapeModel, facecolors=color_s, edgecolors='k'); ax.axis('off')
        fig.savefig( frontViewFilePath, bbox_inches='tight')
        ax.cla()

        ax.set_aspect('equal'); ax.set_xlim([y0_min, y0_max]); ax.set_ylim([z0_min, z0_max])
        ax.tripcolor( y, z, facesShapeModel, facecolors=color_s, edgecolors='k'); ax.axis('off')
        fig.savefig( sideViewFilePath, bbox_inches='tight')
        ax.cla()
        
        np.savetxt( betasFilePath, randomParams, fmt='%10.5f' )
        
faces_shape_model = scipy.io.loadmat( faces_shape_model )['faces']
faces_shape_model = faces_shape_model - 1 #Indices in Matlab start from 1 in Python from 0

model_mean_points_numpy = scipy.io.loadmat( os.path.join( model_input_data_dir, 'meanShape.mat' ))['points'  ]
model_evectors_numpy    = scipy.io.loadmat( os.path.join( model_input_data_dir, 'evectors.mat'  ))['evectors']
model_evalues_numpy     = scipy.io.loadmat( os.path.join( model_input_data_dir, 'evalues.mat'   ))['evalues' ]

model_mean_points_numpy = model_mean_points_numpy.reshape( -1 )
model_evalues_numpy     = model_evalues_numpy    .reshape( -1 ) 

mat_filename_s = glob.glob(mesh_input_data_dir+'/*.mat') 

total_points_list = list()
for mat_filename in tqdm( mat_filename_s ):
    xyz_mat  = scipy.io.loadmat( mat_filename )
    try :
        total_points_list.append( xyz_mat['points'] )
    except:
        print('Couldn\'t process: ', mat_filename )

total_nbr_of_samples = len( total_points_list )
total_points_numpy   = np.array( total_points_list ).reshape( total_nbr_of_samples, -1 )
pca = PCA(n_components=50)
pca.fit_transform( total_points_numpy )

mesh_mean_points_numpy = pca.mean_
mesh_evectors_numpy    = pca.components_
mesh_evalues_numpy     = pca.explained_variance_

#meanShape       = model_mean_points_numpy.reshape(-1,3)
#eigenShape_s    = model_evectors_numpy
#eigenVariance_s = model_evalues_numpy 

meanShape       = mesh_mean_points_numpy.reshape(-1,3)
eigenShape_s    = mesh_evectors_numpy
eigenVariance_s = mesh_evalues_numpy 

eigenShape_s    = eigenShape_s   [:n_components] 
eigenVariance_s = eigenVariance_s[:n_components]

if verbose : 
    randomShape, randomParam_s = generateRandomShape( meanShape, eigenShape_s, eigenVariance_s )
    plotRandomShape( randomShape, faces_shape_model )
else :
    print("eigenVariance_s = ", eigenVariance_s)
    randomShape_s, randomParams_s = generateRandomShapes( number, meanShape, eigenShape_s, eigenVariance_s )
    saveShapesData( randomShape_s, faces_shape_model, randomParams_s, output )
