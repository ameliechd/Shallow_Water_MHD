"""
Plot map outputs.

Usage:
    plot_2d_map.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames_2d_mhd]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm

#this has been changed to match the mhd code, see commments to change back

def main(filename, start, count, output):
    ''' Function that does the plotting and mapping for a 2D map.'''

    #plot settings 
    task = 'geopotential'
    task_velocity = 'velocity'
    cmap = plt.cm.viridis #nipy_spectral for Prez-Becker colors
    dpi = 100
    figsize = (9,6)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    meter = 1.0
    a = 8.2e7*meter #radius of HJ

    #opening files and plotting results 
    with h5py.File(filename, mode ='r') as file: 
        dset = file['tasks'][task]
        dset_vector = file['tasks'][task_velocity] #for arrows

        if dset.ndim == 4: #vector field-like (ex: velocity)
            phi = dset.dims[2][0][:].ravel() /a  #longitude, DIVISION BY a
            theta = dset.dims[3][0][:].ravel()/a #latitude, used to be colatitude 
        else:
            phi = dset.dims[1][0][:].ravel() /a#longitude , DIVISION BY a
            theta = dset.dims[2][0][:].ravel() /a#ltitude, used to be colatitude
            
        #from colatitude to latitude (and degree units)
        lat = np.degrees(theta) #used to be np.degrees(np.pi/2 - theta)
        lon = np.degrees(phi)
        Lon, Lat= np.meshgrid(lon, lat, indexing ='ij') 

        #vector version
        phi_vec =dset_vector.dims[2][0][:].ravel()/a
        theta_vec = dset_vector.dims[3][0][:].ravel()/a
        lat_vec = np.degrees(theta_vec) # used to be lat_vec = np.degrees(np.pi/2 - theta_vec)
        lon_vec = np.degrees(phi_vec)
        Lon_vec, Lat_vec = np.meshgrid(lon_vec, lat_vec, indexing ='ij')

        #looping over write indices
        for index in range(start, start+count):
            data_slices = (index, slice(None), slice(None))

            u_phi = dset_vector[index, 0, :, :]
            u_theta = dset_vector[index, 1, :, :]

            if dset.ndim == 4:
                #magnitude cmputation for plotting 
                data = np.sqrt(u_phi**2+u_theta**2)
            else:
                data = dset[data_slices] 
            
            #for plotting the arrows
            stride = 6 
            Lon_sub = Lon_vec[::stride, ::stride]
            Lat_sub = Lat_vec[::stride, ::stride]
            u_theta_sub = np.roll(u_theta, u_theta.shape[0]//2, axis=0)[::stride, ::stride]
            u_phi_sub = np.roll(u_phi, u_phi.shape[0]//2, axis=0)[::stride, ::stride]

            #color 
            vmin = np.min(data)
            vmax = np.max(data)
            norm=Normalize(vmin=vmin, vmax=vmax)

            fig, ax = plt.subplots(figsize=figsize)
            pcm = ax.pcolormesh(Lon, Lat, np.roll(data, data.shape[0]//2, axis=0), cmap=cmap, shading='auto')
            fig.colorbar(pcm, ax = ax, orientation = 'vertical', pad = 0.05, location = 'right')
            fig.suptitle(task)

            #overlay velocity arrows 
            ax.quiver(Lon_sub, Lat_sub, u_phi_sub, u_theta_sub, scale_units='xy', width =0.002, color ='k') #ax.quiver(Lon_sub, Lat_sub, u_theta_sub, -u_phi_sub, scale_units='xy', width =0.002, color ='k')

            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
            ax.set_xlim(-180,180) #used to be ax.set_xlim(0,360)
            ax.set_ylim(-90,90)

            write_num = file['scales/write_number'][index]
            savename = savename_func(write_num)
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            plt.close(fig)

        

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    import shutil 

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if output_path.exists():
                #Clear existing contents 
                for item in output_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

        else:
            output_path.mkdir()
            
    post.visit_writes(args['<files>'], main, output=output_path)