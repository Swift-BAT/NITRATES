"""
This fine allows the user to configure where the data is located on their system and keeps the changes persistent through the whole code.
"""

import os


NITRATES_RESP_DIR="/Users/tparsota/Documents/BAT_SCRIPTS/NITRATES_BAT_RSP_FILES/" # env variable can be used or this can be set
if NITRATES_RESP_DIR is None:
    NITRATES_RESP_DIR=os.getenv('NITRATES_RESP_DIR')
if NITRATES_RESP_DIR is None:
    # if NITRATES_RESP_DIR is not set here or as an env var then
    # it's assumed to be in the current working direc
    NITRATES_RESP_DIR='.'
    
    # ray traces directory
rt_dir = os.path.join(NITRATES_RESP_DIR,'ray_traces_detapp_npy')
# Resp Table with direct response directory
RESP_TAB_DNAME = os.path.join(NITRATES_RESP_DIR,'resp_tabs_ebins')
# Directory with Compton + Flor response
COMP_FLOR_RESP_DNAME = os.path.join(NITRATES_RESP_DIR,'comp_flor_resps')
# Directory with Flor response only
HP_FLOR_RESP_DNAME = os.path.join(NITRATES_RESP_DIR,'hp_flor_resps')
# DPI with st per det exposed to sky
solid_angle_dpi_fname = os.path.join(NITRATES_RESP_DIR,'solid_angle_dpi.npy')


#get the directory that the data directory is located in
dir = os.path.split(__file__)[0]

# Directory with the element cross section data files
ELEMENT_CROSS_SECTION_DNAME = os.path.join(dir, "data",'element_cross_sections')

# Table of bright known sources from the Trans Monitor
bright_source_table_fname = os.path.join(dir, "data",'bright_src_cat.fits')

EBINS0 = [15., 24., 35., 48., 64., 84., 120., 171.5, 245.]
EBINS1 = [24., 35., 48., 64., 84., 120., 171.5, 245., 350.]


###########################
###   Old Stuff ###
###########################

#bat_ml_dir='/storage/work/jjd330/local/bat_data/BatML/'
ftool_wrap='/storage/work/jjd330/local/bat_data/BatML/HeasoftTools/run_ftool.sh' #referenced in gen_tools.py

# rt_dir='/gpfs/scratch/jjd330/bat_data/ray_traces_detapp_npy/'
fp_dir = '/gpfs/scratch/jjd330/bat_data/footprints_npy/'
# #drm_dir='/gpfs/scratch/jjd330/bat_data/drms/'
# drm_dir='/storage/work/jjd330/local/bat_data/drms/'
# drm_quad_dir='/gpfs/scratch/jjd330/bat_data/drms4quads/'
# solid_angle_dpi_fname='/storage/work/jjd330/local/bat_data/solid_angle_dpi.npy'
# bright_source_table_fname='/storage/work/jjd330/local/bat_data/BatML/bright_src_cat.fits'

#HEADAS="/storage/work/jjd330/heasoft/heasoft-6.21/x86_64-unknown-linux-gnu-libc2.12"
#HEADAS_INIT="/storage/work/jjd330/heasoft/heasoft-6.21/x86_64-unknown-linux-gnu-libc2.12/headas-init.sh"
#CALDB="/storage/work/jjd330/caldb_files"
#CALDB_INIT="/storage/work/jjd330/caldb_files/software/tools/caldbinit.sh"

# EBINS0 = [14., 24., 36.3, 55.4, 80.0, 120.7]
# EBINS1 = [24., 36.3, 55.4, 80.0, 120.7, 194.9]

quad_dicts = {'all':{'quads':[0,1,2,3],
                     'drm_fname':'drm_0.200_0.150_.fits',
                    'imx':0.2, 'imy':0.15, 'id':0},
                    'left':{'quads':[0,1],
                    'drm_fname':'drm_1.000_0.150_.fits',
                    'imx':1.0, 'imy':0.15, 'id':1},
                    'top':{'quads':[1,2],
                    'drm_fname':'drm_-0.000_-0.500_.fits',
                    'imx':0.0, 'imy':-0.5, 'id':2},
                    'right':{'quads':[2,3],
                    'drm_fname':'drm_-1.000_0.150_.fits',
                    'imx':-1.0, 'imy':0.15, 'id':3},
                    'bottom':{'quads':[3,0],
                    'drm_fname':'drm_-0.000_0.450_.fits',
                    'imx':0.0, 'imy':0.45, 'id':4},
                    'quad0':{'quads':[0],
                    'drm_fname':'drm_1.000_0.500_.fits',
                    'imx':1.0, 'imy':0.5, 'id':5},
                    'quad1':{'quads':[1],
                    'drm_fname':'drm_0.800_-0.400_.fits',
                    'imx':0.8, 'imy':-0.4, 'id':6},
                    'quad2':{'quads':[2],
                    'drm_fname':'drm_-0.750_-0.450_.fits',
                    'imx':-0.75, 'imy':-0.45, 'id':7},
                    'quad3':{'quads':[3],
                    'drm_fname':'drm_-1.100_0.500_.fits',
                    'imx':-1.1, 'imy':0.5, 'id':8}
                }

