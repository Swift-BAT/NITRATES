# Non-Imaging Transient Reconstruction And TEmporal Search (NITRATES)

This repo contains the full NITRATES pipeline for maximum likelihood-driven discovery and localization of Gamma Ray Bursts in the Neil Gehrels Swift Observatory's Burst Alert Telescope (BAT) instrument. 

A description of the method can be found in [DeLaunay & Tohuvavohu (2021)](https://arxiv.org/abs/2111.01769). We ask scientific users of this code to cite the paper.

The BAT instrumental response functions necessary for this pipeline can be found in [this Zenodo community](https://zenodo.org/communities/swift-bat).

This codebase is under active cleanup, and development, and at present time simply presents a snapshot of the entire autonomous pipeline (from listeners to results). Readability is low. \
We welcome questions, comments, issues, and pull requests.

# Response Files

There are many internal data files (mostly used for generating the detector response) used in the codebase. Some are found in this repo and others are found in the [Zenodo community](https://zenodo.org/communities/swift-bat). 

The simplest way to organize these files is to put all of them into a single directory, and setting the path to that directory as an environment variable named `NITRATES_RESP_DIR`. If `NITRATES_RESP_DIR` is set and all of the response files are placed in there with their original file names (and directory structure for tarred directories) then `config.py` should be able to find the full path to all of the necessary files. `NITRATES_RESP_DIR` can also be hard coded into the `config.py` file instead of being set as an env variable. 

In the [Zenodo community](https://zenodo.org/communities/swift-bat), the [Swift-BAT Response Files for NITRATES](https://zenodo.org/record/5634207) dataset contains sevaral .tar.gz files that each contain a folder of data files to be downloaded and extracted into `NITRATES_RESP_DIR`. In order to perform the NITRATES likelihood analysis and run the example notebook `Example_LLH_setup_fixed_dirs.ipynb`, the tarred directories: `resp_tabs_ebins.tar.gz`, `comp_flor_resps.tar.gz`, `hp_flor_resps.tar.gz` need to be downloaded and extracted into `NITRATES_RESP_DIR`, along with the file `solid_angle_dpi.npy`. The other files either contain more energy bins or are used only for the seeding analyses. 

The .tar.gz files in the datasets [Swift-BAT Response Files for NITRATES: Forward Ray Tracings at IMX > 0](https://zenodo.org/record/5639481) and [Swift-BAT Response Files for NITRATES: Forward Ray Tracings at IMX < 0](https://zenodo.org/record/5639084) also need to be downloaded and extracted into `NITRATES_RESP_DIR`. These are the forward ray tracing files and are a few hundred GBs uncompressed. They're split up into seperate tarred files do to size limits, but should end up in the same directory, `NITRATES_RESP_DIR/ray_traces_detapp_npy/`. 

The `bright_src_cat.fits` file and the `element_cross_sections` folder in this repo should also be copied into `NITRATES_RESP_DIR`. 

Paths to these files can instead be given as arguments to some of the analysis objects, such as Source_Model_InOutFoV(). 

# Current Full Pipeline Orchestration Scripts

`run_stuff_grb2.sh`
* Used to launch the full targeted analysis. 
  * Runs `mkdb.py`, `do_data_setup.py`, `do_full_rates.py`, then `do_manage2.py`.
  * The first arg is the trigger time, second arg is the Name of the trigger, and the optional third arg is the minimum duration to use

`mkdb.py` 
* Creates a sqlite DB that contains the trigger time and important file names.
* DB not used much in the analysis, used to be used to store results and is kind of a relic now.

`do_data_setup.py`
* Gathers the event, attitude, and enabled detectors files.
* Chooses which dets to mask, based on any hot or cold dets or any det glitches.
* Makes a "filtered" event file that has the events removed outside the usable energy range or far away from the analysis time.
* Adds a GTI table to the event file for when Swift not slewing and no multi-detector glitches.
* Also makes a partial coding image if there's a usable set of HEASOFT tools.

`do_full_rates.py`
* Runs the full rates analysis to pick time bins as seeds for the full likelihood analysis.

`do_manage2.py` 
* Manages the rest of the analysis. Submits jobs to the cluster, organizes results, and emails out top results.
  * First submits a job for the bkg fit to off-time data. 
  * Then submits several jobs for the split detector rates analysis. 
  * Gathers the split rates results and makes the final set of position and time seeds. 
  * Assigns which jobs will processes which seeds and writes them to rate_seeds.csv (for inside FoV jobs) and out_job_table.csv (for out of FoV jobs). 
  * Submits several jobs to the cluster for both inside FoV and outside FoV analysis. 
  * Gathers results and emails out top results when all of the jobs are done.

`do_bkg_estimation_wPSs_mp2.py` 
* Script to perform the bkg fit to off-time data. 
* Run as a single job, usually with 4 procs.

`do_rates_mle_InOutFoV2.py`
* Script to perform the split rates analysis.
* Run as several single proc jobs.

`do_llh_inFoV4realtime2.py`
* Script to perform the likelihood analysis for seeds that are inside the FoV. \
* Run as several single proc jobs.

`do_llh_outFoV4realtime2.py`
* Script to perform the likelihood analysis for seeds that are outside the FoV. \
* Run as several single proc jobs.


# Important Modules 

`LLH.py`
* Has class and functions to compute the LLH
* The `LLH_webins` class handles the data and LLH calculation for a given model and paramaters
  * It takes a model object, the event data, detmask, and start and stop time for inputs 
  * Converts the event data within the start and stop time into a 2D histogram in det and energy bins
  * Can then compute the LLH for a given set of paramaters for the model
  * Can do a straight Poisson likelihood or Poisson convovled with a Gaussian error

`minimizers.py`
* Funtctions and classes to handle numerically minimizing the NLLH
* Most minimizer objects are subclasses of `NLLH_Minimizer`
  * Contains functions for doing parameter transformations and setting bounds
  * Also handles mapping the tuple of paramter values used for a standard scipy minimizer to the dict of paramater names and values used by the LLH and model objects

`models.py`
* Has the models that convert input paramaters into the count rate expectations for each det and energy bin in the LLH
* The models are sub classes of the `Model` class
* Currently used diffuse model is `Bkg_Model_wFlatA`
* Currently used point source model is `Source_Model_InOutFoV`, which supports both in and out of FoV positions
* Currently used simple point source model for known sources is `Point_Source_Model_Binned_Rates`
* `CompoundModel` takes a list of models to make a single model object that can give the total count expectations from all models used

`flux_models.py`
* Has functions and classes to handle computing fluxes for different flux models
* The different flux model object as subclasses of `Flux_Model`
  * `Flux_Model` contains methods to calculate the photon fluxes in a set of photon energy bins
  * Used by the response and point source model objects
* The different available flux models are:
  * `Plaw_Flux` for a simple power-law
  * `Cutoff_Plaw_Flux` for a power-law with an exponential cut-off energy
  * `Band_Flux` for a Band spectrum 

`response.py`
* Contains the functions and objects for the point source model
* Most current response object is `ResponseInFoV2` and is used in the `Source_Model_InOutFoV` model

`ray_trace_funcs.py`
* Contains the functions and objects to read and perform bilinear interpolation of the foward ray trace images that give the shadowed fraction of detectors at different in FoV sky positions
* `RayTraces` class manages the reading and interpolation and is used by the point source response function and simple point source model

