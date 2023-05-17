# PhD_Code_Final
The code contained in this folder was used in the thesis entitled "On the Emergence of Organic Economic Institutions
and the Impact of Legal Rules" by Gregory James Fisher.  The publication date was 16 May 2023 and the thesis can be
found via the University of Southampton's library.

Note that the code is licensed under a creative commons license - see LICENSE file that comes alongside this readme
file.

The code is contained in two files: (1) main.py; and (2) a version of my library, gjf_lib_v1_0.py.

The code was written in Python v3. 

A note on the major functions in main.py:

- run_sim(), which is the main function, manages a single simulation.
- multi_sims() manages a set of simulations (same parameters); and
- run_sim_suite() manages a suite of (sets of) simulations.

All the other functions and objects in main (and gjf_lib_v1_0) mostly provide code that supports these three functions.

A note on directories: some of the code at the beginning of 'main' identifies the directory this file is located in
and then creates relevant data folders (if they don't exist already), which run data is output to.  The project data is
recorded in /project_data: if it is a single run (from run_sim()) then the data is saved in /project_data/single_sims/
and if it's a set of simulations (from multi_sims()), it's saved in /project_data/sim_sets/.

Parameters: the default parameters in run_sim() assume the first model (market emergence).  If the parameter
respect_property_rights = 0 then the second model is run: code at the beginning of run_sim() changes the parameters
to be equal to the default parameters of the second model.

Finally, there is a fair bit of redundant code, e.g., related to parallelism (which I played with), neural networks, and
various things I experimented with but which never made it in the final thesis.

Any questions can be directed at me via gjf1g13@soton.ac.uk or (if this dies) via gregfisherhome@gmail.com.

Greg Fisher, 17 May 2023