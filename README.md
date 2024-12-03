# Not So Griddy
 Code for paper ["Not so griddy: Internal representations of RNNs path integrating more than one agent"](https://www.biorxiv.org/content/10.1101/2024.05.29.596500v1.abstract) Redman et al. NeurIPS (2024). This codes is built upon the [code framework](https://github.com/ganguli-lab/grid-pattern-formation/tree/master) of [Sorscher et al. NeurIPS (2019)](https://proceedings.neurips.cc/paper/2019/hash/6e7d5d259be7bf56ed79029c4e621f44-Abstract.html) and adds a second agent that the RNN must simultaneously path integrate. 

Scripts to train your own single and dual agent RNN can be found in the **Code** folder. Depending on RNN architecture, choice of training hyper-parameters, and computing resources, this can take several hours to run. 

Saved trained models can be found at https://drive.google.com/file/d/1EJjnLdDnOtkj5ZyN3QJGQaPDnVZrKBhF/view?usp=sharing. Download these models and save them in a **Models** folder on your local machine. 

Code to visualize the internal representations and re-make the plots from the paper can be found in the **Plotting** folder. These can be run on the saved trained models (see above) or on your own trained RNNs. The scripts and the corresponding figures they generate are as follows: ``plotting_network_performance.py`` (Fig. 2); ``plotting_functional_classes.py`` (Fig. 4A-B); ``plotting_ablations.py`` (Fig. 4C); ``plotting_relative_spatial_information.py`` (Fig. 5). 

Feel free to reach out with any comments/questions to will.redman@jhuapl.edu. 
 
