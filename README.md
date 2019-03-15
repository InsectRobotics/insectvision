# InsectVision

Computer Vision Toolbox for Insect Vision

## Version 1.0.1

This version has been created as support for the article:

Gkanias E., Risse B., Mangan M., and Webb B. (2019) From skylight input to behavioural
output: a computational model of the insect polarised light compass. PLOS Computational
Biology.

### Requirements

To be able to run all the experiments and replicate the results presented in the article,
the [compmodels](https://github.com/evgkanias/compmodels) package is needed. If you
don't have access to this package, please contact the authors.

Clone both repositories and set the **compmodels** as a dependence to the
**insectvision** package.

### Observe the plots presented in the article

To see the results, it is not necessary to run the code. By simply opening the
[notebooks/plos.ipynb](https://github.com/InsectRobotics/insectvision/blob/version-1.0.1/notebooks/plos.ipynb)
file, the plots should be automatically generated for you to observe them.


### Replicate the results from the article

To create the plots by yourself, you need to start a **Jupyter notebook** kernel at
the root of the package. Then run the [notebooks/plos.ipynb](https://github.com/InsectRobotics/insectvision/blob/version-1.0.1/notebooks/plos.ipynb)
file, which already contains all the plots and the respective code to replicate them.
Some plots (especially the ones related to the global optimisation) may need a long 
time to run; this does not mean that they do not work.

### Author

All the code has been implemented by [Evripidis Gkanias](http://homepages.inf.ed.ac.uk/s1514920/).

### Copyright

Copyright &copy; 2019, Insect Robotics Group, Institude of Perception, Action and Behaviour, School of Informatics, the University of Edinburgh
