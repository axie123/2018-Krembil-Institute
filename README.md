# 2018-Krembil-Institute

**Update** : *Dr. Tokar has left Krembil as of 2019. As far as I know, the Catharine project is suspended.*

This repository is made for the work I have done for summer research at the Krembil Research Institute (Toronto Western Hospital) during the summer of 2018. 

The main focus of our project was to develop machine learning models that can predict transcription factor (TF)-DNA interactions better than traditional wet lab experiments can. This has significance since that the ability to accurately predict TF-DNA pairs can help with identifying certain pathways which cancer is propagated.

The work done in this repository is supervised and contributed with Dr. Igor Jurisica and Dr. Tomas Tokar, and is the first part of the Catharine Project (TF pathways and cancer).

Jurisica Lab Website: https://www.cs.toronto.edu/~juris/home.html
________________________________________________________________________________________________________________________________________

The lab work was done with Python v2.7. The Numpy, Pandas, and Scikit-Learn libraries were used. 

This repository has three branches:

### master
  
This branch contains the linear and tree-based models, as well as the files that contain the data used for the models. The proof-of-concept graphs produced which plotted the models with the experiments are also given in a .zip file. The data originally comes from ChIPBase v2.0: http://rna.sysu.edu.cn/chipbase/index.php

The green and black plots on the graphs represent clusters of wet lab experiments, and the dashed line is the computer-generated model.

The papers used as background research is also given the 'source' file.

### mappingandpivot
  
This branch contains all the files that were used for map the data from one dataset to the main ones and pivoting them for the right shape. 

The code is all over the place unfortunately. This is mainly because the processing of data was also partly trial and error on different methods. This was necessary as we didn't have jupyter notebook at the time. The work was done with Pandas.

### databasemerge
  
This branch contains the files that merges all the smaller databases in the original dataset into one big one used for machine learning models. os library was used and so was Pandas.
