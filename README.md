# Masters_Thesis
A Hybrid Approach to Melanoma Classification: Improving medical diagnosis by combining machine learning with human expertise


Welcome to my Master's Thesis in Data Science. I created this project at Univeristy of Southern Denmark (SDU) in the Spring of 2022. This Github includes:
- The actual Thesis in PDF format
- All code used to create the Thesis
- All evaluations and visualizations used for the Thesis
- The best of the 16 models which was trained during the process

I recommend skimming through the Thesis to get an understanding of what is going on. If you are already a trained Data Scientist with experience with CNNs and ROC/AUC analysis, skim the data section then go quickly to the evaluation and the hybrid algorithms sections - this is where the magic happens ;)

If you skip reading the paper, you should know that the project used two data sets. The first data set includes the 2019 and 2020 competition images for the ISIC challenge. They can be downloaded at:
2019: https://challenge.isic-archive.com/
2020: https://www.kaggle.com/c/siim-isic-melanoma-classification/overview
The second dataset was borrowed from the authors of a 2006 study on dermatology. The data is at the moment proprietary so I cannot upload it. Instead I include a reference to the study in the bottom of this document, they may let others use it for academic purposes.


The code is not production ready, but is intended to run throught a GUI such as PyCharm or Spyder which lets you explore the objects.
The scripts are supposed to be run in this following order:

The "Kaggle_import.py" is first. This script puts some strucure to all the training images. I left instructions in the script to where you can download the images and what do call the folder. 

The next script is called "initial_training.py". No real need to run this, it serves as a learning experience on what not to do.

Next comes the scripts "CNN_binary_deep.py", "CNN_binary_shallow.py", "CNN_multiclass_deep.py", and "CNN_multiclass_shallow.py". These all serve as one big grid search, but each script took 8-14 days to complete on a 32-core CPU which meant I did not want to wait for a sequential run to terminate. At this point the training is done and the prediction arrays for the training, val, test, and evaluation data sets are created.

Next is "Evaluation.py" which serves to describe the evaluation data set and establish baseline performances.

The final script is "Ensembling.py" where the hybrid algorithms are used on the evaluation datasets to create new results.


Since the evaluation dataset is not public, you cannot use the evaluation and ensembling scripts much. The model training scripts can still be made to work, just comment out all the sections related to the evaluation dataset and you should be fine. If you wish to apply the hybrid algorithms to other datasets, you can do your own data processing and use the functions "averaging_ROC" to generate the augmented algorithm and "hybrid_majority_vote" to perform the hybrid majority vote algorithm. These functions can be found in the class called "evaluation_functions" from the "Custom_functions.py" in the Helper functions foler.


I hope you enjoy my work! Feel free to reach out to me with questions.


*Zalaudek, I., Argenziano, G., Soyer, H. P., Corona, R., Sera, F., Blum, A., . . . GROUP, D. W. (2006).
Three-point checklist of dermoscopy: an open internet study. British journal of dermatology (1951),
154 (3), 431-437.
Page 51
