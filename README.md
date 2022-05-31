# A Hybrid Approach to Melanoma Classification: Improving medical diagnosis by combining machine learning with human expertise

Welcome to my Master's Thesis in Data Science. I created this project at Univeristy of Southern Denmark (SDU) in the Spring of 2022.
This Github contains everything except the training data sets and the trained models.

I recommend reading the Thesis first to get an understanding of what is going on. If you are already a trained Data Scientist and have experience with CNNs and ROC/AUC analysis, my advice is to skim the data section and go quickly to the evaluation and the hybrid algorithms sections - this is where the magic happens ;)

The code is not production ready, but is intended to run throught a GUI such as PyCharm or Spyder which lets you explore the objects.
The scripts are supposed to be run in this following order:
- "Kaggle_import.py". This script puts some strucure to all the training images. I left instructions in the script to where you can download the images and what do call the folder. 
- "initial_training.py". No real need to run this, it mostly serves as a learning experience on what not to do.
- "CNN_binary_deep.py", "CNN_binary_shallow.py", "CNN_multiclass_deep.py", and "CNN_multiclass_shallow.py". These all serve as one big grid search, but each script took 8-14 days to complete on a 32-core CPU which meant I did not want to wait for a sequential run to terminate.
- "Evaluation.py". This which serves to describe the evaluation data set and establish baseline performances.
- "Ensembling.py". This final script is where the hybrid algorithms are used on the evaluation datasets to create new results.

If you skip reading the paper, you should know that the project used two data sets. The first data set includes the 2019 and 2020 competition images for the ISIC challenge. They can be downloaded at:
2019: https://challenge.isic-archive.com/
2020: https://www.kaggle.com/c/siim-isic-melanoma-classification/overview
Instructions to download the training sets and set them up for code is included in "initial_training.py".
The second dataset was borrowed from the authors of a 2006 study on dermatology* who have allowed it to be shared for academic purposes only.

I hope you enjoy my work! Feel free to reach out to me with questions.

*Zalaudek, I., Argenziano, G., Soyer, H. P., Corona, R., Sera, F., Blum, A., . . . GROUP, D. W. (2006).
Three-point checklist of dermoscopy: an open internet study. British journal of dermatology (1951),
154 (3), 431-437.
Page 51
