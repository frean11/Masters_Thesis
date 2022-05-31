# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:19:01 2022

@author: frede
"""

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

class evaluation_functions:

    def csv_import(truth_path: str, guess_path: str):
        """
        Function used to import and pre-process the data from the Zadaulek study.

        Parameters
        ----------
        truth_path : str
            A path to the file containing the ground truths for each picture in the study.
        guess_path : str
            A path to the file containing answers from the participants in the study.

        Returns
        -------
        merged_test : DataFrame
            The result of merging the ground truths and guesses.
        useful_participants : list
            A subset of the full participant list containing only those who are experienced in Dermoscopy and have answered at least 126 test questions.
        scoring : dict
            Descriptive statistics on the employment profession and number of completed pictures.
        another_scoring : dict
            Descriptive statistics on the experience level and number of completed pictures.

        """
        
        # Loading in the truth and guesses and merging the two
        
        truth = pd.read_excel(truth_path)
        
        guess = pd.read_excel(guess_path)
        
        merged_test = truth.merge(guess,
                                  left_on = 'NUM',
                                  right_on = 'CASE Number',
                                  how = 'left')
        
        # Binary encoding the ground truth
        
        merged_test['Ground_truth'] = [1 if x=='Suspicious' else 0 for x in merged_test['EXPERT_EVALUATION']]
        
        # Creating a "final verdict" on the guesses
        
        merged_test['Guess_score'] = merged_test['ASYMMETRY'] + merged_test['ATYPICAL NETWORK'] + merged_test['BLUE-WHITE STRUCTURES']
        merged_test['Guess_suspicioun'] = [1 if x >= 2 else 0 for x in merged_test['Guess_score']]
        merged_test['Correct'] = merged_test['Ground_truth'] == merged_test['Guess_suspicioun']
        
        # Finding distribution of participants in the study
        
        all_participants = merged_test['OBSERVER ID'].unique()
        
        participant_group = []
        another_group = []

        for i in merged_test['OBSERVER ID'].unique():
            dat = merged_test[merged_test['OBSERVER ID'] == i]
            
            # Getting the category of test images answered
            
            dat_len = len(dat)
            train_len = len(dat[dat['TRAINING'] == 'yes'])
            test_len = dat_len-train_len
            if test_len == 0:
                test_cat = '0'
            elif test_len > 0 and test_len <= 25:
                test_cat = '1-25'
            elif test_len > 25 and test_len <= 75:
                test_cat = '26-75'
            elif test_len > 75 and test_len <= 125:
                test_cat = '76-125'
            else:
                test_cat = '126-150'
            
            # Profession and experience in dermascopy
            
            profession = dat['OBSERVER PROFESSION'].iloc[0]
            experience = dat['PREVIOUS USE OF DERMOSCOPY'].iloc[0]
            
            participant_group.append((profession, test_cat))
            another_group.append((experience, test_cat))


        scoring = {}
        
        participant_set = set(participant_group)
        for j in participant_set:
            scoring[j] = participant_group.count(j)
        
        another_scoring = {}
        
        another_set = set(another_group)
        for j in another_set:
            another_scoring[j] = another_group.count(j)
        
        # Finding those people who have experience in dermoscopy and have answered at least 126 questions
        
        merged_test = merged_test[merged_test['TRAINING'] != 'yes'] # We only want to return information on the training data
        
        useful_participants = []
        
        for i in tqdm(all_participants):
            dat = merged_test[merged_test['OBSERVER ID'] == i]
            if len(dat) > 125:
                if dat['PREVIOUS USE OF DERMOSCOPY'].iloc[0] == 'Yes':
                    useful_participants.append(i)
        
        return merged_test, useful_participants, scoring, another_scoring
 
    
    def import_X_and_y(im_path: str, im_shape, dataframe):
        """
        Function for loading the images, labels, and image indexed for the pictures in the Zadaulek study.

        Parameters
        ----------
        im_path : str
            A path to the location of the images.
        im_shape : tuple
            The desired image resolution.
        dataframe : DataFrame
            A DataFrame containing the names of the target images in the folder. This can be created using function "csv_import"

        Returns
        -------
        images: array
            The images stored as 3D numpy arrays
        return_labels: list
            The labels for each picture, useful for evaluation.
        image_list : list
            The image number corresponding to each image and return label.

        """
        
        
        return_images = []
        return_labels = []
        
        data = dataframe
        image_list = data['NUM'].astype(str).unique()
        
        # Labels first, images require some additional processing
        
        for i in tqdm(image_list):
            dat = data[data['NUM'].astype(str)==i].reset_index()
            label = dat['Ground_truth'][0]
            return_labels.append(label)
        
        # Now images
        
        image_list = [x if len(x)==3 else '0'+x for x in image_list]
        
        for i in tqdm(image_list):
            digits = load_img(im_path+'/'+i+'.jpg', target_size = im_shape)
            digits = img_to_array(digits)
            digits = digits / 255 # To standardize in inverval [0,1]
            
            return_images.append(digits)
            
        return np.asarray(return_images), np.asarray(return_labels), image_list
    
    
    def Zadaulek_eval(model, X_eval, idx, mel_num: int):
        """
        Function used to predict on the Zadaulek images using a trained Keras model.

        Parameters
        ----------
        model : Keras Model
            The trained model.
        X_eval : 3D array
            The images you want to evaluate
        idx : list
            The image ID's corresponding to the images in X_eval
        mel_num : int
            The index number in the model prediction outputs corresponding to the "MEL" category

        Returns
        -------
        predict_df : DataFrame
            A dataframe containing the predictions for each outcome for each category.
        mel_df : DataFrame
            The predictions for only the "MEL" category along with image names for each image. This is used for Kagel submission.

        """
        
        predictions = model.predict(X_eval)
        mel_predictions = [x[mel_num] for x in predictions]
        
        predict_df = pd.DataFrame(predictions)
        mel_df = pd.DataFrame(zip(idx,mel_predictions), columns = ['Image_ID', 'prediction'])
        
        return predict_df, mel_df
        
    
    def prediction_dataframes(model, binary: bool, train_gen, test_gen):
        """
        Function used to predict on images loaded using an ImageDataGenerator with trained Keras model.

        Parameters
        ----------
        model : Keras Model
            The trained model which shall be evaluated
        binary : bool
            To clarify whether the model has been trained on Binary or Multiclass data.
        train_gen : ImageDataGenerator
            The ImageDataGenerator which generates the training images. This is used to get the label associated with the "MEL" class
        test_gen : ImageDataGenerator
            The ImageDataGenerator which generates the you want to evaluate.

        Returns
        -------
        predict_df : DataFrame
            A dataframe containing the predictions for each outcome for each category.
        submit_df : DataFrame
            The predictions for only the "MEL" category along with image names for each image. This is used for Kagel submission.

        """
        
        mod = model
        binary = binary
        
        # Making predictions using the model
        
        predict = mod.predict(test_gen,
                              verbose = 1)
        
        if binary == False:
            mel_num = train_gen.class_indices.get('MEL')
            mel_predict = [x[mel_num] for x in predict]
        else:
            mel_predict = [x[0] for x in predict]
        
        # Generating corresponding imagenames
        
        file_names = test_gen.filenames
        file_names = [x[5:-4] for x in file_names]
        
        # Creating dataframes
        
        predict_df = pd.DataFrame(predict)
        submit_df = pd.DataFrame(zip(file_names, mel_predict), columns = ['image_name', 'target'])
        
        return predict_df, submit_df
    
    
    def ROC_eval(predictions, y_true):
        """
        Transforming a prediction and a ground truth array into coordinates for a ROC curve and corresponding AUC value.

        Parameters
        ----------
        predictions : array
            The predictions from a model in interval [0,1].
        y_true : array
            A binary array encoding the ground truths corresponding to the predictions.

        Returns
        -------
        df : DataFrame
            A DataFrame containing pairs of FPR and TPR values along with the threshold corresponding to each pair.
        auc : float
            The AUC score for the analysis.

        """
        
        auc = roc_auc_score(y_true, predictions, average = None)
        
        fpr, tpr, thresholds = roc_curve(y_true, predictions)
        
        df = pd.DataFrame(zip(fpr,tpr,thresholds), columns = ['False positive rate', 'True positive rate', 'Thresholds'])
        
        return df, auc
  
   
    
    def load_model(path):
        return tf.keras.models.load_model(path)

    
    def inner_outer(predictions: list, ratio: int, y_truth: list, image_idx: list):
    
        pred = predictions        
        pred_sorted = sorted(pred)
    
        # Defining the boundaries
    
        lower_bound = int(len(pred_sorted)/ratio)
        upper_bound = int(len(pred_sorted)*(ratio-1)/ratio)
        
        # Rearranging into inner and outer list
        
        lower = pred_sorted[:lower_bound]
        middle = pred_sorted[lower_bound:upper_bound]
        upper = pred_sorted[upper_bound:]
        
        pred_outer = lower + upper
        pred_inner = middle
        
        # Getting the indexes corresponding to each value
        
        indexes_outer = []
        
        for val in pred_outer:
            indexes_outer.append(pred.index(val))
        
        indexes_inner = []
        
        for val in pred_inner:
            indexes_inner.append(pred.index(val))
        
        # New prediction and label list using the indexes
        
        pred_outer = [pred[i] for i in indexes_outer]
        y_outer = [y_truth[i] for i in indexes_outer]
        
        pred_inner = [pred[i] for i in indexes_inner]
        y_inner = [y_truth[i] for i in indexes_inner]
        
        # And getting the corresponding image numbers for each image in the lists
        
        idx = image_idx
        
        idx_inner = [int(idx[i]) for i in indexes_inner]
        idx_outer = [int(idx[i]) for i in indexes_outer]
    
        return pred_inner, pred_outer, y_inner, y_outer, indexes_inner, indexes_outer, idx_inner, idx_outer


    def fpr_tpr(DataFrame):
        """
        Creates a pair of FPR and TPR values from a DataFrame. Is used to access individual performance by subsetting the Zadaulek data for each participant.

        Parameters
        ----------
        DataFrame : DataFrame
            A DataFrame containing the truth and guesses for each picture.

        Returns
        -------
        fpr : float
            The False Positive Rate
        tpr : float
            The True positive Rate

        """
        
        dat = DataFrame
        matrix = confusion_matrix(dat['Ground_truth'], dat['Guess_suspicioun'])
        tn, fp, fn, tp = matrix.ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
    
        return fpr, tpr

    def human_fpr_tpr(dataframe, participant_list: list, image_idx: list):
        """
        Uses the fpr_tpr function to create a dictionary of performances. Used to get an overview of the individual performances from the Zadaulek study.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing truth values and guesses for each picture for each participant.
        participant_list : list
            A list with the numbers of the relevant participants. Used if not all particpants in the DataFrame are desired.
        image_idx : list
            A list of the relevant image indexes. Used if not all images in the DataFrame are desired.

        Returns
        -------
        x_values : list
            The FPR for each participant in the participant list.
        x_avg : float
            The average FPR value.
        y_values : list
            The TPR for each particpant in the participant list.
        y_avg : float
            The average TPR value.

        """
        
        scoring= {}
        data = dataframe
        # print(len(image_idx))
        
        for i in tqdm(participant_list):
            dat = data[(data['OBSERVER ID'] == i)
                      & (data['NUM'].isin(image_idx))]
            # print(len(dat))
            scoring[i] = evaluation_functions.fpr_tpr(dat)
    
        # Organizing scores and calculating average
        
        scoring_values = list(scoring.values())
        x_values = [x[0] for x in scoring_values]
        x_avg = np.average(x_values)
        y_values = [y[1] for y in scoring_values]
        y_avg = np.average(y_values)

        return x_values, x_avg, y_values, y_avg


    def fpr_tpr_arrays(y_true, predictions):
        """
        Used to generate FPR and TPR pairs using prediction and ground truth arrays. This is used to evaluate the Keras models.

        Parameters
        ----------
        y_true : array
            Binary encoded ground truth values.
        predictions : array
            A prediction array in range [0,1]

        Returns
        -------
        fpr : float
            The False Positive Rate
        tpr : float
            The True Positive Rate

        """
        
        matrix = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = matrix.ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        
        return fpr, tpr

    def human_fpr_tpr_full_ensemble(dataframe, participant_list: list, image_idx: list, y_true):
        """
        Uses fpr_tpr_arrays to calculate the FPR and TPR score for the full human ensemble.

        Parameters
        ----------
            
        dataframe : DataFrame
            The DataFrame containing truth values and guesses for each picture for each participant.
        participant_list : list
            A list with the numbers of the relevant participants. Used if not all particpants in the DataFrame are desired.
        image_idx : list
            A list of the relevant image indexes. Used if not all images in the DataFrame are desired.
        y_true : array
            An array containing the truth value for each picture.

        Returns
        -------
        fpr : float
            The False Positive Rate for the full ensemble.
        tpr : float
            The True Positive Rate for the full ensemble.

        """
        
        data = dataframe
        predictions = []

        for i in image_idx:
            num = int(i)
            dat = data[(data['NUM'] == num)
                       & (data['OBSERVER ID'].isin(participant_list))]
            avg_ans = dat['Ground_truth'].mean()
            freq_ans = np.round(avg_ans,0)
            predictions.append(freq_ans)
        
        fpr, tpr = evaluation_functions.fpr_tpr_arrays(y_true,predictions)
        
        return fpr, tpr

    
    def human_fpr_tpr_three_ensemble(dataframe, participant_list: list, image_idx: list, y_true, num_iterations):
        """
        Uses fpr_tpr_arrays to generate an average FPR and TPR score over a number of iterations using a three-human ensembling strategy.
        Additionally calculates how many times the third opinion is relevant.
        
        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing truth values and guesses for each picture for each participant.
        participant_list : list
            A list with the numbers of the relevant participants. Used if not all particpants in the DataFrame are desired.
        image_idx : list
            A list of the relevant image indexes. Used if not all images in the DataFrame are desired.
        y_true : array
            An array containing the truth value for each picture.
        num_iterations : int
            The number of iterations to average over. I recommond at least 1000.

        Returns
        -------
        fpr_avg : float
            The average False Positive Rate from the iterations.
        tpr_avg : float
            the Average True Positive Rate from the iterations.
        third_opinions_avg : float
            The average number of third opinions required to achieve majority on all pictures.

        """
        
        data = dataframe
        n = num_iterations
        
        fpr_list = []
        tpr_list = []
        third_opinions_list = []
        
        for i in tqdm(range(n)):
        
            predictions = []
            third_opinions = 0
            
            for i in image_idx:
                
                bla = data[(data['NUM'] == i)
                           & (data['OBSERVER ID'].isin(participant_list))]
                guesses = bla['Guess_suspicioun'].tolist()
                sample = random.sample(guesses, 3)
                
                # Checking if the first two people agree
                
                if sample[0] == sample[1]:
                    predictions.append(sample[0])
                
                else:
                    predictions.append(sample[2])
                    third_opinions += 1
            
            # Calculating fpr and tpr from precictions
            
            fpr, tpr = evaluation_functions.fpr_tpr_arrays(y_true, predictions)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            third_opinions_list.append(third_opinions)
        
        # Now finding the average scores
        
        fpr_avg = np.mean(fpr_list)
        tpr_avg = np.mean(tpr_list)
        third_opinions_avg = np.mean(third_opinions_list)
        
        return fpr_avg, tpr_avg, third_opinions_avg
        

    def human_answers(dataframe, participant_list: list, image_list):
        """
        Picks a random human answer to each picture in the relevant list using only the answers from the relevant humans from the Zadaulek study.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing the Zadaulek study - can be generated with function "csv_import"
        participant_list : list
            List of the relevant participants - can be generated with function "csv_import"
        image_list : list
            A list with the numbers of the images in the Zadaulek study.

        Returns
        -------
        human_answers : dict
            A dictionary containing the human answer to each image.

        """
        
        data = dataframe
        
        human_answers = {}
    
        for i in image_list:
            dat = data[data['NUM'] == int(i)]
            people = dat['OBSERVER ID'].unique()
            random_person = random.choice(people)
            dat = dat[dat['OBSERVER ID'] == random_person]
            answer = dat['Guess_suspicioun'].iloc[0]
            human_answers[i-16] = answer
        
        return human_answers

    def new_predictions(dataframe, participant_list: list, image_list, predictions: list):
        """
        Uses function 'human_answers' to replace the the relevant prediction from a list with human answers.
        This is central to the augmented hybrid approach.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing the Zadaulek study - can be generated with function "csv_import"
        participant_list : list
            List of the relevant participants - can be generated with function "csv_import"
        image_list : list
            A list with the numbers of the images in the Zadaulek study.
        predictions : list
            Array containing predictions in range [0,1].

        Returns
        -------
        pred : array
            An imputed array of predictions containing a mix of the original predictions list and human answers.

        """
        
        answers = evaluation_functions.human_answers(dataframe, participant_list, image_list)
        
        pred = predictions.copy()
        
        for i in answers.keys():
            pred[i] = answers[i]

        return pred


    def averaging_ROC(iterations: int, data, useful_participants: list, index, predictions, y_true):
        """
        Using the new_predictions method to generate imputed predicions lists and perform ROC/AUC analysis on these.
        The ROC curves and AUC scores are averaged over a number of iterations.

        Parameters
        ----------
        iterations : int
            The number of iterations to run. I recommend at least 1000.
        dataframe : DataFrame
            The DataFrame containing the Zadaulek study - can be generated with function "csv_import"
        useful_participants : list
            List of the relevant participants - can be generated with function "csv_import"
        index : list
            A list with the numbers of the images in the Zadaulek study.
        predictions : list
            Array containing predictions in range [0,1].
        y_true : array
            Array containing the truth values for each image.

        Returns
        -------
        averages : DataFrame
            A DataFrame containing the average FPR and TPR values over all iterations for each threshold.
        auc_average : float
            The average AUC score over all iterations.

        """
        
        dfs = []
        aucs = []
        thresholds = [] # So we can see that it uses the same threshold every time
        
        for i in tqdm(range(iterations)):
            
            pred_sub_inner = evaluation_functions.new_predictions(data, useful_participants, index, predictions)
            df_sub, auc = evaluation_functions.ROC_eval(pred_sub_inner, y_true)
            dfs.append(df_sub)
            thresholds.append(len(df_sub))
            aucs.append(auc)
        
        assert len(set(thresholds)) == 1 # Verifying the number of thresholds are actually the same each time
        
        auc_average = np.mean(aucs)
        
        averages = pd.concat([each.stack() for each in dfs],axis=1)\
                     .apply(lambda x:x.mean(),axis=1)\
                     .unstack()
        
        return averages, auc_average
    
    
    def hybrid_majority_vote(dataframe, predictions, y_true, image_idx: list, num_iterations: int):
        """
        Function which compares human and machine answers and if necessary uses a second human opinion to break the tie.
        It then evaluates the resulting prediction array to the ground truth to generate FPR and TPR values.
        The procedure is averaged through a number of iterations.
        This is the procedure used to create hybrid majority vote results.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing the Zadaulek study - can be generated with function "csv_import"
        predictions : array
            A prediction array in range [0,1]. This is thougth to be the prediction output from a Keras Model.
        y_true : list
            An array containing the ground truths associated with each prediction.
        image_idx : list
            A list containing the relevant image numbers.
        num_iterations : int
            The number of times to use the approach.

        Returns
        -------
        roc_df : DataFrame
            The baseline ROC performance for the computer model created from the original predictions and ground truths.
        thresholds : list
            The thresholds used to perform the ROC analysis.
        scoring : dict
            The average FPR and TPR values for each threshold.
        third_opinions : float
            The average number of third opinions necessary to break ties for the relevant images.

        """
        
        data = dataframe
        
        roc_df, _ = evaluation_functions.ROC_eval(predictions, y_true)
        thresholds = roc_df['Thresholds'].tolist()
        
        scoring = {}
        third_opinions = []
        
        for threshold in tqdm(thresholds):
        
            fpr_list = []
            tpr_list = []
            third_opinion_list = []
            
            for i in range(num_iterations): # Running a lot of iterations to mediate the stochasticity associated with the random sampling
                
                # To keep track of the outcomes
                
                new_pred = []
                third_opinion_count = 0
                
                for i in range(len(image_idx)):
                
                    # Getting the guesses from humans on one picture at a time
                
                    dat = data[data['NUM'] == image_idx[i]]
                    answers = dat['Guess_suspicioun'].tolist()
                
                    # Randomly selecting two answers without replacement
                
                    selection = random.sample(answers, 2)
                
                    # Getting a prediction from the computer based on the threshold
                
                    if predictions[i] > threshold:
                        computer_answer = 1
                    else:
                        computer_answer = 0
                
                    # Assuming human and machine answer the same, we have our ensemble guess
                
                    if selection[0] == computer_answer:
                        ensemble_guess = selection[0]
                
                    # If they disagree, we need a third opinion to break the tie. This yields the ensemble guess
                
                    else:
                        third_opinion_count += 1
                        ensemble_guess = selection[1]
                
                    new_pred.append(ensemble_guess)
                
                # Creating a dataframe from the predictions and the correct answers - hacky solution so I can use the tpr_fpr function.
                
                new_pred_df = pd.DataFrame(list(zip(y_true, new_pred)), columns = ['Ground_truth' , 'Guess_suspicioun'])
                fpr, tpr = evaluation_functions.fpr_tpr(new_pred_df)
            
                # Appending the results
            
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                third_opinion_list.append(third_opinion_count)
            
            # And appending so we have the fpr and tpr score for this specific threshold
            
            scoring[threshold] = np.mean(fpr_list), np.mean(tpr_list)
            third_opinions.append(np.mean(third_opinion_list))
        
        return roc_df, thresholds, scoring, third_opinions


class training_functions:
    
    
    def get_lr_callback(batch_size=8):
        """
        A learning rate schedule recommended by Chris Deotte.

        Parameters
        ----------
        batch_size : int
            The batch size used by the ImageDataGenerator. Defaults to 8.

        Returns
        -------
        lr_callback: Keras Callback
            Returns a Keras Callback instance which controls the learning rate in a model fitting.

        """
        
        lr_start   = 0.000005
        lr_max     = 0.00000125 * batch_size
        lr_min     = 0.000001
        lr_ramp_ep = 5
        lr_sus_ep  = 0
        lr_decay   = 0.8
       
        def lrfn(epoch):
            if epoch < lr_ramp_ep:
                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
                
            elif epoch < lr_ramp_ep + lr_sus_ep:
                lr = lr_max
                
            else:
                lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
                
            return lr
    
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
        return lr_callback


    def category_distribution(path: str):
        """
        Generates descriptive statistics on the training and validation datasets.

        Parameters
        ----------
        path : str
            A path to the primary folder where the images are placed.

        Returns
        -------
        df : DataFrame
            A DataFrame showing the count and relative frequency of each category in the relevant dataset.

        """
        
        FOLDER = os.listdir(path)
        INSTANCES = []
        
        for i in FOLDER:
            INSTANCES.append([i, len(os.listdir(path+'/'+i))])
        
        df = pd.DataFrame(INSTANCES, columns = ['Category', 'Count'])
        df['Relative freq %'] = df['Count'] / df['Count'].sum()*100
        
        return df
    