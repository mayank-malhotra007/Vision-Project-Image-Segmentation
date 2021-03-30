# Vision-Project-Image-Segmentation
Code for the Computer Vision project in Neural Networks.

The solution to task 1 is in the notebook Q1.ipynb and the created weights will be saved in a folder models1/ that should lie in the same directory in which Q1.ipynb will be executed.

The solution to task 2 are the scripts "Q2_NN_Project" and "Q2_Evaluation_scipt_NN_Project" , for training/visualization and the evaluation part respectively.
There is a little ambiguity, both of our saved models are called "best.pt". For task2 we have uploaded the model on the google drive and made the repo public.
The link is available for the repo: https://drive.google.com/drive/folders/1xYF-EB5Dwt_TRivMNnqaU1KRoO3qP6GY?usp=sharing


The solution to task 3 are the script q3_train.py, which will train the models and store the trained model (multiple if the line to save weights per epoch will be uncommented) in a folder called models/.
The other Solution for task 3 is the script q3_eval.py, that loads the best model called "best.pt" that lies inside the folder models/ and computes the metrics on the validation set.
Since the model was too big to upload to Github (125MB) the fully trained model for Q3 is provided in the Teams assignments upload of the project of Tom Fischer (2563286).
To use the model, please place it in the models/ folder, which we provided in this Github repo, since this is where the load_model() function in the Q3_eval script will search for the best models.

