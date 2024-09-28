For this week's mini-project, you will participate in this Kaggle competition: 
Histopathologic Cancer Detection

This Kaggle competition is a binary image classification problem where you will identify metastatic cancer in small image patches taken from larger digital pathology scans.

You will submit three deliverables: 

Deliverable 1 — A Jupyter notebook with a description of the problem/data, exploratory data analysis (EDA) procedure, analysis (model building and training), result, and discussion/conclusion. 

Suppose your work becomes so large that it doesn’t fit into one notebook (or you think it will be less readable by having one large notebook). In that case, you can make several notebooks or scripts in a GitHub repository (as deliverable 3) and submit a report-style notebook or pdf instead. 

If your project doesn’t fit into Jupyter notebook format (E.g., you built an app that uses ML), write your approach as a report and submit it in a pdf form. 

Deliverable 2 — A public project GitHub repository with your work (please also include the GitHub repo URL in your notebook/report).

Deliverable 3 — A screenshot of your position on the Kaggle competition leaderboard for your top-performing model.

## Project Description
We first start by testing a simple CNN using only numpy matrices. Since memory is limited on my system and I was at first having challenges getting CUDA working for linux, I used a small subset of the training data and validated it against some reserved examples. A screenshot is available for the validation accuracy, but this was not used to generate the final submission.

The next part of the project I used tensor flow with adam optimizer and binary cross entropy loss function. The model is a simple CNN with 3 convolutional layers and 3 max pooling layers. The model is trained for 10 epochs. I achieved an accuracy ~70% across multiple runs. After saving the model, I used it to generate predictions on the test data and submitted the results to Kaggle.
