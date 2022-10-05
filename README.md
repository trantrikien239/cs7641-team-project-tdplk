This repo including programming and modeling codes for CS 7641 class. 

Team members:
- Tanmay Kenjale
- Duncan Siebert
- Palash Choudhary
- Xiang Li
- Kien Tran

Proposal:
We are conducting our machine learning project on essay grading, based on a Kaggle competition where the goal is to evaluate the essays of 8th-12th grade English as second language learners. In accordance with the competition and its description, we will be predicting six analytic measures for each essay, namely: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. We plan to use a popular pre-trained general language model BERT and utilizing its output to train our own model (or potentially models) on the Kaggle data. The dataset contains 3911 essays with a mean of 461 words.

Essay scoring is a relevant and applicable usage of Machine Learning as it can tangibly save hours required to manually grade essays. Our scoring algorithm will be particularly useful as it will score essays on specific dimensions, something that has lagged automatic “Holistic” or general scoring of essays. Our project is NLP(Natural Language Processing) focused, where we will utilize TR(Transfer Learning), and apply BERT(Bidirectional Encoder Representations from Transformers). TR means that we will reuse the pre-trained model and apply a new problem statement to it. NLP, as a branch of Artificial Intelligence, is making computers have the ability to process text like humans. Therefore, BERT stands out for our projects because BERT is a transformer-based technique for NLP applications. It is a pre-trained language model which helps machines learn excellent representations of text concerning the context in many natural language tasks. It utilizes a bidirectional encoder to contextualize the sentence from Left-to-Right and Right-to-Left. We plan to apply BERT as a pre-trained model and add downstream layers to predict the metric scores. 

Additionally, we will conduct a comparison between our main BERT model and several additional models including RNN (Recurrent Neural Network) based models and traditional ML models (e.g. Tree models, SVM (Support Vector Machine), etc.) with hand designed features to evaluate the efficacy of our main model. 

Once we’ve finished with developing our BERT model, we plan on clustering essays into different meaningful categories based on their scores in each of the analytic measures. This will not only provide us insight on the data, but also potentially give teachers a way to give more customized feedback to students even if they’re not directly involved in the scoring process.

Our final result will be two models: our BERT model, that will predict the six analytic measures as its output, as well as one that is able to perform clustering on the dataset, putting students into contextually meaningful clusters.
 
We will evaluate the performance of our Supervised model using the quadratic weighted kappa (an accuracy rating that accounts for chance) and the mean column wise root mean squared error, since it is the metric used for scoring in the competition. Additionally, we will evaluate the clusters generated by our clustering algorithm to ensure they are contextually relevant.
