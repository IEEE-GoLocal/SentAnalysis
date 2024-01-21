## Customer review sentiment analysis

We have used a publicly available kaggle <a href="https://www.kaggle.com/datasets/bittlingmayer/amazonreviews">dataset</a>\
We used RNN model to train our data and achived an accuracy of 94%

We have saved our model structure in model.json and stored weights of model in model.h5. Also we have stored tokenizer in pickle file tokenizer.pkl

To use our model:
1. You just need to run predictions.ipynb
2. Change the testing sequences or add new sentences in the list and run the code
3. Final ouput will show the label of each sentences of the testing list
