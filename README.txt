# 06
Sentiment analysis with tweets

The initial dataset is "ISO-8859-1" encoded and has no columns names. We import the dataset accordingly. We also highlight that the target feature is sentiment and has one of 3 values, {0: "negative", 2: "neutral", 4: "positive"}.

01. Pre-Processing:
01.00. The target features are converted to integers.
01.01. Using NLTK we shall create a function that removes links,user tags and special characters.
01.02. Using Scikit-learn, we shall split the dataset for train/test. The standard is 80/20.
01.03. The sentiment threshold is a the function below. It will be used to give meaning to our trained model with scores. Label will be determined by the comparison of score and threshold values above. 
'Neutral' if 0.4<score<0.7,
'Negative' if score<=0.4, 
'Positive' if score>=0.7. 

02. Processing: 
02.01. Embeddings:
The model will initialize using an embedding layer that is developed via the dataset itself. Using W2V we will create an embedding layer such that: 300 length, 7 windows, the minimum account is 10 and we shall have 32 epochs. This will take a while.
02.02. Tokenization:
This will be a standard process using Keras.
02.03. Padding: 
Not all tweets have the same length, therefore, we shall pad the missed via a standard process using Keras. 
02.04. Label encoder:
This will be a standard process using Keras to label the categorical features. 

03. Model:
03.01. Building on a sequential layer we shall add the following layers in this order:
- The embedding layer (using Keras. This will require an embedding matrix which is dependent on the W2V model that has been developed earlier)
- A dropout layer (randomly sets input units to 0 with a frequency of 0.5).
- An LSTM layer (taking 100 units, dropout on both standards and linear at 0.2 ).
- A dense layer sigmoid activated.
03.02. Compile the model with the binary cross entropy loss, optimize by adam and value by accuracy.
03.03. Call-back the model on a list of two functions: (most recommended and can be optimized later if need be)
- ReduceLROnPlateau with the following settings: monitor='val_loss', patience=5, cooldown=0.
- EarlyStopping with the following settings: monitor='val_accuracy', min_delta=1e-4, patience=5)
03.04. Train the model with 1024 batches, on 3 epochs, and validate-split at 0.1.
03.05. Evaluate and Predict.

04. Classification report on a confusion matrix.
