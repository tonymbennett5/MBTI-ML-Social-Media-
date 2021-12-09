# Meyers Briggs Personality Predictor 
## Meyers Briggs in the Office 
The Meyers-Briggs Type indicator is a questionaire indicating differing psychological preferences in how people perceieve the world and make decisions. It was developed by American writer Katherine Briggs and based on the work of Swiss psychiatrist Carl Jung. While not backed by an exact science, the test can provide value to indviduals and businesses. Once you know someones type you can learn ways to communicate with them more effectively. They also help companies understand their employees strengths , weaknesses and how they process information.  About 80% of Fortune 500 companies rely on tests like the Meyers Briggs test to build stronger and more effective teams. 

More info on the MBTI in the workplace context:

https://www.forbes.com/sites/elenabajic/2015/09/28/how-the-mbti-can-help-you-build-a-stronger-company/?sh=43cc492bd93c

https://www.pacificprime.com/blog/how-to-use-the-mbti-to-enhance-the-workplace.html


## Business Problem 
 In this project we are seeing if a machine learning model could be built to predict someones personality and what personalities can be predicted based on the features of the social media text. The head of hr at Cartex wants to be able to group employees based on their Meyers Briggs Type. He wants to group them for small social group meetings before the work day starts. However, he wants make sure the groups contain people they wouldn't normally interact with. To help achieve this goal he wants to make sure that there are different MB types in each group. He asks the employees to submit their social media links and tasks the data science team to build the model. 

## Data Understanding
The data set is from Kaggle and is contained in one file. It contains over 8600 rows containing two columns of a person’s Meyers Briggs code/type and a section of the last 50 things they have posted on an online internet forum. The posts are separated by ‘|||’. 

## Data Download
The link to the data is here:
https://www.kaggle.com/datasnaek/mbti-type/code

In order to begin working on the dataset you only have to import the set as csv with pandas. From there you can move it into your data folder and begin to work from there. 

## Tools Used 
NLP, Scikit learn, NLTK, VADER Sentiment Analyzer,  WordNetLemmatizer, CountVectorizer, TfidfVectorizer, LogisticRegression, DecisionTreeClassifier,  KNeighborsClassifier, RandomForestClassifier, MultinomialNB, RandomUnderSampler

## Methods Used

### Dealing With Class Imbalance 

- Divided single type classes into four features  So each part of the MBTI test was either a 1 or 0
    - Extroversion vs. Introversion
    I - 0
    E - 1
    - Sensing vs. Intuition
    N - 0
    S - 1
    - Thinking vs. Feeling
    F - 0
    T - 1
    - Judging vs. Perceiving
    P - 0
    J - 1
- I also used a Random Under Sampler in the machine learning pipeline 
## Feature Engineering

### EDA and Cleaning 
- post data was made lower case
- the '|||' separators were stripped and replaced with white space
- punctuations and emails addresses were dropped 
- Post Data was lemmatized and stopwords were removed 

### Sentiment, Post Tagging , Counting 
- Used Vader Sentiment Intensity Analyzer to find the compound, postive, neutral and negative scores of each 
- Used Post Tagging to tag each part of speech in the dataset.
- Some of the tags used were: Noun, Verb, Adjective, Prepostions ,Interjections and Determiners. 
- Counted unique words, emojis, colon, question marks, exclamation points, upper case words, links, ellipses and images. These were used as additonal features for the models. 

## Modeling 

- The data was split into X(features) and y(target)
- The features(X) were the sentiment scores, pos tags and other counts 
- The target(y) was set to four target features : Extrovert, Sensing, Thinking, Judging 
- Imbalanced Learn Pipeline was used because the dataset was unbalanced 
- The modeling pipeline was composed of preprocessed features, random under sampler, and the specfic machine learning model.
- Models tested:
    - TF-IDF Logistic Regression 
    - Count Vectorized Logistic Regression
    - TF-IDF Logistic Ridge
    - Count Vectorized Logistic Ridge
    - TD-IDF Decision Tree Classifier
    - Count Vectorized Decision Tree Classifier
    - TF-IDF Support Vector Classifier
    - Count Vectorized Support Vector Classifier
    - TF-IDF Random Forest
    - Count Vectorized Random Forest
    - TF - IDF Naive Bayes
    - Count Vectorized Naive Bayes
- Model Evaluations used:
    - Accuracy
    - Precision 
    - ROC-AUC
    - Average Precision-Recall Score 
    - Classfication Report Imbalanced (pre, rec, spe, f1, geo, iba,sup)
- Final Model: Based on the evaluation metrics used the TF-IDF model was selected as the final model. 

## Conclusion 
- Dataset was heavily imbalanced which caused problems while classifying. More people were introverted and intuitive then Extroverted and Senstive.
- Models had a tough time discerning between Extroversion vs Introversion and Sensitivity vs Intuition.
- Random UnderSampling did not help as much as hoped. 
- Meyers Briggs is a pretty basic test. People often come up in the middle which can cause problems when trying to classify people people. Cartex's groups may not be as equal as they would like. 
- Would consider model a relative success as even humans have a tough time discerning someones MBTI. Personality is alot more complex then just words and text expression. 
### Recomendation 
- The company should implement the model to group employees. 
-  Hold off on implementing the model on anything that is serious like a marketing campaign which would involve more factors outside of the model's capability. 

## In the Future 
- Add more data for the types that were undersampled. 
- Continue to try different model types, possiblt nueral network based model.  

# Repository Structure 
Images
- Contains some images used and charts

Notebooks                                                 

- Count Vectorizing File 
- Counting File 
- EDA and Cleaning File 
- Modeling File 
- Sentiment Analysis and POS tagging   

Gitignore

Final Notebook

README

Environment File

Presentation PDF 



By: Tony Bennett 