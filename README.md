# CUSTOMER CHURN PREDICTION PROJECT

## Project Overview.

Business growth and development remains a central motivator in organizational decision-making and policy making. Although every business leader aspires to achieve growth in revenues, clientele, and profitability, they must try as much as possible to avoid making losses.

In recent years, such leaders, as well as business experts, have identified customer satisfaction as an important factor to ensuring such growth and development. Without customers, a business would not make any sales, record any cash inflows in terms of revenues, nor make any profits. This underscores the the need for organizations to implement measures that retain existing customers.

Recent technological advancements have also contributed to an increased business rivalry, especially due to increased startups and entrants. Such competition, coupled with an augmented saturation of markets, means that it has become harder and more expensive for businesses in most sectors to acquire new clients, which means they must shift their focus to cementing relationships with existing customers.

## Business Understanding.

With new technical advancements providing consumers with more communication options, the telecom sector has grown extremely competitive. In an effort to preserve overall development and profitability, we want to develop a predictive business model that will allow them to implement methods that will lower attrition and retain and expand their clientele. As the primary stakeholder, the business will gain from this model by lowering customer attrition rates, which might boost sales and profits, encourage expansion, and maintain—or rather, strengthen—its position in the market. Better customer service and enhanced telecommunications services would also benefit the customers. The stockholders will also benefit as the business grows in terms of sales, profitability, clientele, and market share.

#### Research Objectives

To identify the key features that determine if a customer is likely to churn. To determine the most suitable model to predict Customer Churn. To establish customer retention strategy to reduce churn.


## Data Preparation.
 The analysis performed on the dataset included the following steps: Data Cleaning: The dataset was checked per column to find missing values, duplicates, and outliers and we dealt with them accordingly Data Transformation: Categorical data in the churn column was converted into numerical data. Exploratory Data Analysis to check the distribution of features and the target and to identify possible correlations between features.
 Certain columns were transformed to enhance their usefulness. This included; Encoding categorical variables into numerical representations e.g area_code, international_plan, voice_mail_plan Normalization and Scaling features to a consistent range using the StandardScaler
By performing these steps, I aimed to gain a comprehensive understanding of the dataset and prepare it for further analysis and modeling

## Modeling.

In order to gather information and generate predictions, I created and assessed three classification models during the investigation. An outline of the models is provided below:

Logistic Regression Model 1 Baseline model


The initial logistic regression model was constructed with default settings. The model achieved 86% accuracy in training and 86% accuracy in testing. The F1 Score was 27%, recall was 18%, and precision was 60%.

Class imbalance is addressed with a logistic model.

By correcting class imbalance using the class weight "balanced," we attempted to enhance the model's performance. Test accuracy was 0.78 and train accuracy was 0.77. The F1 score was 51%, the precision was 39%, and the recall was 77%. The model chosen for logistic regression was this one.

Model 2: KNN Baseline Model: K-Nearest Neighbors

By default, the initial K-Nearest Neighbors model was constructed.
The model obtained an F1-score of 36.3%, recall of 23.7%, accuracy of 87.4%, and precision of 77.42%. 91% is the model's training score, and 87.4% is its test score.

KNN Grid search and tweaking of hyperparameters

Here, my goal was to determine the model's ideal parameters. The weight, distance metric, and n_neighbors(K value) parameter to employ. To prevent overfitting, we also used cross-validation. With hyperparameters {'n_neighbors': 7, 'p': 1, and 'weights': 'distance'}, the K-nearest neighbors (KNN) model's best-performing parameters obtained an accuracy of 88.4%, precision of 80%, recall of 31.6%, and F1-score of 45.39% on the test data, with a 100% training score and an 88% testing score. This model wasn't the best because it overfit the training data.
KNN Group Techniques

By combining several KNN models using the ensemble technique Bagging, i had the chance to enhance the KNN models and produce a more reliable and accurate classifier. The following are the model's performance metrics: F1-score: 41.17%; recall: 27.71%; accuracy: 88%; precision: 80%. Furthermore, the model has an 88% test score and a 91.22% training score. Despite its very low recall score, this turned out to be the best KNN model.
Model 3: Random Forest Classifier
Baseline Model

This project utilizes the Random Forest machine learning algorithm to predict customer churn. It analyzes customer behavior and key features to determine the likelihood of customers leaving the service. The Random Forest model is chosen for its accuracy and ability to handle complex relationships in the data.

Using k-fold cross-validation to address overfitting

To address overfitting, K-fold cross-validation was employed for the Random Forest model. This technique helps assess the model's robustness and generalization by splitting the data into K subsets and iteratively training and testing the model on different combinations.

Random Forest classifier with reduced n_estimators and limited max_depth

To tackle overfitting, adjustments were made to the parameters of the Random Forest classifier, reducing the number of trees (n_estimators) and constraining the maximum depth of each tree (max_depth). These modifications aimed to create a simpler and less complex model, improving its ability to generalize to new data and reduce overfitting.

The ROC curves for Logistic Regression, K Nearest Neighbors, and Random Forest models were analyzed. The Random Forest model outperformed the others, showing a higher Area Under the Curve (AUC) and better classification performance, making it the most effective model.

In conclusion, the analysis suggests that i can accurately predict customer churn using a machine learning model, with the Random Forest Classifier being our recommended model due to its strong overall performance. As this is the best-performing model with a ROC curve that hugs the upper left corner of the graph, hence giving us the largest AUC (Area Under the curve).



