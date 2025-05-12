# Data Science Project on European Bank Customer Churns
## Table Of Content

Part 1. Case Descriptions.
Part 2. EDA Findings.
Part 3. 1. Machine Learning Model – Logistic Regression.
Part 3. 2. Machine Learning Model – Decision Tree.
Part 3. 3.  Machine Learning Model – Deep Learning.
Part 4. Cluster Model – K Means Clustering.
Part 5. Conclusions & Recommendations. 

## Part 1. Case Descriptions
The project had studied a dataset of customers from a European bank that operates in Germany, France, and Spain. The goal was to discover the pattern of features for customers who churned. Last but not the least, a cluster model would be performed to further group the customers, so the company would be able to customize its marketing strategy for different groups.

## Part 2. EDA Findings.
1.	The number of total unique customers is 10000. 20.37% or 2037 of customers had churned or had left the bank. 
2.	The age of clients ranges from 19 to 92 with the average age of 39. The estimated salaries went from 11.58 and 200,000 Euros with an average of 100,090 Euros. The bank balance was between 0 Euros and 250,898 Euros with a mean of 76,485 Euros. 75% of customers have credits, and 51% of customers are active members. Customers had spent as long as 10 years and as short as less than one year with the bank with an average tenure of 5 years.
3.	The credit scores and age is normally distributed.  For Balance, many clients have 0 balances in their accounts, and for the rest, the balance was randomly distributed. Many customers have only 1 and 2 products. Estimated Salary appeared to be very uniformly distributed.

![image](https://github.com/user-attachments/assets/1d2ae52f-2ba4-45db-bfe8-d947a17c539d)

 
4.	There were no null values in the dataset.  

![image](https://github.com/user-attachments/assets/01d529a9-44f0-46c7-8edd-6647d040c931)


5.	Churned customers’ average age was 44 comparing to that of un churned customers, which was 37 years old. Churned customers have a higher balance mean of 91,108 Euros whereas the un churned customers have a balance mean of 72745 Euros. 36% the churned are active members and 55% of the un churned are active members.  That is churned customers tend to be older, have a higher balance and less likely to be an active member than those who stayed. 
 
![image](https://github.com/user-attachments/assets/6d002129-bcbe-49a8-b6c6-8f3f0aaff8b7)
![image](https://github.com/user-attachments/assets/43c3ed7b-0e74-4283-9e53-d4b286f2baf2)
![image](https://github.com/user-attachments/assets/54bc03bc-c294-4d95-8862-97892065864a)

 
6.	Correlation Analysis. Balance was negatively correlated with number of products. Age has a strong correlation with churn. That is the older a person is, the more likely they would close their accounts.

![image](https://github.com/user-attachments/assets/741c1ae0-a58d-4aff-8b8e-79ea5a3b37f5)

 
7.	Churn rate for different features. For age, clients between age of 45 and 56 and 84 have very high churn rate. German clients have a churn rate of 30% twice as much as those from France and Spain. Customers with 3 and 4 products had almost churned 100%. Clients who are not active members have a churn rate twice as high as those who are active member. 
 
![image](https://github.com/user-attachments/assets/e9184e40-2157-4cba-910d-d80444030cc2)
![image](https://github.com/user-attachments/assets/b5256f05-7232-41e8-854d-46d73887f39b)
![image](https://github.com/user-attachments/assets/c6009be2-ed69-4aec-99e4-369e7d581a64)
![image](https://github.com/user-attachments/assets/0c9c12a9-eb4c-4875-9051-a61d9f32d27b)


 


8.	For different nations, they have similar distribution for the estimated salary, however, Germany has a much higher balance than its peers. For different genders, they do have very similar distribution of balance and estimated salaries.

![image](https://github.com/user-attachments/assets/ebad4e3c-8750-440a-9edc-b41829d3c27b)

![image](https://github.com/user-attachments/assets/7e51ed3e-3ff0-4d98-88d4-2d75dd1402f2)


## Part 3.1 Logistic Regression
9.	Applied one hot encoding to Gender, Geography. Performed model tuning with grid search cv with C as np.linspace(0.1,2,20), and penalty as [“l1”, “l2”]. The best parameters are C as 0.1 and penalty as l1.

10.	Model evaluation. The training accuracy was 0.708875, and the testing accuracy was 0.72. The model is not overfitting and the accuracy was acceptable. The precision and recall for the positive cases are 39% and 72%. It means that the model is better at predicting negative cases than positive cases. AUC was 78%, which means it is better than random guessing which is 50%.

![image](https://github.com/user-attachments/assets/94d6690d-e391-4bcd-99ef-e94d803379e5)


## Part 3.2 Decision Trees
11.	Applied Grid search cv for the best parameters. Also retrained the  model with important features including Age, NumOfProducts,Balance, IsActiveMember, CreditScore, EstimmatedSalary. The params were set as such:
•	"max_depth":[5,10,15,20],
* "min_samples_leaf":np.arange(5,21,5),
•	"criterion":["entropy","gini"],

![image](https://github.com/user-attachments/assets/dfd3abdd-44ea-4773-b260-e7772957b4a7)


12.	The best params were {criterion: entropy, max_depth: 5, min_sample_leaf: 10}.
13.	Model evaluation. The training accuracy was 77%. The testing was 76%. It's not overfitting and with better accuracy. However, once again, the model is better at predicting negative cases than positive cases. AUC was improved a lot as well. The result was much better than logistic regression.

![image](https://github.com/user-attachments/assets/ef075a81-544d-44ac-a2c7-09416f41943b)


## Part 3.3 Deep Learning Model
14.	Deep Learning arkitect. Set up input and output layers with 2 hidden layers in the middle. 20 neuros in the input layer, 10 for the hidden layers, and 1 for the output layer. 

![image](https://github.com/user-attachments/assets/d06fcef1-945e-4cb8-984d-f31b75ef57f0)

 
15.	Model Evaluation. The training accuracy was 87% and the testing accuracy was 86%. The precision and the recall for positive cases was 0.72 and 0.47. Once again, it’s good at predicting negative cases but not so with positive cases. It’s the best model so far.

![image](https://github.com/user-attachments/assets/fd93d68b-e551-4778-984b-2616c08daed1)


## Part 4. K Means Clustering Models
16.	Applied K Means clustering. Calculated a series of inertias and silhouette scores for different Ks. Eventually, set k=6. And Each cluster was labeled as the following:
0:"0 Female. Low Number of Products. All Credit Cards.",
1:"1 Low Balance. High Number of Products. All Credit Cards.",
2:"2 Low Balance. Low Cradit cards.",
3:"3 Older People. Most are Active Members.",
4:"4 Males. Low number of products. All Credit Cards.",
5:"5 High Balance. Low Number of products. No Credit Cards."
Cluster 0, 3, and 5 has the highest attrition rate. Cluster 1 has the lowest churn rate. Cluster 1 and 2 have low balances, the bank should help them to save.

![image](https://github.com/user-attachments/assets/3abc8b20-057b-4118-a347-4a11803ce917)

![image](https://github.com/user-attachments/assets/302fe1de-13e6-4ac6-bd44-7f1b7c500f9f)

![image](https://github.com/user-attachments/assets/73e88fc7-1e72-4c87-8b4e-be87435db810)

17. Cluster Insights through histograms. Cluster 3 has the highest churn rate. It has the most senior members. They also have a low balance. Most of them have credit cards. And most of them are active members. It has confirmed the insights we get from DEA. The seniors have a tendency to leave the bank.
![image](https://github.com/user-attachments/assets/a6ee5a5c-b32c-4c6a-ace4-d4b6e0ba5de4)

![image](https://github.com/user-attachments/assets/6ccf441f-e5f7-4b30-82f4-2de3bfdeb891)

 
18.	PCA visualization. The clusters were not clustered clearly. They seem to be mixed up among one another. Along the x-axis, toward the plus direction, the cluster tends to have more number of products, and toward the minus direction, the cluster tends to have high balances. Along the y-axis, towards the positive direction, the cluster tends to have more active members, older members.

![image](https://github.com/user-attachments/assets/c488d5bb-1d3d-4eb8-b2ca-ef61f2390c72)


## Conclusions & Recommendations
1.	Insights. Churned customers tend to be older, have a higher balance and less likely to be an active member than those who stayed. They are also likely to have 3 and 4 products. Additionally, they have higher balances. However, they all have similar credit card scores and estimated salaries.
2.	The best model is the deep learning model with a training accuracy of 87% and a testing accuracy of 86%.
#### Recommendation for each cluster
•	Group 0. Female. Low Number of Products. All Credit Cards: High Attrition (27%) with high balances. Talk to them on how bank can help them grow their savings. 
•	Group 1. Low Balance. High Number of Products. All Credit Cards: Low Attrition rate (11%). Monitor their balance make sure they ae not running out of their cash.
•	Group 2. Low Balance. Low Cradit cards.: Low attrition rate (17%). High earners. Sign them up for credit cards and more financial products to help them save.
•	Group 3. Older People. All Active Members: Highest Attritions (30%). It makes sense since there are lots of seniors. Wish them a good retirement.
•	Group 4. Males. Low number of products. All Credit Cards: Low attritions (17%). Talk to them on how bank can help them grow their savings.
•	Group 5. High Balance. Low Number of products. No Credit Cards.: High Attritions (22%). Make an appointment. Recommend credit cards and more financial products.

## The End & Thank You For Reading.

