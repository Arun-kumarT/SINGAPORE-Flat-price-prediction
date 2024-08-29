
# Price Prediction of Singapore Flats Using Machine Learning: A Streamlit-Based Web Application

This project involves the creation and deployment of a machine learning-based web application designed to predict the resale prices of flats in Singapore.

## Documentation

The model is trained on a dataset of 930,000 transactions using a RandomForest Regressor , considering factors like location, flat type, floor area, and lease duration.

The project encompasses data preprocessing, feature engineering, model training, and thorough evaluation achieving an R² score of 98.4%.


The Streamlit-based web application, deployed on Render cloud platform for application deployment.
, integrates artificial intelligence to provide accurate resale price predictions based on user inputs, demonstrating effective coding and communication in presenting results.

[Documentation](https://docs.google.com/document/d/1mPb68zw8G-iFNcFr4hSAp7yIXc3-0JVlFVKBPf-0Hxo/edit)


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

-streamlit==1.32.0

-pandas==2.2.1

-scikit-learn==1.1.3


## Demo

 link to demo

https://drive.google.com/file/d/1t5xSMcnLBjNA_DXcknPCUHEehOrByQIE/view?usp=sharing
## Features

- Prediction of Real Resale Flat Price
- With custom input from user



## 🚀 About Me
As a recent graduate with a Master of Data Science from IIT-Madras, you have acquired a solid foundation in data science, statistics, and machine learning. Your education has been complemented by hands-on experience through various projects, where you've applied theoretical knowledge to solve real-world problems. This combination of academic excellence and practical exposure equips you with the necessary skills to excel in a fast-paced, data-driven environment.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
https://github.com/Arun-kumarT
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
https://www.linkedin.com/in/arunkumar-t-8745242ba



## 🛠 Skills
Programming: Python.

Database: MySQL,Mongodb

Data Visualization: PoweBI, Plotly, Seaborn

ML models: scikit-learn

Deep Learning TensorFlow, Pytorch

Natural Language Processing (NLP)
LLM Models and Tools


## Lessons Learned

Here are some key lessons learned from the project:

1. Importance of Data Quality and Preprocessing
Lesson: The accuracy of the model heavily depends on the quality of the input data. Handling missing values, outliers, and irrelevant features is crucial for building a robust model.
Experience: During the project, extensive data cleaning and preprocessing were necessary to ensure that the features such as location, flat type, and lease duration were accurately represented in the model.

2. Feature Engineering is Key to Model Performance
Lesson: Creating new features or modifying existing ones can significantly improve model performance. In this project, feature engineering played a vital role in achieving a high R² score.
Experience: Features like the proximity to amenities, public transport, and the age of the flat were derived and fine-tuned, which enhanced the predictive power of the model.

3. Model Selection and Hyperparameter Tuning
Lesson: Choosing the right model and tuning its hyperparameters can make a substantial difference in the model’s accuracy. RandomForest Regressor was chosen for its ability to handle large datasets and complex relationships.
Experience: Through multiple iterations of model training and hyperparameter tuning, the project achieved an impressive R² score of 98.4%, highlighting the importance of this step in predictive modeling.

4. Deployment Challenges and Solutions
Lesson: Deploying a machine learning model as a web application introduces new challenges, such as optimizing for speed and ensuring the user interface is intuitive.
Experience: Deploying the model using Streamlit required careful consideration of how user inputs were processed and how predictions were displayed, ensuring that the application was both responsive and user-friendly.

5. Communication and Presentation of Results
Lesson: Effectively communicating the results and the model's decision-making process is crucial, especially in domains like real estate, where end-users may not be familiar with machine learning concepts.
Experience: Building the Streamlit web application provided a platform to present the predictions in a clear and accessible manner, demonstrating the practical value of the model to users.

6. Real-World Applicability and Ethical Considerations
Lesson: Applying machine learning in real-world scenarios requires an understanding of the broader impact, including ethical considerations and potential biases in the model.
Experience: The project emphasized the need to be mindful of how factors like location and property type could inadvertently introduce biases into the predictions, prompting careful evaluation and adjustment to ensure fairness and accuracy.


These lessons have not only enhanced my technical skills but also provided valuable insights into the practical aspects of deploying machine learning models in real-world applications.


## Run Locally

Clone the project

```bash
  git clone https://github.com/Arun-kumarT/SINGAPORE-Flat-price-prediction.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install all requirement.txt
```

Start the server

```bash
  streamlit run app.py
```


## Related

Here are some related projects

[Awesome README](https://github.com/Arun-kumarT?tab=repositories)

