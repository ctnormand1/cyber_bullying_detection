# Project 5 - Cyber Bullying Detection!
### Problem Statement
Many content providers on the internet give users the ability to write comments
with the goal of promoting healthy discussion. Unfortunately, comment sections
are prone to abuse, and many users experience cyberbullying in the form of
toxic, aggressive, or attack-laden comments. Programmatic detection of
cyberbullying has a strong use case for content providers who seek to remove
harmful content from their comment sections.  In this project, we address this
problem by developing a set of models to predict the likelihood that a comment
contains a personal attack, toxic language, or aggressive tone.

### Timeline
| Date           | Description                       | Status        |
| -------------- | --------------------------------- | ------------- |
| Tue 2021-08-03 | Project kickoff                   | Complete      |
| Fri 2021-08-06 | Choose topic & get data           | Complete      |
| Thu 2021-08-12 | Cleaning & EDA complete           | Complete      |
| Mon 2021-08-16 | Preliminary modeling complete     | Complete      |
| Wed 2021-08-18 | Final model selection             | Complete      |
| Fri 2021-08-20 | Project complete (soft deadline)  | Complete      |
| Mon 2021-08-23 | Presentations                     | In progress   |

### Methodology
We used data from the Wikimedia Research: Detox project, which was a 2017
endeavor by the Wikimedia Foundation (WMF) to programmatically recognize
toxicity, attacks, and aggression on it's discussion boards. WMF published a
paper in which they used logistic regression and a multilayer perceptron neural
net to identify posts containing an attack. We expand on this work by training a
logistic regression, Naive Bayes, support vector classifier, and XGBoost models
on the toxicity, attacks, and aggression datasets.

The source data already went through some level of preprocessing, but we
implemented a couple more steps to improve compatibility with our models. These
steps included:
- Removing non-ascii characters
- Replacing "NEWLINE_TOKEN" and "TAB_TOKEN" with whitespace
- Removing repeated informational comments that were likely posted by
wiki-admins.

We then trained logistic regression, Naive Bayes, support vector classifier, and
XGBoost models on the three datasets. After a full gridsearch over
hyperparameters for each model, we determined that XGBoost was producing the
most accurate results. At this point, we performed additional gridsearches on
the XGBoost model, but any accuracy improvements over first model iteration were
negligible.

We created a Streamlit app to demonstrate our cyberbullying detection model.
XGBoost is the model we decided to deploy in the app because it produced the
highest accuracy scores.

### Results

The table below shows the testing accuracy of each model for the toxicity,
attack, and aggression dataset.

| Model               | Toxicity | Attack    | Aggression |
| ------------------- | -------- | --------- | ---------- |
| Baseline            | 88.3%    | 86.7%     | 85.3%      |
| Logistic regression | 89.7%    | 88.5%     | 87.1%      |
| Naive Bayes         | 88.6%    | 89.1%     | 86.9%      |
| SVC                 | 79.1%    | 71.6%     | 72.0%      |
| XGBoost             | 91.7%    | 90.6%     | 89.4%      |

Accuracy is useful for a general evealuation of model performance. However, if
we were to further tune our models, we would focus on maximizing **recall**.
By doing so, we would be doing our best to ensure any comments that contain
cyberbullying *do not* evade detection. The table below shows the recall scores
for each model.

| Model               | Toxicity | Attack    | Aggression |
| ------------------- | -------- | --------- | ---------- |
| Logistic regression | 86.1%    | 83.9%     | 82.9%      |
| Naive Bayes         | 83.6%    | 77.9%     | 76.4%      |
| SVC                 | 77.7%    | 63.9%     | 61.9%      |
| XGBoost             | 79.6%    | 78.5%     | 76.8%      |

The recall scores are significantly lower than the accuracy scores. This
indicates that although there's a relatively low chance of the models
flagging a non-toxic/attack/aggressive comment, it is *quite likely* for the
models to incorrectly ignore a comment that *is* a toxic/attack/aggressive
comment. Optimizing the models for recall would be an important step for a
continuation of this project.

### Conclusions
Cyberbullying, and more generally harassment online, negatively impacts a
significant amount of people on the internet. The results of this project show
it is possible to use machine learning to detect personal attacks, toxic
language, and aggressive comments online. A company implementing even the
basic models we developed in this project would see improvement over baseline,
leading to cost savings and better resource allocation. Future improvements to
these models, focused especially on recall rather than accuracy, could have
even greater benefits for the internet community at large.

### References
**Data Sources**
- [Wikipedia Talk Data](https://figshare.com/projects/Wikipedia_Talk/16731)

**Other References**
- [Wikimedia Research: Detox Project Page](https://meta.wikimedia.org/wiki/Research:Detox)
- [Wikimedia Research: Detox Data Description](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release)
- [Wikimedia Research: Detox Project Report](https://arxiv.org/pdf/1610.08914.pdf)
- [Google Jigsaw Report on Toxicity](https://jigsaw.google.com/the-current/toxicity/)
- [Pew Research: State of Online Harassment](https://www.pewresearch.org/internet/2021/01/13/the-state-of-online-harassment/)
- [Anti-Defamation League: Hate and Harassment Online](https://www.adl.org/onlineharassment)
