![Logo](https://e7.pngegg.com/pngimages/756/750/png-clipart-data-analysis-business-analytics-data-science-big-data-business-text-resume-thumbnail.png)

# The Analysis of Job Change of Data Scientists

In this project, our goal is to classify if 
candidates want to work at a company after joining 
the training course for a Data Scientist position.
Another aim is to determine which features impact 
candidates' decisions to not continue working for 
the company.


## Description

### Context and Content
A company which is active in Big Data and Data Science 
wants to hire data scientists among people who successfully 
pass some courses which conduct by the company. Many people 
signup for their training. Company wants to know which of 
these candidates are really wants to work for the company after 
training or looking for a new employment because it helps to 
reduce the cost and time as well as the quality of training 
or planning the courses and categorization of candidates. 
Information related to demographics, education, experience 
are in hands from candidates signup and enrollment.

This dataset designed to understand the factors that lead 
a person to leave current job for HR researches too. 
By model(s) that uses the current credentials, demographics,
experience data you will predict the probability of a 
candidate to look for a new job or will work for the company, 
as well as interpreting affected factors on employee decision.

### Features
|variable               |description |
|:---|:-----------|
|enrollee_id            | Unique ID for candidate|
|city                   | City code|
|city_development_index | Developement index of the city (scaled)|
|gender                 | Gender of candidate|
|relevent_experience    | Relevant experience of candidate|
|enrolled_university    | Type of University course enrolled if any|
|education_level        | Education level of candidate|
|major_discipline       | Education major discipline of candidate|
|experience             | Candidate total experience in years|
|company_size           | No of employees in current employer's company|
|company_type           | Type of current employer|
|lastnewjob             | Difference in years between previous job and current job|
|training_hours         | Training hours completed|
|target                 | 0 – Not looking for job change, 1 – Looking for a job change|
  
  
## Installation

This project requires **R version 4.0.3** 
with **R Studio Version 1.3.1093** installed.

### R
[Install R](https://www.r-project.org/)

### R Studio
[Install R Studio](https://www.rstudio.com/products/rstudio/download/)

    
## Appendix

Libraries used:
```{r}
library(ggplot2)
library(tidyverse)
library(caret)
library(mice)
library(VIM)
library(lattice)
library(ROSE)
library(pROC)
library(ROCR)
library(knitr)
library(kableExtra)
library(gridExtra)
library(gbm)
```

 
## Acknowledgements
The retrieved data is [Coronavirus (Covid-19) Data in the United States](https://github.com/nytimes/covid-19-data)
