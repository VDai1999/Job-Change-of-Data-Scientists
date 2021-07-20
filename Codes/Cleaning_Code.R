# Load libraries
library(ggplot2)
library(tidyverse)
library(mice)
library(VIM)
library(lattice)
library(ROSE)
library(knitr)
library(gridExtra)



# Read the data
data <- read.csv("aug_train.csv") %>%
  # Delete enroll_id and city variable
  select(-c(enrollee_id, city))
summary(data)
str(data)


# CLEAN DATA
# Recode variables
data <- data %>%
  # Recode variables as factor type
  mutate(gender = as.factor(gender),
         relevent_experience = as.factor(relevent_experience),
         enrolled_university = as.factor(enrolled_university),
         education_level = as.factor(enrolled_university),
         major_discipline = as.factor(major_discipline),
         experience = as.factor(experience), #
         company_size = as.factor(company_size),
         company_type = as.factor(company_type),
         last_new_job = as.factor(last_new_job),
         target = as.factor(target)) %>%
  # Reorder levels of categorical variables in an appropriate order
  mutate(gender =  factor(gender, levels = c("Male", "Female", "Other")),
         enrolled_university = factor(enrolled_university, 
                                      levels = c("no_enrollment", 
                                                 "Part time course", 
                                                 "Full time course")),
         education_level = factor(education_level, 
                                  levels = c("no_enrollment", 
                                             "Part time course", 
                                              "Full time course")),
         major_discipline = factor(major_discipline, 
                                   levels = c("Arts", 
                                              "Business Degree", 
                                              "Humanities",
                                              "STEM", "Other", "No Major")),
         company_size = ifelse(as.character(company_size) == "10/49", "10-49", 
                               as.character(company_size)),
         company_size = as.factor(company_size),
         company_size = factor(company_size, 
                               levels = c("<10", "10-49", "50-99", 
                                          "100-500", "500-999",
                                          "1000-4999", "5000-9999", "10000+")),
         experience = factor(experience, 
                             levels = c("<1", "1", "2", "3", "4", "5", "6", 
                                        "7", "8", "9", "10", "11", "12", "13", 
                                        "14", "15", "16", "17", "18", 
                                        "19", "20", ">20")),
         company_type = factor(company_type, 
                               levels = c("Early Stage Startup", 
                                          "Funded Startup", "Pvt Ltd", 
                                          "Public Sector", "NGO", "Other")),
         last_new_job = factor(last_new_job, 
                               levels = c("never", "1", "2", "3", "4", ">4")))

str(data)


# Collapse levels of experience variable
# The experience variable has 22 levels; thus, we decide to collapse this variable.
# experience = <1 ~ <1
# experience = [1-5] ~ <=5
# experience = [6-10] ~ <=10
# experience = [11-15] ~ <=15
# experience = [16-20] ~ <=20
# experience = >20 ~ >20
data <- data %>%
  mutate(experience = as.character(experience),
         experience = case_when(experience %in% 
                                  c("1", "2", "3", "4", "5") ~ "<=5",
                                experience %in% 
                                  c("6", "7", "8", "9", "10") ~ "<=10",
                                experience %in% 
                                  c("11", "12", "13", "14", "15") ~ "<=15",
                                experience %in% 
                                  c("16", "17", "18", "19", "20") ~ "<=20",
                                experience %in% c("<1", ">20") ~ experience),
         experience = factor(experience, 
                             levels = c("<1", "<=5", "<=10", 
                                        "<=15", "<=20", ">20")))
summary(data$experience)


# Missing values
# A function to calculate the number of missing values and the percentage 
# of records for each variable that has missing values.
missing_values <- function(dat) {
  df <- as.data.frame(cbind(lapply(lapply(dat, is.na), sum))) %>%
    rename(numberOfObservations = V1)
  
  # Change the index into the first column
  df <- cbind(Variables = rownames(df), df)
  rownames(df) <- 1:nrow(df)
  
  # Calculate percentage of missing values
  df <- df %>%
    mutate(numberOfObservations = as.integer(numberOfObservations),
           missingPerc = round(numberOfObservations/nrow(data)*100, 2)) %>%
    filter(numberOfObservations != 0) %>%
    arrange(desc(numberOfObservations))
  
  return (df)
}

missing_df <- missing_values(data)
missing_df
# Plot the number of missing values by variables
missing_df %>%
  arrange(numberOfObservations) %>%
  mutate(Variables = factor(Variables, levels = Variables)) %>%
  ggplot(aes(x=numberOfObservations, y=Variables, 
             label = numberOfObservations)) +
  geom_col(color = "black", fill = "white") +
  geom_text(hjust = -0.5, fontface = "bold") +
  scale_x_continuous(limits = c(0, 6500),
                     breaks = seq(0, 6200, by = 1000)) +
  scale_y_discrete(labels = c("Experience", "Enrolled level", 
                              "Education university", "Last new job", 
                              "Major discipline", "Gender",
                              "Company size", "Company type")) +
  labs(x = "Number of observations",
       y = "Variable") +
  theme_bw()

# Because both enrolled_university and experience variables 
# have the same number of missing value, we doubt that those missing values 
# are from the same observations.
miss_subset_df <- data[rowSums(is.na(data[,c("enrolled_university", 
                                             "education_level")])) > 0, ] %>%
  select(c("enrolled_university", "education_level"))
sum(is.na(miss_subset_df)) 
# 722 => it means that the whole data set contains missing values.
# This confirms our assumption. Since the number of missing values is 
# not really large 

# (386 observations out of 19158 observations), we decide to delete records 
# that have missing values in both columns.
data <- data %>% filter(complete.cases(enrolled_university))
missing_df <- missing_values(data)
missing_df

# For the rest of missing values, we decided to do imputation using 
# MICE package. We assume that the 
# missing data are missing at random (MAR).
# m = no of imputed data set
# maxit = no. of iterations taken to impute missing values
imputed_data <- mice(data, m=2, maxit = 2, seed = 500)
# Get completed data
complete_data <- complete(imputed_data)
sum(is.na(complete_data))

summary(complete_data)

# Save an object to a file
saveRDS(complete_data, file = "complete_data.rds")

# Reread the object
dat <- readRDS(file = "complete_data.rds")
str(dat)
summary(dat)

# Plot the frequency of gender variable before and after imputation.
# Create a data frame.
gender_df <- data.frame(id = seq(1, nrow(dat), by = 1),
                        bf_impute = data$gender,
                        af_impute = dat$gender)
summary(gender_df)
# Change from wide to long format
gender_df <- gender_df %>%
  pivot_longer(cols = -id, names_to = "Imputation", values_to = "Gender")
gender_df <- as.data.frame(table(gender_df$Imputation, gender_df$Gender))
gender_df <- gender_df %>%
  rename(Imputation = Var1,
         Gender = Var2) %>%
  mutate(Imputation = factor(Imputation, levels = c("bf_impute", "af_impute")))

# Plot
ggplot(data=gender_df, aes(x = Gender, y=Freq, fill = Imputation)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_text(aes(label = Freq), fontface = "bold", vjust = -0.5,
            position = position_dodge(.9), size = 4) +
  labs(y = "Number of observations") +
  scale_y_continuous(limits = c(0, 17500)) +
  scale_fill_discrete(name = "",
                      labels = c("Before Imputation", "After Imputation")) +
  theme_bw()

# Percentage changes
# Male
(16947-13041)/13041
# Female
(1592-1218)/1218
# Other
(273/178)/178


# Imbalanced data set
dat <- dat %>%
  mutate(target = ifelse(target == "0", "No", "Yes"),
         target = as.factor(target))

#Frequency of the target variable
table(dat$target)
# Proportion of 2 classes
prop.table(table(dat$target))


