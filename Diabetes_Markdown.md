Characterization and Classification of Diabetic Retinopathy
================
Santosh Gummidipundi
1/7/2020

  - [Dataset Information](#dataset-information)
      - [Data Dictionary](#data-dictionary)
  - [Objective](#objective)
  - [Descriptive Analyses](#descriptive-analyses)
      - [Table 1](#table-1)
      - [Descriptive Plots](#descriptive-plots)
  - [Predictions](#predictions)
      - [Feature Selection](#feature-selection)
      - [Model Training](#model-training)
  - [Results](#results)

# Dataset Information

The data analyzed here has been sourced from the UCI Machine learning
repository.
<https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set>.

“The dataset consists of features extracted from the Messidor image set
to predict whether an image contains signs of diabetic retinopathy or
not.” - UCI Machine Learning Repository

## Data Dictionary

``` r
#This pulls in another CSV which specifies the data dictionary
dd <- read.csv("data/data_dictionary.csv")

kable(dd, 
      format = formatSet,
      col.names = c("Col #","Variable","Description","Type"),
      booktabs = TRUE) %>%
  kable_styling(bootstrap_option = "striped",
                position = "center")
```

    ## Warning in kable_styling(., bootstrap_option = "striped", position =
    ## "center"): Please specify format in kable. kableExtra can customize either
    ## HTML or LaTeX outputs. See https://haozhu233.github.io/kableExtra/ for
    ## details.

| Col \# | Variable                             | Description                                                           | Type                                                                            |
| :----- | :----------------------------------- | :-------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| 1      | Quality Assessment                   |                                                                       | integer; 0 = bad quality, 1 = sufficient quality                                |
| 2      | Pre-Screening Result                 |                                                                       | integer; 0 = lack of severe retinal abnormality, 1 = severe retinal abnormality |
| 3-8    | Results of MA Detection              | Number of microaneurysms founds at different confidence intervals     | Integer                                                                         |
| 9-16   | Exudate                              | Number of lesions divided by diameter or ROI                          | Numeric                                                                         |
| 17     | Distance – Center of Macula          | Distance from center of macula to center of optic disc divided by ROI | Continuous                                                                      |
| 18     | Diameter – Optic Disc                |                                                                       | Continuous                                                                      |
| 19     | AM/FM Classification                 |                                                                       | Binary                                                                          |
| 19     | Class label for diabetic retinopathy | Tells us if there is signs of DR                                      | Binary                                                                          |

``` r
#Read in data
df <- read.arff("data/messidor_features-1.arff")
```

``` r
#Set column names because there is no header in this data
colnames(df) <- c("quality",
                  "pre_screen",
                  "ma_1",
                  "ma_2",
                  "ma_3",
                  "ma_4",
                  "ma_5",
                  "ma_6",
                  "ex_1",
                  "ex_2",
                  "ex_3",
                  "ex_4",
                  "ex_5",
                  "ex_6",
                  "ex_7",
                  "ex_8",
                  "dist",
                  "diam",
                  "am_fm",
                  "class")

#Create a copy of the data for later use in model training and testing
df_copy <- df

#Change attributes for variables for use in descriptive analyses
df <- df %>%
      mutate_if(names(.) %in% c("quality","pre_screen","am_fm","class"), yes_no) 

#These variable dictionaries will be used in creating table 1
numeric_labels <- c("ma_1" = "alpha = 0.5",
                    "ma_2" = "alpha = 0.6",
                    "ma_3" = "alpha = 0.7",
                    "ma_4" = "alpha = 0.8",
                    "ma_5" = "alpha = 0.9",
                    "ma_6" = "alpha = 1.0",
                    "ex_1" = "alpha = 0.3",
                    "ex_2" = "alpha = 0.4",
                    "ex_3" = "alpha = 0.5",
                    "ex_4" = "alpha = 0.6",
                    "ex_5" = "alpha = 0.7",
                    "ex_6" = "alpha = 0.8",
                    "ex_7" = "alpha = 0.9",
                    "ex_8" = "alpha = 1.0",
                    "dist" = "Distance – Center of Macula",
                    "diam" = "Diameter – Optic Disc")

categorical_labels <- c("am_fm" = "AM/FM Classification",
                        "quality" = "Quality Assessment",
                        "pre_screen" = "Pre-Screening Result")
```

# Objective

The objective of this analysis is to describe and characterize the image
features extracted from the Messidor image set in addition to performing
an array of ML models to predict whether a patient has diabetic
retinopathy.

# Descriptive Analyses

Ordinarily, a study of predictive nature a descriptive analyses should
not be performed for on a whole dataset and rather just the training
dataset. However, for the purposes of this demonstration of programming
and reproducibility it will be performed.

## Table 1

``` r
#This creates a very basic table 1 plot for getting an overview of predictors and class balance.
#Summaries for overall group
summary_numeric <- df %>%
                   select_if(is.numeric) %>%
                   desctable(stats =  list("N" = length,
                                           "% or Mean (SD)" = is.numeric ~ mean_sd),
                   labels = numeric_labels) %>%
                   as.data.frame

summary_cat <- df %>%
               select(-class) %>%
               select_if(is.factor) %>%
               desctable(stats =  list("N" = length,
                                       "% or Mean (SD)" = is.factor ~ percent),
               labels = categorical_labels) %>%
               as.data.frame %>%
               mutate_if(is.numeric, funs(round(., digits = 1)))

summary <- rbind(summary_numeric, summary_cat) 
colnames(summary)[1] <- "var"


#Summaries stratified by class
summary_numeric_class <- df %>%
                         group_by(class) %>%
                         select_if(is.numeric) %>%
                         desctable(stats =  list("N" = length,
                                                 "% or Mean (SD)" = is.numeric ~ mean_sd),
                         labels = numeric_labels) %>%
                         as.data.frame

summary_cat_class <- df %>%
                     select_if(is.factor) %>%
                     group_by(class) %>%
                     desctable(stats =  list("N" = length,
                                             "% or Mean (SD)" = is.factor ~ percent),
                     labels = categorical_labels) %>%
                     as.data.frame %>%
                     mutate_if(is.numeric, funs(round(., digits = 1)))

summary_class <- rbind(summary_numeric_class, summary_cat_class)
colnames(summary_class)[1] <- "var"



summary <- cbind(summary, summary_class %>% select(-var))
summary$var <- gsub(".*:","",summary$var, perl = TRUE)

           
y <- kable(summary,
           format = "html",
           booktabs = TRUE,
           col.names = c(" ","N","%","N","%","N","%","p-value","test"),
           align = c("l",rep("c",8)),
           caption = "Table 1") %>%
    add_header_above(header = c(" " = 1, " " = 2, "DR Positive (n = 611)" = 2, "DR Negative (n = 540)" = 2,"Tests" = 2), align = "c") %>%
    add_header_above(header = c(" " = 1, "Overall" = 2, "Stratified" = 6), align = "c") %>%
    kable_styling(full_width = TRUE) %>%
    group_rows("Microaneurysms",1,6) %>%
    group_rows("Exudate Pixels",7,14) %>%
    group_rows("Quality Assessment",17,19) %>%
    group_rows("Pre-Screening Result",20,22) %>%
    group_rows("AM/FM Classification",23,25) %>%
    

print(y)
```

<table class="table" style="margin-left: auto; margin-right: auto;">

<caption>

Table 1

</caption>

<thead>

<tr>

<th style="border-bottom:hidden" colspan="1">

</th>

<th style="border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="2">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

Overall

</div>

</th>

<th style="border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="6">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

Stratified

</div>

</th>

</tr>

<tr>

<th style="border-bottom:hidden" colspan="1">

</th>

<th style="border-bottom:hidden" colspan="2">

</th>

<th style="border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="2">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

DR Positive (n = 611)

</div>

</th>

<th style="border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="2">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

DR Negative (n = 540)

</div>

</th>

<th style="border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="2">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

Tests

</div>

</th>

</tr>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:center;">

N

</th>

<th style="text-align:center;">

%

</th>

<th style="text-align:center;">

N

</th>

<th style="text-align:center;">

%

</th>

<th style="text-align:center;">

N

</th>

<th style="text-align:center;">

%

</th>

<th style="text-align:center;">

p-value

</th>

<th style="text-align:center;">

test

</th>

</tr>

</thead>

<tbody>

<tr grouplength="6">

<td colspan="9" style="border-bottom: 1px solid;">

<strong>Microaneurysms</strong>

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.5

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

38.43 (25.6)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

45.47 (27.4)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

30.46 (20.7)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.6

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

36.91 (24.1)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

42.94 (25.4)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

30.08 (20.5)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.7

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

35.14 (22.8)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

40.17 (23.8)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

29.45 (20.2)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.8

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

32.3 (21.1)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

36.22 (21.9)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

27.86 (19.3)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.9

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

28.75 (19.5)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

31.71 (20.1)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

25.39 (18.3)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 1.0

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

21.15 (15.1)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

22.97 (15.6)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

19.1 (14.3)

</td>

<td style="text-align:center;">

0.0000086

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr grouplength="8">

<td colspan="9" style="border-bottom: 1px solid;">

<strong>Exudate Pixels</strong>

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.3

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

64.1 (58.5)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

67.29 (64.4)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

60.49 (50.8)

</td>

<td style="text-align:center;">

0.4905083

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.4

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

23.09 (21.6)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

23.1 (23.2)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

23.08 (19.7)

</td>

<td style="text-align:center;">

0.0873105

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.5

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

8.7 (11.6)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

9.12 (12.4)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

8.23 (10.6)

</td>

<td style="text-align:center;">

0.9262427

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.6

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

1.84 (3.9)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

2.22 (4.7)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

1.4 (2.8)

</td>

<td style="text-align:center;">

0.0234166

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.7

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

0.56 (2.5)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

0.89 (3.3)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

0.18 (0.6)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.8

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

0.21 (1.1)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

0.36 (1.4)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

0.04 (0.2)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 0.9

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

0.09 (0.4)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

0.15 (0.5)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

0.01 (0)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

alpha = 1.0

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

0.04 (0.2)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

0.07 (0.2)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

0 (0)

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left;">

Distance – Center of Macula

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

0.52 (0)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

0.52 (0)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

0.52 (0)

</td>

<td style="text-align:center;">

0.7615216

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr>

<td style="text-align:left;">

Diameter – Optic Disc

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

0.11 (0)

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

0.11 (0)

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

0.11 (0)

</td>

<td style="text-align:center;">

0.2628535

</td>

<td style="text-align:center;">

wilcox.test

</td>

</tr>

<tr grouplength="3">

<td colspan="9" style="border-bottom: 1px solid;">

<strong>Quality Assessment</strong>

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

Quality Assessment

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

fisher.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

Yes

</td>

<td style="text-align:center;">

1147

</td>

<td style="text-align:center;">

99.7

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

100

</td>

<td style="text-align:center;">

536

</td>

<td style="text-align:center;">

99.3

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

No

</td>

<td style="text-align:center;">

4

</td>

<td style="text-align:center;">

0.3

</td>

<td style="text-align:center;">

0

</td>

<td style="text-align:center;">

0

</td>

<td style="text-align:center;">

4

</td>

<td style="text-align:center;">

0.7

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

</td>

</tr>

<tr grouplength="3">

<td colspan="9" style="border-bottom: 1px solid;">

<strong>Pre-Screening Result</strong>

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

Pre-Screening Result

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

0.0000000

</td>

<td style="text-align:center;">

fisher.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

Yes

</td>

<td style="text-align:center;">

1057

</td>

<td style="text-align:center;">

91.8

</td>

<td style="text-align:center;">

549

</td>

<td style="text-align:center;">

89.9

</td>

<td style="text-align:center;">

508

</td>

<td style="text-align:center;">

94.1

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

No

</td>

<td style="text-align:center;">

94

</td>

<td style="text-align:center;">

8.2

</td>

<td style="text-align:center;">

62

</td>

<td style="text-align:center;">

10.1

</td>

<td style="text-align:center;">

32

</td>

<td style="text-align:center;">

5.9

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

</td>

</tr>

<tr grouplength="3">

<td colspan="9" style="border-bottom: 1px solid;">

<strong>AM/FM Classification</strong>

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

AM/FM Classification

</td>

<td style="text-align:center;">

1151

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

611

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

540

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

0.2000000

</td>

<td style="text-align:center;">

fisher.test

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

Yes

</td>

<td style="text-align:center;">

387

</td>

<td style="text-align:center;">

33.6

</td>

<td style="text-align:center;">

194

</td>

<td style="text-align:center;">

31.8

</td>

<td style="text-align:center;">

193

</td>

<td style="text-align:center;">

35.7

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

</td>

</tr>

<tr>

<td style="text-align:left; padding-left: 2em;" indentlevel="1">

No

</td>

<td style="text-align:center;">

764

</td>

<td style="text-align:center;">

66.4

</td>

<td style="text-align:center;">

417

</td>

<td style="text-align:center;">

68.2

</td>

<td style="text-align:center;">

347

</td>

<td style="text-align:center;">

64.3

</td>

<td style="text-align:center;">

</td>

<td style="text-align:center;">

</td>

</tr>

</tbody>

</table>

## Descriptive Plots

``` r
#This section details select graphical plots for variables of interest
#----------------------------Mean Exudates/ROI-------------------------------#
df_summary <- df %>%
              select(starts_with("ex"),"class") %>%
              group_by(class) %>%
              summarize_all(mean) %>%
              gather(ex, value, ex_1:ex_8)
df_summary$ex <- factor(df_summary$ex,
                        levels = c("ex_1",
                                   "ex_2",
                                   "ex_3",
                                   "ex_4",
                                   "ex_5",
                                   "ex_6",
                                   "ex_7",
                                   "ex_8"),
                        labels = seq(0.3,1.0,0.1))
              
plot <- ggplot(df_summary,
               aes(x = ex,
                   y = value,
                   fill = ex)) +
        geom_bar(position = "dodge",
                 stat = "identity",
                 aes(y = value,
                     x = factor(ex))) +
        theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
              legend.position = "none") +
        xlab("CI Alpha") +
        ylab("Mean Exudate/ROI") +
        ggtitle("Mean Exudate/ROI Stratified by Diabetic Retinopathy Outcome") +
  facet_grid(class ~ .)
plot
```

![](Diabetes_Markdown_files/figure-gfm/DA%20Plots-1.png)<!-- -->

``` r
#----------------------------Mean Microaneurysms-------------------------------#
df_summary <- df %>%
              select(starts_with("ma"),"class") %>%
              group_by(class) %>%
              summarize_all(mean) %>%
              gather(ma, value, ma_1:ma_6)
df_summary$ma <- factor(df_summary$ma,
                        levels = c("ma_1",
                                   "ma_2",
                                   "ma_3",
                                   "ma_4",
                                   "ma_5",
                                   "ma_6"),
                        labels = seq(0.5,1.0,0.1))
              
plot <- ggplot(df_summary,
               aes(x = ma,
                   y = value,
                   fill = ma)) +
        geom_bar(position = "dodge",
                 stat = "identity",
                 aes(y = value,
                     x = factor(ma))) +
        theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
              legend.position = "none") +
        xlab("CI Alpha") +
        ylab("Mean Microaneurysms") +
        ggtitle("Mean Microaneurysms Stratified by Diabetic Retinopathy Outcome") +
  facet_grid(class ~ .)
plot
```

![](Diabetes_Markdown_files/figure-gfm/DA%20Plots-2.png)<!-- -->

``` r
#-------------------AM/FM Classification---------------------------------#
df_summary <- df %>%
              select("am_fm","class") %>%
              mutate(am_fm = as.character(am_fm)) %>%
              group_by(class, am_fm) %>%
              tally %>%
              mutate(sum = sum(n)) %>%
              mutate(f = n/sum, am_fm = factor(am_fm, levels = c("Yes","No"), labels = c("AM","FM")))

plots <- ggplot(df_summary,
                aes(x = factor(am_fm),
                    weight = f,
                    fill = class)) +
         geom_bar(position = "dodge",
                  stat = "identity",
                  aes(y = f,
                      x = factor(am_fm))) +
        theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
              legend.position = "none") +
        facet_grid(class ~., scales = "free") +
        xlab("AM/FM Classification") +
        ylab("Proportion") +
        ggtitle("Proportion AM/FM Stratified by Diabetic Retinopathy Outcome") +
        scale_y_continuous(limits = c(0,1))
plots
```

![](Diabetes_Markdown_files/figure-gfm/DA%20Plots-3.png)<!-- -->

``` r
#-------------------Pre-Screening Result---------------------------------#
df_summary <- df %>%
              select("pre_screen","class") %>%
              mutate(am_fm = as.character(pre_screen)) %>%
              group_by(class, pre_screen) %>%
              tally %>%
              mutate(sum = sum(n)) %>%
              mutate(f = n/sum, pre_screen = factor(pre_screen, levels = c("Yes","No"), labels = c("Severe Retinoabnormality","None")))

plots <- ggplot(df_summary,
                aes(x = factor(pre_screen),
                    weight = f,
                    fill = class)) +
         geom_bar(position = "dodge",
                  stat = "identity",
                  aes(y = f,
                      x = factor(pre_screen))) +
        theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
              legend.position = "none") +
        facet_grid(class ~., scales = "free") +
        xlab("Severe Retinoabnormality") +
        ylab("Proportion") +
        ggtitle("Proportion with Severe Retinoabnormality Stratified by Diabetic Retinopathy Outcome") +
        scale_y_continuous(limits = c(0,1))
plots
```

![](Diabetes_Markdown_files/figure-gfm/DA%20Plots-4.png)<!-- -->

# Predictions

A train-test set is created by random shuffling and a 75/25 split.

``` r
set.seed(seed)
index <- sample(seq(1,nrow(df_copy)))
df_copy <- df_copy[index,]

df_copy$split <- sample.split(df_copy$ex_4, SplitRatio = 0.75)

train <- df_copy %>% 
         filter(split == TRUE) %>% 
         select(-split, -quality)

test <- df_copy %>% 
        filter(split == FALSE) %>% 
        select(-split, -quality)
```

## Feature Selection

``` r
set.seed(seed)
#Create an object to specify details of model selection
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      number = 5,
                      repeats = 3)

results <- rfe(x = train[,c(1:18)],
               y = train[,19],
               sizes = c(1:18),
               rfeControl = control)
results
```

    ## 
    ## Recursive feature selection
    ## 
    ## Outer resampling method: Cross-Validated (5 fold, repeated 3 times) 
    ## 
    ## Resampling performance over subset size:
    ## 
    ##  Variables Accuracy  Kappa AccuracySD KappaSD Selected
    ##          1   0.5744 0.1446    0.02504 0.05439         
    ##          2   0.6315 0.2655    0.02924 0.06051         
    ##          3   0.6612 0.3247    0.03614 0.07141         
    ##          4   0.6621 0.3214    0.02696 0.05247         
    ##          5   0.6655 0.3288    0.03358 0.06669         
    ##          6   0.6767 0.3502    0.02749 0.05576         
    ##          7   0.6790 0.3548    0.03977 0.07805         
    ##          8   0.6755 0.3468    0.03662 0.07320         
    ##          9   0.6809 0.3575    0.02969 0.05880         
    ##         10   0.6871 0.3707    0.03653 0.07223        *
    ##         11   0.6837 0.3629    0.03642 0.07128         
    ##         12   0.6829 0.3621    0.02992 0.05734         
    ##         13   0.6852 0.3681    0.03367 0.06543         
    ##         14   0.6782 0.3531    0.03523 0.06854         
    ##         15   0.6763 0.3505    0.03547 0.06863         
    ##         16   0.6736 0.3452    0.04303 0.08380         
    ##         17   0.6701 0.3384    0.03822 0.07414         
    ##         18   0.6763 0.3507    0.04011 0.07765         
    ## 
    ## The top 5 variables (out of 10):
    ##    ma_1, ex_7, ex_8, ma_2, ex_1

``` r
#Predictors to use for model training and testing
predictors <- c("ma_1", "ex_7", "ma_2", "ex_1", "ex_8")
```

From this we found the top 5 variables to be:

1.  Microaneurysm, alpha = 0.5
2.  Microaneurysm, alpha = 0.6
3.  Exudates/ROI, alpha = 0.3
4.  Exudates/ROI, alpha = 0.9
5.  Exudates/ROI, alpha = 1.0

## Model Training

From here, we train several select models using the above predictors.

``` r
#Create a control object to specify training details
control <- trainControl(method = 'cv',
                        number = 3,
                        returnResamp = "none",
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
```

### Logistic Regression

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 104  39
    ##          1  42 103
    ##                                          
    ##                Accuracy : 0.7188         
    ##                  95% CI : (0.663, 0.7699)
    ##     No Information Rate : 0.5069         
    ##     P-Value [Acc > NIR] : 1.996e-13      
    ##                                          
    ##                   Kappa : 0.4376         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.8241         
    ##                                          
    ##             Sensitivity : 0.7123         
    ##             Specificity : 0.7254         
    ##          Pos Pred Value : 0.7273         
    ##          Neg Pred Value : 0.7103         
    ##              Prevalence : 0.5069         
    ##          Detection Rate : 0.3611         
    ##    Detection Prevalence : 0.4965         
    ##       Balanced Accuracy : 0.7188         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

### Random Forest

``` r
set.seed(seed)
#Train Random Forest
rf <- train(train[,predictors], factor(train[,"class"], levels = c(1,0), labels = c("yes","no")),
                 method = "rf",
                 trControl = control,
                 metric = "ROC",
                 preProc = c("center","scale"),
                 tuneLength = 3)

#Caclulate confusion matrix for predictions
confusionMatrix(data = predict(rf, test),
                reference = factor(test$class, levels = c(1,0), labels = c("yes","no")))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  95 62
    ##        no   47 84
    ##                                           
    ##                Accuracy : 0.6215          
    ##                  95% CI : (0.5628, 0.6778)
    ##     No Information Rate : 0.5069          
    ##     P-Value [Acc > NIR] : 5.91e-05        
    ##                                           
    ##                   Kappa : 0.244           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.1799          
    ##                                           
    ##             Sensitivity : 0.6690          
    ##             Specificity : 0.5753          
    ##          Pos Pred Value : 0.6051          
    ##          Neg Pred Value : 0.6412          
    ##              Prevalence : 0.4931          
    ##          Detection Rate : 0.3299          
    ##    Detection Prevalence : 0.5451          
    ##       Balanced Accuracy : 0.6222          
    ##                                           
    ##        'Positive' Class : yes             
    ## 

### Gradient Boosted Trees

``` r
set.seed(seed)
#Train gradient boosted machines
gbm <- train(train[,predictors], factor(train[,"class"], levels = c(1,0), labels = c("yes","no")),
             method = "gbm",
             trControl = control,
             metric = "ROC",
             preProc = c("center","scale"),
             tuneLength = 3,
             verbose = FALSE)
```

``` r
#Caclulate confusion matrix for predictions
confusionMatrix(data = predict(gbm, test),
                reference = factor(test$class, levels = c(1,0), labels = c("yes","no")))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  79 56
    ##        no   63 90
    ##                                           
    ##                Accuracy : 0.5868          
    ##                  95% CI : (0.5275, 0.6443)
    ##     No Information Rate : 0.5069          
    ##     P-Value [Acc > NIR] : 0.003927        
    ##                                           
    ##                   Kappa : 0.1729          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.582306        
    ##                                           
    ##             Sensitivity : 0.5563          
    ##             Specificity : 0.6164          
    ##          Pos Pred Value : 0.5852          
    ##          Neg Pred Value : 0.5882          
    ##              Prevalence : 0.4931          
    ##          Detection Rate : 0.2743          
    ##    Detection Prevalence : 0.4688          
    ##       Balanced Accuracy : 0.5864          
    ##                                           
    ##        'Positive' Class : yes             
    ## 

### k-Nearest Neighbors

``` r
set.seed(seed)
#Train KNN
knn <- train(train[,predictors], factor(train[,"class"], levels = c(1,0), labels = c("yes","no")),
             method = "knn",
             trControl = control,
             metric = "ROC",
             preProcess = c("center","scale"),
             tuneLength = 3)

#Caclulate confusion matrix for predictions
confusionMatrix(data = predict(knn, test),
                reference = factor(test$class, levels = c(1,0), labels = c("yes","no")))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  74 49
    ##        no   68 97
    ##                                          
    ##                Accuracy : 0.5938         
    ##                  95% CI : (0.5346, 0.651)
    ##     No Information Rate : 0.5069         
    ##     P-Value [Acc > NIR] : 0.001891       
    ##                                          
    ##                   Kappa : 0.1859         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.096092       
    ##                                          
    ##             Sensitivity : 0.5211         
    ##             Specificity : 0.6644         
    ##          Pos Pred Value : 0.6016         
    ##          Neg Pred Value : 0.5879         
    ##              Prevalence : 0.4931         
    ##          Detection Rate : 0.2569         
    ##    Detection Prevalence : 0.4271         
    ##       Balanced Accuracy : 0.5928         
    ##                                          
    ##        'Positive' Class : yes            
    ## 

### Naive Bayes

``` r
set.seed(seed)
#Train Naive Bayes
nb <- train(train[,predictors], factor(train[,"class"], levels = c(1,0), labels = c("yes","no")),
            method = "nb",
            tuneLength = 3,
            preProcess = c("center","scale"),
            trControl = control,
            verbose = FALSE)
```

``` r
#Caclulate confusion matrix for predictions
confusionMatrix(data = predict(nb, test),
                reference = factor(test$class, levels = c(1,0), labels = c("yes","no")))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes  no
    ##        yes  27   5
    ##        no  115 141
    ##                                          
    ##                Accuracy : 0.5833         
    ##                  95% CI : (0.524, 0.6409)
    ##     No Information Rate : 0.5069         
    ##     P-Value [Acc > NIR] : 0.00555        
    ##                                          
    ##                   Kappa : 0.1576         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2e-16        
    ##                                          
    ##             Sensitivity : 0.19014        
    ##             Specificity : 0.96575        
    ##          Pos Pred Value : 0.84375        
    ##          Neg Pred Value : 0.55078        
    ##              Prevalence : 0.49306        
    ##          Detection Rate : 0.09375        
    ##    Detection Prevalence : 0.11111        
    ##       Balanced Accuracy : 0.57795        
    ##                                          
    ##        'Positive' Class : yes            
    ## 

### Neural Network

``` r
set.seed(seed)
#Train Neural Network
anNet <- train(train[,predictors], factor(train[,"class"], levels = c(1,0), labels = c("yes","no")),
            method = "nnet",
            tuneLength = 3,
            preProcess = c("center","scale"),
            trControl = control,
            verbose = FALSE)
```

    ## # weights:  8
    ## initial  value 401.728715 
    ## iter  10 value 338.749068
    ## iter  20 value 297.138379
    ## iter  30 value 295.366388
    ## iter  40 value 294.814329
    ## final  value 294.813540 
    ## converged
    ## # weights:  22
    ## initial  value 415.434620 
    ## iter  10 value 311.979588
    ## iter  20 value 280.702383
    ## iter  30 value 276.227054
    ## iter  40 value 274.862705
    ## iter  50 value 273.036509
    ## iter  60 value 272.637496
    ## iter  70 value 272.401336
    ## iter  80 value 272.235580
    ## iter  90 value 271.991122
    ## iter 100 value 271.962971
    ## final  value 271.962971 
    ## stopped after 100 iterations
    ## # weights:  36
    ## initial  value 395.243997 
    ## iter  10 value 337.154912
    ## iter  20 value 281.161639
    ## iter  30 value 271.685783
    ## iter  40 value 264.998811
    ## iter  50 value 262.549438
    ## iter  60 value 259.914162
    ## iter  70 value 258.615021
    ## iter  80 value 258.032567
    ## iter  90 value 257.749803
    ## iter 100 value 257.484837
    ## final  value 257.484837 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 402.515202 
    ## iter  10 value 357.192875
    ## iter  20 value 338.635547
    ## iter  30 value 322.294270
    ## iter  40 value 321.208233
    ## final  value 321.206665 
    ## converged
    ## # weights:  22
    ## initial  value 399.919938 
    ## iter  10 value 355.540168
    ## iter  20 value 324.869344
    ## iter  30 value 310.867955
    ## iter  40 value 308.670116
    ## iter  50 value 307.928152
    ## iter  60 value 306.914909
    ## iter  70 value 306.413132
    ## iter  80 value 306.372142
    ## final  value 306.371166 
    ## converged
    ## # weights:  36
    ## initial  value 416.115568 
    ## iter  10 value 338.232592
    ## iter  20 value 312.397265
    ## iter  30 value 308.770856
    ## iter  40 value 307.329595
    ## iter  50 value 305.352445
    ## iter  60 value 303.348334
    ## iter  70 value 302.783659
    ## iter  80 value 302.516633
    ## iter  90 value 302.286911
    ## iter 100 value 302.181789
    ## final  value 302.181789 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 394.367556 
    ## iter  10 value 333.333094
    ## iter  20 value 295.740370
    ## iter  30 value 295.424786
    ## iter  40 value 294.947909
    ## iter  50 value 294.879666
    ## iter  60 value 294.864772
    ## iter  70 value 294.854719
    ## iter  70 value 294.854717
    ## iter  70 value 294.854717
    ## final  value 294.854717 
    ## converged
    ## # weights:  22
    ## initial  value 399.949287 
    ## iter  10 value 325.315122
    ## iter  20 value 284.209388
    ## iter  30 value 278.014785
    ## iter  40 value 275.597434
    ## iter  50 value 274.721049
    ## iter  60 value 273.384513
    ## iter  70 value 271.124307
    ## iter  80 value 269.884689
    ## iter  90 value 269.477471
    ## iter 100 value 269.442986
    ## final  value 269.442986 
    ## stopped after 100 iterations
    ## # weights:  36
    ## initial  value 433.897367 
    ## iter  10 value 340.394800
    ## iter  20 value 283.488719
    ## iter  30 value 266.041259
    ## iter  40 value 262.481185
    ## iter  50 value 260.224731
    ## iter  60 value 259.214106
    ## iter  70 value 258.072089
    ## iter  80 value 256.300205
    ## iter  90 value 255.428753
    ## iter 100 value 253.103993
    ## final  value 253.103993 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 409.991802 
    ## iter  10 value 354.792766
    ## iter  20 value 310.967823
    ## iter  30 value 309.042313
    ## iter  40 value 308.578158
    ## iter  50 value 308.043406
    ## iter  60 value 307.843407
    ## iter  70 value 307.674607
    ## iter  80 value 307.637913
    ## iter  90 value 307.587758
    ## iter 100 value 307.571063
    ## final  value 307.571063 
    ## stopped after 100 iterations
    ## # weights:  22
    ## initial  value 463.235143 
    ## iter  10 value 358.714493
    ## iter  20 value 303.571658
    ## iter  30 value 291.085473
    ## iter  40 value 288.089065
    ## iter  50 value 286.301617
    ## iter  60 value 283.104400
    ## iter  70 value 280.898049
    ## iter  80 value 280.423816
    ## iter  90 value 280.161733
    ## iter 100 value 280.138321
    ## final  value 280.138321 
    ## stopped after 100 iterations
    ## # weights:  36
    ## initial  value 419.652349 
    ## iter  10 value 342.813348
    ## iter  20 value 301.110770
    ## iter  30 value 285.610870
    ## iter  40 value 279.202277
    ## iter  50 value 273.453202
    ## iter  60 value 268.990199
    ## iter  70 value 267.230313
    ## iter  80 value 266.505480
    ## iter  90 value 265.461324
    ## iter 100 value 264.018303
    ## final  value 264.018303 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 425.252168 
    ## iter  10 value 338.066882
    ## iter  20 value 329.458329
    ## final  value 329.179175 
    ## converged
    ## # weights:  22
    ## initial  value 398.745693 
    ## iter  10 value 338.405699
    ## iter  20 value 321.398901
    ## iter  30 value 317.923994
    ## iter  40 value 317.062038
    ## iter  50 value 316.555135
    ## iter  60 value 316.318633
    ## iter  70 value 316.313471
    ## final  value 316.313459 
    ## converged
    ## # weights:  36
    ## initial  value 404.668032 
    ## iter  10 value 342.235840
    ## iter  20 value 322.217304
    ## iter  30 value 315.156939
    ## iter  40 value 314.131356
    ## iter  50 value 313.826953
    ## iter  60 value 313.307280
    ## iter  70 value 313.052036
    ## iter  80 value 312.999790
    ## iter  90 value 312.984750
    ## iter 100 value 312.982941
    ## final  value 312.982941 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 397.343641 
    ## iter  10 value 328.450601
    ## iter  20 value 308.774572
    ## iter  30 value 308.022365
    ## iter  40 value 307.839674
    ## iter  50 value 307.715191
    ## iter  60 value 307.673436
    ## iter  70 value 307.633382
    ## iter  80 value 307.626514
    ## iter  90 value 307.614152
    ## iter 100 value 307.611333
    ## final  value 307.611333 
    ## stopped after 100 iterations
    ## # weights:  22
    ## initial  value 404.950486 
    ## iter  10 value 347.212979
    ## iter  20 value 306.611761
    ## iter  30 value 297.143713
    ## iter  40 value 292.350323
    ## iter  50 value 290.026047
    ## iter  60 value 289.769083
    ## iter  70 value 289.354831
    ## iter  80 value 288.858853
    ## iter  90 value 288.663344
    ## iter 100 value 288.575029
    ## final  value 288.575029 
    ## stopped after 100 iterations
    ## # weights:  36
    ## initial  value 402.049985 
    ## iter  10 value 327.594727
    ## iter  20 value 289.076455
    ## iter  30 value 278.916409
    ## iter  40 value 275.345013
    ## iter  50 value 272.845632
    ## iter  60 value 270.606681
    ## iter  70 value 269.320889
    ## iter  80 value 269.004418
    ## iter  90 value 268.738954
    ## iter 100 value 268.230175
    ## final  value 268.230175 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 397.445470 
    ## iter  10 value 310.889621
    ## iter  20 value 301.483856
    ## iter  30 value 299.576335
    ## iter  40 value 298.459775
    ## iter  50 value 298.034765
    ## iter  60 value 297.950147
    ## iter  70 value 297.855460
    ## iter  80 value 297.844635
    ## iter  90 value 297.835711
    ## iter 100 value 297.830071
    ## final  value 297.830071 
    ## stopped after 100 iterations
    ## # weights:  22
    ## initial  value 406.655146 
    ## iter  10 value 323.747978
    ## iter  20 value 293.928811
    ## iter  30 value 292.377146
    ## iter  40 value 290.626394
    ## iter  50 value 289.071411
    ## iter  60 value 288.863231
    ## iter  70 value 288.494025
    ## iter  80 value 287.631092
    ## iter  90 value 287.352983
    ## iter 100 value 287.103116
    ## final  value 287.103116 
    ## stopped after 100 iterations
    ## # weights:  36
    ## initial  value 398.560222 
    ## iter  10 value 350.286767
    ## iter  20 value 303.910058
    ## iter  30 value 279.373677
    ## iter  40 value 270.875398
    ## iter  50 value 264.223921
    ## iter  60 value 261.386099
    ## iter  70 value 257.350023
    ## iter  80 value 254.228407
    ## iter  90 value 253.432083
    ## iter 100 value 252.844192
    ## final  value 252.844192 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 406.652974 
    ## iter  10 value 350.521778
    ## iter  20 value 329.726893
    ## iter  30 value 326.718056
    ## final  value 326.704516 
    ## converged
    ## # weights:  22
    ## initial  value 402.586736 
    ## iter  10 value 349.007512
    ## iter  20 value 321.548091
    ## iter  30 value 314.345180
    ## iter  40 value 313.763073
    ## iter  50 value 313.739691
    ## iter  60 value 313.725472
    ## iter  70 value 313.718958
    ## final  value 313.718936 
    ## converged
    ## # weights:  36
    ## initial  value 484.205044 
    ## iter  10 value 349.973855
    ## iter  20 value 322.985569
    ## iter  30 value 310.416722
    ## iter  40 value 307.563303
    ## iter  50 value 306.625429
    ## iter  60 value 306.190912
    ## iter  70 value 305.111070
    ## iter  80 value 304.682998
    ## iter  90 value 304.557137
    ## iter 100 value 304.550629
    ## final  value 304.550629 
    ## stopped after 100 iterations
    ## # weights:  8
    ## initial  value 444.390987 
    ## iter  10 value 347.601745
    ## iter  20 value 298.518870
    ## iter  30 value 298.199478
    ## iter  40 value 298.023400
    ## iter  50 value 297.936287
    ## iter  60 value 297.905041
    ## iter  70 value 297.878729
    ## iter  80 value 297.877217
    ## final  value 297.873284 
    ## converged
    ## # weights:  22
    ## initial  value 428.040869 
    ## iter  10 value 345.866564
    ## iter  20 value 287.143429
    ## iter  30 value 276.829733
    ## iter  40 value 275.715059
    ## iter  50 value 275.173061
    ## iter  60 value 274.646259
    ## iter  70 value 273.207512
    ## iter  80 value 270.840633
    ## iter  90 value 269.969276
    ## iter 100 value 269.884647
    ## final  value 269.884647 
    ## stopped after 100 iterations
    ## # weights:  36
    ## initial  value 390.763429 
    ## iter  10 value 342.869920
    ## iter  20 value 307.643941
    ## iter  30 value 273.355882
    ## iter  40 value 266.422810
    ## iter  50 value 262.460124
    ## iter  60 value 260.612196
    ## iter  70 value 258.534196
    ## iter  80 value 257.434028
    ## iter  90 value 257.036894
    ## iter 100 value 256.935384
    ## final  value 256.935384 
    ## stopped after 100 iterations
    ## # weights:  36
    ## initial  value 608.790450 
    ## iter  10 value 521.741947
    ## iter  20 value 486.943165
    ## iter  30 value 461.325366
    ## iter  40 value 454.798308
    ## iter  50 value 453.184111
    ## iter  60 value 452.562184
    ## iter  70 value 452.393430
    ## iter  80 value 452.185467
    ## iter  90 value 451.077750
    ## iter 100 value 449.129026
    ## final  value 449.129026 
    ## stopped after 100 iterations

``` r
#Caclulate confusion matrix for predictions
confusionMatrix(data = predict(anNet, test),
                reference = factor(test$class, levels = c(1,0), labels = c("yes","no")))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes  no
    ##        yes  98  34
    ##        no   44 112
    ##                                           
    ##                Accuracy : 0.7292          
    ##                  95% CI : (0.6739, 0.7796)
    ##     No Information Rate : 0.5069          
    ##     P-Value [Acc > NIR] : 1.178e-14       
    ##                                           
    ##                   Kappa : 0.4577          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.3082          
    ##                                           
    ##             Sensitivity : 0.6901          
    ##             Specificity : 0.7671          
    ##          Pos Pred Value : 0.7424          
    ##          Neg Pred Value : 0.7179          
    ##              Prevalence : 0.4931          
    ##          Detection Rate : 0.3403          
    ##    Detection Prevalence : 0.4583          
    ##       Balanced Accuracy : 0.7286          
    ##                                           
    ##        'Positive' Class : yes             
    ## 

# Results

Based on these results, we find that the model best suited for accurate
predictions of Diabetic Retinopathy is the Neural Network.
