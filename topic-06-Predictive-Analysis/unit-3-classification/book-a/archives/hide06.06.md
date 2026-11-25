The [Cancer Diagnosis](./archives/Cancer-Diagnosis.xlsx) spreadsheet works the same way as the bombers and seagulls scenario.
It is designed to provide a realistic simulation of the cost-benefit assumptions that must be made to set the classification threshold for a medical diagnostic product.
Download the spreadsheet and investigate the contents, below are details of what is contained in the file:

- The file has 10,0000 rows of data.
- The ranked scores in column A are the level of a certain protein as measured by the diagnostic test.
- The true condition for each protein level is given in column C [Cancer=1, No Cancer=0].
- A threshold for a positive classification can be set between any two protein levels.
- For each threshold in column A, the resulting number of false negative classification errors is given in column H, and the false positive classification errors in column F.

There are 100 correct cancer diagnosis out of the 10,000, resulting in 9900 correct no cancer diagnosis.
If you look at row 19, protein value 18636.922. If the threshold is set to greater than this amount there would be 1 true positive diagnosis and 9 false positive diagnosis.
This can also be read as saying at that threshold there would be for the sample, 9 false positives and 99 false negatives (only one true positive at that threshold out of the 100 total).

This spreadsheet is designed to allow you to observe how changing cost inputs impacts both:

- the overall costs of using a cancer diagnostic test at each threshold, and
- what threshold should be chosen as optimal thus minimizing costs.

Cell G3  contains the cost per False Negative (missing a cancer case) and cell H3 for the cost per False positive (a false alarm).
Total costs as each threshold are given in column K.

The minimum total cost, and minimum cost per event (per diagnosed classification reported), are displayed in cells K4 , and L4,

The optimum threshold - the lowest protein level score that should be classified positive is displayed in cell M4. The formula finds the threshold with the lowest overall cost (in false positives and false negatives). In this case it is Row 2103 where at the default costs of €50,000 per false negative error and €500 per false positive error, the minimum cost per event is €119.90, and the optimal threshold for positive classification is 16551.930.

##Answer the following questions:

Change the cost per FN to $15,000 and change the cost per FP to $1,500

- What is the new minimum cost per event (rounded to the nearest dollar)?
- What is the lowest level of protein that should be classified "Positive" to achieve the minimum cost per test at the new cost per error given the above?
- Can a change in classification threshold change a diagnostic test's True Positive Rate? Use logic - no need to calculate any numbers.
- "Condition Incidence" is the portion of a population that has the condition being studied. Can a change in the threshold change the condition incidence? Use logic.
- Does the change in threshold change the test's Area under the ROC curve? Use logic.

##AOC
Calculate the area under the curve for the soldier data

[Soldier Data](./archives/Forecasting-Soldier-Performance-single-ca2.xlsx)


- You will need to calculate the true positives and false positives for each threshold in the 12 instances of height data.

- Then calculation the true positive rate (y) and false positive rate (x) at each threshold for the height.

- Then plot the x and y pairs (false positive rate, true positive rate) on a scatter chart.

- Identify each rectangle on the chart and calculate the area of each rectangle by multiplying height by weight to find out the AUC. Based on the result is this a good model?


This work is to be submitted to Moodle and is worth 10% of your continuous assessment marks. The work is to be done individually.

- AUC.xlsx
- Cancer-Diagnosis.xlsx
- Forecasting-Solider-Performance-single-ca2.xlsx
