...


This file will contain the explanation for the project and the folder structure.

Project Explanation:
in this project developed the code for PU learning and Confidence learning while doing label flipping poisoning.

so what we do is that each iteration we change the label of 2% of the data and try to use PU learning and Confidence learning technique to return the data back to be correct.

In the PU learning, in each iteration we change 2% of the positive data to be unlabelled.
In Confidence Learning, in each iteration we change positive labels to be negative and 2% of the negative data to be positive.


Folder Structure:

pu_learning and confidence_learning files which are .py are the classes themselfs which are used in the 2 jupyter notebook files named implementation which are the main implementations of the code. 
The logreg and the logreg confidence files are the results for logistic regression in pu_learning and confidence learning respectively 