# Additive-Sequence-Forest-ASForest

Abstract - Boosting and bootstrap aggregation (Bagging) are two famous ensemble learning approaches, which combine multiple base learners to generate a composite learner. This paper proposes additive sequence tree (ASTree) and additive sequence forest (ASForest). ASTree combines the idea of gradient boosting and tree models, which performs a Newton update in each node to optimize the objective function. When a new sample comes in ASTree, it is first sorted down to a leaf, then the final prediction can be obtained by summing up the outputs of all models along the path from the root node to that leaf. ASForest integrates multiple ASTrees into a forest. The performance of ASForest is compared to several other state-of-the-art ensemble algorithms on 30 datasets from various application scenarios, suggesting that ASForest generates a stronger and more robust composite learner.

ASTree:
![maze](https://github.com/zhaochangming/Additive-Sequence-Forest-ASForest-/blob/master/FigASTree.png)  

ASForst:
![maze](https://github.com/zhaochangming/Additive-Sequence-Forest-ASForest-/blob/master/FigASForest.png)  

Experiment_1_Classification and Experiment_1_Regression: Codes of Experiment 1

Experiment_2_ClfLearners and Experiment_2_RegLearners: Codes of Experiment 2

Experiment_3_ClfLeafs and Experiment_3_RegLeafs: Codes of Experiment 3

Experiment_4_LMT: Codes of Experiment 4

src/ASForest: Codes of ASForest

output1: Outputs of Experiment 1

output2: Outputs of Experiment 2
![maze](https://github.com/zhaochangming/Additive-Sequence-Forest-ASForest-/blob/master/ex_2_clf.png)  
![maze](https://github.com/zhaochangming/Additive-Sequence-Forest-ASForest-/blob/master/ex_2_reg.png)  

output3: Outputs of Experiment 3
![maze](https://github.com/zhaochangming/Additive-Sequence-Forest-ASForest-/blob/master/ex_3_clf.png)  
![maze](https://github.com/zhaochangming/Additive-Sequence-Forest-ASForest-/blob/master/ex_3_reg.png)  

output4: Codes of Experiment 4
