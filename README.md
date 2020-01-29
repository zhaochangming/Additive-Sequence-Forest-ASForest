# Additive-Sequence-Forest-ASForest-
The codes of ASForest
Abstract - Boosting and bootstrap aggregation (Bagging) are two famous ensemble learning approaches, which combine multiple base learners to generate a composite learner. This paper proposes additive sequence tree (ASTree) and additive sequence forest (ASForest). ASTree combines the idea of gradient boosting and tree models, which performs a Newton update in each node to optimize the objective function. When a new sample comes in ASTree, it is first sorted down to a leaf, then the final prediction can be obtained by summing up the outputs of all models along the path from the root node to that leaf. ASForest integrates multiple ASTrees into a forest. The performance of ASForest is compared to several other state-of-the-art ensemble algorithms on 30 datasets from various application scenarios, suggesting that ASForest generates a stronger and more robust composite learner.
