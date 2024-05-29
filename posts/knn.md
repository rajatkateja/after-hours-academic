---
title: Book sales predictions gone wrong or how (not) to use a kNN model
link: knn
published_date: 2024-05-29
meta_description: what is a kNN model and how (and how not) to use it
tags: knn, machine-learning
---
An entrepreneurial friend of mine wanted to get on the AI hype (he didn't know anything about AI, but when has that stopped anyone). He came up with an idea -- BookSalesPredictor. Before a book is launched, authors and publishers can use BookSalesPredictor to get a prediction for the sales of the book. They can use the prediction to print only the required number of copies. He launched the startup with much fanfare and it worked! He got his first client who sent in an advance copy of a book to know about its predicted sales. 

My friend figured that the number of pages in the book would be the best predictor of the sales (who wants to read long books anyway?). He convinced the client that all they need to know is whether the sales would be `High` (more than 100,000 copies) or `Low` (less than 100,000 copies). 

He still didn't know much about AI, but he knew that it needs data. So he collected some data about book lengths and sales. 

| # pages | sales | 
|---------|-------|
| 70 | High | 
| 30 | High | 
| 127 | High | 
| 400 | Low | 
| 318 | Low | 
| 1024 | Low | 

The client's book was 93 pages long. By eyeballing the data, he figured it should have `High` sales. 

What my friend did was use the [**duck test**](https://en.wikipedia.org/wiki/Duck_test). Duck test states that if something looks like a duck, swims like a duck, and quacks like a duck, it probably is a duck.  

Turns out the duck test is the insight behind an actual classification algorithm, known as the k nearest neighbor (kNN) model. A kNN model classifies a data point based on what its neighbors (k nearest neighbors, to be precise) are classified as. In effect, it is trying to figure out what the the data point "looks like" to classify it 
 
In the above example of book sales, my friend used a 1NN model. The closest neighbor to 93 pages is 70 pages. Because 70 has `High` sales, the 1NN model would predict that the book would have `High` sales as well. 

What about a book with 263 pages. It's closest neighbor is the book with 127 pages and hence it would have `High` predicted sales. A book with 264 pages would have `Low` predicted sales because its closest neighbor is the book with 400 pages. What the 1NN model is doing is finding a boundary around which to split the decision. Anything less than 264 pages would have `High` sales. Anything more than 264 pages would have `Low` sales. 


Let's revisit the idea of k=1 and how that affects this boundary. Let's say we collected some more data so that it looked like below:

| # pages | sales | 
|---------|-------|
| 70 | High | 
| 30 | High | 
| 127 | High | 
| 250 | Low |
| 260 | High |
| 270 | Low |
| 280 | Low |
| 400 | Low | 
| 318 | Low | 
| 1024 | Low | 

What would the predicted sales for a book with 257 pages be? It's nearest neighbor is the book with 260 pages. Based on the 1NN model, it should be `High`, but that does not seem correct. (Of course, nothing about predicting book sales based on the number of pages is correct, but we are here already, so let's keep going.) This is the problem with the `1` in 1NN model. Instead, let's say we use 3NN and look at the 3 nearest neighbors (250, 260, and 270) and take a majority vote. Now we would end up with `Low` predicted sales. That sounds more reasonable. 

This is because the data point of `260, High` is probably noise and does not fit the underlying data pattern. Using 1NN caused the algorithm to overfit to the noise. Using 3NN corrected that. The `k` is thus a knob for the inductive bias of the kNN model. [Recall](/decision-trees) that inductive bias of a model is its ability to generalize to unseen data (i.e., its propensity to not overfit to the training data).

Why 3NN and not 2NN though? If we were to use 2NN, the nearest neighbors would be 250 and 260 and we would end up with a tie. We would need a tie breaker. Potential tie breakers could be to go with the nearest neighbor, to poll the next nearest neighbor (in which case it becomes 3NN), or to assign weights to the neighbors based on their distances. 

Choice of k and the tie breaking mechanism are example of **hyperparameters**. Hyperparameters are not learned by the model through the training data, but instead are decided before training. In some sense, hyperparameters define the model itself. A 1NN model is different from a 3NN model because it has a different hyperparameter.

Another hyperparameter in the kNN model is the distance metric. In this example, we had just one input parameter (the number of pages in the book). So we intuitively used the difference in the number of pages as the distance metric. In general though (in an n-dimensional input space), there could be different distance metrics, e.g., [euclidian distance](https://en.wikipedia.org/wiki/Euclidean_distance) or [manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry).  

Hyperparameters themselves can be tuned. Let's say we know that we want to use a certain model but not sure what parameters to use for that model. We can set aside a part of our training dataset and assign it to be the **validation dataset**. We can then train different models (same model with different hyperparameters) using just the remainder of the training dataset. Finally, we can use the validation dataset to test the models (we already know what the "correct" prediction for data points validation dataset because it came from the training dataset). We can then choose the hyperparameters that had the lowest error on the validation dataset.

In general, kNN models can be used to to predict a categorical variable (high or low sales) based on continuous variable inputs (number of pages in the book). This is in contrast to [decision trees](/decision-trees) where the inputs parameters were also categorical. In the next post, we will look at linear regression, in which both input parameters and outputs are continuous variables. 

References
- Lectures from the ["Introduction to Machine Learning (10-701)"](https://www.cs.cmu.edu/~hchai2/courses/10701/) course from Carnegie Mellon University
