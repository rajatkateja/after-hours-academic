---
title: Decision trees, mutual information, and inductive bias
link: decision-trees
published_date: 2024-05-22
meta_description: Basics of decision trees
tags: machine-learning, decision-tree
---

Let's say I want to train an algorithm that 
can predict whether I will bike or drive to work on a 
given day. I come up with a list of factors that I think 
might affect this decision. They are: (i) is it raining?, 
(ii) do I have to carry my backpack?, 
and (iv) am I tired?

To train a model that can predict my transportation medium, I collect some historical data to use it as training data.

| Raining | Backpack | Tired | Transportation Medium |
|---------|---------------------|-------|-----------------------|
| Yes | Backpack | No | Drive |
| No | Neither | No | Bike | 
| No | Backpack | No | Bike | 
| No | Backpack | Yes | Drive | 

I intuitively know that rain is probably going to be the main 
deciding factor, so I try to simplify the problem. 
Limit the data just to rain, I get the following dataset. 

| Raining | Transportation Medium |
|---------|-----------------------|
| Yes | Drive |
| No | Bike | 
| No | Bike | 
| No | Drive |

Based on the data, I come up with a simple algorithm. 
```
if raining: 
drive
else
bike
```

This algorithm gives the correct prediction for 3 out of the 4 
observations we see here. 3 out of 4 is not bad, but can I improve it? How about I add the tiredness factor too? 

| Raining | Tired | Transportation Medium |
| ------- |---------|-----------------------|
| Yes | Yes | Drive |
| No | No | Bike | 
| No | No | Bike | 
| No | Yes| Drive |

Now, I can update the algorithm to account for both the factors.   

```
if raining:
drive 
else: 
if tired:
drive
else:
bike
```
Now the algorithm predicts all the 4 observations correctly. Awesome! 


What I built above is a **decision tree** that predicts an 
outcome (`bike or drive`) based on certain inputs (`raining, backpack, tired`).
To get a prediction for a new observation, 
I will start at the tree's root (`raining`), 
follow the branches (`yes` or `no`) and 
other splits (`tired`, `yes` or `no`) 
until I reach a leaf (`drive` or `bike`). 

In this particular example, my intuition 
guided my choice of the root node as well 
as the other node that to base the decision on. 

A more generalized
problem is training a decision tree to 
predict an outcome `y` based on _n_ input parameters
`x_1, x_2, ..., x_n`. 
To start with, we have some observational data.

| x_1 | x_2 | ... | x_n | y | 
| --- | --- | --- | --- | --- | 
| a_1 | a_2 | ... | a_n | y_a |
| b_1 | b_2 | ... | b_n | y_b |
| c_1 | c_2 | ... | c_n | y_c |
| ... | ... | ... | ... | ... | 

To build a decision tree to predict `y` based 
on `x_1, x_2, ... x_n`, we need to choose the 
parameters to split on (`raining` and `tired` 
in the above example) and the order in which to 
consider those (split on `raining` before 
on `tired` in the above example).

The answer to both of these comes from 
the idea of **mutual information**. Mutual 
information of the decision `y` with respect to 
a parameter (say) `x_1` is a measure the decision's 
information that is contained  
in the parameter. This is a fancy way of 
saying that the mutual information 
of `y` with respect to `x_1` refers to how "similar" 
`y` is to `x_1` or how much of `y` can be inferred 
from `x_1`. 

Mathematically, mutual information 
of `y` with respect to `x_1` is the difference in entropy 
of `y` and the entropy of `y` given `x_1`.
```
mutual_information(y, x_1) = entropy(y) - entropy(y | x_1)
```
Entropy is a measure of the randomness of a 
variable. A high value of mutual 
information means that the entropy of `y` given `x_1` 
is significantly less than entropy of `y`. That 
is, knowing `x_1` significantly reduced the 
randomness of `y`. 

Now let's see how to use mutual information 
to train our decision tree. 

Similar to our intuition with the bike or drive 
example above, we want to use the input parameters that 
will help us get to the answer the fastest. 
This is exactly what mutual information tells us. 
A parameter with large mutual information implies 
that it can significantly reduce the randomness of 
the output, i.e., it can predict the output with a 
high degree of confidence. 
So, we choose the parameter with the largest mutual 
information as the first parameter to split 
on. 

Let's say we decide to split on `x_1`. 
Our decision tree would start to shape up like this.

```
switch x_1: 
case value_1:
...
case value_2: 
...
...
case value_k:
...
```

`x_1` can take multiple values (and not just 
        `true` or `false`), so end up with as many branches as the number 
of values that `x_1` can take. 
Now we need to populate each of these branches with 
the next parameter to base the decision on. 

Simple! We will just repeat the same process for each of the 
branches. For a branch where `x_1` takes the 
value `value_1`, we have effectively 
reduced the problem and now need to consider 
only consider the training data where `x_1 = value_1`. 
In this subset of data, we need to find the parameter 
to base the decision on. Mutual information to 
the rescue again. We compute the mutual information of 
`y` with all the remaining parameters `x_2, x_3`, ..., x_n`. 
And choose the parameter with the highest 
mutual information as the parameter to split upon. 

We will continue this process for all the branches 
and sub-branches till we have populated the entire 
tree. 

As an example, for a three input problem (`x_1`, `x_2`, `x_3`) where 
each input can take only two values (`true` or `false`), the 
decision tree would look like below.

```
if x_1:
if x_2:
if x_3:
prediction_true_true_true
else:
prediction_true_true_false
else:
if x_3:
prediction_true_false_true
else:
prediction_true_false_false
else:
if x_2:
if x_3:
prediction_false_true_true
else:
prediction_false_true_false
else:
if x_3:
prediction_false_false_true
else:
prediction_false_false_false
```            

Done? Not quite. 

There are two problems with the model. 
First is that if it requires `2^n` observations for 
`n` input parameters (e.g., 8 observations for the 
        3 input parameters above). This isn't always 
feasible. 

Let's say we solve this problem by ignoring the paths 
for which we don't have training data. The second 
problem with the model is that it is prone to overfitting. 

To understand overfitting, let's go back to our bike or drive 
example. Let's say we had this observation in 
the training dataset.   

| Raining | Backpack | Tired | Transportation Medium |
|---------|---------------------|-------|-----------------------|
| Yes | Yes | Yes | Bike |

If we follow the above algorithm, we might 
end up with a path in the tree that says: 

```
if raining:
if tired: 
if backpack
bike
```

However, this does not seem like a reasonable path.
In the future, if there is a day when it is raining, 
I am tired, and I have a backpack to carry, I would 
expect the model to predict that I would drive to 
work. However, the model we have just trained 
would predict that I would bike. 

This is because the model has overfit to the 
training data. This observation in the training data 
could have been because of some other unknown factor. For example, 
maybe the car was being serviced and thus unavailable 
that day, which caused me to commute via bike. 
Such observations are called noise in the data. 

Overfitting refers to when the model fits the 
training data extremely well but generalizes 
to unseen data poorly. In trying to fit the 
training data well, the model starts fitting 
to the noise instead of the underlying pattern. 

Complex models are more prone to overfitting because 
they tend to start modeling the noise. 
The principle of reducing the complexity of the models to 
avoid overfitting is called **inductive bias**. 

For decision trees, inductive bias is trying to find the 
shortest tree that performs well on the training 
data and would also generalize to unseen data. To that end, 
we stop splitting the tree's branches when 
either of the following 
becomes true: (i) if the entropy of the output `y` is low enough, 
or (ii) if the mutual information with either of the remaining 
parameter is low enough. 
In either of these scenarios, 
splitting any further does not provide a lot of value and 
runs the risk of overfitting to the training data. 
The threshold of "low enough" is a parameter that 
can be tuned to get low error rates on test data.

Decision trees are useful for models where the input 
parameters are categorical, i.e., they can take values 
only from a set of values. Examples of categorical 
variables are `raining` or `t-shirt sizes`. These 
are different from continuous parameters that can take any 
value from a range. E.g., `amount of rain (in cm)` or `length of t-shirt (in inches)`. 
Continuous variables are not good candidates for decision tree 
models because each split can take an infinite number of 
values. Of course, we could group the output 
of continuous variables into ranges and convert them 
to categorical values if it makes sense for the problem at hand.

To recap:
- Decision trees work by greedily choosing the parameters that best describe the output and splitting on those. 
- Mutual information is a way to measure the similarity of the output to an input parameter. 
- Inductive bias is a way to avoid overfitting by choosing a simpler model. For decision trees, finding the smallest tree that provides a "low enough" error rate is the inductive bias. 

References
- Lectures from the ["Introduction to Machine Learning (10-701)"](https://www.cs.cmu.edu/~hchai2/courses/10701/) course from Carnegie Mellon University


