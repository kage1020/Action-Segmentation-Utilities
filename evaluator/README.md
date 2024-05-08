# Evaluator for Action Segmentation

This module provides the evaluation tools for action segmentation.

## How to Use

Put this directory in your project root directory, and import the `Evaluator` class.

```python
from evaluator import Evaluator

evaluator = Evaluator()

for data, gt in dataloader:
    pred = model(data)
    evaluator.add(gt, pred)
    metrics = evaluator.get()
```

if you want to use this class from same directory, you can import like this:

```python
from main import Evaluator
```

You can rename `main.py` to any name you want, but don't forget to change the import statement.

## Metrics

In these metrics, we define ground truth as $y$, prediction as $\hat{y}$, frame count as $N$, class List $C$. Then, we define the following terms:

- True Positive (TP): The number of frames where the ground truth and prediction are both positive.
    $$
    \text{TP} = \sum_{c=1}^{C} \sum_{i=1}^{N} \mathbf{1}(y_i = c \land \hat{y}_i = c) = \sum_{c=1}^{C} \text{TP}_c
    $$
- True Negative (TN): The number of frames where the ground truth and prediction are both negative.
    $$
    \text{TN} = \sum_{c=1}^{C} \sum_{i=1}^{N} \mathbf{1}(y_i \neq c \land \hat{y}_i \neq c) = \sum_{c=1}^{C} \text{TN}_c
    $$
- False Positive (FP): The number of frames where the ground truth is negative but the prediction is positive.
    $$
    \text{FP} = \sum_{c=1}^{C} \sum_{i=1}^{N} \mathbf{1}(y_i \neq c \land \hat{y}_i = c) = \sum_{c=1}^{C} \text{FP}_c
    $$
- False Negative (FN): The number of frames where the ground truth is positive but the prediction is negative.
    $$
    \text{FN} = \sum_{c=1}^{C} \sum_{i=1}^{N} \mathbf{1}(y_i = c \land \hat{y}_i \neq c) = \sum_{c=1}^{C} \text{FN}_c
    $$

Where $\mathbf{1}(\cdot)$ is the indicator function. It returns 1 if the condition is true, otherwise 0. $(\cdot)_c$ is the value for class $c$.

Then, we define action segments of ground truth as $Y$ and prediction as $\hat{Y}$. The number of action segments is $|Y|$ and $|\hat{Y}|$ respectively.

### Frame-wise Accuracy (Mean of Frame or MoF)

The frame-wise accuracy is the mean of the frame-wise accuracy for all classes. Note that this metric calculates all videos together.

$$
\text{Frame-wise Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{\text{TP} + \text{TN}}{N} = \frac{\text{correct frames in all videos}}{\text{all frames in all videos}}
$$

### Class-wise Accuracy

The class-wise accuracy is the mean of the frame-wise accuracy for each class. Note that this metric calculates all videos together.

$$
\text{Class-wise Accuracy} = \frac{\text{TP}_c + \text{TN}_c}{\text{TP}_c + \text{TN}_c + \text{FP}_c + \text{FN}_c} = \frac{\text{TP}_c + \text{TN}_c}{N_c} = \frac{\text{correct frames in all videos for each class}}{\text{all frames in all videos for each class}}
$$

### Edit Score

The edit score is the mean of the edit score for all videos. This metric calculates the Levenshtein distance (LD) between the ground truth segments and prediction segments for each video. **Note that this metric calculates mean of normalized edit score for all videos, not mean of levenstein distance.**

$$
\begin{align*}
    \text{LD}(Y, \hat{Y}) &= \left\{\begin{aligned}
        &|Y| & \text{if } |\hat{Y}| = 0,\\
        &|\hat{Y}| & \text{if } |Y| = 0,\\
        &\text{LD}(Y_{\backslash 1}, \hat{Y_{\backslash 1}}) & \text{if } Y_1 = \hat{Y}_1,\\
        &1+\min\left(\begin{aligned}
            &\text{LD}(Y_{\backslash 1}, \hat{Y})\\
            &\text{LD}(Y, \hat{Y}_{\backslash 1})\\
            &\text{LD}(Y_{\backslash 1}, \hat{Y}_{\backslash 1})
        \end{aligned}\right)& \text{otherwise.}
    \end{aligned}\right.\\
    \text{Edit Score} &= \frac{1}{V} \sum_{v=1}^{V} \left(1-\frac{\text{LD}(Y_v, \hat{Y}_v)}{\max(Y_v, \hat{Y}_v)}\right)\times 100
\end{align*}
$$

Where $V$ is the number of videos, $(\cdot)_{\backslash 1}$ is the segments without the first element.

### F1 Score

The F1 score is the mean of the F1 score for all classes. Note that this metric calculates all videos together.

We re-define the terms for the F1 score:

- True Positive (TP): The number of segments where the ground truth and prediction are both positive.
    $$
    \text{TP}@\tau = \sum_{c=1}^{C} \sum_{i=1}^{|\hat{Y}|} \mathbf{1}(Y_i = c \land \hat{Y}_i = c)
    $$
- False Positive (FP): The number of segments where the ground truth is negative but the prediction is positive.
    $$
    \text{FP}@\tau = \sum_{c=1}^{C} \sum_{i=1}^{|\hat{Y}|} \mathbf{1}(Y_i \neq c \land \hat{Y}_i = c)
    $$
- False Negative (FN): The number of segments where the ground truth is positive but the prediction is negative.
    $$
    \text{FN}@\tau = \sum_{c=1}^{C} \sum_{i=1}^{|\hat{Y}|} \mathbf{1}(Y_i = c \land \hat{Y}_i \neq c)
    $$

Where $\tau$ is the threshold for intersection over union (IoU) of the segments.

Then, we define precision, recall, and F1 score:

$$
\begin{align*}
    \text{Precision}@\tau &= \sum_{v=1}^{V}\frac{\text{TP}@\tau}{\text{TP}@\tau + \text{FP}@\tau}\\
    \text{Recall}@\tau &= \sum_{v=1}^{V}\frac{\text{TP}@\tau}{\text{TP}@\tau + \text{FN}@\tau}\\
    \text{F1 Score}@\tau &= \frac{2 \times \text{Precision}@\tau \times \text{Recall}@\tau}{\text{Precision}@\tau + \text{Recall}@\tau}
\end{align*}
