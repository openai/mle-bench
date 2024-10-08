## Description

What do doctors do when a patient has trouble breathing? They use a ventilator to pump oxygen into a sedated patient's lungs via a tube in the windpipe. But mechanical ventilation is a clinician-intensive procedure, a limitation that was prominently on display during the early days of the COVID-19 pandemic. At the same time, developing new methods for controlling mechanical ventilators is prohibitively expensive, even before reaching clinical trials. High-quality simulators could reduce this barrier.

Current simulators are trained as an ensemble, where each model simulates a single lung setting. However, lungs and their attributes form a continuous space, so a parametric approach must be explored that would consider the differences in patient lungs.

Partnering with Princeton University, the team at Google Brain aims to grow the community around machine learning for mechanical ventilation control. They believe that neural networks and deep learning can better generalize across lungs with varying characteristics than the current industry standard of PID controllers.

In this competition, you’ll simulate a ventilator connected to a sedated patient's lung. The best submissions will take lung attributes compliance and resistance into account.

If successful, you'll help overcome the cost barrier of developing new methods for controlling mechanical ventilators. This will pave the way for algorithms that adapt to patients and reduce the burden on clinicians during these novel times and beyond. As a result, ventilator treatments may become more widely available to help patients breathe.

Photo by [Nino Liverani](https://unsplash.com/@ninoliverani?utm_source=unsplash) on [Unsplash](https://unsplash.com/s/photos/lung?utm_source=unsplash)

## Evaluation

The competition will be scored as the mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored. The score is given by:

$$
|X-Y|
$$
where $X$ is the vector of predicted pressure and $Y$ is the vector of actual pressures across all breaths in the test set.

## Submission File

For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Timeline

- **September 22, 2021** - Start Date.
- **October 27, 2021** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **October 27, 2021** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **November 3, 2021** - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Prizes

- 1st Place - $2,500
- 2nd Place - $2,500
- 3rd Place - $2,500

## Citation

Addison Howard, alexjyu, Daniel Suo, Will Cukierski. (2021). Google Brain - Ventilator Pressure Prediction. Kaggle. https://kaggle.com/competitions/ventilator-pressure-prediction

# Data

## Dataset Description

The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

In this competition, participants are given numerous time series of breaths and will learn to predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

![2020-10-02 Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files

- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns

- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.