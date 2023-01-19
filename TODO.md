# TODO

## Step 1 - Read the documentation: ✓

## Step 2 - Vanilla RL agent:

### Step 2.2.a: ✓

### Step 2.3:

Run (again) ablation study (classical grid search) with following parameters:

    - learning rate: [1e-3, 2e-3, 5e-3]
    - batch size: [128, 256, 512]
    - gamma: [0.9, 0.99]
    - lr_schedule: ["constant", "linear"]

    TOTAL: 36 trials * 2 env = 72 learnings

**NB: Remember to generate the `tensorboard` plots**

**_OPTIONAL: Render a video with the best hparams_**

## Step 3 - Uniform Domain Randomization: 

 - Run both simulations with and without UDR using the best hyperparameters from **Step 2.3**
 - Run again an ablation study with some possible hyperparameters (_we found out that the best configuration for noUDR isn't the best also for UDR_)

**NB: Remember to generate the `tensorboard` plots**

**_OPTIONAL: Render a video with the best hparams (with UDR)_**

## Step 4 - Vision-based RL:
Given:

    S := state space
    A := action space
    I := image-based observation space
    E_S := an embedding space for the state space (possibly 
           with no physical meaning)

Mapping NNs have to learn

    1. S -> A (✓)
    2. I -> S
    3. I -> E_S -> A

### Step 4.1

 - Decide the (CNN) architecture
 - Produce a dataset for the supervised learning part ✓
 - Understand how to use a custom NN with `stable_baselines3` models
 - Train the mapping ``2.``
 - Train the mapping ``3.`` (the policy)

**NB: Remember to consider different values for the hparams in the trainig phase**

**NB: Remember to generate the `tensorboard` plots**

**_OPTIONAL: Render a video with the best hparams (with UDR + )_**

### Step 4.2 (optional improvements)

 - Try different distributions for Domain Randomization
 - Try DR on images (data augmentation?)
 - Other

## Step 5 - Write the report + fill the provided document

 - Title
 - Abstract
 - Introduction
 - Related Work
 - Method
 - Experiments
 - Conclusions
 - References
 - Images + plots
