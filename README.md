<div align="center">
  <h1>GTA1: GUI Test-time Scaling Agent</h1> 
</div>

Exploring some code for my 2025 winter break, christmas, and new year!

## TL;DR

I spent $230 on Lambda compute (most of it went into setting up the environment and debugging things I thought were easy to debug 😭) to run a very *very* small experiment.

I trained Qwen2.5-VL-3B-Instruct with the GTA1 training code on *only* 8 examples from the Aria-UI dataset they used (originally wanted to do 100 examples but it would take too long and compute was expensive). I evaluated on 100 random ScreenSpot Pro examples.

I also tested with my implementation of the [GroundCUA](https://github.com/ServiceNow/GroundCUA) reward function trained on the same 8 examples. 

wandb project: https://wandb.ai/vincenttu/gta1

### Findings

This experiment is way too small to make any sense out of, but we can speculate. 

I included epoch step 80 because it had the highest reward. 

| Model/Setting              | # Correct |
|----------------------------|-----------|
| base model                 | 5         |
| baseline (80 epochs in)    | 15        |
| baseline (100 epochs)      | 16        |
| distance reward (80 epochs)| 8         |
| distance reward (100 epochs)| 9        |

- from the reward plots, the slope of both rewards is roughly the same ~0.007-0.008 of reward per step; both show a strong upward trend but very choppy and slow 
- the GroundCUA reward function has a larger reward range and starts lower, heavily penalizing far-off points and continually optimizing points within the bbox till it reaches the center (whereas the hit reward just checks if the point is in the box)
- the hit reward achieves more correct much sooner in training; it seems the distance (GroundCUA) reward is about twice as slow in achieving the same performance (but it could also be that I implemented it incorrectly)

#### Correct Examples & Error Analysis

I took a quick look on both the 100 epoch baseline and 100 epoch distance reward and they share **8** correct examples, which leads me to think these 8 evaluation examples are easy. Coordinate placement for the correct examples are pretty much the same. In both evaluation results, mac had the most correct by an overwhelming majority. Maybe these models have strong Mac understanding? And/or the buttons are bigger/wider?
 
There were no formatting issues for either run (baseline/distance reward 100 epochs). I didn't look at all all the incorrect examples for both of the 2 runs. 

## Credit

I originally forked this project from https://github.com/Yan98/GTA1 — please take a moment to check out the original repo and give it an upvote! Thanks GTA1 for making this repo open-source. 