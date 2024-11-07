
# WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts

Chong Zhang\*, Wenli Xiao\*, Tairan He, Guanya Shi

CoRL 2024 (oral)


<div align="center">

[[Website]](https://lecar-lab.github.io/wococo/)
[[Arxiv]](https://arxiv.org/abs/2406.06005)
[[Video]](https://www.youtube.com/watch?v=_S6DNhPDuTw&t=1s&ab_channel=LeCARLabatCMU)

<img src="./media/WoCoCo.gif" width="600px"/>

</div>



## License 

> [!IMPORTANT]
> This codebase is under CC BY-NC 4.0 license, with inherited license in Legged Gym and RSL RL from ETH Zurich, Nikita Rudin and NVIDIA CORPORATION & AFFILIATES. You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products.

## Quick Start

### Install
Similar to basic legged-gym based codebase, the install pipeline is:

1. install isaac gym
2. install packages
    ```
    pip install -r requirements.txt
    ```


### Training

- go to `legged_gym/legged_gym`

    ```
    python3 scripts/train.py --task=h1:jumpjack
    ```

### Play

- go to `legged_gym/legged_gym`
    ```
    python scripts/play.py --task=h1:jumpjack --num_envs=3
    ```

## Note

Since our framework is quite intuitive and we *de-engineered* a lot of things, (i.e., we do not use certain engineering tricks that can be applied to specific tasks such as terrain curriculum) to showcase the framework's capability, we provide a clap-and-dance example here for the reward and MDP implementations, and encourage anyone to engineer the specific environments for better performance on their own applications.

The engineering is encouraged for terrain design, curiosity observation space design (which can be further reduced based on specific tasks), and sim-to-real pipeline (where tricks like teacher-student and system identification may make training and deployment more stable).

## Cite

```
@inproceedings{
    zhang2024wococo,
    title={WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts},
    author={Chong Zhang and Wenli Xiao and Tairan He and Guanya Shi},
    booktitle={8th Annual Conference on Robot Learning},
    year={2024},
    url={https://openreview.net/forum?id=Czs2xH9114}
}
```