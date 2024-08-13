# Improve the Worst-Case Robustness in Natural and Adversarial Training through Group Distributionally Robust Optimization
This is the official implementation of the paper "Improve the Worst-Case Robustness in Natural and Adversarial Training through Group Distributionally Robust Optimization"
![image](model.jpg)

## Abstract

It is well known that deep neural networks(DNN) have achieved high accuracy in many challenging tasks. Due to the traditional training strategy, it is vulnerable to adversarial attacks as well as suffers from poor performance on minority groups. Recent works show that group distributionally robust optimization (Group-DRO) can minimize worst-case loss and focus on core features instead of spurious features, which is still vulnerable to adversarial attacks. Additionally, while adversarial training has shown promising results in defending against adversarial attacks, it can cause a significant drop in accuracy on minority groups. To address these issues, we propose the GDRO-AT training framework, which improves the Group-DRO with adversarial training based strategies. The key insight of our proposed algorithm lies in leveraging regularized loss across groups for both natural and adversarial training from the perspective of distributionally robust optimization. Theoretically, we establish a convergence guarantee for our algorithm, ensuring it reaches first-order stationary points in a convex setting. Empirically, we perform experiments on two benchmark datasets and achieve better performance than baselines. In addition, we also extend our study to cover a real-world medical application, where our method remains robust against hospital specific spurious markers. To further demonstrate the relation between adversarial attacks and core features, we propose a Grad-CAM based method for visualizing adversarial attacks.



## Installation 
conda create -n myenv python=3.7
conda activate myenv
pip install -r requirements.txt



## Dataset
- Waterbirds: see instructions [here](https://github.com/kohpangwei/group_DRO#waterbirds).
- CelebA: see instruction [here](https://github.com/kohpangwei/group_DRO#celeba).

Waterbirds:
For running the codes, following files/folders should be in the [root_dir]/cub directory:
- `data/waterbird_complete95_forest2water2/`
- 
CelebA:
For running the codes, following files/folders should be in the [root_dir]/celebA directory:
- `data/list_eval_partition.csv`
- `data/list_attr_celeba.csv`
- `data/img_align_celeba/`

## training

### Waterbirds
Run the following command for Waterbirds


### CelebA
Run the following command for CelebA




