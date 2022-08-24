# Computer-Vision-MoCo-Adaptation
## An Adaptation of the Deep Learning Model MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
## Deep Learning Spring 2021 Competition
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

This is a PyTorch adaptation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```

### Background
This team composed of **Robert Melikyan, Jash Tejaskumar Doshi, and formerly Omkar Ajit Darekar**; was tasked with creating a self-supervised deep learning model for Spring 2021. Given the impressive results by MoCo and a few initial tests between comparative models like SIMCLR, BOYL, or SWAV, the team settled on MoCo as the basis of its architecture to satisfy the tradeoff between code dependencies, computation intensity and model effectiveness.

### Preparation

Ensure PyTorch is installed along with 96x96 dataset of RGB images. Although given an original dataset of both labelled and unlabelled images, 12800 images were labelled during the process. Their respective identification filenames and labels can be found [here](https://drive.google.com/drive/folders/1SxcXDGZpbkNeIScJA3dFK1aMcimK8XxF?usp=sharing). The methodology of how the 12800 images were chosen was after running the unsupervised model initially, and deciding to randomly sample images afterwards to get a representative balance over high and low energy images that were either in cluster centers or edges.

The files for this project should be maintained and executed in the same directory, the main files including [main_moco.py](https://github.com/robml/Deep-Learning-SSL-MoCo-SP21/blob/main/main_moco.py) for unsupervised learning and [main_linclssupervised.py](https://github.com/robml/Deep-Learning-SSL-MoCo-SP21/blob/main/main_linclssupervised.py) for the linear classifier. Additionally there is [main_linclsnewlabel.py](https://github.com/robml/Deep-Learning-SSL-MoCo-SP21/blob/main/main_linclsnewlabel.py) which takes into account the newly labelled data linked above, however since then this code has been deprecated in favor of the former, by simply replacing the directory name with the folder containing the new labelled data linked above.

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on the dataset in an 2-gpu machine, run a sbatch file as specified below:
```
#!/bin/bash                                                                     

#SBATCH --gres=gpu:2                                                            
#SBATCH --partition=n1s16-t4-2                                                  
#SBATCH --account=dl17                                                          
#SBATCH --time=20:00:00                                                         
#SBATCH --output=final%j.out                                                    
#SBATCH --error=final%j.err                                                     
#SBATCH --exclusive                                                             
#SBATCH --requeue                                                               

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"


cd $HOME/test/repo/NYU_DL_comp/moco/

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c " 
source /ext3/env.sh                                                             
conda activate dev                                                              
CUDA_VISIBLE_DEVICES=0,1 python3 main_moco.py   -a resnet50   --lr 0.06 --batch\
-size 512 --epochs 200  --dist-url 'tcp://localhost:10004' --resume $SCRATCH/ch\
eckpoints/demo/moco_unsupervised_0065.pth.tar  --multiprocessing-distributed --\
world-size 1 --rank 0 -data /dataset --mlp --moco-t 0.2 --aug-plus --cos  --wor\
kers 4 --checkpoint_dir $SCRATCH/checkpoints/moco"
```

### HyperParemeters

Initially, this script uses all the default hyper-parameters as described in the MoCo v2 paper. Note both 200, 250 and 300 epochs were run. For Linear Classification 100 epochs were used.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 2-gpu machine, run the following sbatch request:
```
#!/bin/bash                                                                     

#SBATCH --gres=gpu:2                                                            
#SBATCH --partition=n1s16-t4-2                                                  
#SBATCH --account=dl17                                                          
#SBATCH --time=20:00:00                                                         
#SBATCH --output=moco%j.out                                                     
#SBATCH --error=moco%j.err                                                      
#SBATCH --exclusive                                                             
#SBATCH --requeue                                                               

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/test/repo/NYU_DL_comp/moco/

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "                
source /ext3/env.sh                                                             
conda activate dev                                                              
CUDA_VISIBLE_DEVICES=0,1                                                        
python3 main_linclssupervised.py \                                              
  -a resnet50 \                                                                 
 -data /dataset \                                                               
  --lr 30.0 \                                                                   
  --batch-size 256 \                                                            
  --pretrained $SCRATCH/checkpoints/demo/moco_unsupervised_0065.pth.tar \       
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size\
 1 --rank 0"
```

### Models
Linear classification results on CSCI-GA.2572 dataset using this repo with 2 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">44 hours</td>
<td align="center">35&plusmn;0.1</td>
</tr>
</tbody></table>

Our pre-trained ResNet-50 models along with supplementary code can be found [here](https://drive.google.com/drive/folders/1JhFI2a_fiUphjgzQHiwKFSTWdy_Kn8a5?usp=sharing)

REQUIRED PACKAGES
-------------------
- Python 3 (tested on 3.8)
- PyTorch (1.8.1)
- CudaToolKit (tested on 11.1)
- SciPy
- Numpy

### License
See [LICENSE](LICENSE) for details.
