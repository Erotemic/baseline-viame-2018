# VIAME Detection Challenge - Baseline
A baseline solution to the 2018 VIAME detection challenge

This repo outlines a baseline solution to the 2018 VIAME Detection Challenge
using the algorithms provided by Detectron system (developed by Facebook
Research).

Challenge Website: [1]
    

The instructions in this script rely on a few predefined directories.
You may overwrite these to fit your personal workflow.

# Getting Started

```bash
CODE_DIR=$HOME/code
DATA_DIR=$HOME/data
WORK_DIR=$HOME/work
```

## Get the data 

First, download the groundtruth (phase0-annotations.tar.gz) and the images (phase0-imagery.tar.gz) from [2].

After downloading the data from challenge.kitware.com, extract it to your data directory
```bash
mkdir -p $DATA_DIR/viame-challenge-2018
tar xvzf $HOME/Downloads/phase0-annotations.tar.gz -C $DATA_DIR/viame-challenge-2018
tar xvzf $HOME/Downloads/phase0-imagery.tar.gz -C $DATA_DIR/viame-challenge-2018

tar xvzf data-challenge-training-imagery.tar.gz
```

## Install the Detectron docker image.

Assuming you already have installed `nvidia-docker`, clone the Detectron repo and build the associated docker image. 

```
DETECTRON=$CODE_DIR/Detectron
if [ ! -d "$DETECTRON" ]; then
    git clone https://github.com/facebookresearch/Detectron.git $DETECTRON
fi
# Build the docker container with caffe2 and detectron (which must use python2 ☹)
cd $DETECTRON/docker
docker build -t detectron:c2-cuda9-cudnn7 .
# test the image to make sure it works
nvidia-docker run -v ~/data:/data --rm -it detectron:c2-cuda9-cudnn7 python2 tests/test_batch_permutation_op.py
```





[1]: http://www.viametoolkit.org/cvpr-2018-workshop-data-challenge/
[2]: https://challenge.kitware.com/girder#collection/5a722b2c56357d621cd46c22/folder/5a9028a256357d0cb633ce20
[3]: https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md
