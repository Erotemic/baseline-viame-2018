# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import glob
import ubelt as ub
from coco_wrangler import CocoDataset, StratifiedGroupKFold
from viame_wrangler.cats_2018 import get_coarse_mapping


def setup_data():
    """
    hacks:
        rsync -avpR acidalia:/home/git/phase0-annotations.tar.gz ~/Downloads

        rm -rf ~/data/viame-challenge-2018/phase0-annotations
        rm -rf ~/data/viame-challenge-2018/phase0-*.mscoco.json

        tar xvzf ~git/phase0-annotations-old-names.tar.gz -C ~/data/viame-challenge-2018
        ls ~/data/viame-challenge-2018

    Challenge Website:
        http://www.viametoolkit.org/cvpr-2018-workshop-data-challenge/

    Challenge Download Data:
        https://challenge.kitware.com/girder#collection/5a722b2c56357d621cd46c22/folder/5a9028a256357d0cb633ce20
        * Groundtruth: phase0-annotations.tar.gz
        * Images: phase0-imagery.tar.gz

    ```
    CODE_DIR=$HOME/code
    DATA_DIR=$HOME/data
    WORK_DIR=$HOME/work

    mkdir -p $DATA_DIR/viame-challenge-2018
    tar xvzf $HOME/Downloads/phase0-annotations.tar.gz -C $DATA_DIR/viame-challenge-2018
    tar xvzf $HOME/Downloads/phase0-imagery.tar.gz -C $DATA_DIR/viame-challenge-2018
    ```

    """
    work_dir = ub.truepath(ub.argval('--work', default='~/work'))
    data_dir = ub.truepath(ub.argval('--data', default='~/data'))
    challenge_data_dir = join(data_dir, 'viame-challenge-2018')
    challenge_work_dir = join(work_dir, 'viame-challenge-2018')
    ub.ensuredir(challenge_work_dir)

    img_root = join(challenge_data_dir, 'phase0-imagery')
    # annot_dir = join(challenge_data_dir, 'phase0-annotations-old-names')
    annot_dir = join(challenge_data_dir, 'phase0-annotations')
    fpaths = list(glob.glob(join(annot_dir, '*.json')))
    # ignore the non-bounding box nwfsc and afsc datasets for now

    # exclude = ('nwfsc', 'afsc', 'mouss', 'habcam')
    # exclude = ('mbari',)
    # fpaths = [p for p in fpaths if not basename(p).startswith(exclude)]

    print('Reading')
    dsets = []
    for fpath in fpaths:
        dset = CocoDataset(fpath, autobuild=False)
        # dset._run_fixes()
        dset._build_index()
        dsets.append(dset)

        # print(ub.repr2([d['name'] for d in dset.cats.values()]))

    for dset in dsets:
        print(dset.img_root)
        # print(ub.repr2(dset.basic_stats()))
        print(ub.repr2(dset.category_annotation_frequency()))

    print('Merging')
    merged = CocoDataset.union(*dsets)
    merged.img_root = img_root
    # merged._run_fixes()
    # print(ub.repr2(merged.category_annotation_frequency()))

    merged.dump(join(challenge_work_dir, 'phase0-merged-raw.mscoco.json'))

    if True:
        # Cleanup the dataset
        merged._remove_bad_annotations()
        merged._remove_radius_annotations()

        merged._remove_keypoint_annotations()
        merged._remove_empty_images()

    print(ub.repr2(merged.category_annotation_frequency()))
    print(sum(list(merged.category_annotation_frequency().values())))

    # Remap to coarse categories
    merged.coarsen_categories(get_coarse_mapping())
    print(ub.repr2(merged.category_annotation_frequency()))
    print(sum(list(merged.category_annotation_frequency().values())))

    print(ub.repr2(merged.basic_stats()))

    merged.dump(join(challenge_work_dir, 'phase0-merged-coarse.mscoco.json'))

    if False:
        # aid = list(merged.anns.values())[0]['id']
        # merged.show_annotation(aid)
        gids = sorted([gid for gid, aids in merged.gid_to_aids.items() if aids])
        # import utool as ut
        # for gid in ut.InteractiveIter(gids):
        for gid in gids:
            from matplotlib import pyplot as plt
            fig = plt.figure(1)
            fig.clf()
            merged.show_annotation(gid=gid)
            fig.canvas.draw()

    # Split into train / test  set
    print('Splitting')
    skf = StratifiedGroupKFold(n_splits=2)
    groups = [ann['image_id'] for ann in merged.anns.values()]
    y = [ann['category_id'] for ann in merged.anns.values()]
    X = [ann['id'] for ann in merged.anns.values()]
    split = list(skf.split(X=X, y=y, groups=groups))[0]
    train_idx, test_idx = split

    print('Taking subsets')
    aid_to_gid = {aid: ann['image_id'] for aid, ann in merged.anns.items()}
    train_aids = list(ub.take(X, train_idx))
    test_aids = list(ub.take(X, test_idx))
    train_gids = sorted(set(ub.take(aid_to_gid, train_aids)))
    test_gids = sorted(set(ub.take(aid_to_gid, test_aids)))

    train_dset = merged.subset(train_gids)
    test_dset = merged.subset(test_gids)

    print('--- Training Stats ---')
    print(ub.repr2(train_dset.basic_stats()))
    print('--- Testing Stats ---')
    print(ub.repr2(test_dset.basic_stats()))


def setup_detectron(train_dset, test_dset):
    work_dir = ub.truepath(ub.argval('--work', default='~/work'))
    data_dir = ub.truepath(ub.argval('--data', default='~/data'))
    challenge_work_dir = join(work_dir, 'viame-challenge-2018')
    ub.ensuredir(challenge_work_dir)

    train_dset._ensure_imgsize()
    test_dset._ensure_imgsize()

    print('Writing')
    train_dset.dump(join(challenge_work_dir, 'phase0-merged-train.mscoco.json'))
    test_dset.dump(join(challenge_work_dir, 'phase0-merged-test.mscoco.json'))

    num_classes = len(train_dset.cats)
    print('num_classes = {!r}'.format(num_classes))

    # Make a detectron yaml file
    config_text = ub.codeblock(
        """
        MODEL:
          TYPE: generalized_rcnn
          CONV_BODY: ResNet.add_ResNet50_conv4_body
          NUM_CLASSES: {num_classes}
          FASTER_RCNN: True
        NUM_GPUS: 1
        SOLVER:
          WEIGHT_DECAY: 0.0001
          LR_POLICY: steps_with_decay
          BASE_LR: 0.01
          GAMMA: 0.1
          # 1x schedule (note TRAIN.IMS_PER_BATCH: 1)
          MAX_ITER: 180000
          STEPS: [0, 120000, 160000]
        RPN:
          SIZES: (32, 64, 128, 256, 512)
        FAST_RCNN:
          ROI_BOX_HEAD: ResNet.add_ResNet_roi_conv5_head
          ROI_XFORM_METHOD: RoIAlign
        TRAIN:
          WEIGHTS: https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
          DATASETS: ('/work/viame-challenge-2018/phase0-merged-train.mscoco.json',)
          IM_DIR: '/data/viame-challenge-2018/phase0-imagery'
          SCALES: (800,)
          MAX_SIZE: 1333
          IMS_PER_BATCH: 1
          BATCH_SIZE_PER_IM: 512
        TEST:
          DATASETS: ('/work/viame-challenge-2018/phase0-merged-test.mscoco.json',)
          IM_DIR: '/data/viame-challenge-2018/phase0-imagery'
          SCALES: (800,)
          MAX_SIZE: 1333
          NMS: 0.5
          FORCE_JSON_DATASET_EVAL: True
          RPN_PRE_NMS_TOP_N: 6000
          RPN_POST_NMS_TOP_N: 1000
        OUTPUT_DIR: /work/viame-challenge-2018/output
        """)
    config_text = config_text.format(
        num_classes=num_classes,
    )
    ub.writeto(join(challenge_work_dir, 'phase0-faster-rcnn.yaml'), config_text)

    docker_cmd = ('nvidia-docker run '
                  '-v {work_dir}:/work -v {data_dir}:/data '
                  '-it detectron:c2-cuda9-cudnn7 bash').format(
                      work_dir=work_dir, data_dir=data_dir)

    train_cmd = ('python2 tools/train_net.py '
                 '--cfg /work/viame-challenge-2018/phase0-faster-rcnn.yaml '
                 'OUTPUT_DIR /work/viame-challenge-2018/output')

    hacks = ub.codeblock(
        """
        git remote add Erotemic https://github.com/Erotemic/Detectron.git
        git fetch --all
        git checkout general_dataset

        python2 tools/train_net.py --cfg /work/viame-challenge-2018/phase0-faster-rcnn.yaml OUTPUT_DIR /work/viame-challenge-2018/output
        """)

    print(docker_cmd)
    print(train_cmd)


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/baseline-viame-2018
        python ~/code/baseline-viame-2018/wrangle.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
