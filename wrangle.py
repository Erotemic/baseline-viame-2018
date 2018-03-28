# -*- coding: utf-8 -*-
"""
hacks:
    rsync -avp acidalia:/home/git/phase0-annotations.tar.gz ~/Downloads
    rsync -avp acidalia:/home/git/noaa-full-datasets.tar.gz ~/Downloads

    rm -rf ~/data/viame-challenge-2018/phase0-annotations
    rm -rf ~/data/viame-challenge-2018/phase0-*.mscoco.json

    tar xvzf ~/Downloads/phase0-annotations.tar.gz -C ~/data/viame-challenge-2018
    ls ~/data/viame-challenge-2018

    tar xvzf ~git/phase0-annotations-old-names.tar.gz -C ~/data/viame-challenge-2018
    ls ~/data/viame-challenge-2018

    tar xvzf ~/Downloads/noaa-full-datasets.tar.gz -C ~/data/viame-challenge-2018
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
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import glob
import ubelt as ub
from coco_wrangler import CocoDataset, StratifiedGroupKFold
import viame_wrangler.mappings


class WrangleConfig(object):
    def __init__(cfg):
        cfg.work_dir = ub.truepath(ub.argval('--work', default='~/work'))
        cfg.data_dir = ub.truepath(ub.argval('--data', default='~/data'))
        cfg.challenge_data_dir = join(cfg.data_dir, 'viame-challenge-2018')
        cfg.challenge_work_dir = join(cfg.work_dir, 'viame-challenge-2018')
        ub.ensuredir(cfg.challenge_work_dir)


def show_low_support_classes(dset):
    """
    dset = merged
    coarse = merged
    """
    # aid = list(dset.anns.values())[0]['id']
    # dset.show_annotation(aid)
    dset._remove_keypoint_annotations()
    gids = sorted([gid for gid, aids in dset.gid_to_aids.items() if aids])

    catfreq = dset.category_annotation_frequency()
    inspect_cids = []
    for name, freq in catfreq.items():
        if freq > 0 and freq < 50:
            cid = dset.name_to_cat[name]['id']
            inspect_cids.append(cid)
    inspect_gids = list(set(ub.flatten(ub.take(dset.cid_to_gids, inspect_cids))))
    # inspect_gids = [gid for gid in inspect_gids if 'habcam' not in dset.imgs[gid]['file_name']]

    import utool as ut
    if ut.inIPython():
        import IPython
        IPython.get_ipython().magic('pylab qt5 --no-import-all')

    print('inspect_gids = {!r}'.format(inspect_gids))
    from matplotlib import pyplot as plt
    for gid in ut.InteractiveIter(inspect_gids):
        img = dset.imgs[gid]
        print('img = {}'.format(ub.repr2(img)))
        aids = dset.gid_to_aids[gid]
        primary_aid = None
        anns = list(ub.take(dset.anns, aids))
        for ann in anns:
            ann = ann.copy()
            ann['category'] = dset.cats[ann['category_id']]['name']
            print('ann = {}'.format(ub.repr2(ann)))
            if primary_aid is None:
                if ann['category_id'] in inspect_cids:
                    primary_aid = ann['id']

        try:
            fig = plt.figure(1)
            fig.clf()
            dset.show_annotation(primary_aid, gid=gid)
            fig.canvas.draw()
        except:
            print('cannot draw')

    # # import utool as ut
    # for gid in gids:
    #     fig = plt.figure(1)
    #     fig.clf()
    #     dset.show_annotation(gid=gid)
    #     fig.canvas.draw()


def setup_data():
    cfg = WrangleConfig()

    img_root = join(cfg.challenge_data_dir, 'phase0-imagery')
    annot_dir = join(cfg.challenge_data_dir, 'phase0-annotations')
    fpaths = list(glob.glob(join(annot_dir, '*.json')))

    print('Reading')
    dsets = []
    for fpath in fpaths:
        dset = CocoDataset(fpath)
        dsets.append(dset)

    print('Merging')
    merged = CocoDataset.union(*dsets)
    merged.img_root = img_root

    def ensure_heirarchy(dset, heirarchy):
        for cat in heirarchy:
            try:
                dset.add_category(**cat)
            except ValueError:
                realcat = dset.name_to_cat[cat['name']]
                realcat['supercategory'] = cat['supercategory']

    ### FINE-GRAIND DSET  ###
    fine = merged.copy()
    FineGrainedChallenge = viame_wrangler.mappings.FineGrainedChallenge
    fine.rename_categories(FineGrainedChallenge.raw_to_cat)
    ensure_heirarchy(fine, FineGrainedChallenge.heirarchy)
    fine.dump(join(cfg.challenge_work_dir, 'phase0-merged-fine-bbox-keypoint.mscoco.json'))
    if True:
        print(ub.repr2(fine.category_annotation_type_frequency(), nl=1, sk=1))
        print(ub.repr2(fine.basic_stats()))
    # remove keypoint annotations
    # Should we remove the images that have keypoints in them?
    fine_bbox = fine.copy()
    fine_bbox._remove_keypoint_annotations()
    fine_bbox.dump(join(cfg.challenge_work_dir, 'phase0-merged-fine-bbox.mscoco.json'))

    ### COARSE DSET  ###
    coarse = merged.copy()
    CoarseChallenge = viame_wrangler.mappings.CoarseChallenge
    coarse.rename_categories(CoarseChallenge.raw_to_cat)
    ensure_heirarchy(coarse, CoarseChallenge.heirarchy)
    print(ub.repr2(coarse.basic_stats()))
    coarse.dump(join(cfg.challenge_work_dir, 'phase0-merged-coarse-bbox-keypoint.mscoco.json'))
    if True:
        print(ub.repr2(coarse.category_annotation_type_frequency(), nl=1, sk=1))
        print(ub.repr2(coarse.basic_stats()))
    # remove keypoint annotations
    coarse_bbox = coarse.copy()
    coarse_bbox._remove_keypoint_annotations()
    coarse_bbox.dump(join(cfg.challenge_work_dir, 'phase0-merged-coarse-bbox.mscoco.json'))
    return fine, coarse


def setup_yolo():
    """
        python ~/code/baseline-viame-2018/wrangle.py setup_yolo
    """
    cfg = WrangleConfig()
    fine, coarse = setup_data()
    train_dset, test_dset = make_test_train(coarse)

    print('Writing')
    train_dset.dump(join(cfg.challenge_work_dir, 'phase0-merged-train.mscoco.json'))
    test_dset.dump(join(cfg.challenge_work_dir, 'phase0-merged-test.mscoco.json'))


def make_test_train(merged):
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
    return train_dset, test_dset


def setup_detectron(train_dset, test_dset):
    cfg = WrangleConfig()

    train_dset._ensure_imgsize()
    test_dset._ensure_imgsize()

    print('Writing')
    train_dset.dump(join(cfg.challenge_work_dir, 'phase0-merged-train.mscoco.json'))
    test_dset.dump(join(cfg.challenge_work_dir, 'phase0-merged-test.mscoco.json'))

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
    ub.writeto(join(cfg.challenge_work_dir, 'phase0-faster-rcnn.yaml'), config_text)

    docker_cmd = ('nvidia-docker run '
                  '-v {work_dir}:/work -v {data_dir}:/data '
                  '-it detectron:c2-cuda9-cudnn7 bash').format(
                      work_dir=cfg.work_dir, data_dir=cfg.data_dir)

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
