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
import viame_wrangler
import viame_wrangler.mappings


def check_images():
    cfg = viame_wrangler.config.WrangleConfig()
    import os

    annot_dir = cfg.annot_dir
    fpaths = list(glob.glob(join(annot_dir, '*.json')))

    print('Reading raw mscoco files')
    for fpath in fpaths:
        print('reading fpath = {!r}'.format(fpath))
        dset = CocoDataset(fpath)
        # dset.img_root = img_root

        for img in dset.imgs.values():
            path = join(cfg.img_root, dset.img_root, img['file_name'])
            if not os.path.exists(path):
                print(path)


def setup_data():
    """
    Create final MSCOCO training files for the 4 challenge types:
        * fine-grained + bbox-only
        * fine-grained + bbox-keypoints
        * coarse-grained + bbox-only
        * coarse-grained + bbox-keypoints

    CommandLine:
        python ~/code/baseline-viame-2018/wrangle.py setup_data --data=$HOME/data --work=$HOME/work --phase=0
    """
    cfg = viame_wrangler.config.WrangleConfig()

    img_root = cfg.img_root
    annot_dir = cfg.annot_dir
    fpaths = list(glob.glob(join(annot_dir, '*.json')))

    print('Reading raw mscoco files')
    dsets = []
    for fpath in fpaths:
        print('reading fpath = {!r}'.format(fpath))
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

    prefix = 'phase{}'.format(cfg.phase)

    ### FINE-GRAIND DSET  ###
    fine = merged.copy()
    FineGrainedChallenge = viame_wrangler.mappings.FineGrainedChallenge
    fine.rename_categories(FineGrainedChallenge.raw_to_cat)
    ensure_heirarchy(fine, FineGrainedChallenge.heirarchy)
    if 1:
        # print(ub.repr2(fine.category_annotation_type_frequency(), nl=1, sk=1))
        print('Dumping fine-bbox-keypoint')
        print(ub.repr2(fine.basic_stats()))
    fine.dump(join(cfg.challenge_work_dir, prefix + '-fine-bbox-keypoint.mscoco.json'))

    # remove keypoint annotations
    # Should we remove the images that have keypoints in them?
    fine_bbox = fine.copy()
    fine_bbox._remove_keypoint_annotations()
    if 1:
        # print(ub.repr2(fine.category_annotation_type_frequency(), nl=1, sk=1))
        print('Dumping fine-bbox-only')
        print(ub.repr2(fine_bbox.basic_stats()))
    fine_bbox.dump(join(cfg.challenge_work_dir, prefix + '-fine-bbox-only.mscoco.json'))

    ### COARSE DSET  ###
    coarse = merged.copy()
    CoarseChallenge = viame_wrangler.mappings.CoarseChallenge
    coarse.rename_categories(CoarseChallenge.raw_to_cat)
    ensure_heirarchy(coarse, CoarseChallenge.heirarchy)
    if 1:
        # print(ub.repr2(coarse.category_annotation_type_frequency(), nl=1, sk=1))
        print('Dumping coarse-bbox-keypoint')
        print(ub.repr2(coarse.basic_stats()))
    coarse.dump(join(cfg.challenge_work_dir, prefix + '-coarse-bbox-keypoint.mscoco.json'))

    # remove keypoint annotations
    coarse_bbox = coarse.copy()
    coarse_bbox._remove_keypoint_annotations()
    if 1:
        # print(ub.repr2(fine.category_annotation_type_frequency(), nl=1, sk=1))
        print('Dumping coarse-bbox-only')
        print(ub.repr2(coarse_bbox.basic_stats()))
    coarse_bbox.dump(join(cfg.challenge_work_dir, prefix + '-coarse-bbox-only.mscoco.json'))
    return fine, coarse, fine_bbox, coarse_bbox


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


def setup_yolo():
    """
    CommandLine:
        python ~/code/baseline-viame-2018/wrangle.py setup_yolo --data=$HOME/data --work=$HOME/work --phase=0
    """
    cfg = viame_wrangler.config.WrangleConfig()
    fine, coarse, fine_bbox, coarse_bbox = setup_data()

    suffix = 'coarse-bbox-only'
    train_dset, test_dset = make_test_train(coarse_bbox)

    print('Writing')
    prefix = 'phase{}'.format(cfg.phase)
    train_fpath = join(cfg.challenge_work_dir, prefix + '-' + suffix + '-train.mscoco.json')
    test_fpath = join(cfg.challenge_work_dir, prefix + '-' + suffix + '-val.mscoco.json')
    print('train_fpath = {!r}'.format(train_fpath))
    print('test_fpath = {!r}'.format(test_fpath))

    train_dset.dump(train_fpath)
    test_dset.dump(test_fpath)


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/baseline-viame-2018
        python ~/code/baseline-viame-2018/wrangle.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
