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
    # Set has_annots=True on all images with at least one annotation
    merged._mark_annotated_images()

    def ensure_heirarchy(dset, heirarchy):
        for cat in heirarchy:
            try:
                dset.add_category(**cat)
            except ValueError:
                realcat = dset.name_to_cat[cat['name']]
                realcat['supercategory'] = cat['supercategory']

    prefix = 'phase{}'.format(cfg.phase)

    def verbose_dump(dset, fpath):
        print('Dumping {}'.format(fpath))
        if False:
            print(ub.repr2(dset.category_annotation_type_frequency(), nl=1, sk=1))
        print(ub.dict_hist([img['has_annots'] for img in dset.imgs.values()]))
        print(ub.repr2(dset.basic_stats()))
        dset.dump(fpath)

    ### FINE-GRAIND DSET  ###
    fine = merged.copy()
    FineGrainedChallenge = viame_wrangler.mappings.FineGrainedChallenge
    fine.rename_categories(FineGrainedChallenge.raw_to_cat)
    verbose_dump(fine, join(cfg.challenge_work_dir, prefix + '-fine-bbox-keypoint.mscoco.json'))

    # remove keypoint annotations
    # Should we remove the images that have keypoints in them?
    fine_bbox = fine.copy()
    fine_bbox._remove_keypoint_annotations()
    verbose_dump(fine_bbox, join(cfg.challenge_work_dir, prefix + '-fine-bbox-only.mscoco.json'))

    ### COARSE DSET  ###
    coarse = merged.copy()
    CoarseChallenge = viame_wrangler.mappings.CoarseChallenge
    coarse.rename_categories(CoarseChallenge.raw_to_cat)
    ensure_heirarchy(coarse, CoarseChallenge.heirarchy)
    verbose_dump(coarse, join(cfg.challenge_work_dir, prefix + '-coarse-bbox-keypoint.mscoco.json'))

    # remove keypoint annotations
    coarse_bbox = coarse.copy()
    coarse_bbox._remove_keypoint_annotations()
    verbose_dump(coarse_bbox, join(cfg.challenge_work_dir, prefix + '-coarse-bbox-only.mscoco.json'))
    return fine, coarse, fine_bbox, coarse_bbox


def make_test_train(merged):
    # Split into train / test  set
    print('Splitting')
    import numpy as np
    rng = np.random.RandomState(0)
    skf = StratifiedGroupKFold(n_splits=2, random_state=rng)
    groups = [ann['image_id'] for ann in merged.anns.values()]
    y = [ann['category_id'] for ann in merged.anns.values()]
    X = [ann['id'] for ann in merged.anns.values()]
    split = list(skf.split(X=X, y=y, groups=groups))[0]
    train_idx, test_idx = split

    print('Taking subsets')
    aid_to_gid = {aid: ann['image_id'] for aid, ann in merged.anns.items()}
    train_aids = list(ub.take(X, train_idx))
    test_aids = list(ub.take(X, test_idx))

    train_gids = set(ub.take(aid_to_gid, train_aids))
    test_gids = set(ub.take(aid_to_gid, test_aids))

    # Include images without any annotations
    gids = set(merged.imgs.keys())

    extra_gids = (gids - (train_gids | test_gids))

    gids_maybe = []
    gids_true = []
    gids_false = []
    for gid in extra_gids:
        if 'has_annots' not in merged.imgs[gid]:
            merged.imgs[gid]['has_annots'] = None

        if str(merged.imgs[gid]['has_annots']).lower() == 'false':
            merged.imgs[gid]['has_annots'] = False
        if str(merged.imgs[gid]['has_annots']).lower() == 'true':
            merged.imgs[gid]['has_annots'] = True
        if str(merged.imgs[gid]['has_annots']).lower() == 'null':
            merged.imgs[gid]['has_annots'] = None

        if merged.imgs[gid]['has_annots'] is None:
            # There might actually be unannoted fish in these images
            gids_maybe.append(gid)
        elif merged.imgs[gid]['has_annots'] is False:
            # We know there should not be any fish in this image
            gids_false.append(gid)
        elif merged.imgs[gid]['has_annots'] is True:
            # There are fish in this image, but the annots were removed
            gids_true.append(gid)
        else:
            assert False, repr(merged.imgs[gid]['has_annots'])

    rng.shuffle(gids_maybe)
    rng.shuffle(gids_false)

    train_gids.update(gids_false[0::2])
    train_gids.update(gids_maybe[0::2])

    test_gids.update(gids_false[1::2])
    test_gids.update(gids_maybe[1::2])

    train_dset = merged.subset(sorted(train_gids))
    test_dset = merged.subset(sorted(test_gids))

    print('--- Training Stats ---')
    print(ub.repr2(train_dset.basic_stats()))
    print('--- Testing Stats ---')
    print(ub.repr2(test_dset.basic_stats()))
    return train_dset, test_dset


def setup_yolo(cfg=None):
    """
    CommandLine:
        python ~/code/baseline-viame-2018/wrangle.py setup_yolo \
            --annots=$HOME/data/viame-challenge-2018/phase1-annotations \
            --work=$HOME/work/viame-challenge-2018

        python ~/code/baseline-viame-2018/wrangle.py setup_yolo \
            --annots=/data/projects/noaa/phase1-annotations/*/*coarse-bbox-only*.json \
            --img_root=/data/projects/noaa/phase1-imagery \
            --work=$HOME/work/viame-challenge-2018

    Ignore:
        cfg = viame_wrangler.config.WrangleConfig()
        cfg.annots = '/data/projects/noaa/phase1-annotations/*/*coarse-bbox-only*.json'
        cfg.img_root = '/data/projects/noaa/phase1-imagery'
        cfg.work = ub.truepath('$HOME/work/viame-challenge-2018')
    """
    if cfg is None:
        cfg = viame_wrangler.config.WrangleConfig()

    fpaths = list(glob.glob(cfg.annots))
    print('fpaths = {!r}'.format(fpaths))

    print('Reading raw mscoco files')
    dsets = []
    for fpath in sorted(fpaths):
        print('reading fpath = {!r}'.format(fpath))
        import os
        hack = os.path.basename(fpath).split('-')[0]
        dset = CocoDataset(fpath, img_root=hack)
        print(ub.repr2(dset.basic_stats()))
        dsets.append(dset)

    print('Merging')
    merged = CocoDataset.union(*dsets)
    merged.img_root = cfg.img_root

    # suffix = 'coarse-bbox-only'
    # prefix = 'phase{}'.format(cfg.phase)
    train_dset, test_dset = make_test_train(merged)

    if 1:
        print(ub.repr2(train_dset.category_annotation_type_frequency(), nl=1, sk=1))
        print(ub.repr2(test_dset.category_annotation_type_frequency(), nl=1, sk=1))

    print('Writing')
    train_fpath = join(cfg.workdir, 'train.mscoco.json')
    test_fpath = join(cfg.workdir, 'vali.mscoco.json')
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
