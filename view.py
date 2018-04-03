from os.path import join
import os
import ubelt as ub
import viame_wrangler
from coco_wrangler import CocoDataset
import glob


def show_keypoint_annots():
    cfg = viame_wrangler.config.WrangleConfig({
        # 'img_root': '~/work/viame-challenge-2018/phase0-imagery',
        # 'annots': '~/work/viame-challenge-2018/phase0-fine*keypoint*.json'
        'img_root': '/data/projects/noaa/phase1-imagery',
        'annots': '/data/projects/noaa/phase1-annotations/*/*fine*-keypoint*',
    })
    annot_fpaths = glob.glob(cfg.annots)
    print('annot_fpaths = {}'.format(ub.repr2(annot_fpaths)))

    print('Reading raw mscoco files')
    dsets = []
    for fpath in sorted(annot_fpaths):
        print('reading fpath = {!r}'.format(fpath))
        try:
            dset = CocoDataset(fpath, tag='', img_root=cfg.img_root)
            assert not dset.missing_images()
        except AssertionError:
            hack = os.path.basename(fpath).split('-')[0].split('.')[0]
            dset = CocoDataset(fpath, tag=hack, img_root=join(cfg.img_root, hack))
            print(ub.repr2(dset.missing_images()))
            assert not dset.missing_images(), 'missing!'
        print(ub.repr2(dset.basic_stats()))
        dsets.append(dset)

    for dset in dsets:
        print({type(i) for s in dset.cid_to_gids.values() for i in s})

    print('Merging')
    merged = CocoDataset.union(*dsets, autobuild=False)
    merged.img_root = cfg.img_root
