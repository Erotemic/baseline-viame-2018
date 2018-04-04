import viame_wrangler
from coco_wrangler import CocoDataset
import glob
from os.path import join
from os.path import os
import ubelt as ub


def fix_dataset_phase1_original():
    cfg = viame_wrangler.config.WrangleConfig({
        'annots': ub.truepath('~/data/viame-challenge-2018/phase1-annotations/*/original_*.json')
    })

    annots = cfg.annots
    fpaths = list(glob.glob(annots))
    print('Reading raw mscoco files')
    fpath_iter = iter(fpaths)

    for fpath in fpath_iter:
        print('reading fpath = {!r}'.format(fpath))
        dset_name = os.path.basename(fpath).replace('original_', '').split('.')[0]
        dset = CocoDataset(fpath, img_root=cfg.img_root)

        did_fix = False
        if dset.missing_images():
            did_fix = True
            print('Fixing missing images')
            for img in dset.dataset['images']:
                if img['file_name'].startswith(dset_name):
                    assert False
                img['file_name'] = join(dset_name, img['file_name'])
            assert not dset.missing_images()
        # dset.dataset.keys()
        # dset.dataset['categories']

        bad_annots = dset._find_bad_annotations()
        if bad_annots:
            print('Fixing bad annots')
            did_fix = True
            for ann in bad_annots:
                dset.remove_annotation(ann)
            dset._build_index()

        bad_hasannots_flags = not all([img.get('has_annots', ub.NoParam) in [True, False, None] for img in dset.imgs.values()])

        if bad_hasannots_flags:
            did_fix = True
            for gid, img in dset.imgs.items():
                aids = dset.gid_to_aids.get(gid, [])

                if True:
                    # SPECIAL CASES
                    if img['file_name'] == 'afsc_seq1/003496.jpg':
                        img['has_annots'] = True

                # if False:
                #     if img['has_annots'] is None:
                #         dset.show_annotation(gid=img['id'])
                #         break
                # If there is at least one annotation, always mark as has_annots
                if img.get('has_annots', None) not in [True, False, None]:
                    if str(img['has_annots']).lower() == 'false':
                        img['has_annots'] = False
                    else:
                        assert False, ub.repr2(img)
                if len(aids) > 0:
                    img['has_annots'] = True
                else:
                    # Otherwise set has_annots to null if it has not been
                    # explicitly labeled
                    if 'has_annots' not in img:
                        img['has_annots'] = None

            print(ub.dict_hist([g['has_annots'] for g in dset.imgs.values()]))

        if did_fix:
            print('manual check')
            break
            # ut.print_difftext(ut.get_textdiff(dset.dumps(), orig_dset.dumps()))
            dset.dump(fpath)


def regenerate_phase1_flavors():
    """
    Assumes original data is in a good format
    """
    cfg = viame_wrangler.config.WrangleConfig({
        'annots': ub.truepath('~/data/viame-challenge-2018/phase1-annotations/*/original_*.json')
    })

    annots = cfg.annots
    fpaths = list(glob.glob(annots))
    print('Reading raw mscoco files')
    for fpath in fpaths:
        print('reading fpath = {!r}'.format(fpath))
        dset_name = os.path.basename(fpath).replace('original_', '').split('.')[0]
        orig_dset = CocoDataset(fpath, img_root=cfg.img_root, tag=dset_name)
        dpath = os.path.dirname(fpath)

        assert not orig_dset.missing_images()
        assert not orig_dset._find_bad_annotations()
        assert all([img['has_annots'] in [True, False, None] for img in orig_dset.imgs.values()])
        print(ub.dict_hist([g['has_annots'] for g in orig_dset.imgs.values()]))

        make_dataset_flavors(orig_dset, dpath, dset_name)


def make_dataset_flavors(orig_dset, dpath, dset_name):
    import viame_wrangler.mappings
    def ensure_heirarchy(dset, heirarchy):
        for cat in heirarchy:
            try:
                dset.add_category(**cat)
            except ValueError:
                realcat = dset.name_to_cat[cat['name']]
                realcat['supercategory'] = cat['supercategory']

    def verbose_dump(dset, fpath):
        print('Dumping {}'.format(fpath))
        print('dset_name = {!r}'.format(dset_name))
        if False:
            print(ub.repr2(dset.category_annotation_type_frequency(), nl=1, sk=1))
        print(ub.dict_hist([img['has_annots'] for img in dset.imgs.values()]))
        print(ub.repr2(dset.basic_stats()))
        dset.dump(fpath)

    ### FINE-GRAIND DSET  ###
    fine = orig_dset.copy()
    FineGrainedChallenge = viame_wrangler.mappings.FineGrainedChallenge
    fine.rename_categories(FineGrainedChallenge.raw_to_cat)
    ensure_heirarchy(fine, FineGrainedChallenge.heirarchy)
    verbose_dump(fine, join(dpath, dset_name + '-fine-bbox-keypoint.mscoco.json'))

    # remove keypoint annotations
    # Should we remove the images that have keypoints in them?
    fine_bbox = fine.copy()
    fine_bbox._remove_keypoint_annotations()
    verbose_dump(fine_bbox, join(dpath, dset_name + '-fine-bbox-only.mscoco.json'))

    ### COARSE DSET  ###
    coarse = orig_dset.copy()
    CoarseChallenge = viame_wrangler.mappings.CoarseChallenge
    coarse.rename_categories(CoarseChallenge.raw_to_cat)
    ensure_heirarchy(coarse, CoarseChallenge.heirarchy)
    verbose_dump(coarse, join(dpath, dset_name + '-coarse-bbox-keypoint.mscoco.json'))

    # remove keypoint annotations
    coarse_bbox = coarse.copy()
    coarse_bbox._remove_keypoint_annotations()
    verbose_dump(coarse_bbox, join(dpath, dset_name + '-coarse-bbox-only.mscoco.json'))


def generate_phase1_data_tables():
    cfg = viame_wrangler.config.WrangleConfig({
        'annots': ub.truepath('~/data/viame-challenge-2018/phase1-annotations/*/*coarse*bbox-keypoint*.json')
    })

    all_stats = {}

    annots = cfg.annots
    fpaths = list(glob.glob(annots))
    print('fpaths = {}'.format(ub.repr2(fpaths)))
    for fpath in fpaths:
        dset_name = os.path.basename(fpath).split('-')[0]
        dset = CocoDataset(fpath, img_root=cfg.img_root, tag=dset_name)

        assert not dset.missing_images()
        assert not dset._find_bad_annotations()
        assert all([img['has_annots'] in [True, False, None] for img in dset.imgs.values()])

        print(ub.dict_hist([g['has_annots'] for g in dset.imgs.values()]))

        stats = {}
        stats.update(ub.dict_subset(dset.basic_stats(), ['n_anns', 'n_imgs']))

        roi_shapes_hist = dict()
        populated_cats = dict()
        for name, item in dset.category_annotation_type_frequency().items():
            if item:
                populated_cats[name] = sum(item.values())
                for k, v in item.items():
                    roi_shapes_hist[k] = roi_shapes_hist.get(k, 0) + v

        stats['n_cats'] = populated_cats
        stats['n_roi_shapes'] = roi_shapes_hist
        stats['n_imgs_with_annots'] = ub.map_keys({None: 'unsure', True: 'has_objects', False: 'no_objects'}, ub.dict_hist([g['has_annots'] for g in dset.imgs.values()]))
        all_stats[dset_name] = stats

    print(ub.repr2(all_stats, nl=3, sk=1))
