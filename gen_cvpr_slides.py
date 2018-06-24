from os.path import join
from os.path import os
import ubelt as ub
import glob
import netharn as nh
from netharn.data import coco_api


class Sampler:
    def __init__(sampler, dset):
        sampler.dset = dset

    def variety_selection(sampler, num=20):
        import numpy as np
        dset = sampler.dset

        gid_to_props = ub.odict()
        for gid, img in dset.imgs.items():
            aids = dset.gid_to_aids[gid]
            annot_types = frozenset(dset.anns[aid]['roi_shape'] for aid in aids)
            annot_cids = frozenset(dset.anns[aid]['category_id'] for aid in aids)
            gid_to_props[gid] = ub.odict()
            gid_to_props[gid]['num_aids'] = len(aids)
            gid_to_props[gid]['annot_types'] = annot_types
            gid_to_props[gid]['annot_cats'] = annot_cids
            gid_to_props[gid]['orig_dset'] = frozenset([img['orig_dset']])

            try:
                from datetime import datetime
                datetime_object = datetime.strptime(img['date'], '%Y-%m-%d %H:%M:%S')
            except Exception as ex:
                print('failed to parse time: {}'.format(img.get('date', None)))
                gid_to_props[gid]['time'] = None
            else:
                gid_to_props[gid]['time'] = datetime_object.toordinal()
                # 735858 + np.random.randn()

        if True:
            # Handle items without a parsable time
            all_ts = []
            for p in gid_to_props.values():
                if p['time'] is not None:
                    all_ts.append(p['time'])
            if len(all_ts) == 0:
                all_ts = [735857, 735859, 735850]
            all_ts = np.array(all_ts)
            mean_t = all_ts.mean()
            std_t = all_ts.std()
            for p in gid_to_props.values():
                if p['time'] is None:
                    p['time'] = mean_t + np.random.randn() * std_t

        basis_values = ub.ddict(set)
        for gid, props in gid_to_props.items():
            for key, value in props.items():
                if ub.iterable(value):
                    basis_values[key].update(value)

        basis_values = ub.map_vals(sorted, basis_values)

        # Build a descriptor to find a "variety" of images
        gid_to_desc = {}
        for gid, props in gid_to_props.items():
            desc = []
            for key, value in props.items():
                if ub.iterable(value):
                    hotvec = np.zeros(len(basis_values[key]))
                    for v in value:
                        idx = basis_values[key].index(v)
                        hotvec[idx] = 1
                    desc.append(hotvec)
                else:
                    desc.append([value])
            gid_to_desc[gid] = list(ub.flatten(desc))

        gids = np.array(list(gid_to_desc.keys()))
        vecs = np.array(list(gid_to_desc.values()))

        from sklearn import cluster
        algo = cluster.KMeans(
            n_clusters=num, n_init=20, max_iter=10000, tol=1e-6,
            algorithm='elkan', verbose=0)
        algo.fit(vecs)
        algo.cluster_centers_

        assignment = algo.predict(vecs)
        grouped_gids = ub.group_items(gids, assignment)

        gid_list = [nh.util.shuffle(gids)[0] for gids in grouped_gids.values()]
        return gid_list

    def images_with_keypoints(sampler):
        keypoint_gids = set()
        for aid, ann in sampler.dset.anns.items():
            if ann['roi_shape'] == 'keypoints':
                keypoint_gids.add(ann['image_id'])

        relevant = ub.dict_subset(sampler.dset.gid_to_aids, keypoint_gids)
        relevant = {
            gid: [a for a in aids
                  if sampler.dset.anns[a]['roi_shape'] == 'keypoints']
            for gid, aids in relevant.items()}

        gid_list = ub.argsort(ub.map_vals(len, relevant))[::-1]
        return gid_list

    def sort_gids_by_nannots(sampler, gids):
        img_aids = ub.dict_subset(sampler.dset.gid_to_aids, gids, default=[])
        img_num_aids = ub.map_vals(len, img_aids)
        return ub.argsort(img_num_aids)[::-1]

    def images_with_keypoints_and_boxes(sampler):
        keypoint_gids = set()
        for aid, ann in sampler.dset.anns.items():
            if ann['roi_shape'] == 'keypoints':
                keypoint_gids.add(ann['image_id'])

        gid_list = []
        for gid in keypoint_gids:
            aids = sampler.dset.gid_to_aids[gid]
            types = set()
            for ann in ub.take(sampler.dset.anns, aids):
                types.add(ann['roi_shape'])
            if len(types) > 1:
                gid_list.append(gid)

        gid_list = sampler.sort_gids_by_nannots(gid_list)
        return gid_list

    def image_from_each_dataset(sampler):
        groups = ub.ddict(list)
        for gid, img in sampler.dset.imgs.items():
            groups[os.path.dirname(img['file_name'])].append(gid)

        gid_groups = []
        for gids in groups.values():
            gids = sampler.sort_gids_by_nannots(gids)
            gid_groups.append(gids)

        # round robin sample
        datas = [gid for x in zip(*gid_groups) for gid in x]
        return datas


def gen_cvpr_images():

    config = {
        # Data on hermes
        'img_root': '/data/projects/noaa/training_data/imagery',
        'annots': list(glob.glob('/data/projects/noaa/training_data/annotations/*/*fine*-keypoint*')),

        'output_dpath': ub.truepath('~/remote/hermes/work/noaa/cvpr_slides')
    }
    output_dpath = ub.ensuredir(ub.truepath(config['output_dpath']))

    annot_fpaths = config['annots']
    img_root = config['img_root']

    print('annot_fpaths = {}'.format(ub.repr2(annot_fpaths)))

    print('Reading raw mscoco files')
    dsets = {}
    for fpath in sorted(annot_fpaths):
        print('reading fpath = {!r}'.format(fpath))
        hack = os.path.basename(fpath).split('-')[0].split('.')[0]
        dset = coco_api.CocoDataset(fpath, tag=hack, img_root=join(img_root, hack))
        # assert not dset.missing_images()
        # assert not dset._find_bad_annotations()
        print(ub.repr2(dset.basic_stats()))
        for img in dset.dataset['images']:
            img['orig_dset'] = dset.tag
        dset._build_index()
        dsets[dset.tag] = dset

    # Separate habcam out because it is so much bigger than the other sets
    habcam = dsets['habcam_seq0']
    # del dsets['habcam_seq0']

    def invert_y_coordinate(nwfsc):
        import numpy as np
        nwfsc._build_index()
        for ann in nwfsc.anns.values():
            # Invert the Y coordinate for NWFSC
            img = nwfsc.imgs[ann['image_id']]
            xyv = np.array(ann['keypoints']).reshape(-1, 3)
            xyv[:, 1] = img['height'] - xyv[:, 1]
            ann['keypoints'] = xyv.ravel().tolist()
    nwfsc = dsets['nwfsc_seq0']
    invert_y_coordinate(nwfsc)

    print('Merging')
    merged = coco_api.CocoDataset.union(*dsets.values(), autobuild=False)

    merged.img_root = img_root
    merged._build_index()
    habcam._build_index()

    nh.util.autompl()

    merged_gid_list = Sampler(merged).variety_selection(num=20)
    habcam_gid_list = Sampler(habcam).variety_selection(num=10)

    def dump_selection(dset, gid_list):
        from matplotlib import pyplot as plt
        for gid in ub.ProgIter(gid_list, verbose=3):
            fig = plt.figure(6)
            fig.clf()
            dset.show_annotation(gid=gid)
            name = os.path.basename(os.path.dirname(dset.imgs[gid]['file_name']))
            ax = plt.gca()
            plt.gca().set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.gca().grid('off')
            fig.canvas.draw()
            dpi = 96
            fig.set_dpi(dpi)
            fig.set_size_inches(1920 / dpi, 1080 / dpi)
            img = nh.util.mplutil.render_figure_to_image(fig, dpi=dpi)
            # print('img = {!r}'.format(img.shape))

            if dset.tag:
                out_fname = dset.tag + '_' + '_'.join(dset.imgs[gid]['file_name'].split('/')[-2:])
            else:
                out_fname = '_'.join(dset.imgs[gid]['file_name'].split('/')[-2:])
            fpath = join(output_dpath, out_fname)
            print('fpath = {!r}'.format(fpath))
            nh.util.imwrite(fpath, img)
            # nh.util.imshow(img, fnum=2)

    dump_selection(merged, merged_gid_list)
    dump_selection(habcam, habcam_gid_list)

    mouss0_gid_list = Sampler(dsets['mouss_seq0']).variety_selection(num=5)
    dump_selection(dsets['mouss_seq0'], mouss0_gid_list)

    nwfsc_seq0_gids = Sampler(dsets['nwfsc_seq0']).variety_selection(num=10)
    dump_selection(dsets['nwfsc_seq0'], nwfsc_seq0_gids)

    if False:
        from matplotlib import pyplot as plt
        import utool as ut
        for gid in ut.InteractiveIter(gid_list):
            try:
                fig = plt.figure(5)
                fig.clf()
                merged.show_annotation(gid=gid)
                name = os.path.basename(os.path.dirname(merged.imgs[gid]['file_name']))
                ax = plt.gca()
                plt.gca().set_title(name)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.gca().grid('off')
                fig.canvas.draw()
            except Exception:
                print('cannot draw')
