# -*- coding: utf-8 -*-
"""
Extended MS-COCO API. Currently only supports keypoints and bounding boxes.

Extends the format to also include line annotations.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
from os.path import splitext
from os.path import basename
from os.path import join
import json
import numpy as np
import ubelt as ub
import six


class CocoDataset(ub.NiceRepr):
    """

    Notes:
        A keypoint annotation
            {
                "image_id" : int,
                "category_id" : int,
                "keypoints" : [x1,y1,v1,...,xk,yk,vk],
                "score" : float,
            }
            Note that `vi` is a visibility flag.

        A bounding box annotation
            {
                "image_id" : int,
                "category_id" : int,
                "bbox" : [x,y,width,height],
                "score" : float,
            }

        We also define a non-standard "line" annotation (which
            our fixup scripts will interpret as the diameter of a circle to
            convert into a bounding box)

        A line* annotation (note this is a non-standard field)
            {
                "image_id" : int,
                "category_id" : int,
                "line" : [x1,y1,x2,y2],
                "score" : float,
            }

        Lastly, note that our datasets will sometimes specify multiple bbox,
        line, and/or, keypoints fields. In this case we may also specify a
        field roi_shape, which denotes which field is the "main" annotation
        type.

    References:
        http://cocodataset.org/#format
        http://cocodataset.org/#download

    CommandLine:
        python -m coco_wrangler.coco_api CocoDataset --show

    Example:
        >>> dataset = demo_coco_data()
        >>> self = CocoDataset(dataset, tag='demo')
        >>> self._run_fixes()
        >>> # xdoctest: +REQUIRES(--show)
        >>> self.show_annotation(gid=2)
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
    """
    def __init__(self, data, tag=None, img_root=None, autobuild=True,
                 autofix=False):
        if isinstance(data, six.string_types):
            fpath = data
            key = basename(fpath).split('.')[0]
            data = json.load(open(fpath, 'r'))
            if tag is None:
                tag = key
            if img_root is None:
                img_root = join('.', key)
        else:
            if not isinstance(data, dict):
                raise TypeError('data must be a dict or path to json file')

        if img_root is None:
            img_root = '.'

        self.tag = tag
        self.dataset = data
        self.img_root = img_root

        if autofix:
            self._run_fixes()

        if autobuild:
            self._build_index()

    def __nice__(self):
        parts = []
        parts.append('tag={}'.format(self.tag))
        if self.dataset is not None:
            info = ub.repr2(self.basic_stats(), kvsep='=', si=1, nobr=1, nl=0)
            parts.append(info)
        return ', '.join(parts)

    def dump(self, fpath, indent=4):
        """
        Writes the dataset out to the json format
        """
        with open(fpath, 'w') as fp:
            json.dump(self.dataset, fp, indent=indent)

    def _build_index(self):
        """
        build reverse indexes

        Notation:
            aid - Annotation ID
            gid - imaGe ID
            cid - Category ID
        """
        # create index
        anns, cats, imgs = {}, {}, {}
        gid_to_aids = ub.ddict(set)
        cid_to_gids = ub.ddict(set)
        cid_to_aids = ub.ddict(set)

        # Build one-to-one self-lookup maps
        for cat in self.dataset.get('categories', []):
            cid = cat['id']
            if cid in cat:
                warnings.warn('Categories have the same id in {}:\n{} and\n{}'.format(
                    self, cats[cid], cat))
            cats[cid] = cat

        for img in self.dataset.get('images', []):
            gid = img['id']
            if gid in imgs:
                warnings.warn('Images have the same id in {}:\n{} and\n{}'.format(
                    self, imgs[gid], img))
            imgs[gid] = img

        for ann in self.dataset.get('annotations', []):
            aid = ann['id']
            if aid in anns:
                warnings.warn('Annotations have the same id in {}:\n{} and\n{}'.format(
                    self, anns[aid], ann))
            anns[aid] = ann

        # Build one-to-many lookup maps
        for ann in anns.values():
            try:
                aid = ann['id']
                gid = ann['image_id']
                cid = ann['category_id']
            except KeyError:
                raise KeyError('Annotation does not have ids {}'.format(ann))

            gid_to_aids[gid].add(aid)
            cid_to_gids[cid].add(gid)
            cid_to_aids[cid].add(aid)
            if gid not in imgs:
                warnings.warn('Annotation {} in {} references '
                              'unknown image_id'.format(ann, self))
            if cid not in cats:
                warnings.warn('Annotation {} in {} references '
                              'unknown category_id'.format(ann, self))

        # create class members
        self.anns = anns
        self.imgs = imgs
        self.cats = cats
        self.gid_to_aids = ub.map_vals(sorted, gid_to_aids)
        self.cid_to_gids = ub.map_vals(sorted, cid_to_gids)
        self.cid_to_aids = ub.map_vals(sorted, cid_to_aids)

    @classmethod
    def union(cls, *others):
        """
        Merges multiple `CocoDataset` items into one. Does not retain old ids.
        """
        def _coco_union(dsets):
            """ union of dictionary based data structure """
            merged = ub.odict([
                ('categories', []),
                ('licenses', []),
                ('info', []),
                ('images', []),
                ('annotations', []),
            ])

            merged_cat_name_to_id = {}

            def update_ifnotin(d1, d2):
                """ copies keys from d2 that doent exist in d1 into d1 """
                for k, v in d2.items():
                    if k not in d1:
                        d1[k] = v
                return d1

            for subdir, old_dset in dsets:
                # Create temporary indexes to map from old to new
                cat_id_map = {}
                img_id_map = {}

                # Add the licenses / info into the merged dataset
                # Licenses / info are unused in our datas, so this might not be
                # correct
                merged['licenses'].extend(old_dset['licenses'])
                merged['info'].extend(old_dset['info'])

                # Add the categories into the merged dataset
                for old_cat in old_dset['categories']:
                    new_id = merged_cat_name_to_id.get(old_cat['name'], None)
                    # The same category might exist in different datasets.
                    if new_id is None:
                        # Only add if it does not yet exist
                        new_id = len(merged_cat_name_to_id) + 1
                        merged_cat_name_to_id[old_cat['name']] = new_id
                        new_cat = ub.odict([
                            ('id', new_id),
                            ('name', old_cat['name']),
                            ('supercategory', old_cat['supercategory']),
                        ])
                        update_ifnotin(new_cat, old_cat)
                        merged['categories'].append(new_cat)
                    cat_id_map[old_cat['id']] = new_id

                # Add the images into the merged dataset
                for old_img in old_dset['images']:
                    new_img = ub.odict([
                        ('id', len(merged['images']) + 1),
                        ('file_name', join(subdir, old_img['file_name'])),
                    ])
                    # copy over other metadata
                    update_ifnotin(new_img, old_img)
                    img_id_map[old_img['id']] = new_img['id']
                    merged['images'].append(new_img)

                # Add the annotations into the merged dataset
                for old_annot in old_dset['annotations']:
                    old_cat_id = old_annot['category_id']
                    old_img_id = old_annot['image_id']
                    new_cat_id = cat_id_map.get(old_cat_id, None)
                    new_img_id = img_id_map.get(old_img_id, None)
                    if new_cat_id is None:
                        print('annot {} in {} has bad category-id {}'.format(
                            old_annot['id'], subdir, old_cat_id))
                    if new_img_id is None:
                        print('annot {} in {} has bad image-id {}'.format(
                            old_annot['id'], subdir, old_img_id))
                    new_annot = ub.odict([
                        ('id', len(merged['annotations']) + 1),
                        ('image_id', new_img_id),
                        ('category_id', new_cat_id),
                    ])
                    update_ifnotin(new_annot, old_annot)
                    merged['annotations'].append(new_annot)
            return merged

        dsets = [(d.img_root, d.dataset) for d in others]
        merged = _coco_union(dsets)
        return cls(merged)

    def subset(self, sub_gids):
        """
        Return a subset of the larger coco dataset by specifying which images
        to port. All annotations in those images will be taken.

        Example:
            >>> dataset = demo_coco_data()
            >>> self = CocoDataset(dataset, tag='demo')
            >>> sub1 = self.subset([1])
            >>> sub2 = self.subset([2])
            >>> sub3 = self.subset([3])
            >>> others = [sub1, sub2, sub3]
            >>> rejoined = CocoDataset.union(*others)
            >>> assert len(sub1.anns) == 9
            >>> assert len(sub2.anns) == 2
            >>> assert len(sub3.anns) == 0
            >>> assert rejoined.basic_stats() == self.basic_stats()
        """
        new_dataset = ub.odict([(k, []) for k in self.dataset])
        new_dataset['categories'] = self.dataset['categories']
        new_dataset['info'] = self.dataset['info']
        new_dataset['licenses'] = self.dataset['licenses']

        sub_aids = sorted([aid for gid in sub_gids
                           for aid in self.gid_to_aids.get(gid, [])])
        new_dataset['annotations'] = list(ub.take(self.anns, sub_aids))
        new_dataset['images'] = list(ub.take(self.imgs, sub_gids))
        sub_dset = CocoDataset(new_dataset, img_root=self.img_root)
        return sub_dset

    def _ensure_imgsize(self):
        """
        Populate the imgsize field if it does not exist.

        Example:
            >>> dataset = demo_coco_data()
            >>> self = CocoDataset(dataset, tag='demo')
            >>> self._ensure_imgsize()
            >>> assert self.imgs[1]['width'] == 512
            >>> assert self.imgs[2]['width'] == 300
            >>> assert self.imgs[3]['width'] == 256
        """
        from PIL import Image
        for img in ub.ProgIter(list(self.imgs.values()), desc='ensure imgsize'):
            gpath = join(self.img_root, img['file_name'])
            if 'width' not in img:
                pil_img = Image.open(gpath)
                w, h = pil_img.size
                pil_img.close()
                img['width'] = w
                img['height'] = h

    def show_annotation(self, primary_aid=None, gid=None):
        """
        Use matplotlib to show an image with annotations overlaid
        """
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from PIL import Image

        if gid is None:
            primary_ann = self.anns[primary_aid]
            gid = primary_ann['image_id']

        img = self.imgs[gid]
        aids = self.gid_to_aids.get(img['id'], [])

        # Collect annotation overlays
        segments = []
        keypoints = []
        rects = []
        texts = []
        for aid in aids:
            ann = self.anns[aid]
            # Note standard coco bbox is [x,y,width,height]
            if 'bbox' in ann:
                x1, y1 = ann['bbox'][0:2]
            elif 'line' in ann:
                x1, y1 = ann['line'][0:2]
            elif 'keypoints' in ann:
                kpts = np.array(ann['keypoints']).reshape(-1, 3)
                xys = kpts.T[0:2].T
                x1, y1 = xys.min(axis=0)

            catname = self.cats[ann['category_id']]['name']
            textkw = {
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
                'backgroundcolor': (0, 0, 0, .3),
                'color': 'white',
                'fontproperties': mpl.font_manager.FontProperties(
                    size=6, family='monospace'),
            }
            texts.append((x1, y1, catname, textkw))

            color = 'orange' if aid == primary_aid else 'blue'
            if 'bbox' in ann:
                [x, y, w, h] = ann['bbox']
                rect = mpl.patches.Rectangle((x, y), w, h, facecolor='none',
                                             edgecolor=color)
                rects.append(rect)
            if 'line' in ann:
                x1, y1, x2, y2 = ann['line']
                segments.append([(x1, y1), (x2, y2)])
            if 'keypoints' in ann:
                kpts = np.array(ann['keypoints']).reshape(-1, 3)
                xys = kpts.T[0:2].T
                keypoints.append(xys)

        # Show image
        gpath = join(self.img_root, img['file_name'])
        with Image.open(gpath) as pil_img:
            np_img = np.array(pil_img)
        plt.imshow(np_img)
        ax = plt.gca()

        # Show all annotations inside it
        for (x1, y1, catname, textkw) in texts:
            ax.text(x1, y1, catname, **textkw)

        if segments:
            line_col = mpl.collections.LineCollection(segments, 2, color='b')
            ax.add_collection(line_col)

        rect_col = mpl.collections.PatchCollection(rects, match_original=True)
        ax.add_collection(rect_col)
        if keypoints:
            xs, ys = np.vstack(keypoints).T
            ax.plot(xs, ys, 'bo')

    def category_annotation_frequency(self):
        """
        Reports the number of annotations of each category

        Example:
            >>> dataset = demo_coco_data()
            >>> self = CocoDataset(dataset, tag='demo')
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.repr2(hist))
            {
                'astronomer': 1,
                'rocket': 1,
                'astronaut': 1,
                'helmet': 1,
                'mouth': 2,
                'star': 5,
            }
        """
        catname_to_nannots = ub.map_keys(lambda x: self.cats[x]['name'],
                                         ub.map_vals(len, self.cid_to_aids))
        catname_to_nannots = ub.odict(sorted(catname_to_nannots.items(),
                                             key=lambda kv: kv[1]))
        return catname_to_nannots

    def basic_stats(self):
        """
        Reports number of images, annotations, and categories.

        Example:
            >>> dataset = demo_coco_data()
            >>> self = CocoDataset(dataset, tag='demo')
            >>> print(ub.repr2(self.basic_stats()))
            {
                'n_anns': 11,
                'n_imgs': 3,
                'n_cats': 6,
            }
        """
        return ub.odict([
            ('n_anns', len(self.dataset['annotations'])),
            ('n_imgs', len(self.dataset['images'])),
            ('n_cats', len(self.dataset['categories'])),
        ])

    # --- The following functions were only defined for debug purposes ---

    def _run_fixes(self):
        """
        Fixes issues in conversion scripts. Published (non-phase0) data should
        not have these issues.
        """
        for ann in self.dataset['annotations']:

            if 'roi_category' in ann:
                cid = ann['roi_category']
                if 'category_id' in ann:
                    assert ann['category_id'] == cid
                ann['category_id'] = cid

            if 'roi_shape' not in ann:
                isect = set(ann).intersection({'bbox', 'keypoints', 'line'})
                if isect == {'bbox'}:
                    ann['roi_shape'] = 'bbox'
                elif isect == {'keypoints'}:
                    ann['roi_shape'] = 'keypoints'
                elif isect == {'line'}:
                    ann['roi_shape'] = 'line'

            if ann['roi_shape'] == 'circle':
                # We should simply remove this annotation
                pass

                # # the circle format is simply a line that defines the radius
                # x1, y1, x2, y2 = ann['bbox']
                # xc = (x1 + x2) / 2
                # yc = (y1 + y2) / 2
                # radius = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # length = radius * 2
                # bbox = [(xc - radius), (yc - radius), length, length]
                # ann['bbox'] = bbox
                # ann['radius'] = [x1, y1, x2, y2]
                # ann['roi_shape'] = 'bbox'

            if ann['roi_shape'] == 'point' and 'keypoints' not in ann:
                x, y, w, h = ann.pop('bbox')
                ann.pop('area', None)
                assert w == 0 and h == 0
                ann['keypoints'] = [x, y, 1]
                ann['roi_shape'] = 'keypoints'

            if ann['roi_shape'] == 'boundingBox':
                # standard coco bbox is [x,y,width,height]
                x1, y1, x2, y2 = ann['bbox']
                assert x2 >= x1
                assert y2 >= y1
                w = x2 - x1
                h = y2 - y1
                ann['bbox'] = [x1, y1, w, h]
                ann['roi_shape'] = 'bbox'

            if ann['roi_shape'] == 'line' and 'line' not in ann:
                # hack in a decent bounding box to fix the roi.
                # Assume the line is the diameter of an enscribed circle
                x1, y1, x2, y2 = ann['bbox']
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                bbox = [(xc - length / 2), (yc - length / 2), length, length]
                ann['bbox'] = bbox
                ann['line'] = [x1, y1, x2, y2]

            if 'line' in ann and 'bbox' not in ann:
                x1, y1, x2, y2 = ann['line']
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                bbox = [(xc - length / 2), (yc - length / 2), length, length]
                ann['bbox'] = bbox
                ann['roi_shape'] = 'line'

            if 'roi_shape' in ann:
                assert ann['roi_shape'] in ['keypoints', 'line', 'bbox'], (
                    ub.repr2(ann))

            if 'keypoints' in ann:
                assert len(ann['keypoints']) % 3 == 0

            # to make detectron happy
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                ann['area'] = w * h

            ann['segmentation'] = []
            ann['iscrowd'] = 0

    def lookup_imgs(self, filename=None):
        """
        Linear search for an images with specific attributes

        Ignore:
            filename = '201503.20150525.101841191.573975.png'
            list(self.lookup_imgs(filename))
            gid = 64940
            img = self.imgs[gid]
            img['file_name'] = filename
        """
        for img in self.imgs.values():
            if filename is not None:
                fpath = img['file_name']
                fname = basename(fpath)
                fname_noext = splitext(fname)[0]
                if filename in [fpath, fname, fname_noext]:
                    print('img = {!r}'.format(img))
                    yield img

    def lookup_anns(self, has=None):
        """
        Linear search for an annotations with specific attributes

        Ignore:
            list(self.lookup_anns(has='radius'))
            gid = 112888
            img = self.imgs[gid]
            img['file_name'] = filename
        """
        for ann in self.anns.values():
            if has is not None:
                if hasattr(ann, has):
                    print('ann = {!r}'.format(ann))
                    yield ann

    def _remove_keypoint_annotations(self):
        to_remove = []
        for ann in self.dataset['annotations']:
            if ann['roi_shape'] == 'keypoints':
                to_remove.append(ann)
        print('Removing {} keypoint annotations'.format(len(to_remove)))
        for ann in to_remove:
            self.dataset['annotations'].remove(ann)
        self._build_index()

    def _remove_bad_annotations(self):
        to_remove = []
        for ann in self.dataset['annotations']:
            if ann['image_id'] is None or ann['category_id'] is None:
                to_remove.append(ann)
        print('Removing {} bad annotations'.format(len(to_remove)))
        for ann in to_remove:
            self.dataset['annotations'].remove(ann)
        self._build_index()

    def _remove_radius_annotations(self):
        to_remove = []
        for ann in self.dataset['annotations']:
            if 'radius' in ann:
                to_remove.append(ann)
        print('Removing {} radius annotations'.format(len(to_remove)))
        for ann in to_remove:
            self.dataset['annotations'].remove(ann)
        self._build_index()

    def _remove_empty_images(self):
        to_remove = []
        for gid in self.imgs.keys():
            aids = self.gid_to_aids.get(gid, [])
            if not aids:
                to_remove.append(self.imgs[gid])
        print('Removing {} empty images'.format(len(to_remove)))
        for img in to_remove:
            self.dataset['images'].remove(img)
        self._build_index()


def demo_coco_data():
    """
    Simple data for testing
    """
    gpath1 = ub.grabdata('https://i.imgur.com/KXhKM72.png')
    gpath2 = ub.grabdata('https://i.imgur.com/flTHWFD.png')
    gpath3 = ub.grabdata('https://i.imgur.com/kCi7C1r.png')
    dataset = {
        'categories': [
            {'id': 1, 'name': 'astronaut', 'supercategory': 'human'},
            {'id': 2, 'name': 'rocket', 'supercategory': 'object'},
            {'id': 3, 'name': 'helmet', 'supercategory': 'object'},
            {'id': 4, 'name': 'mouth', 'supercategory': 'human'},
            {'id': 5, 'name': 'star', 'supercategory': 'object'},
            {'id': 6, 'name': 'astronomer', 'supercategory': 'human'},
        ],
        'images': [
            {'id': 1, 'file_name': gpath1},
            {'id': 2, 'file_name': gpath2},
            {'id': 3, 'file_name': gpath3},
        ],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1,
             'bbox': [10, 10, 360, 490]},
            {'id': 2, 'image_id': 1, 'category_id': 2,
             'bbox': [350, 5, 130, 290]},
            {'id': 3, 'image_id': 1, 'category_id': 3,
             'line': [326, 369, 500, 500]},
            {'id': 4, 'image_id': 1, 'category_id': 4,
             'keypoints': [202, 139, 1, 215, 150, 1, 229, 150, 1, 244, 142, 1]},
            {'id': 5, 'image_id': 1, 'category_id': 5,
             'keypoints': [37, 65, 1]},
            {'id': 6, 'image_id': 1, 'category_id': 5,
             'keypoints': [37, 16, 1]},
            {'id': 7, 'image_id': 1, 'category_id': 5,
             'keypoints': [3, 9, 1]},
            {'id': 8, 'image_id': 1, 'category_id': 5,
             'keypoints': [2, 111, 1]},
            {'id': 9, 'image_id': 1, 'category_id': 5,
             'keypoints': [2, 60, 1]},
            {'id': 10, 'image_id': 2, 'category_id': 6,
             'bbox': [37, 6, 230, 240]},
            {'id': 11, 'image_id': 2, 'category_id': 4,
             'bbox': [124, 96, 45, 18]}
        ],
        'licenses': [],
        'info': [],
    }
    return dataset


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m coco_wrangler.coco_api
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
