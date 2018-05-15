"""
Train a baseline model using a pytorch implementation of YOLO2

Ignore:
    rsync -avzP ~/data/./viame-challenge-2018/phase0-imagery hermes:data/viame-challenge-2018/
    rsync -avzP ~/data/viame-challenge-2018/phase0-imagery/./mbari_seq0 hermes:data/viame-challenge-2018/phase0-imagery
    rsync -avzP ~/data/viame-challenge-2018/phase0-imagery/./mouss_seq1 hermes:data/viame-challenge-2018/phase0-imagery

    rsync -avzP ~/data/viame-challenge-2018/phase0-imagery/./mouss_seq1 hermes:data/viame-challenge-2018/phase0-imagery

    tar -xvzf /data/jowens/noaa
    tar -xvzf /data/jowens/noaa/phase1-imagery.tar.gz -C /data/projects/noaa
    tar -xvzf /data/jowens/noaa/phase1-annotations.tar.gz -C /data/projects/noaa

"""
import os
from os.path import join
import glob
import torch
import cv2
import ubelt as ub
import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
import torch.utils.data as torch_data
import torch.utils.data.sampler as torch_sampler
import netharn as nh
from netharn.models.yolo2 import multiscale_batch_sampler
from netharn.models.yolo2 import light_region_loss
from netharn.models.yolo2 import light_yolo

from coco_wrangler import coco_api


class TorchCocoDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Example:
        >>> dset = coco_api.CocoDataset(coco_api.demo_coco_data())
        >>> self = TorchCocoDataset(dset)
    """
    def __init__(self, coco_dset):
        self.dset = coco_dset

        self.label_names = sorted(self.dset.name_to_cat,
                                  key=lambda n: self.dset.name_to_cat[n]['id'])
        self._class_to_ind = ub.invert_dict(dict(enumerate(self.label_names)))
        self.base_size = np.array([416, 416])

        self.num_images = len(self.dset.imgs)

        self.num_classes = len(self.label_names)
        try:
            self.input_id = ub.hash_data(self.dset.dataset)
        except TypeError:
            self.input_id = ub.hash_data(ub.repr2(self.dset.dataset, nl=0))
        # self.input_id = os.path.basename(self.coco_fpath)

        if False:
            # setup heirarchy
            import networkx as nx
            g = nx.DiGraph()
            for cat in self.dset.cats.values():
                g.add_node(cat['name'])
                if 'supercategory' in cat:
                    g.add_edge(cat['supercategory'], cat['name'])
            for key, val in g.adj.items():
                print('node = {!r}'.format(key))
                print('    * neighbs = {!r}'.format(list(val)))

    def _training_sample_weights(self):
        """
        Assigns weighting to each image to includence sample probability.

        We want to see very frequent categories less often,
        but we also don't really care about the rarest classes to the point
        where we should smaple them more than uncommon classes.  We also don't
        want to sample images without any or with too many annotations very
        often.
        """
        index_to_gid = [img['id'] for img in self.dset.dataset['images']]
        index_to_aids = list(ub.take(self.dset.gid_to_aids, index_to_gid))
        index_to_cids = [[self.dset.anns[aid]['category_id'] for aid in aids]
                         for aids in index_to_aids]

        catname_to_cid = {
            cat['name']: cid
            for cid, cat in self.dset.cats.items()}

        # median frequency weighting with minimum threshold
        min_examples = 20
        cat_freq = pd.Series(self.dset.category_annotation_frequency())

        valid_freq = cat_freq[cat_freq > min_examples]
        normal_mfw = valid_freq.median() / valid_freq

        # Draw anything under the threshold with probability equal to the median
        too_few = cat_freq[(cat_freq <= min_examples) & (cat_freq > 0)]
        too_few[:] = 1.0
        category_mfw = pd.concat([normal_mfw, too_few])

        cid_to_mfw = category_mfw.rename(catname_to_cid)

        cid_to_mfw_dict = cid_to_mfw.to_dict()

        index_to_weights = [list(ub.take(cid_to_mfw_dict, cids)) for cids in index_to_cids]
        index_to_nannots = np.array(list(map(len, index_to_weights)))

        # Each image becomes represented by the category with maximum median
        # frequency weight. This allows us to assign each image a proxy class
        # We make another proxy class to represent images without anything in
        # them.
        EMPTY_PROXY_CID = -1
        index_to_proxyid = [
            # cid_to_mfw.loc[cids].idxmax()
            ub.argmax(ub.dict_subset(cid_to_mfw_dict, cids))
            if len(cids) else EMPTY_PROXY_CID
            for cids in index_to_cids
        ]

        proxy_freq = pd.Series(ub.dict_hist(index_to_proxyid))
        proxy_root_mfw = proxy_freq.median() / proxy_freq
        power = 0.878
        proxy_root_mfw = proxy_root_mfw ** power
        # We now have a weight for each item in out dataset
        index_to_weight = np.array(list(ub.take(proxy_root_mfw.to_dict(), index_to_proxyid)))

        if False:
            # Figure out how the likelihoods of each class change
            xy = {}
            for power in [0, .5, .878, 1]:
                proxy_root_mfw = proxy_freq.median() / proxy_freq
                # dont let weights get too high
                # proxy_root_mfw = np.sqrt(proxy_root_mfw)
                # power = .88
                proxy_root_mfw = proxy_root_mfw ** power
                # proxy_root_mfw = np.clip(proxy_root_mfw, a_min=None, a_max=3)

                index_to_weight = list(ub.take(proxy_root_mfw.to_dict(), index_to_proxyid))

                if 1:
                    # what is the probability we draw an empty image?
                    df = pd.DataFrame({
                        'nannots': index_to_nannots,
                        'weight': index_to_weight,
                    })
                    df['prob'] = df.weight / df.weight.sum()

                    prob_empty = df.prob[df.nannots == 0].sum()

                    probs = {'empty': prob_empty}
                    for cid in cid_to_mfw.index:
                        flags = [cid in cids for cids in index_to_cids]
                        catname = self.dset.cats[cid]['name']
                        p = df[flags].prob.sum()
                        probs[catname] = p
                    xy['p{}'.format(power)] = pd.Series(probs)
            xy['freq'] = {}
            for cid in cid_to_mfw.index:
                catname = self.dset.cats[cid]['name']
                xy['freq'][catname] = proxy_freq[cid]
            print(pd.DataFrame(xy))

        # index_to_prob = index_to_weight / index_to_weight.sum()
        return index_to_weight

    def check_images_exist(self):
        """
        Example:
            >>> coco_dsets = load_coco_datasets()
            >>> self = TorchCocoDataset(coco_dsets['train'])
            >>> self.check_images_exist()
            >>> self = TorchCocoDataset(coco_dsets['vali'])
            >>> self.check_images_exist()
        """
        bad_paths = []
        for index in ub.ProgIter(range(len(self))):
            img = self.dset.dataset['images'][index]
            gpath = join(self.dset.img_root, img['file_name'])
            if not os.path.exists(gpath):
                bad_paths.append((index, gpath))
        if bad_paths:
            print(ub.repr2(bad_paths, nl=1))
            raise AssertionError('missing images')

    def __nice__(self):
        return '{} {}'.format(self.input_id, len(self))

    def __len__(self):
        # return 800
        limit = ub.argval('--limit', default=None)
        if limit:
            return int(limit)
        return self.num_images

    def __getitem__(self, index):
        """
        Returns:
            image, (bbox, class_idxs)

            bbox and class_idxs are variable-length
            bbox is in x1,y1,x2,y2 (i.e. tlbr) format

        CommandLine:
            python ~/code/baseline-viame-2018/yolo_viame.py TorchCocoDataset.__getitem__ --show

        Example:
            >>> dset = coco_api.CocoDataset(coco_api.demo_coco_data())
            >>> self = TorchCocoDataset(dset)
            >>> index = 0
            >>> chw, label = self[index]
            >>> hwc = chw.numpy().transpose(1, 2, 0)
            >>> boxes, class_idxs = label
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> mplutil.imshow(hwc, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.numpy(), box_format='tlbr')
            >>> mplutil.show_if_requested()
        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, inp_size = index
        else:
            inp_size = self.base_size
        hwc255, boxes, gt_classes = self._load_item(index, inp_size)

        chw = torch.FloatTensor(hwc255.transpose(2, 0, 1))
        gt_classes = torch.LongTensor(gt_classes)
        boxes = torch.LongTensor(boxes.astype(np.int32))
        label = (boxes, gt_classes,)
        return chw, label

    def _load_item(self, index, inp_size):
        imrgb_255 = self._load_image(index)
        annot = self._load_annotation(index)

        boxes = annot['boxes']
        gt_classes = annot['gt_classes']

        # squish the bounding box and image into a standard size
        # THIS CODE IS UNUSED ANYWAY
        # DEPRICATE: DONT SQUISH, USE LETTERBOX PADDING
        w, h = inp_size
        sx = float(w) / imrgb_255.shape[1]
        sy = float(h) / imrgb_255.shape[0]
        boxes[:, 0::2] *= sx
        boxes[:, 1::2] *= sy
        interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
        hwc255 = cv2.resize(imrgb_255, (w, h), interpolation=interpolation)
        return hwc255, boxes, gt_classes

    def _load_image(self, index):
        img = self.dset.dataset['images'][index]
        gpath = join(self.dset.img_root, img['file_name'])
        imbgr = cv2.imread(gpath)
        if imbgr is None:
            if not os.path.exists(gpath):
                raise IOError('Image path {} does not exist!'.format(gpath))
            else:
                raise IOError('Error reading image path {}'.format(gpath))

        if 'habcam' in gpath:
            # HACK: habcam images are stereo and we only have annots for the
            # left side. Crop off the left side
            # TODO: the left and right sides should be aligned, so
            # we could actually make use of this data
            if imbgr.shape[1] > 2000:
                imbgr = imbgr[:, 0:imbgr.shape[1] // 2, :]
            else:
                print('imbgr.shape = {!r}'.format(imbgr.shape))

        imrgb_255 = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)

        return imrgb_255

    def _load_annotation(self, index):
        img = self.dset.dataset['images'][index]
        gid = img['id']
        aids = self.dset.gid_to_aids.get(gid, [])
        boxes = []
        gt_labels = []
        for aid in aids:
            ann = self.dset.anns[aid]
            # HACK
            # if 'bbox' not in ann:
            #     continue
            xywh = nh.util.Boxes(np.array(ann['bbox']), 'xywh')
            if xywh.area[0] == 0:
                continue
            tlbr = xywh.toformat('tlbr')
            cid = ann['category_id']
            cname = self.dset.cats[cid]['name']
            cind = self._class_to_ind[cname]
            gt_labels.append(cind)
            boxes.append(tlbr.data)

        annot = {
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'gt_classes': np.array(gt_labels),
            'gid': gid,
            'aids': aids,
        }
        return annot


class YoloCocoDataset(TorchCocoDataset):
    """
    Extends CocoDataset localization dataset (which simply loads the images
    with minimal processing) for multiscale training.
    """

    def __init__(self, coco_dset, train=False):
        super().__init__(coco_dset)

        self.factor = 32  # downsample factor of yolo grid
        self.base_wh = np.array([416, 416], dtype=np.int)
        assert np.all(self.base_wh % self.factor == 0)

        # self.multi_scale_inp_size = np.array([
        #     self.base_wh + (self.factor * i) for i in range(-3, 7)])
        self.multi_scale_inp_size = np.array([
            self.base_wh + (self.factor * i) for i in range(-3, 6)])
        self.multi_scale_out_size = self.multi_scale_inp_size // self.factor

        self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                                   (6.63, 11.38), (9.42, 5.11),
                                   (16.62, 10.52)],
                                  dtype=np.float)
        self.num_anchors = len(self.anchors)
        self.augmenter = None

        if train:
            import netharn.data.transforms  # NOQA
            from netharn.data.transforms import HSVShift
            augmentors = [
                HSVShift(hue=0.1, sat=1.5, val=1.5),
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5),
            ]
            self.augmenter = iaa.Sequential(augmentors)
        self.letterbox = nh.data.transforms.Resize(None, mode='letterbox')

    # @profiler.profile
    def __getitem__(self, index):
        """
        CommandLine:
            python ~/code/baseline-viame-2018/yolo_viame.py TorchCocoDataset.__getitem__ --show

        Example:
            >>> dset = coco_api.CocoDataset(coco_api.demo_coco_data())
            >>> self = YoloCocoDataset(dset, train=1)
            >>> index = 0
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> target, gt_weights, orig_size, index, bg_weight = label
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> inp_size = hwc01.shape[0:2][::-1]
            >>> boxes = nh.util.Boxes(np.atleast_2d(target.numpy())[:, -4:], 'cxywh').scale(inp_size)
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.toformat('xywh').data, 'xywh')

        Ignore:
            >>> harn = setup_harness()
            >>> self = harn.hyper.make_loaders()['train'].dataset
            >>> weights = self._training_sample_weights()
            >>> index = ub.argsort(weights)[-1000]
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> target, gt_weights, orig_size, index, bg_weight = label
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> inp_size = hwc01.shape[0:2][::-1]
            >>> boxes = nh.util.Boxes(nh.util.atleast_nd(target.numpy(), 2)[:, -4:], 'cxywh').scale(inp_size)
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.toformat('xywh').data, 'xywh')

            dset = self.dset

            annot = self._load_annotation(index)
            gid = annot['gid']

            gid = ub.argmax(ub.map_vals(len, dset.gid_to_aids))

            index = dset.dataset['images'].index(self.dset.imgs[6407])
            gid = dset.dataset['images'][index]['id']
            len(dset.gid_to_aids[gid])

            self.dset.gid_to_aids[gid]

        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, size_index = index
            if size_index is None:
                inp_size = self.base_wh
            else:
                inp_size = self.multi_scale_inp_size[size_index]
        else:
            inp_size = self.base_wh
        inp_size = np.array(inp_size)

        image, tlbr, gt_classes, gt_weights = self._load_item(index)
        orig_size = np.array(image.shape[0:2][::-1])
        bbs = nh.util.Boxes(tlbr, 'tlbr').to_imgaug(shape=image.shape)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            image = seq_det.augment_image(image)
            bbs = seq_det.augment_bounding_boxes([bbs])[0]

            # Clip any bounding boxes that went out of bounds
            h, w = image.shape[0:2]
            tlbr = nh.util.Boxes.from_imgaug(bbs)
            tlbr = tlbr.clip(0, 0, w - 1, h - 1, inplace=True)

            # Remove any boxes that are no longer visible or out of bounds
            flags = (tlbr.area > 0).ravel()
            tlbr = tlbr.compress(flags, inplace=True)
            gt_classes = gt_classes[flags]
            gt_weights = gt_weights[flags]

            bbs = tlbr.to_imgaug(shape=image.shape)

        # Apply letterbox resize transform to train and test
        self.letterbox.target_size = inp_size
        image = self.letterbox.augment_image(image)
        bbs = self.letterbox.augment_bounding_boxes([bbs])[0]
        tlbr_inp = nh.util.Boxes.from_imgaug(bbs)

        # Remove any boxes that are no longer visible or out of bounds
        flags = (tlbr_inp.area > 0).ravel()
        tlbr_inp = tlbr_inp.compress(flags, inplace=True)
        gt_classes = gt_classes[flags]
        gt_weights = gt_weights[flags]

        chw01 = torch.FloatTensor(image.transpose(2, 0, 1) / 255.0)

        # Lightnet YOLO accepts truth tensors in the format:
        # [class_id, center_x, center_y, w, h]
        # where coordinates are noramlized between 0 and 1
        cxywh_norm = tlbr_inp.toformat('cxywh').scale(1 / inp_size)
        _target_parts = [gt_classes[:, None], cxywh_norm.data]
        target = np.concatenate(_target_parts, axis=-1)
        target = torch.FloatTensor(target)

        # Return index information in the label as well
        orig_size = torch.LongTensor(orig_size)
        index = torch.LongTensor([index])
        # how much do we care about each annotation in this image?
        gt_weights = torch.FloatTensor(gt_weights)
        # how much do we care about the background in this image?
        bg_weight = torch.FloatTensor([1.0])
        label = (target, gt_weights, orig_size, index, bg_weight)

        return chw01, label

    def _load_item(self, index):
        # load the raw data from VOC
        image = self._load_image(index)
        annot = self._load_annotation(index)
        # VOC loads annotations in tlbr
        tlbr = annot['boxes'].astype(np.float)
        gt_classes = annot['gt_classes']
        # Weight samples so we dont care about difficult cases
        gt_weights = np.ones(len(tlbr))
        return image, tlbr, gt_classes, gt_weights

    def _load_image(self, index):
        return super()._load_image(index)

    def _load_annotation(self, index):
        return super()._load_annotation(index)

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=False):
        """
        Example:
            >>> torch.random.manual_seed(0)
            >>> dset = coco_api.CocoDataset(coco_api.demo_coco_data())
            >>> self = YoloCocoDataset(dset, train=1)
            >>> loader = self.make_loader(batch_size=1)
            >>> train_iter = iter(loader)
            >>> # training batches should have multiple shapes
            >>> shapes = set()
            >>> for batch in train_iter:
            >>>     shapes.add(batch[0].shape[-1])
            >>>     if len(shapes) > 1:
            >>>         break
            >>> assert len(shapes) > 1

            >>> vali_loader = iter(loaders['vali'])
            >>> vali_iter = iter(loaders['vali'])
            >>> # vali batches should have one shape
            >>> shapes = set()
            >>> for batch, _ in zip(vali_iter, [1, 2, 3, 4]):
            >>>     shapes.add(batch[0].shape[-1])
            >>> assert len(shapes) == 1
        """
        assert len(self) > 0, 'must have some data'
        if shuffle:
            if True:
                # If the data is not balanced we need to balance it
                index_to_weight = self._training_sample_weights()
                num_samples = len(self)
                index_to_weight = index_to_weight[:num_samples]
                sampler = torch_sampler.WeightedRandomSampler(index_to_weight,
                                                              num_samples,
                                                              replacement=True)
                sampler.data_source = self  # hack for use with multiscale
            else:
                sampler = torch_sampler.RandomSampler(self)
            resample_freq = 10
        else:
            sampler = torch_sampler.SequentialSampler(self)
            resample_freq = None

        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            sampler, batch_size=batch_size, resample_freq=resample_freq,
        )
        # torch.utils.data.sampler.WeightedRandomSampler
        loader = torch_data.DataLoader(self, batch_sampler=batch_sampler,
                                       collate_fn=nh.data.collate.padded_collate,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)
        if loader.batch_size != batch_size:
            try:
                loader.batch_size = batch_size
            except Exception:
                pass
        return loader


class YoloHarn(nh.FitHarn):
    def __init__(harn, **kw):
        super().__init__(**kw)
        harn.batch_confusions = []
        harn.aps = {}

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        batch_inputs, batch_labels = raw_batch

        inputs = harn.xpu.variable(batch_inputs)
        labels = [harn.xpu.variable(d) for d in batch_labels]

        batch = (inputs, labels)
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader

        Example:
            >>> harn = setup_harness(bsize=2)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
        """

        # Compute how many images have been seen before
        bsize = harn.loaders['train'].batch_sampler.batch_size
        nitems = len(harn.datasets['train'])
        bx = harn.bxs['train']
        n_seen = (bx * bsize) + (nitems * harn.epoch)

        inputs, labels = batch
        outputs = harn.model(inputs)
        # torch.cuda.synchronize()
        target, gt_weights, orig_sizes, indices, bg_weights = labels
        loss = harn.criterion(outputs, target, seen=n_seen)
        # torch.cuda.synchronize()
        return outputs, loss

    def on_batch(harn, batch, outputs, loss):
        """
        custom callback

        Example:
            >>> harn = setup_harness(bsize=1)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> #weights_fpath = light_yolo.demo_voc_weights()
            >>> #state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> #harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, loss)
            >>> # xdoc: +REQUIRES(--show)
            >>> postout = harn.model.module.postprocess(outputs)
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> harn.visualize_prediction(batch, outputs, postout, idx=0,
            >>>                           thresh=0.2)
            >>> mplutil.show_if_requested()
        """
        if harn.current_tag != 'train':
            # Dont worry about computing mAP on the training set for now
            inputs, labels = batch
            inp_size = np.array(inputs.shape[-2:][::-1])

            postout = harn.model.module.postprocess(outputs)

            for y in harn._measure_confusion(postout, labels, inp_size):
                harn.batch_confusions.append(y)

        metrics_dict = ub.odict()
        metrics_dict['L_bbox'] = float(harn.criterion.loss_coord)
        metrics_dict['L_iou'] = float(harn.criterion.loss_conf)
        metrics_dict['L_cls'] = float(harn.criterion.loss_cls)
        for k, v in metrics_dict.items():
            if not np.isfinite(v):
                raise ValueError('{}={} is not finite'.format(k, v))
        return metrics_dict

    def on_epoch(harn):
        """
        custom callback

        Example:
            >>> harn = setup_harness(bsize=4)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'vali')
            >>> #weights_fpath = light_yolo.demo_voc_weights()
            >>> #state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> #harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> # run a few batches
            >>> harn.on_batch(batch, outputs, loss)
            >>> harn.on_batch(batch, outputs, loss)
            >>> harn.on_batch(batch, outputs, loss)
            >>> # then finish the epoch
            >>> harn.on_epoch()
        """
        tag = harn.current_tag
        if tag == 'vali':
            harn._dump_chosen_validation_data()
        if harn.batch_confusions:
            y = pd.concat([pd.DataFrame(y) for y in harn.batch_confusions])
            # TODO: write out a few visualizations
            loader = harn.loaders[tag]
            num_classes = len(loader.dataset.label_names)
            labels = list(range(num_classes))
            aps = nh.metrics.ave_precisions(y, labels, use_07_metric=True)
            harn.aps[tag] = aps
            mean_ap = np.nanmean(aps['ap'])
            max_ap = np.nanmax(aps['ap'])
            harn.log_value(tag + ' epoch mAP', mean_ap, harn.epoch)
            harn.log_value(tag + ' epoch max-AP', max_ap, harn.epoch)
            harn.batch_confusions.clear()
            metrics_dict = ub.odict()
            metrics_dict['max-AP'] = max_ap
            metrics_dict['mAP'] = mean_ap
            return metrics_dict

    # Non-standard problem-specific custom methods

    def _measure_confusion(harn, postout, labels, inp_size, **kw):
        targets = labels[0]
        gt_weights = labels[1]
        # orig_sizes = labels[2]
        # indices = labels[3]
        bg_weights = labels[4]

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        bsize = len(labels[0])
        for bx in range(bsize):
            postitem = asnumpy(postout[bx])
            target = asnumpy(targets[bx]).reshape(-1, 5)
            true_cxywh   = target[:, 1:5]
            true_cxs     = target[:, 0]
            true_weight  = asnumpy(gt_weights[bx])

            # Remove padded truth
            flags = true_cxs != -1
            true_cxywh  = true_cxywh[flags]
            true_cxs    = true_cxs[flags]
            true_weight = true_weight[flags]

            # orig_size    = asnumpy(orig_sizes[bx])
            # gx           = int(asnumpy(indices[bx]))

            # how much do we care about the background in this image?
            bg_weight = float(asnumpy(bg_weights[bx]))

            # Unpack postprocessed predictions
            sboxes = postitem.reshape(-1, 6)
            pred_cxywh = sboxes[:, 0:4]
            pred_scores = sboxes[:, 4]
            pred_cxs = sboxes[:, 5].astype(np.int)

            true_tlbr = nh.util.Boxes(true_cxywh, 'cxywh').to_tlbr()
            pred_tlbr = nh.util.Boxes(pred_cxywh, 'cxywh').to_tlbr()

            true_tlbr = true_tlbr.scale(inp_size)
            pred_tlbr = pred_tlbr.scale(inp_size)

            # TODO: can we invert the letterbox transform here and clip for
            # some extra mAP?
            true_boxes = true_tlbr.data
            pred_boxes = pred_tlbr.data

            y = nh.metrics.detection_confusions(
                true_boxes=true_boxes,
                true_cxs=true_cxs,
                true_weights=true_weight,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_cxs=pred_cxs,
                bg_weight=bg_weight,
                bg_cls=-1,
                ovthresh=harn.hyper.other['ovthresh'],
                **kw
            )
            # y['gx'] = gx
            yield y

    def _postout_to_coco(harn, postout, labels, inp_size):
        """
        -[ ] TODO: dump predictions for the test set to disk and score using
             someone elses code.
        """
        targets = labels[0]
        gt_weights = labels[1]
        orig_sizes = labels[2]
        indices = labels[3]
        # bg_weights = labels[4]

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        def undo_letterbox(cxywh):
            boxes = nh.util.Boxes(cxywh, 'cxywh')
            letterbox = harn.datasets['train'].letterbox
            return letterbox._boxes_letterbox_invert(boxes, orig_size, inp_size)

        predictions = []
        truth = []

        bsize = len(labels[0])
        for bx in range(bsize):
            postitem = asnumpy(postout[bx])
            target = asnumpy(targets[bx]).reshape(-1, 5)
            true_cxywh   = target[:, 1:5]
            true_cxs     = target[:, 0]
            true_weight  = asnumpy(gt_weights[bx])

            # Remove padded truth
            flags = true_cxs != -1
            true_cxywh  = true_cxywh[flags]
            true_cxs    = true_cxs[flags]
            true_weight = true_weight[flags]

            orig_size = asnumpy(orig_sizes[bx])
            gx        = int(asnumpy(indices[bx]))

            # how much do we care about the background in this image?
            # bg_weight = float(asnumpy(bg_weights[bx]))

            # Unpack postprocessed predictions
            sboxes = postitem.reshape(-1, 6)
            pred_cxywh = sboxes[:, 0:4]
            pred_scores = sboxes[:, 4]
            pred_cxs = sboxes[:, 5].astype(np.int)

            true_xywh = undo_letterbox(true_cxywh).toformat('xywh').data
            pred_xywh = undo_letterbox(pred_cxywh).toformat('xywh').data

            for xywh, cx, score in zip(pred_xywh, pred_cxs, pred_scores):
                pred = {
                    'image_id': gx,
                    'category_id': cx,
                    'bbox': list(xywh),
                    'score': score,
                }
                predictions.append(pred)

            for xywh, cx, weight in zip(true_xywh, true_cxs, gt_weights):
                true = {
                    'image_id': gx,
                    'category_id': cx,
                    'bbox': list(xywh),
                    'weight': weight,
                }
                truth.append(true)
        return predictions, truth

    def visualize_prediction(harn, batch, outputs, postout, idx=0,
                             thresh=None, orig_img=None):
        """
        Returns:
            np.ndarray: numpy image
        """
        # xdoc: +REQUIRES(--show)
        inputs, labels = batch
        targets, gt_weights, orig_sizes, indices, bg_weights = labels
        chw01 = inputs[idx]
        target = targets[idx]
        postitem = postout[idx]
        orig_size = orig_sizes[idx].cpu().numpy()
        # ---
        hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
        # TRUE
        true_cxs = target[:, 0].long()
        true_cxywh = target[:, 1:5]
        flags = true_cxs != -1
        true_cxywh = true_cxywh[flags]
        true_cxs = true_cxs[flags]
        # PRED
        pred_cxywh = postitem[:, 0:4]
        pred_scores = postitem[:, 4]
        pred_cxs = postitem[:, 5]

        if thresh is not None:
            flags = pred_scores > thresh
            pred_cxs = pred_cxs[flags]
            pred_cxywh = pred_cxywh[flags]
            pred_scores = pred_scores[flags]

        pred_clsnms = list(ub.take(harn.datasets['train'].label_names,
                                   pred_cxs.long().cpu().numpy()))
        pred_labels = ['{}@{:.2f}'.format(n, s)
                       for n, s in zip(pred_clsnms, pred_scores)]

        true_labels = list(ub.take(harn.datasets['train'].label_names,
                                   true_cxs.long().cpu().numpy()))
        # ---
        inp_size = np.array(hwc01.shape[0:2][::-1])
        target_size = inp_size

        true_boxes_ = nh.util.Boxes(true_cxywh.cpu().numpy(), 'cxywh').scale(inp_size)
        pred_boxes_ = nh.util.Boxes(pred_cxywh.cpu().numpy(), 'cxywh').scale(inp_size)

        letterbox = harn.datasets['train'].letterbox
        img = letterbox._img_letterbox_invert(hwc01, orig_size, target_size)
        img = np.clip(img, 0, 1)
        if orig_img is not None:
            # we are given the original image, to avoid artifacts from
            # inverting a downscale
            assert orig_img.shape == img.shape

        true_cxywh_ = letterbox._boxes_letterbox_invert(true_boxes_, orig_size, target_size)
        pred_cxywh_ = letterbox._boxes_letterbox_invert(pred_boxes_, orig_size, target_size)

        from netharn.util import mplutil
        shift, scale, embed_size = letterbox._letterbox_transform(orig_size, target_size)

        fig = mplutil.figure(doclf=True, fnum=1)
        mplutil.imshow(img, colorspace='rgb')
        mplutil.draw_boxes(true_cxywh_.data, color='green', box_format='cxywh', labels=true_labels)
        mplutil.draw_boxes(pred_cxywh_.data, color='blue', box_format='cxywh', labels=pred_labels)
        return fig

        # mplutil.show_if_requested()

    def _pick_dumpcats(harn):
        """
        Hack to pick several images from the validation set to monitor each
        epoch.
        """
        vali_dset = harn.loaders['vali'].dataset
        chosen_gids = set()
        for cid, gids in vali_dset.dset.cid_to_gids.items():
            for gid in gids:
                if gid not in chosen_gids:
                    chosen_gids.add(gid)
                    break
        for gid, aids in vali_dset.dset.gid_to_aids.items():
            if len(aids) == 0:
                chosen_gids.add(gid)
                break

        gid_to_index = {
            img['id']: index
            for index, img in enumerate(vali_dset.dset.dataset['images'])}

        chosen_indices = list(ub.take(gid_to_index, chosen_gids))
        harn.chosen_indices = sorted(chosen_indices)

    def _dump_chosen_validation_data(harn):
        """
        Dump a visualization of the validation images to disk
        """
        if not hasattr(harn, 'chosen_indices'):
            harn._pick_dumpcats()

        vali_dset = harn.loaders['vali'].dataset
        for indices in ub.chunks(harn.chosen_indices, 16):
            to_collate = [vali_dset[index] for index in indices]
            raw_batch = nh.data.collate.padded_collate(to_collate)
            batch = harn.prepare_batch(raw_batch)
            outputs, loss = harn.run_batch(batch)
            postout = harn.model.module.postprocess(outputs)

            for idx, index in enumerate(indices):
                orig_img = vali_dset._load_image(index)
                fig = harn.visualize_prediction(batch, outputs, postout, idx=idx,
                                                thresh=0.2, orig_img=orig_img)
                img = nh.util.mplutil.render_figure_to_image(fig)
                dump_dpath = ub.ensuredir((harn.train_dpath, 'dump'))
                dump_fname = 'pred_{:04d}_{:08d}.png'.format(index, harn.epoch)
                fpath = os.path.join(dump_dpath, dump_fname)
                nh.util.imwrite(fpath, img)

    def dump_batch_item(harn, batch, outputs, postout):
        fig = harn.visualize_prediction(batch, outputs, postout, idx=0,
                                        thresh=0.2)
        img = nh.util.mplutil.render_figure_to_image(fig)
        dump_dpath = ub.ensuredir((harn.train_dpath, 'dump'))
        dump_fname = 'pred_{:08d}.png'.format(harn.epoch)
        fpath = os.path.join(dump_dpath, dump_fname)
        nh.util.imwrite(fpath, img)

    def deploy(harn):
        """
        Experimental function that will deploy a standalone predictor
        """
        pass


def load_coco_datasets():
    import wrangle
    # annot_globstr = ub.truepath('~/data/viame-challenge-2018/phase0-annotations/*.json')
    # annot_globstr = ub.truepath('~/data/viame-challenge-2018/phase0-annotations/mbari_seq0.mscoco.json')
    # img_root = ub.truepath('~/data/viame-challenge-2018/phase0-imagery')

    # Contest training data on hermes
    annot_globstr = ub.truepath('~/data/noaa/training_data/annotations/*/*-coarse-bbox-only*.json')
    img_root = ub.truepath('~/data/noaa/training_data/imagery/')

    fpaths = sorted(glob.glob(annot_globstr))
    # Remove keypoints annotation data (hack)
    fpaths = [p for p in fpaths if not ('nwfsc' in p or 'afsc' in p)]

    cacher = ub.Cacher('coco_dsets', cfgstr=ub.hash_data(fpaths),
                       appname='viame')
    coco_dsets = cacher.tryload()
    if coco_dsets is None:
        print('Reading raw mscoco files')
        import os
        dsets = []
        for fpath in sorted(fpaths):
            print('reading fpath = {!r}'.format(fpath))
            dset = coco_api.CocoDataset(fpath, tag='', img_root=img_root)
            try:
                assert not dset.missing_images()
            except AssertionError:
                print('fixing image file names')
                hack = os.path.basename(fpath).split('-')[0].split('.')[0]
                dset = coco_api.CocoDataset(fpath, tag=hack, img_root=join(img_root, hack))
                assert not dset.missing_images(), ub.repr2(dset.missing_images()) + 'MISSING'
            print(ub.repr2(dset.basic_stats()))
            dsets.append(dset)

        print('Merging')
        merged = coco_api.CocoDataset.union(*dsets)
        merged.img_root = img_root

        # HACK: wont need to do this for the released challenge data
        # probably wont hurt though
        # if not REAL_RUN:
        #     merged._remove_keypoint_annotations()
        #     merged._run_fixes()

        train_dset, vali_dset = wrangle.make_train_vali(merged)

        coco_dsets = {
            'train': train_dset,
            'vali': vali_dset,
        }

        cacher.save(coco_dsets)

    return coco_dsets


def setup_harness(bsize=16, workers=0):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py setup_harness

    Example:
        >>> harn = setup_harness()
        >>> harn.initialize()
    """

    xpu = nh.XPU.cast('argv')

    nice = ub.argval('--nice', default='Yolo2Baseline')
    batch_size = int(ub.argval('--batch_size', default=bsize))
    bstep = int(ub.argval('--bstep', 1))
    workers = int(ub.argval('--workers', default=workers))
    decay = float(ub.argval('--decay', default=0.0005))
    lr = float(ub.argval('--lr', default=0.001))
    workdir = ub.argval('--workdir', default=ub.truepath('~/work/viame/yolo'))
    ovthresh = 0.5

    coco_dsets = load_coco_datasets()

    datasets = {
        'train': YoloCocoDataset(coco_dsets['train'], train=True),
        'vali': YoloCocoDataset(coco_dsets['vali']),
    }

    datasets['train'].check_images_exist()
    datasets['vali'].check_images_exist()
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=True)
        for key, dset in datasets.items()
    }

    # simulated_bsize = bstep * batch_size
    hyper = nh.HyperParams(**{
        'nice': nice,
        'workdir': workdir,
        'datasets': datasets,

        'xpu': xpu,

        # a single dict is applied to all datset loaders
        'loaders': loaders,

        'model': (light_yolo.Yolo, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors,
            'conf_thresh': 0.001,
            'nms_thresh': 0.5,
        }),

        'criterion': (light_region_loss.RegionLoss, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,
            'class_scale': 1.0,
            'coord_scale': 1.0,
            'thresh': 0.6,  # iou_thresh
        }),

        'initializer': (nh.initializers.Pretrained, {
            'fpath': light_yolo.initial_imagenet_weights(),
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': lr / 10,
            'momentum': 0.9,
            'weight_decay': decay,
        }),

        'scheduler': (nh.schedulers.ListedLR, {
            'points': {
                0:  lr / 10,
                1:  lr,
                59: lr * 1.1,
                60: lr / 10,
                90: lr / 100,
            },
            'interpolate': True
        }),

        'monitor': (nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['mAP'],
            'patience': 160,
            'max_epoch': 160,
        }),

        'augment': datasets['train'].augmenter,

        'dynamics': {
            # Controls how many batches to process before taking a step in the
            # gradient direction. Effectively simulates a batch_size that is
            # `bstep` times bigger.
            'batch_step': bstep,
        },

        'other': {
            # Other params are not used internally, so you are free to set any
            # extra params specific to your algorithm, and still have them
            # logged in the hyperparam structure. For YOLO this is `ovthresh`.
            'batch_size': batch_size,
            'nice': nice,
            'ovthresh': ovthresh,  # used in mAP computation
            'input_range': 'norm01',
        },
    })
    harn = YoloHarn(hyper=hyper)
    harn.config['use_tqdm'] = False
    harn.intervals['log_iter_train'] = 1
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None
    return harn


def train():
    """
    python ~/code/baseline-viame-2018/yolo_viame.py train --nice dummy --batch_size=4 --workers=0 --gpu=0

    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase0_b16 --phase=0 --batch_size=16 --workers=2 --gpu=0
    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase1_b16 --phase=1 --batch_size=16 --workers=2 --gpu=0

    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase1_b32 --phase=1 --batch_size=32 --workers=4 --gpu=2,3

    srun -c 4 -p community --gres=gpu:1 \
            python ~/code/baseline-viame-2018/yolo_viame.py train \
            --nice baseline1 --batch_size=16 --workers=4 --gpu=0

    srun -c 4 -p community --gres=gpu:1 \
            python ~/code/baseline-viame-2018/yolo_viame.py train \
            --nice dummy --batch_size=16 --limit=128 --workers=0 --gpu=0 --lr=0.000001

    """
    harn = setup_harness()
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/baseline-viame-2018/yolo_viame.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
