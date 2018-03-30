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
from clab.util import profiler  # NOQA
import psutil
import torch
import cv2
import ubelt as ub
import numpy as np
import imgaug as ia
import pandas as pd
import imgaug.augmenters as iaa
from clab.models.yolo2.utils import yolo_utils as yolo_utils
from clab.models.yolo2 import multiscale_batch_sampler
from clab import hyperparams
from clab import xpu_device
from clab import fit_harness
from clab import monitor
from clab import nninit
from clab.data import voc
from clab.lr_scheduler import ListedLR
from clab.models.yolo2 import darknet
from clab.models.yolo2 import darknet_loss
import torch.utils.data as torch_data


class DataConfig(object):
    def __init__(cfg):
        pass

    @classmethod
    def phase0(DataConfig):
        # import viame_wrangler
        # other = viame_wrangler.config.WrangleConfig()

        # cfg = DataConfig()
        # cfg.workdir = other.workdir
        # cfg.img_root = other.img_root
        # cfg.train_fpath = join(other.challenge_work_dir, 'phase0-coarse-bbox-only-train.mscoco.json')
        # cfg.vali_fapth = join(other.challenge_work_dir, 'phase0-coarse-bbox-only-val.mscoco.json')
        # return cfg
        cfg = DataConfig()
        cfg.workdir = ub.truepath(ub.argval('--work', default='~/work/viame-challenge-2018'))
        cfg.img_root = ub.truepath(ub.argval('--img_root', default='~/data/viame-challenge-2018/phase0-imagery'))
        cfg.train_fpath = join(cfg.workdir, 'train.mscoco.json')
        cfg.vali_fapth = join(cfg.workdir, 'vali.mscoco.json')
        return cfg

    @classmethod
    def phase1(DataConfig):
        # import viame_wrangler
        # other = viame_wrangler.config.WrangleConfig()
        # cfg.workdir = other.workdir
        # cfg.img_root = other.img_root

        cfg = DataConfig()
        cfg.workdir = ub.truepath(ub.argval('--work', default='~/work/viame-challenge-2018'))
        cfg.img_root = ub.truepath(ub.argval('--img_root', default='~/data/viame-challenge-2018/phase1-imagery'))
        cfg.train_fpath = join(cfg.workdir, 'train.mscoco.json')
        cfg.vali_fapth = join(cfg.workdir, 'vali.mscoco.json')
        return cfg


class TorchCocoDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Example:
        >>> self = TorchCocoDataset()
        >>> index = 139

    Ignore:
        for index in ub.ProgIter(range(len(self))):
            self._load_annotation(index)

        self.check_images_exist()

        self = TorchCocoDataset()

        # hack
        img = np.random.rand(1000, 1000, 3)
        import types
        types.MethodType
        def _load_image(self, index):
            return img

        inject_method(self, _load_image)

        for index in ub.ProgIter(range(len(self)), freq=1, adjust=0):
            try:
                self[index]
                # self._load_item(index, (10, 10))
            except IOError as ex:
                print('index = {!r}'.format(index))
                print('ex = {!r}\n'.format(ex))

        img = self.dset.dataset['images'][index]

    """
    def __init__(self, coco_fpath=None, img_root=None):

        if coco_fpath is None and img_root is None:
            cfg = DataConfig.phase1()
            coco_fpath = cfg.train_fpath
            img_root = cfg.img_root

        from coco_wrangler import CocoDataset
        self.coco_fpath = coco_fpath
        self.dset = CocoDataset(coco_fpath, img_root=img_root)

        self.label_names = sorted(self.dset.name_to_cat,
                                  key=lambda n: self.dset.name_to_cat[n]['id'])
        self._class_to_ind = ub.invert_dict(dict(enumerate(self.label_names)))
        self.base_size = np.array([416, 416])

        self.num_images = len(self.dset.imgs)

        self.num_classes = len(self.label_names)
        self.input_id = os.path.basename(self.coco_fpath)

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

    def check_images_exist(self):
        """
        Example:
            >>> cfg = DataConfig.phase1()
            >>> self = YoloCocoDataset(cfg.train_fpath, cfg.img_root)
            >>> self.check_images_exist()
            >>> self = YoloCocoDataset(cfg.vali_fapth, cfg.img_root)
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

    def make_loader(self, *args, **kwargs):
        """
        We need to do special collation to deal with different numbers of
        bboxes per item.

        Args:
            batch_size (int, optional):
            shuffle (bool, optional):
            sampler (Sampler, optional):
            batch_sampler (Sampler, optional):
            num_workers (int, optional):
            pin_memory (bool, optional):
            drop_last (bool, optional):
            timeout (numeric, optional):
            worker_init_fn (callable, optional):

        References:
            https://github.com/pytorch/pytorch/issues/1512

        Example:
            >>> self = TorchCocoDataset()
            >>> #inbatch = [self[i] for i in range(10)]
            >>> loader = self.make_loader(batch_size=10)
            >>> batch = next(iter(loader))
            >>> images, labels = batch
            >>> assert len(images) == 10
            >>> assert len(labels) == 2
            >>> assert len(labels[0]) == len(images)
        """
        # def custom_collate_fn(inbatch):
        #     # we know the order of data in __getitem__ so we can choose not to
        #     # stack the variable length bboxes and labels
        #     default_collate = torch_data.dataloader.default_collate
        #     inimgs, inlabels = list(map(list, zip(*inbatch)))
        #     imgs = default_collate(inimgs)

        #     # Just transpose the list if we cant collate the labels
        #     # However, try to collage each part.
        #     n_labels = len(inlabels[0])
        #     labels = [None] * n_labels
        #     for i in range(n_labels):
        #         simple = [x[i] for x in inlabels]
        #         if ub.allsame(map(len, simple)):
        #             labels[i] = default_collate(simple)
        #         else:
        #             labels[i] = simple

        #     batch = imgs, labels
        #     return batch
        # # kwargs['collate_fn'] = custom_collate_fn
        from clab.data import collate
        kwargs['collate_fn'] = collate.list_collate
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader

    def __len__(self):
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
            >>> self = TorchCocoDataset()
            >>> index = 100
            >>> chw, label = self[index]
            >>> hwc = chw.numpy().transpose(1, 2, 0)
            >>> boxes, class_idxs = label
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
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
            if ann['area'] == 0:
                continue
            box = np.array(ann['bbox']).copy()
            box[2:4] += box[0:2]
            cid = ann['category_id']
            cname = self.dset.cats[cid]['name']
            cind = self._class_to_ind[cname]
            gt_labels.append(cind)
            boxes.append(box)

        annot = {
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'gt_classes': np.array(gt_labels),
        }
        return annot


class YoloCocoDataset(TorchCocoDataset):
    """
    Extends CocoDataset localization dataset (which simply loads the images
    with minimal processing) for multiscale training.

    Example:
        >>> assert len(YoloCocoDataset(split='train')) == 2501
        >>> assert len(YoloCocoDataset(split='test')) == 4952
        >>> assert len(YoloCocoDataset(split='val')) == 2510

    Example:
        >>> self = CocoDataset()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)
    """

    def __init__(self, *args, **kw):
        super(YoloCocoDataset, self).__init__(*args, **kw)

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

        if 'train' in self.input_id:
            augmentors = [
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Affine(
                    scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-15, 15),
                    shear=(-7, 7),
                    order=[0, 1, 3],
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode=ia.ALL,
                    # Note: currently requires imgaug master version
                    backend='cv2',
                ),
                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
            ]
            self.augmenter = iaa.Sequential(augmentors)

    # @profiler.profile
    def __getitem__(self, index):
        """
        CommandLine:
            python ~/code/baseline-viame-2018/yolo_viame.py TorchCocoDataset.__getitem__ --show

        Example:
            >>> self = YoloCocoDataset()
            >>> index = 831
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> boxes, gt_classes, orig_size, index, gt_weights = label
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.numpy(), box_format='tlbr')
        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, size_index = index
            inp_size = self.multi_scale_inp_size[size_index]
        else:
            inp_size = self.base_size

        # load the raw data
        # hwc255, boxes, gt_classes = self._load_item(index, inp_size)

        # load the raw data from VOC
        image = self._load_image(index)
        annot = self._load_annotation(index)

        # VOC loads annotations in tlbr, but yolo expects xywh
        tlbr = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        # Weight samples so we dont care about difficult cases
        gt_weights = np.ones(len(tlbr))

        # squish the bounding box and image into a standard size
        w, h = inp_size
        im_w, im_h = image.shape[0:2][::-1]
        sx = float(w) / im_w
        sy = float(h) / im_h
        tlbr[:, 0::2] *= sx
        tlbr[:, 1::2] *= sy
        interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
        hwc255 = cv2.resize(image, (w, h), interpolation=interpolation)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            try:

                hwc255 = seq_det.augment_image(hwc255)

                bbs = ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1, y1, x2, y2)
                     for x1, y1, x2, y2 in tlbr], shape=hwc255.shape)
                bbs = seq_det.augment_bounding_boxes([bbs])[0]

                tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                                  for bb in bbs.bounding_boxes])
                tlbr = yolo_utils.clip_boxes(tlbr, hwc255.shape[0:2])

            except Exception:
                print('\n\n!!!!!!!!!!!\n\n')
                print('tlbr = {}'.format(ub.repr2(tlbr, nl=1)))
                print('ERROR index = {!r}'.format(index))
                print('\n\n!!!!!!!!!!!\n\n')
                raise

        chw01 = torch.FloatTensor(hwc255.transpose(2, 0, 1) / 255)
        gt_classes = torch.LongTensor(gt_classes)

        # The original YOLO-v2 works in xywh, but this implementation seems to
        boxes = torch.FloatTensor(tlbr)

        # Return index information in the label as well
        orig_size = torch.LongTensor([im_w, im_h])
        index = torch.LongTensor([index])
        gt_weights = torch.FloatTensor(gt_weights)
        label = (boxes, gt_classes, orig_size, index, gt_weights)

        return chw01, label

    # @ub.memoize_method
    def _load_image(self, index):
        return super(YoloCocoDataset, self)._load_image(index)

    # @ub.memoize_method
    def _load_annotation(self, index):
        return super(YoloCocoDataset, self)._load_annotation(index)


def make_loaders(datasets, batch_size=16, workers=0):
    """
    Example:
        >>> torch.random.manual_seed(0)
        >>> datasets = {'train': YoloCocoDataset()}
        >>> loaders = make_loaders(datasets)
        >>> train_iter = iter(loaders['train'])
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
    loaders = {}
    for key, dset in datasets.items():
        assert len(dset) > 0, 'must have some data'
        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            dset, batch_size=batch_size, shuffle=(key == 'train')
        )
        loader = dset.make_loader(batch_sampler=batch_sampler,
                                  num_workers=workers)
        loader.batch_size = batch_size
        loaders[key] = loader
    return loaders


def ensure_ulimit():
    # NOTE: It is important to have a high enought ulimit for DataParallel
    try:
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if rlimit[0] <= 8192:
            resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
    except Exception:
        print('Unable to fix ulimit. Ensure manually')
        raise


def setup_harness():
    """
    CommandLine:
        python ~/code/baseline-viame-2018/yolo_viame.py setup_harness
        python ~/code/baseline-viame-2018/yolo_viame.py setup_harness --profile

    Example:
        >>> harn = setup_harness()
        >>> harn.initialize()
        >>> harn.dry = True
        >>> harn.run()
    """
    if int(ub.argval('--phase', default=1)) == 1:
        cfg = DataConfig.phase1()
    else:
        assert False
        cfg = DataConfig.phase0()

    workdir = cfg.workdir
    datasets = {
        'train': YoloCocoDataset(cfg.train_fpath, cfg.img_root),
        'vali': YoloCocoDataset(cfg.vali_fapth, cfg.img_root),
    }

    datasets['train'].check_images_exist()
    datasets['vali'].check_images_exist()

    n_cpus = psutil.cpu_count(logical=True)
    workers = int(n_cpus / 2)

    nice = ub.argval('--nice', default=None)

    pretrained_fpath = darknet.initial_weights()

    # NOTE: XPU implicitly supports DataParallel just pass --gpu=0,1,2,3
    xpu = xpu_device.XPU.cast('argv')

    ensure_ulimit()

    postproc_params = dict(
        conf_thresh=0.001,
        nms_thresh=0.5,
        ovthresh=0.5,
    )

    max_epoch = 160

    lr_step_points = {
        # warmup learning rate
        0:  0.0001,
        1:  0.0001,
        2:  0.0002,
        3:  0.0003,
        4:  0.0004,
        5:  0.0005,
        6:  0.0006,
        7:  0.0007,
        8:  0.0008,
        9:  0.0009,
        10: 0.0010,
        # cooldown learning rate
        60: 0.0001,
        90: 0.00001,
    }

    batch_size = int(ub.argval('--batch_size', default=16))
    workers = int(ub.argval('--workers',
                            default=int(psutil.cpu_count(logical=True) / 2)))

    loaders = make_loaders(datasets, batch_size=batch_size,
                           workers=workers if workers is not None else workers)

    hyper = hyperparams.HyperParams(

        model=(darknet.Darknet19, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors
        }),

        criterion=(darknet_loss.DarknetLoss, {
            'anchors': datasets['train'].anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,
            'class_scale': 1.0,
            'coord_scale': 1.0,
            'iou_thresh': 0.6,
            'reproduce_longcw': ub.argflag('--longcw'),
            'denom': ub.argval('--denom', default='num_boxes'),
        }),

        optimizer=(torch.optim.SGD, dict(
            lr=lr_step_points[0],
            momentum=0.9,
            weight_decay=0.0005
        )),

        # initializer=(nninit.KaimingNormal, {}),
        initializer=(nninit.Pretrained, {
            'fpath': pretrained_fpath,
        }),

        scheduler=(ListedLR, dict(
            step_points=lr_step_points
        )),

        other=ub.dict_union({
            'nice': str(nice),
            'batch_size': loaders['train'].batch_sampler.batch_size,
        }, postproc_params),
        centering=None,

        # centering=datasets['train'].centering,
        augment=datasets['train'].augmenter,
    )

    harn = fit_harness.FitHarness(
        hyper=hyper, xpu=xpu, loaders=loaders, max_iter=max_epoch,
        workdir=workdir,
    )
    harn.postproc_params = postproc_params
    harn.nice = nice
    harn.monitor = monitor.Monitor(min_keys=['loss'],
                                   # max_keys=['global_acc', 'class_acc'],
                                   patience=max_epoch)

    @harn.set_batch_runner
    def batch_runner(harn, inputs, labels):
        """
        Custom function to compute the output of a batch and its loss.

        Example:
            >>> harn = setup_harness(workers=0)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> inputs, labels = batch
            >>> criterion = harn.criterion
            >>> weights_fpath = darknet.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['model_state_dict']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn._custom_run_batch(harn, inputs, labels)
        """
        outputs = harn.model(*inputs)

        # darknet criterion needs to know the input image shape
        inp_size = tuple(inputs[0].shape[-2:])

        aoff_pred, iou_pred, prob_pred = outputs
        gt_boxes, gt_classes, orig_size, indices, gt_weights = labels

        loss = harn.criterion(aoff_pred, iou_pred, prob_pred, gt_boxes,
                              gt_classes, gt_weights=gt_weights,
                              inp_size=inp_size, epoch=harn.epoch)
        return outputs, loss

    @harn.add_batch_metric_hook
    def custom_metrics(harn, output, labels):
        metrics_dict = ub.odict()
        criterion = harn.criterion
        metrics_dict['L_bbox'] = float(criterion.bbox_loss.data.cpu().numpy())
        metrics_dict['L_iou'] = float(criterion.iou_loss.data.cpu().numpy())
        metrics_dict['L_cls'] = float(criterion.cls_loss.data.cpu().numpy())
        return metrics_dict

    # Set as a harness attribute instead of using a closure
    harn.batch_confusions = []

    @harn.add_iter_callback
    def on_batch(harn, tag, loader, bx, inputs, labels, outputs, loss):
        """
        Custom hook to run on each batch (used to compute mAP on the fly)

        Example:
            >>> harn = setup_harness(workers=0)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> inputs, labels = batch
            >>> criterion = harn.criterion
            >>> loader = harn.loaders['train']
            >>> weights_fpath = darknet.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['model_state_dict']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn._custom_run_batch(harn, inputs, labels)
            >>> tag = 'train'
            >>> on_batch(harn, tag, loader, bx, inputs, labels, outputs, loss)
        """
        # Accumulate relevant outputs to measure
        gt_boxes, gt_classes, orig_size, indices, gt_weights = labels
        # aoff_pred, iou_pred, prob_pred = outputs
        im_sizes = orig_size
        inp_size = inputs[0].shape[-2:][::-1]

        conf_thresh = harn.postproc_params['conf_thresh']
        nms_thresh = harn.postproc_params['nms_thresh']
        ovthresh = harn.postproc_params['ovthresh']

        postout = harn.model.module.postprocess(outputs, inp_size, im_sizes,
                                                conf_thresh, nms_thresh)
        # batch_pred_boxes, batch_pred_scores, batch_pred_cls_inds = postout
        # Compute: y_pred, y_true, and y_score for this batch
        batch_pred_boxes, batch_pred_scores, batch_pred_cls_inds = postout
        batch_true_boxes, batch_true_cls_inds = labels[0:2]
        batch_orig_sz, batch_img_inds = labels[2:4]

        y_batch = []
        for bx, index in enumerate(batch_img_inds.data.cpu().numpy().ravel()):
            pred_boxes  = batch_pred_boxes[bx]
            pred_scores = batch_pred_scores[bx]
            pred_cxs    = batch_pred_cls_inds[bx]

            # Group groundtruth boxes by class
            true_boxes_ = batch_true_boxes[bx].data.cpu().numpy()
            true_cxs = batch_true_cls_inds[bx].data.cpu().numpy()
            true_weights = gt_weights[bx].data.cpu().numpy()

            # Unnormalize the true bboxes back to orig coords
            orig_size = batch_orig_sz[bx]
            sx, sy = np.array(orig_size) / np.array(inp_size)
            if len(true_boxes_):
                true_boxes = np.hstack([true_boxes_, true_weights[:, None]])
                true_boxes[:, 0:4:2] *= sx
                true_boxes[:, 1:4:2] *= sy

            y = voc.EvaluateVOC.image_confusions(true_boxes, true_cxs,
                                                 pred_boxes, pred_scores,
                                                 pred_cxs, ovthresh=ovthresh)
            y['gx'] = index
            y_batch.append(y)

        harn.batch_confusions.extend(y_batch)

    @harn.add_epoch_callback
    def on_epoch(harn, tag, loader):
        y = pd.concat(harn.batch_confusions)
        num_classes = len(loader.dataset.label_names)

        mean_ap, ap_list = voc.EvaluateVOC.compute_map(y, num_classes)

        harn.log_value(tag + ' epoch mAP', mean_ap, harn.epoch)
        max_ap = np.nanmax(ap_list)
        harn.log_value(tag + ' epoch max-AP', max_ap, harn.epoch)
        harn.batch_confusions.clear()

    return harn


def train():
    """
    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase0_b16 --phase=0 --batch_size=16 --workers=2 --gpu=0
    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase1_b16 --phase=1 --batch_size=16 --workers=2 --gpu=0
    """
    harn = setup_harness()
    with harn.xpu:
        harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/clab/examples
        python ~/code/clab/examples/yolo_viame.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
