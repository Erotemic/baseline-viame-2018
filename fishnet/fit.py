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
import torch
import cv2
import ubelt as ub
import pandas as pd
import numpy as np
import netharn as nh
from netharn.models.yolo2 import light_region_loss
from netharn.models.yolo2 import light_yolo
from fishnet.data import load_coco_datasets, YoloCocoDataset


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
        labels = {k: harn.xpu.variable(d) for k, d in batch_labels.items()}

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
            >>> batch = harn._demo_batch(0, 'vali')
            >>> #weights_fpath = light_yolo.demo_voc_weights()
            >>> #state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> #harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
        """

        # Compute how many images have been seen before
        bsize = harn.loaders['train'].batch_sampler.batch_size
        nitems = len(harn.datasets['train'])
        bx = harn.bxs['train']
        n_seen = (bx * bsize) + (nitems * harn.epoch)

        inputs, labels = batch
        outputs = harn.model(inputs)
        target = labels['targets']
        gt_weights = labels['gt_weights']
        loss = harn.criterion(outputs, target, seen=n_seen,
                              gt_weights=gt_weights)
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

            try:
                postout = harn.model.module.postprocess(outputs)
            except Exception as ex:
                harn.error('\n\n\n')
                harn.error('ERROR: FAILED TO POSTPROCESS OUTPUTS')
                harn.error('DETAILS: {!r}'.format(ex))
                raise

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
        targets = labels['targets']
        gt_weights = labels['gt_weights']
        # orig_sizes = labels['orig_sizes']
        # indices = labels['indices']
        bg_weights = labels['bg_weights']

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        bsize = len(targets)
        for bx in range(bsize):
            postitem = asnumpy(postout[bx])
            target = asnumpy(targets[bx]).reshape(-1, 5)
            true_cxywh = target[:, 1:5]
            true_cxs = target[:, 0]
            true_weight = asnumpy(gt_weights[bx])

            # Remove padded truth
            flags = true_cxs != -1
            true_cxywh = true_cxywh[flags]
            true_cxs = true_cxs[flags]
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
        targets = labels['targets']
        gt_weights = labels['gt_weights']
        indices = labels['indices']
        orig_sizes = labels['orig_sizes']
        # bg_weights = labels['bg_weights']

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        def undo_letterbox(cxywh):
            boxes = nh.util.Boxes(cxywh, 'cxywh')
            letterbox = harn.datasets['train'].letterbox
            return letterbox._boxes_letterbox_invert(boxes, orig_size, inp_size)

        predictions = []
        truth = []

        bsize = len(targets)
        for bx in range(bsize):
            postitem = asnumpy(postout[bx])
            target = asnumpy(targets[bx]).reshape(-1, 5)
            true_cxywh = target[:, 1:5]
            true_cxs = target[:, 0]
            true_weight = asnumpy(gt_weights[bx])

            # Remove padded truth
            flags = true_cxs != -1
            true_cxywh = true_cxywh[flags]
            true_cxs = true_cxs[flags]
            true_weight = true_weight[flags]

            orig_size = asnumpy(orig_sizes[bx])
            gx = int(asnumpy(indices[bx]))

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
        from netharn.util import mplutil
        inputs, labels = batch

        targets = labels['targets']
        # gt_weights = labels['gt_weights']
        orig_sizes = labels['orig_sizes']
        # indices = labels['indices']
        # bg_weights = labels['bg_weights']

        chw01 = inputs[idx]
        target = targets[idx].cpu().numpy().reshape(-1, 5)
        postitem = postout[idx].cpu().numpy().reshape(-1, 6)
        orig_size = orig_sizes[idx].cpu().numpy()
        # ---
        hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
        # TRUE
        true_cxs = target[:, 0].astype(np.int)
        true_cxywh = target[:, 1:5]
        flags = true_cxs != -1
        true_cxywh = true_cxywh[flags]
        true_cxs = true_cxs[flags]
        # PRED
        pred_cxywh = postitem[:, 0:4]
        pred_scores = postitem[:, 4]
        pred_cxs = postitem[:, 5].astype(np.int)

        if thresh is not None:
            flags = pred_scores > thresh
            pred_cxs = pred_cxs[flags]
            pred_cxywh = pred_cxywh[flags]
            pred_scores = pred_scores[flags]

        label_names = harn.datasets['train'].label_names

        true_clsnms = list(ub.take(label_names, true_cxs))
        pred_clsnms = list(ub.take(label_names, pred_cxs))
        pred_labels = ['{}@{:.2f}'.format(n, s)
                       for n, s in zip(pred_clsnms, pred_scores)]
        # ---
        inp_size = np.array(hwc01.shape[0:2][::-1])
        target_size = inp_size

        true_boxes_ = nh.util.Boxes(true_cxywh, 'cxywh').scale(inp_size)
        pred_boxes_ = nh.util.Boxes(pred_cxywh, 'cxywh').scale(inp_size)

        letterbox = harn.datasets['train'].letterbox
        img = letterbox._img_letterbox_invert(hwc01, orig_size, target_size)
        img = np.clip(img, 0, 1)
        if orig_img is not None:
            # we are given the original image, to avoid artifacts from
            # inverting a downscale
            assert orig_img.shape == img.shape

        true_cxywh_ = letterbox._boxes_letterbox_invert(true_boxes_, orig_size, target_size)
        pred_cxywh_ = letterbox._boxes_letterbox_invert(pred_boxes_, orig_size, target_size)

        shift, scale, embed_size = letterbox._letterbox_transform(orig_size, target_size)

        fig = mplutil.figure(doclf=True, fnum=1)
        mplutil.imshow(img, colorspace='rgb')
        mplutil.draw_boxes(true_cxywh_.data, color='green', box_format='cxywh', labels=true_clsnms)
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
        harn.debug('DUMP CHOSEN INDICES')

        if not hasattr(harn, 'chosen_indices'):
            harn._pick_dumpcats()

        vali_dset = harn.loaders['vali'].dataset
        for indices in ub.chunks(harn.chosen_indices, 16):
            harn.debug('PREDICTING CHUNK')
            inbatch = [vali_dset[index] for index in indices]
            raw_batch = nh.data.collate.padded_collate(inbatch)
            batch = harn.prepare_batch(raw_batch)
            outputs, loss = harn.run_batch(batch)
            postout = harn.model.module.postprocess(outputs)

            for idx, index in enumerate(indices):
                orig_img = vali_dset._load_image(index)
                fig = harn.visualize_prediction(batch, outputs, postout, idx=idx,
                                                thresh=0.1, orig_img=orig_img)
                img = nh.util.mplutil.render_figure_to_image(fig)
                dump_dpath = ub.ensuredir((harn.train_dpath, 'dump'))
                dump_fname = 'pred_{:04d}_{:08d}.png'.format(index, harn.epoch)
                fpath = os.path.join(dump_dpath, dump_fname)
                harn.debug('dump viz fpath = {}'.format(fpath))
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


def setup_harness(bsize=16, workers=0, **kw):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py setup_harness

    Example:
        >>> harn = setup_harness()
        >>> harn.initialize()
    """

    xpu = nh.XPU.cast('argv')

    def _argval(arg, default):
        return ub.argval(arg, kw.get(arg.lstrip('-'), default))

    nice = _argval('--nice', default='Yolo2Baseline')
    batch_size = int(_argval('--batch_size', default=bsize))
    bstep = int(_argval('--bstep', 1))
    workers = int(_argval('--workers', default=workers))
    decay = float(_argval('--decay', default=0.0005))
    lr = float(_argval('--lr', default=0.001))
    workdir = _argval('--workdir', default=ub.truepath('~/work/viame/yolo'))
    ovthresh = 0.5

    coco_dsets = load_coco_datasets()

    datasets = {
        'train': YoloCocoDataset(coco_dsets['train'], train=True),
        'vali': YoloCocoDataset(coco_dsets['vali']),
    }

    anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                          (6.63, 11.38), (9.42, 5.11),
                          (16.62, 10.52)],
                         dtype=np.float)

    datasets['train'].check_images_exist()
    datasets['vali'].check_images_exist()

    if workers > 0:
        cv2.setNumThreads(0)

    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=False)
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
            'anchors': anchors,
            'conf_thresh': 0.001,
            'nms_thresh': 0.5,
        }),

        'criterion': (light_region_loss.RegionLoss, {
            'num_classes': datasets['train'].num_classes,
            'anchors': anchors,
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
    harn.intervals['log_iter_train'] = None
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None
    return harn


def train():
    """
    python ~/code/baseline-viame-2018/yolo_viame.py train --nice dummy --batch_size=4 --workers=0 --gpu=0

    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase0_b16 --phase=0 --batch_size=16 --workers=2 --gpu=0
    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase1_b16 --phase=1 --batch_size=16 --workers=2 --gpu=0

    python ~/code/baseline-viame-2018/yolo_viame.py train --nice phase1_b32 --phase=1 --batch_size=32 --workers=4 --gpu=2,3

    srun -c 4 -p priority --gres=gpu:1 \
            python ~/code/baseline-viame-2018/yolo_viame.py train \
            --nice baseline1 --batch_size=16 --workers=4 --gpu=0
    """

    harn = setup_harness(nice='baseline1', batch_size=16, workers=4)
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest ~/code/baseline-viame-2018/yolo_viame.py all
        python ~/code/baseline-viame-2018/yolo_viame.py all
    """
    train()
    # import xdoctest
    # xdoctest.doctest_module(__file__)
