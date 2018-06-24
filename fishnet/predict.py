import glob
import netharn as nh
import numpy as np
import ubelt as ub
from fishnet import coco_api
from netharn.models.yolo2 import light_yolo
from fishnet.data import load_coco_datasets, YoloCocoDataset


def predict():
    """
    Currently hacked in due to limited harness support.

    srun -c 4 -p priority --gres=gpu:1 \
            python ~/code/baseline-viame-2018/yolo_viame.py predict \
            --gpu=0
    """

    # HACK: Load the training dataset to extract the categories
    # INSTEAD: Should read the categories from a deployed model
    coco_dsets = load_coco_datasets()
    categories = coco_dsets['train'].dataset['categories']

    # Create a dataset to iterate through the images to predict on
    test_gpaths = glob.glob(ub.truepath('~/data/noaa/test_data/*/*.png'))
    predict_coco_dataset = {
        'licenses': [],
        'info': [],
        'categories': categories,
        'images': [
            {
                'id': idx,
                'file_name': fpath,
            }
            for idx, fpath in enumerate(test_gpaths)
        ],
        'annotations': [],
    }
    predict_coco_dset = coco_api.CocoDataset(predict_coco_dataset, tag='predict')
    predict_dset = YoloCocoDataset(predict_coco_dset, train=False)

    # HACK: Define the path to the model weights
    # INSTEAD: best weights should be packaged in a model deployment
    load_path = ub.truepath(
        '~/work/viame/yolo/fit/nice/baseline1/best_snapshot.pt')
    # load_path = ub.truepath(
    #     '~/work/viame/yolo/fit/nice/baseline1/torch_snapshots/_epoch_00000080.pt')

    # HACK: Define the model topology (because we know what we trained with)
    # INSTEAD: model deployment should store and abstract away the topology
    model = light_yolo.Yolo(**{
        'num_classes': predict_dset.num_classes,
        'anchors': np.asarray([(1.08, 1.19), (3.42, 4.41),
                               (6.63, 11.38), (9.42, 5.11),
                               (16.62, 10.52)],
                              dtype=np.float),
        'conf_thresh': 0.001,
        'nms_thresh': 0.5,
    })

    # Boilerplate code that could be abstracted away in a prediction harness
    xpu = nh.XPU.cast('gpu')
    print('xpu = {!r}'.format(xpu))
    model = xpu.mount(model)
    snapshot_state = xpu.load(load_path)
    model.load_state_dict(snapshot_state['model_state_dict'])

    batch_size = 16
    workers = 4
    predict_loader = predict_dset.make_loader(batch_size=batch_size,
                                              num_workers=workers,
                                              shuffle=False, pin_memory=False)

    letterbox = predict_dset.letterbox

    # HACK: Main prediction loop
    # INSTEAD: Use a prediction harness to abstract these in a similar way to
    # the fit harness.
    predictions = []

    with nh.util.grad_context(False):
        _iter = ub.ProgIter(predict_loader, desc='predicting')
        for bx, raw_batch in enumerate(_iter):
            batch_inputs, batch_labels = raw_batch

            inputs = xpu.variable(batch_inputs)
            labels = {k: xpu.variable(d) for k, d in batch_labels.items()}

            outputs = model(inputs)

            # Transform yolo outputs into the coco format
            postout = model.module.postprocess(outputs)

            indices = labels['indices']
            orig_sizes = labels['orig_sizes']
            inp_size = np.array(inputs.shape[-2:][::-1])
            bsize = len(inputs)
            for ix in range(bsize):
                postitem = postout[ix].data.cpu().numpy()

                orig_size = orig_sizes[ix].data.cpu().numpy()
                gx = int(indices[ix].data.cpu().numpy())
                gid = predict_dset.dset.dataset['images'][gx]['id']

                # Unpack postprocessed predictions
                sboxes = postitem.reshape(-1, 6)
                pred_cxywh = sboxes[:, 0:4]
                pred_scores = sboxes[:, 4]
                pred_cxs = sboxes[:, 5].astype(np.int)

                sortx = pred_scores.argsort()
                pred_scores = pred_scores.take(sortx)
                pred_cxs = pred_cxs.take(sortx)
                pred_cxywh = pred_cxywh.take(sortx, axis=0)

                norm_boxes = nh.util.Boxes(pred_cxywh, 'cxywh')
                boxes = norm_boxes.scale(inp_size)
                pred_box = letterbox._boxes_letterbox_invert(boxes, orig_size,
                                                             inp_size)
                pred_box = pred_box.clip(0, 0, orig_size[0], orig_size[1])

                pred_xywh = pred_box.toformat('xywh').data

                # print(ub.repr2(pred_cxywh.tolist(), precision=2))
                # print(ub.repr2(pred_xywh.tolist(), precision=2))

                for xywh, cx, score in zip(pred_xywh, pred_cxs, pred_scores):
                    if score > 0.1:
                        cid = predict_dset.dset.dataset['categories'][cx]['id']
                        pred = {
                            'id': len(predictions) + 1,
                            'image_id': gid,
                            'category_id': cid,
                            'bbox': list(xywh),
                            'score': score,
                        }
                        predictions.append(pred)
            # if bx > 1:
            #     break

    predict_coco_dset.dataset['annotations'] = predictions
    predict_coco_dset._build_index()

    with open('./viame_pred_dump.mscoco.json') as file:
        predict_coco_dset.dump(file)

    if False:
        import utool as ut
        from matplotlib import pyplot as plt
        gids = set([a['image_id'] for a in predict_coco_dset.anns.values()])
        for gid in ut.InteractiveIter(list(gids)):

            try:
                fig = plt.figure(1)
                fig.clf()
                predict_coco_dset.show_annotation(gid=gid)
                fig.canvas.draw()
            except Exception:
                print('cannot draw')

        z = inputs[0].cpu().numpy().transpose(1, 2, 0)
        nh.util.imshow(z, fnum=2, colorspace='rgb')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/baseline-viame-2018/fishnet/predict.py all
    """
    predict()
