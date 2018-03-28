
def show_low_support_classes(dset):
    """
    dset = merged
    coarse = merged
    """
    # aid = list(dset.anns.values())[0]['id']
    # dset.show_annotation(aid)
    dset._remove_keypoint_annotations()
    gids = sorted([gid for gid, aids in dset.gid_to_aids.items() if aids])

    catfreq = dset.category_annotation_frequency()
    inspect_cids = []
    for name, freq in catfreq.items():
        if freq > 0 and freq < 50:
            cid = dset.name_to_cat[name]['id']
            inspect_cids.append(cid)
    inspect_gids = list(set(ub.flatten(ub.take(dset.cid_to_gids, inspect_cids))))
    # inspect_gids = [gid for gid in inspect_gids if 'habcam' not in dset.imgs[gid]['file_name']]

    import utool as ut
    if ut.inIPython():
        import IPython
        IPython.get_ipython().magic('pylab qt5 --no-import-all')

    print('inspect_gids = {!r}'.format(inspect_gids))
    from matplotlib import pyplot as plt
    for gid in ut.InteractiveIter(inspect_gids):
        img = dset.imgs[gid]
        print('img = {}'.format(ub.repr2(img)))
        aids = dset.gid_to_aids[gid]
        primary_aid = None
        anns = list(ub.take(dset.anns, aids))
        for ann in anns:
            ann = ann.copy()
            ann['category'] = dset.cats[ann['category_id']]['name']
            print('ann = {}'.format(ub.repr2(ann)))
            if primary_aid is None:
                if ann['category_id'] in inspect_cids:
                    primary_aid = ann['id']

        try:
            fig = plt.figure(1)
            fig.clf()
            dset.show_annotation(primary_aid, gid=gid)
            fig.canvas.draw()
        except:
            print('cannot draw')

    # # import utool as ut
    # for gid in gids:
    #     fig = plt.figure(1)
    #     fig.clf()
    #     dset.show_annotation(gid=gid)
    #     fig.canvas.draw()


def setup_detectron(train_dset, test_dset):
    cfg = viame_wrangler.config.WrangleConfig()

    train_dset._ensure_imgsize()
    test_dset._ensure_imgsize()

    print('Writing')
    train_dset.dump(join(cfg.challenge_work_dir, 'phase0-merged-train.mscoco.json'))
    test_dset.dump(join(cfg.challenge_work_dir, 'phase0-merged-test.mscoco.json'))

    num_classes = len(train_dset.cats)
    print('num_classes = {!r}'.format(num_classes))

    # Make a detectron yaml file
    config_text = ub.codeblock(
        """
        MODEL:
          TYPE: generalized_rcnn
          CONV_BODY: ResNet.add_ResNet50_conv4_body
          NUM_CLASSES: {num_classes}
          FASTER_RCNN: True
        NUM_GPUS: 1
        SOLVER:
          WEIGHT_DECAY: 0.0001
          LR_POLICY: steps_with_decay
          BASE_LR: 0.01
          GAMMA: 0.1
          # 1x schedule (note TRAIN.IMS_PER_BATCH: 1)
          MAX_ITER: 180000
          STEPS: [0, 120000, 160000]
        RPN:
          SIZES: (32, 64, 128, 256, 512)
        FAST_RCNN:
          ROI_BOX_HEAD: ResNet.add_ResNet_roi_conv5_head
          ROI_XFORM_METHOD: RoIAlign
        TRAIN:
          WEIGHTS: https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
          DATASETS: ('/work/viame-challenge-2018/phase0-merged-train.mscoco.json',)
          IM_DIR: '/data/viame-challenge-2018/phase0-imagery'
          SCALES: (800,)
          MAX_SIZE: 1333
          IMS_PER_BATCH: 1
          BATCH_SIZE_PER_IM: 512
        TEST:
          DATASETS: ('/work/viame-challenge-2018/phase0-merged-test.mscoco.json',)
          IM_DIR: '/data/viame-challenge-2018/phase0-imagery'
          SCALES: (800,)
          MAX_SIZE: 1333
          NMS: 0.5
          FORCE_JSON_DATASET_EVAL: True
          RPN_PRE_NMS_TOP_N: 6000
          RPN_POST_NMS_TOP_N: 1000
        OUTPUT_DIR: /work/viame-challenge-2018/output
        """)
    config_text = config_text.format(
        num_classes=num_classes,
    )
    ub.writeto(join(cfg.challenge_work_dir, 'phase0-faster-rcnn.yaml'), config_text)

    docker_cmd = ('nvidia-docker run '
                  '-v {work_dir}:/work -v {data_dir}:/data '
                  '-it detectron:c2-cuda9-cudnn7 bash').format(
                      work_dir=cfg.work_dir, data_dir=cfg.data_dir)

    train_cmd = ('python2 tools/train_net.py '
                 '--cfg /work/viame-challenge-2018/phase0-faster-rcnn.yaml '
                 'OUTPUT_DIR /work/viame-challenge-2018/output')

    hacks = ub.codeblock(
        """
        git remote add Erotemic https://github.com/Erotemic/Detectron.git
        git fetch --all
        git checkout general_dataset

        python2 tools/train_net.py --cfg /work/viame-challenge-2018/phase0-faster-rcnn.yaml OUTPUT_DIR /work/viame-challenge-2018/output
        """)

    print(docker_cmd)
    print(train_cmd)

