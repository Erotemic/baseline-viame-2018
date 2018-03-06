# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import glob
import ubelt as ub
from coco_wrangler import CocoDataset, StratifiedGroupKFold


def get_coarse_mapping():
    simple_maps = [
        # habcam
        {
            'scallop': [
                'live sea scallop',
                'live sea scallop width',
                'live sea scallop inexact',
                'swimming sea scallop',
                'probable live sea scallop',
                'probable live sea scallop inexact',
                'swimming sea scallop width',
                'probable dead sea scallop inexact',
                'dead sea scallop',
                'sea scallop clapper',
                'sea scallop clapper width',
                'sea scallop clapper inexact',
                'dead sea scallop inexact',
                'probable dead sea scallop',
                'swimming sea scallop inexact',
                'probable swimming sea scallop',
                'probable swimming sea scallop inexact',
                'probable dead sea scallop width'
            ],

            'round_fish': [
                # Roundfish - A fish classification, including species such as
                # trout, bass, cod, pike, snapper and salmon
                'unidentified roundfish',
                'unidentified roundfish (less than half)',
                'convict worm',  # Perciformes Pholidichthyidae Pholidichthys leucotaenia  # eel-like
            ],

            'flat_fish': [
                # A flatfish is a member of the order Pleuronectiformes of
                # ray-finned demersal fishes,
                'unidentified flatfish',
                'unidentified flatfish (less than half)',
            ],

            'other_fish': [
                'unidentified fish',
                'unidentified fish (less than half)',
                'monkfish',  # Lophius
            ],

            'skate': [
                'unidentified skate',
                'unidentified skate (less than half)',
            ],

            'crab': [
                'jonah or rock crab',
            ],

            'snail': [
                'waved whelk',  # snail - Buccinidae Buccinum undatum
            ],

            'lobster': [
                'American Lobster',
            ],

            'squid': [
                'squid',
            ],

            'ignore': [
                'probably didemnum',
                'dust cloud',
                'probable scallop-like rock',
            ]
        },

        # mouss_seq0
        {
            # 'shark': [
            'other_fish': [
                'carcharhinus plumbeus',  # Carcharhiniformes Carcharhinidae Carcharhinus plumbeus - sandbar shark (shark)
            ],
        },

        # mouss_seq1
        {
            'round_fish': [
                'pristipomoides filamentosus',  # Perciformes Lutjanidae Pristipomoides filamentosus (crimson jobfish)
                'pristipomoides sieboldii',  # lavandar jobfish
            ]
        },

        # mouss_seq2
        {
            'ignore': [
                'negative'
            ],

            # 'shark': [
            'other_fish': [
                'carcharhinus plumbeus',  # Carcharhiniformes Carcharhinidae Carcharhinus plumbeus - sandbar shark (shark)
            ],

            'round_fish': [
                'pristipomoides filamentosus',  # crimson jobfish (round)
                'pristipomoides sieboldii',  # lavandar jobfish (round)
                'merluccius_productus',  # north pacific hake (round)
                'lycodes_diapterus',  # black eelpout
            ],

            'rock_fish': [
                'sebastes_2species',  # Scorpaeniformes
                'sebastolobus_altivelis',  # Scorpaeniformes
            ],

            'flat_fish': [
                'glyptocephalus_zachirus'  # red sole flat
            ],

            'echinoderm': [
                'psolus_squamatus',  # sea cucumber
                'rathbunaster_californicus',  # starfish
            ]
        },

        # nwfsc_seq0
        {
            'rock_fish': [
                'greenstriped rockfish',
                'unknown rockfish',
                'yelloweye rockfish',  # Scorpaeniformes Sebastes ruberrimus
                'rosethorn rockfish',
                'stripetail rockfish',
                'pygmy/puget sound rockfish',
                'redstripe rockfish',
                'wsr/sebastomus',
                'unknown sebastomus',

                'unknown sculpin',  # Scorpaeniformes Cottoidei Cottoidea
                'threadfin sculpin',

                'poacher/cottid',  # Scorpaeniformes Cottoidea Agonidae
                'unknown poacher',
            ],

            'round_fish': [
                'unknown roundfish',
                'spotted ratfish',  # Chimaeriformes Chimaeridae Hydrolagus colliei

                'unknown eelpout',  # Perciformes Zoarcidae Lycodes
                'blackbelly eelpout',  # Perciformes Zoarcidae Lycodes pacificus
            ],

            'flat_fish': [
                'unknown flatfish',
                'petrale sole',
                'dover sole',  # Pleuronectiformes Soleidae Solea solea
                'english sole',
            ],

            'other_fish': [
                'unknown fish',
            ]
        },

        # afsc_seq0
        {
            'round_fish': [
                'Searcher',  # Perciformes Bathymasteridae Bathymaster signatus
                'Herring',  # Clupeiformes Clupeidea Clupea harengus
                'Pollock',  # Gadiformes Gadidae Pollachius

                'Smelt Unid.',  # Osmeriformes Osmeridae

                'Lingcod',
                'Pacific Cod',  # Gadiformes Gadidae Gadus macrocephalus
                'Gadoid Unid.',

                'Prickleback',  # Perciformes Zoarcoidei Stichaeidae - eel-like
                'Stichaeidae',  # a prickleback

                'Eelpout Unid.',  # Perciformes Zoarcidae
                'Ronquil Unid.',  # related to eelpouts, but not the same
                'Prowfish',  # Perciformes Zaproridae Zaprora silenus
            ],

            'flat_fish': [
                'Pacific Halibut',  # Pleuronectiformes Pleuronectidae Hippoglossus stenolepis
                'Arrowtooth Flounder',  # Pleuronectiformes Pleuronectidae Atheresthes stomias
                'Flatfish Unid.',
                'Rex Sole',
                'Rock Sole Unid.',
                'Flathead Sole',
                'Dover Sole',   # Pleuronectiformes Soleidae Solea solea
            ],

            'rock_fish': [
                'Poacher Unid.',  # Scorpaeniformes Cottoidea Agonidae

                'Snailfish Unid.',  # Scorpaeniformes Cyclopteroidea Liparidae
                'Shortspine Thornyhead',  # Scorpaeniformes Sebastidae Sebastolobus alascanus
                'Thornyhead Unid.',  # Scorpaeniformes Sebastidae Sebastolobus

                'Hexagrammidae sp.',  # Scorpaeniformes Hexagrammoidei Hexagrammidae

                'Sablefish',  # Scorpaeniformes  Anoplopomatidae Anoplopoma fimbria

                'Rockfish Unid.',
                'Northern Rockfish',  # Scorpaeniformes Sebastes polyspinis
                'Dusky Rockfish',     # Scorpaeniformes Sebastes ciliatus
                'Harlequin Rockfish',
                'Sharpchin Rockfish',
                'Blackspotted Rockfish',
                'Black Rockfish',
                'Redstripe Rockfish',
                'Shortraker Rockfish',
                'Silvergray Rockfish',
                'Rosethorn Rockfish',
                'Yelloweye Rockfish',  # Scorpaeniformes Sebastes ruberrimus
                'Quillback Rockfish',
                'Darkblotched Rockfish',

                'Sculpin Unid.',
                'Irish Lord',  # Scorpaeniformes Cottidae Hemilepidotus hemilepidotus

                'Greenling Unid.',
                'Kelp Greenling',  # Scorpaeniformes Hexagrammidae decagrammus

                'Atka Mackerel',  # Scorpaeniformes Hexagrammidae Pleurogrammus Pleurogrammus monopterygius

                'Pacific Ocean Perch',  # Scorpaeniformes Sebastidae Sebastes alutus
            ],

            'other_fish': [
                'Fish Unid.',
            ],

            'shrimp': [
                'Shrimp',
                'Shrimp Unid.',
            ],

            'echinoderm': [
                'Starfish Unid.',
                'nudibranch',
            ],

            'skate': [
                'Skate Unid.',
                'Longnose Skate',  # Rajiformes Rajidae Dipturus oxyrinchus
            ],

            'crab': [
                'Crab Unid.',
                'Tanner Crab Unid.',  # Chionoecetes bairdi
                'Bairdi Tanner Crab',
            ],

            'ignore': [
                'Octopus Unid.',
                'spc D',
            ],
        }
    ]

    mapping = {}
    for submap in simple_maps:
        for key, vals in submap.items():
            for val in vals:
                if val in mapping:
                    assert key == mapping[val]
                mapping[val] = key
    return mapping

    # newfreq = {}
    # for k, v in self.category_annotation_frequency().items():
    #     key = mapping[k]
    #     newfreq[key] = newfreq.get(key, 0) + v

    # newfreq = ub.odict(sorted(newfreq.items(),
    #                           key=lambda kv: kv[1]))

    # print(ub.repr2(newfreq))


def fix_full_truthfiles():
    """
    rsync -avpR acidalia:/home/git/phase0-annotations.tar.gz ~/Downloads
    """
    data_dir = ub.truepath('~/data')
    full_annots = join(data_dir, 'viame_full_annotation_files')
    fpaths = list(glob.glob(join(full_annots, '*.json')))

    dsets = []
    for fpath in fpaths:
        dset = CocoDataset(fpath, autobuild=False)
        dset._run_fixes()
        dset._build_index()
        dsets.append(dset)

    self = CocoDataset.union(*dsets)


def make_baseline_truthfiles():
    work_dir = ub.truepath('~/work')
    data_dir = ub.truepath('~/data')

    challenge_data_dir = join(data_dir, 'viame-challenge-2018')
    challenge_work_dir = join(work_dir, 'viame-challenge-2018')

    ub.ensuredir(challenge_work_dir)

    img_root = join(challenge_data_dir, 'phase0-imagery')
    annot_dir = join(challenge_data_dir, 'phase0-annotations')
    fpaths = list(glob.glob(join(annot_dir, '*.json')))
    # ignore the non-bounding box nwfsc and afsc datasets for now

    # exclude = ('nwfsc', 'afsc', 'mouss', 'habcam')
    # exclude = ('nwfsc', 'afsc', 'mouss',)
    # fpaths = [p for p in fpaths if not basename(p).startswith(exclude)]

    print('Reading')
    dsets = []
    for fpath in fpaths:
        dset = CocoDataset(fpath, autobuild=False)
        dset._run_fixes()
        dset._build_index()
        dsets.append(dset)

        # print(ub.repr2([d['name'] for d in dset.cats.values()]))
        # print(dset.img_root)
        # print(ub.repr2(dset.basic_stats()))
        # print(ub.repr2(dset.category_annotation_frequency()))

    print('Merging')
    self = CocoDataset.union(*dsets)
    self.img_root = img_root
    self._run_fixes()
    print(ub.repr2(self.category_annotation_frequency()))

    self.dump(join(challenge_work_dir, 'phase0-merged.mscoco.json'))

    if True:
        # Cleanup the dataset
        self._remove_bad_annotations()

        self._remove_radius_annotations()

        self._remove_keypoint_annotations()

        self._remove_empty_images()

    print(ub.repr2(self.category_annotation_frequency()))
    print(sum(list(self.category_annotation_frequency().values())))
    self.coarsen_categories(get_coarse_mapping())
    print(ub.repr2(self.category_annotation_frequency()))
    print(sum(list(self.category_annotation_frequency().values())))

    print(ub.repr2(self.basic_stats()))

    if False:
        # aid = list(self.anns.values())[0]['id']
        # self.show_annotation(aid)
        gids = sorted([gid for gid, aids in self.gid_to_aids.items() if aids])
        # import utool as ut
        # for gid in ut.InteractiveIter(gids):
        for gid in gids:
            from matplotlib import pyplot as plt
            fig = plt.figure(1)
            fig.clf()
            self.show_annotation(gid=gid)
            fig.canvas.draw()

    # Split into train / test  set
    print('Splitting')
    skf = StratifiedGroupKFold(n_splits=2)
    groups = [ann['image_id'] for ann in self.anns.values()]
    y = [ann['category_id'] for ann in self.anns.values()]
    X = [ann['id'] for ann in self.anns.values()]
    split = list(skf.split(X=X, y=y, groups=groups))[0]
    train_idx, test_idx = split

    print('Taking subsets')
    aid_to_gid = {aid: ann['image_id'] for aid, ann in self.anns.items()}
    train_aids = list(ub.take(X, train_idx))
    test_aids = list(ub.take(X, test_idx))
    train_gids = sorted(set(ub.take(aid_to_gid, train_aids)))
    test_gids = sorted(set(ub.take(aid_to_gid, test_aids)))

    train_dset = self.subset(train_gids)
    test_dset = self.subset(test_gids)

    print('--- Training Stats ---')
    print(ub.repr2(train_dset.basic_stats()))
    print('--- Testing Stats ---')
    print(ub.repr2(test_dset.basic_stats()))

    train_dset._ensure_imgsize()
    test_dset._ensure_imgsize()

    print('Writing')
    train_dset.dump(join(challenge_work_dir, 'phase0-merged-train.mscoco.json'))
    test_dset.dump(join(challenge_work_dir, 'phase0-merged-test.mscoco.json'))

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
          RPN_PRE_NMS_TOP_N: 6000
          RPN_POST_NMS_TOP_N: 1000
        OUTPUT_DIR: /work/viame-challenge-2018/output
        """)
    config_text = config_text.format(
        num_classes=len(self.cats),
    )
    ub.writeto(join(challenge_work_dir, 'phase0-faster-rcnn.yaml'), config_text)

    docker_cmd = ('nvidia-docker run '
                  '-v {work_dir}:/work -v {data_dir}:/data '
                  '-it detectron:c2-cuda9-cudnn7 bash').format(
                      work_dir=work_dir, data_dir=data_dir)

    train_cmd = ('python2 tools/train_net.py '
                 '--cfg /work/viame-challenge-2018/phase0-faster-rcnn.yaml '
                 'OUTPUT_DIR /work/viame-challenge-2018/output')

    hacks = ub.codeblock(
        """
        git remote add Erotemic https://github.com/Erotemic/Detectron.git
        git fetch --all
        git checkout general_dataset

        # curl https://github.com/Erotemic/Detectron/blob/42d44b2d155c775dc509b6a44518d0c582f8cdf5/tools/train_net.py
        # wget https://github.com/Erotemic/Detectron/blob/42d44b2d155c775dc509b6a44518d0c582f8cdf5/lib/core/config.py
        """)

    print(docker_cmd)
    print(train_cmd)


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/baseline-viame-2018
        python ~/code/baseline-viame-2018/wrangle.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
