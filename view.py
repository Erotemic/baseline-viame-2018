from os.path import join
import os
import ubelt as ub
import viame_wrangler
from fishnet.coco_api import CocoDataset
import glob


def _read(cfg):
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

        bad_annots = dset._find_bad_annotations()
        if bad_annots:

            print(ub.repr2(bad_annots))
            assert False, 'bad annotatinos'

        print(ub.repr2(dset.basic_stats()))
        dsets.append(dset)

    print('Merging')
    merged = CocoDataset.union(*dsets, autobuild=False)
    # merged._remove_bad_annotations()
    merged.img_root = cfg.img_root
    merged._build_index()
    return merged


def read_fine_merged():
    cfg = viame_wrangler.config.WrangleConfig({
        # 'img_root': '~/data/viame-challenge-2018/phase0-imagery',
        # 'annots': '~/data/viame-challenge-2018/phase0-fine*keypoint*.json'
        # 'img_root': '~/data/viame-challenge-2018/phase1-imagery',
        # 'annots': '~/data/viame-challenge-2018/phase1-annotations/*/*fine*keypoint*.json',
        'img_root': '/data/projects/noaa/training_data/imagery',
        'annots': '/data/projects/noaa/training_data/annotations/*/*fine*-keypoint*',
    })
    merged = _read(cfg)
    return merged


def read_coarse_merged():
    cfg = viame_wrangler.config.WrangleConfig({
        # 'img_root': '~/work/viame-challenge-2018/phase0-imagery',
        # 'annots': '~/work/viame-challenge-2018/phase0-fine*keypoint*.json'
        # 'img_root': '~/data/viame-challenge-2018/phase1-imagery',
        # 'annots': '~/data/viame-challenge-2018/phase1-annotations/*/*coarse*-keypoint*',
        'img_root': '/data/projects/noaa/training_data/imagery',
        'annots': '/data/projects/noaa/training_data/annotations/*/*fine*-keypoint*',
    })
    merged = _read(cfg)
    return merged


def show_keypoint_annots():
    merged = read_fine_merged()

    def images_with_keypoints():
        keypoint_gids = set()
        for aid, ann in merged.anns.items():
            if ann['roi_shape'] == 'keypoints':
                keypoint_gids.add(ann['image_id'])

        relevant = ub.dict_subset(merged.gid_to_aids, keypoint_gids)
        relevant = {gid: [a for a in aids if merged.anns[a]['roi_shape'] == 'keypoints'] for gid, aids in relevant.items()}

        gid_list = ub.argsort(ub.map_vals(len, relevant))[::-1]
        return gid_list

    def sort_gids_by_nannots(gids):
        return ub.argsort(ub.map_vals(len, ub.dict_subset(merged.gid_to_aids, gids, default=[])))[::-1]

    def images_with_keypoints_and_boxes():
        keypoint_gids = set()
        for aid, ann in merged.anns.items():
            if ann['roi_shape'] == 'keypoints':
                keypoint_gids.add(ann['image_id'])

        gid_list = []
        for gid in keypoint_gids:
            aids = merged.gid_to_aids[gid]
            types = set()
            for ann in ub.take(merged.anns, aids):
                types.add(ann['roi_shape'])
            if len(types) > 1:
                gid_list.append(gid)

        gid_list = sort_gids_by_nannots(gid_list)
        return gid_list

    def image_from_each_dataset():
        groups = ub.ddict(list)
        for gid, img in merged.imgs.items():
            groups[os.path.dirname(img['file_name'])].append(gid)

        gid_groups = []
        for gids in groups.values():
            gids = sort_gids_by_nannots(gids)
            gid_groups.append(gids)

        # round robin sample
        datas = [gid for x in zip(*gid_groups) for gid in x]
        return datas

    # gid_list = images_with_keypoints()
    gid_list = images_with_keypoints_and_boxes()
    gid_list = image_from_each_dataset()

    # gid = gid_list[2]
    # import matplotlib.pyplot as plt
    # plt.gcf().clf()
    # merged.show_annotation(gid=gid)

    import utool as ut
    if ut.inIPython():
        import IPython
        IPython.get_ipython().magic('pylab qt5 --no-import-all')

    from matplotlib import pyplot as plt
    for gid in ut.InteractiveIter(gid_list):
        try:
            fig = plt.figure(1)
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


def nx_ascii_tree(graph, key=None):
    """
    Creates an printable ascii representation of a directed tree / forest.

    Args:
        graph (nx.DiGraph): each node has at most one parent (
            i.e. graph must be a directed forest)
        key (str): if specified, uses this node attribute as a label instead of
            the id

    References:
        https://stackoverflow.com/questions/32151776/visualize-tree-in-bash-like-the-output-of-unix-tree

    Example:
        >>> import networkx as nx
        >>> graph = nx.dfs_tree(nx.balanced_tree(2, 2), 0)
        >>> text = nx_ascii_tree(graph)
        >>> print(text)
        └── 0
           ├── 1
           │  ├── 3
           │  └── 4
           └── 2
              ├── 5
              └── 6
    """
    import six
    import networkx as nx
    branch = '├─'
    pipe = '│'
    end = '└─'
    dash = '─'

    assert nx.is_forest(graph)
    assert nx.is_directed(graph)

    lines = []

    def _draw_tree_nx(graph, node, level, last=False, sup=[]):
        def update(left, i):
            if i < len(left):
                left[i] = '   '
            return left

        initial = ['{}  '.format(pipe)] * level
        parts = six.moves.reduce(update, sup, initial)
        prefix = ''.join(parts)
        if key is None:
            node_label = str(node)
        else:
            node_label = str(graph.nodes[node]['label'])

        suffix = '{} '.format(dash) + node_label
        if last:
            line = prefix + end + suffix
        else:
            line = prefix + branch + suffix
        lines.append(line)

        children = list(graph.succ[node])
        if children:
            level += 1
            for node in children[:-1]:
                _draw_tree_nx(graph, node, level, sup=sup)
            _draw_tree_nx(graph, children[-1], level, True, [level] + sup)

    def draw_tree(graph):
        source_nodes = [n for n in graph.nodes if graph.in_degree[n] == 0]
        if source_nodes:
            level = 0
            for node in source_nodes[:-1]:
                _draw_tree_nx(graph, node, level, last=False, sup=[])
            _draw_tree_nx(graph, source_nodes[-1], level, last=True, sup=[0])

    draw_tree(graph)
    text = '\n'.join(lines)
    return text


def printable_heirarchy():
    fine = read_fine_merged()
    coarse = read_coarse_merged()
    # from viame_wrangler import mappings

    # COARSE CAT

    import networkx as nx
    g = nx.DiGraph()
    for cat in coarse.cats.values():
        # for cat in dset.dataset['fine_categories']:
        # for cat in mappings.CoarseChallenge.heirarchy:
        g.add_node(cat['name'])
        if 'supercategory' in cat:
            g.add_edge(cat['supercategory'], cat['name'])

    for n in g.nodes:
        cat = coarse.name_to_cat[n]
        cid = cat['id']
        n_examples = len(coarse.cid_to_aids[cid])
        g.node[n]['label'] = '"{}":{}'.format(n, n_examples)

    print(nx_ascii_tree(g, 'label'))

    # FINE CAT
    # dset = merged

    import networkx as nx
    g = nx.DiGraph()
    for cat in fine.cats.values():
        # for cat in dset.dataset['fine_categories']:
        # for cat in mappings.FineGrainedChallenge.heirarchy:
        g.add_node(cat['name'])
        if 'supercategory' in cat:
            g.add_edge(cat['supercategory'], cat['name'])

    for n in g.nodes:
        try:
            cat = fine.name_to_cat[n]
            cid = cat['id']
            n_examples = len(fine.cid_to_aids[cid])
        except Exception:
            n_examples = 0
        g.node[n]['label'] = '"{}":{}'.format(n, n_examples)

    print(nx_ascii_tree(g, 'label'))
