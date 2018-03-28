from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join, dirname
import glob
import os
import ubelt as ub
import coco_wrangler
import viame_wrangler

DRAW = True


class WrangleConfig(object):
    def __init__(cfg):
        cfg.work_dir = ub.truepath(ub.argval('--work', default='~/work'))
        cfg.data_dir = ub.truepath(ub.argval('--data', default='~/data'))
        cfg.challenge_data_dir = join(cfg.data_dir, 'viame-challenge-2018')
        cfg.challenge_work_dir = join(cfg.work_dir, 'viame-challenge-2018')
        ub.ensuredir(cfg.challenge_work_dir)

        cfg.phase = int(ub.argval('--phase', default='0'))
        if cfg.phase == 0:
            cfg.img_root = join(cfg.challenge_data_dir, 'phase0-imagery')
            cfg.annot_dir = join(cfg.challenge_data_dir, 'full-datasets')
        elif cfg.phase == 1:
            pass
        else:
            raise KeyError(cfg.phase)


def download_phase0_annots():
    """
    CommandLine:
        pip install girder-client
        python ~/code/baseline-viame-2018/standardize.py download_phase0_annots
    """
    cfg = WrangleConfig()
    dpath = cfg.challenge_data_dir
    fname = 'phase0-annotations.tar.gz'
    dest = os.path.join(dpath, fname)
    if not os.path.exists(dest):
        from girder_client.cli import main  # NOQA
        command = 'girder-cli --api-url https://challenge.kitware.com/api/v1 download 5a9d839456357d0cb633d0e3 {}'.format(dpath)
        info = ub.cmd(command, verbout=1, verbose=1, shell=True)
        assert info['ret'] == 0
    unpacked = join(dpath, 'phase0-annotations')
    if not os.path.exists(unpacked):
        info = ub.cmd('tar -xvzf "{}" -C "{}"'.format(dest, dpath), verbose=2, verbout=1)
        assert info['ret'] == 0
    return dest


# @ub.memoize
def read_raw_categories():
    cfg = WrangleConfig()
    img_root = cfg.img_root
    annot_dir = cfg.annot_dir
    fpaths = list(glob.glob(join(annot_dir, '*.json')))

    print('Reading')
    dsets = [coco_wrangler.CocoDataset(fpath, autobuild=True)
             for fpath in fpaths]

    if 0:
        for dset in dsets:
            print(dset.img_root)
            # print(ub.repr2([d['name'] for d in dset.cats.values()]))
            # print(ub.repr2(dset.basic_stats()))
            print(ub.repr2(dset.category_annotation_frequency()))

    print('Merging')
    merged = coco_wrangler.CocoDataset.union(*dsets)
    merged.img_root = img_root
    # merged._run_fixes()
    # print(ub.repr2(merged.category_annotation_frequency()))

    tree0 = viame_wrangler.lifetree.LifeCatalog(autoparse=True)
    mapper = viame_wrangler.cats_2018.make_raw_category_mapping(merged, tree0)
    merged.rename_categories(mapper)

    print('Building')
    node_to_freq = merged.category_annotation_frequency()
    for node in tree0.G.nodes():
        tree0.G.node[node]['freq'] = node_to_freq.get(node, 0)
    tree0.accumulate_frequencies()
    tree0.remove_unsupported_nodes()
    if DRAW:
        tree0.draw('c0-fine-classes-raw.png')
    return tree0, merged, mapper


def define_fine_challenge_categories():
    """
    Use the full dataset to reduce to a set of standard categories and output
    those mappings.

    python ~/code/baseline-viame-2018/standardize.py define_fine_challenge_categories
    """
    tree0, merged0, mapper0 = read_raw_categories()
    raw_to_freq = merged0.category_annotation_frequency()

    print('Build fine cats')
    # Build fine-tree0
    tree2 = tree0.copy()
    tree2.reduce_paths()
    # tree2.draw('fine-classes-reduced.png')
    # tree2.remove_all_freq0_nodes()
    # This "tree0" represents the final fine-grained class heirarchy.
    # (sans any changes)
    if DRAW:
        tree2.draw('c2-fine-classes-minimized.png')
    tree0 = viame_wrangler.lifetree.LifeCatalog(autoparse=True)
    mapper2 = viame_wrangler.cats_2018.make_raw_category_mapping(merged0, tree0, tree2)
    for k, v in mapper2.items():
        if raw_to_freq[k] == 0:
            mapper2[k] = 'ignore'
    merged2 = merged0.copy()
    merged2.rename_categories(mapper2)
    cats2 = []
    cats2 += [{'name': 'ignore'}]
    for n in tree2.G.node:
        assert len(tree2.G.pred[n]) <= 1
        if len(tree2.G.pred[n]):
            cats2 += [{
                'name': n,
                'supercategory': list(tree2.G.pred[n])[0],
            }]
        else:
            cats2 += [{
                'name': n,
            }]
    cats2 = sorted(cats2, key=lambda c: c['name'])

    if 1:
        # print('Fine-Grained Classification Grouping')
        # print(ub.repr2(ub.invert_dict(mapper2, False)))
        print('Fine-Grained Classification Class Frequency')
        print(ub.repr2(merged2.category_annotation_frequency()))
        print('Fine-Grained Classification Annotation Type Frequency')
        print(ub.repr2(merged2.category_annotation_type_frequency(), sk=1))
        print('Fine-Grained Classification Class Hierarchy')
        print(ub.repr2(cats2))

    return mapper0, mapper2, cats2


def define_coarse_challenge_categories():
    """
    python ~/code/baseline-viame-2018/standardize.py define_coarse_challenge_categories
    """
    # Build coarse tree0
    import networkx as nx
    tree0, merged0, mapper0 = read_raw_categories()
    raw_to_freq = merged0.category_annotation_frequency()

    print('Build coarse cats')
    tree1 = tree0.copy()
    groups = []
    for n, d in tree1.G.nodes(data=True):
        if d.get('rank', 'null') == 'order':
            groups.append(list(nx.descendants(tree1.G, n)) + [n])
    tree1.collapse(groups)
    groups = []
    for n, d in tree1.G.nodes(data=True):
        if d.get('rank', 'null') == 'class':
            desc = list(nx.descendants(tree1.G, n))
            if not any(tree1.G.node[d].get('rank', 'null') == 'order' for d in desc):
                # collapse classes if orders are unknown
                groups.append(desc + [n])
    tree1.collapse(groups)
    tree1.collapse([['Rock', 'DustCloud', 'NonLiving']])

    tree1.collapse_descendants('Echinodermata')
    tree1.collapse_descendants('Gastropoda')
    tree1.collapse_descendants('Cephalopoda')

    tree1.collapse([['Chimaeriformes', 'Clupeiformes', 'Lophiiformes',
                    'NotPleuronectiformes']])
    tree1.reduce_paths()

    if DRAW:
        tree1.draw('c1-coarse-classes-collapsed.png')

    mapper1 = viame_wrangler.cats_2018.make_raw_category_mapping(merged0, tree0, tree1)
    for k, v in mapper1.items():
        if raw_to_freq[k] == 0:
            mapper1[k] = 'ignore'
    merged1 = merged0.copy()
    merged1.rename_categories(mapper1)

    cats1 = []
    cats1 += [{'name': 'ignore'}]
    for n in tree1.G.node:
        assert len(tree1.G.pred[n]) <= 1
        if len(tree1.G.pred[n]):
            cats1 += [{
                'name': n,
                'supercategory': list(tree1.G.pred[n])[0],
            }]
        else:
            cats1 += [{
                'name': n,
            }]
    cats1 = sorted(cats1, key=lambda c: c['name'])

    if 1:
        # print('Coarse Classification Grouping')
        # print(ub.repr2(ub.invert_dict(mapper1, False)))
        print('Coarse Classification Class Frequency')
        print(ub.repr2(merged1.category_annotation_frequency()))
        print('Coarse Classification Annotation Type Frequency')
        print(ub.repr2(merged1.category_annotation_type_frequency(), sk=1))
        print('Coarse Classification Class Hierarchy')
        print(ub.repr2(cats1))

    return mapper0, mapper1, cats1


def main():
    mapper0, mapper2, cats2 = define_fine_challenge_categories()
    mapper0, mapper1, cats1 = define_coarse_challenge_categories()

    mapper2['shrimp']
    mapper1['shrimp']

    assert all([k == v for k, v in mapper2.items()])
    assert not all([k != v for k, v in mapper1.items()])

    raw_to_fine_cat = {k: mapper2[v] for k, v in mapper0.items()}
    raw_to_coarse_cat = {k: mapper1[v] for k, v in mapper0.items()}

    fine_to_coarse_cat = {}
    fine_to_raws = ub.invert_dict(mapper0, 0)
    for fine, raws in fine_to_raws.items():
        for raw in raws:
            coarse = raw_to_coarse_cat[raw]
            fine_to_coarse_cat[fine] = coarse

    print(ub.repr2(ub.invert_dict(raw_to_fine_cat, False)))
    print(ub.repr2(ub.invert_dict(raw_to_coarse_cat, False)))

    # Write a python file that contains coarse mappings
    text = ub.codeblock(
        '''
        """ autogenerated file defining the viame challenge 2018 categories """


        class FineGrainedChallenge(object):
            raw_to_cat = {raw_to_fine_cat}
            heirarchy = {cats2}


        class CoarseChallenge(object):
            raw_to_cat = {raw_to_coarse_cat}
            fine_to_cat = {fine_to_coarse_cat}
            heirarchy = {cats1}
        ''').format(
            raw_to_fine_cat=ub.repr2(raw_to_fine_cat),
            raw_to_coarse_cat=ub.repr2(raw_to_coarse_cat),
            fine_to_coarse_cat=ub.repr2(fine_to_coarse_cat),
            cats1=ub.repr2(cats1),
            cats2=ub.repr2(cats2)
    )
    import autopep8
    pep8_options = {}
    new_text = autopep8.fix_code(text, pep8_options)
    # print(new_text)
    ub.writeto(join(dirname(viame_wrangler.__file__), 'mappings.py'), new_text)


if __name__ == '__main__':
    r"""
    python ~/code/baseline-viame-2018/standardize.py main

    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/baseline-viame-2018
        python ~/code/baseline-viame-2018/standardize.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
