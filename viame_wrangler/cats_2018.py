import ubelt as ub
import re


def normalize_name(name):
    # norm = custom_fine_grained_map.get(name, name).lower()
    norm = name.lower()
    norm = norm.replace(' (less than half)', '')
    norm = norm.replace('probable', '')
    norm = norm.replace('probably', '')
    norm = norm.replace('width', '')
    norm = norm.replace('inexact', '')
    norm = norm.replace('unid.', 'unclassified')
    norm = norm.replace('unknown', 'unclassified')
    norm = norm.replace('unidentified', 'unclassified')
    if norm.strip().endswith(' sp.'):
        norm = norm.replace('sp.', 'unclassified')
    norm = norm.replace('_', ' ').replace('-', ' ')
    if 'unclassified' in norm:
        # move unclassified to the front
        norm = 'unclassified ' + norm.replace('unclassified', '')

    # norm = norm.replace('unid.', '')
    # norm = norm.replace('unknown', '')
    # norm = norm.replace('unidentified', '')
    norm = re.sub('  *', ' ', norm)
    norm = norm.strip()
    return norm


def normalize_categories(merged):
    from viame_wrangler import lifetree
    import networkx as nx
    tree = lifetree.LifeCatalog(autoparse=True)

    common_name_to_node = {}
    node_to_common_names = nx.get_node_attributes(tree.G, 'common_names')
    for node_id, common_names in node_to_common_names.items():
        norm_node_id = normalize_name(node_id)
        common_name_to_node[node_id] = node_id
        common_name_to_node[norm_node_id] = node_id
        for name in common_names:
            norm = normalize_name(name)
            norm2 = norm.replace(' ', '')
            common_name_to_node[name] = node_id
            common_name_to_node[norm] = node_id
            common_name_to_node[norm2] = node_id
            if norm.startswith('unclassified '):
                norm3 = norm.replace('unclassified ', '')
                common_name_to_node[norm3] = node_id

    # Hack in some special cats
    common_name_to_node['negative'] = 'NonLiving'
    common_name_to_node['scallop like rock'] = 'NonLiving'
    common_name_to_node['dust cloud'] = 'NonLiving'
    common_name_to_node['spc d'] = 'NonLiving'

    mapper = {}
    for cat in merged.cats.values():
        norm = normalize_name(cat['name'])
        # hacks
        if norm not in common_name_to_node:
            if norm.startswith('unclassified'):
                if norm[len('unclassified') + 1:] in common_name_to_node:
                    norm = norm[len('unclassified') + 1:]
            if 'unclassified ' + norm in common_name_to_node:
                norm = 'unclassified ' + norm
            if norm == 'unclassified shrimp' or norm == 'shrimp':
                norm = 'caridean shrimp'
            if 'sea scallop' in norm:
                norm = norm.replace('live', '')
                norm = norm.replace('dead', '')
                norm = norm.replace('swimming', '')
                norm = norm.replace('clapper', '')
                norm = normalize_name(norm)
            if 'jonah or rock crab' == norm:
                norm = 'jonah crab'
        node_id = common_name_to_node[norm]
        mapper[cat['name']] = node_id

    merged.coarsen_categories(mapper)

    # Get number of examples of each category
    G = tree.G.copy()
    node_to_freq = merged.category_annotation_frequency()
    for node in G.nodes():
        G.node[node]['freq'] = node_to_freq.get(node, 0)

    # Sum number of examples of each category
    for node in G.nodes():
        freq = G.node[node].get('freq', 0)
        total = freq
        for sub in list(nx.descendants(G, node)):
            total += G.node[sub].get('freq', 0)
        G.node[node]['total'] = total
        if total:
            print('node = {!r}'.format(node))
            print('total = {!r}'.format(total))
            G.node[node]['label'] = G.node[node]['label'] + '\nfreq={},total={}'.format(freq, total)

    bad_nodes = []
    for node in G.nodes():
        if G.node[node]['total'] == 0:
            bad_nodes.append(node)
    G.remove_nodes_from(bad_nodes)

    from networkx.algorithms.connectivity.edge_augmentation import collapse

    source = 'Physical'
    def dfs_streaks(G, source):
        visited = set()
        # stack = [(source, iter(G[source]))]
        stack = [(None, iter([source]))]

        streaks = []
        streak_cls = ub.oset
        streak_cls = list
        streak = streak_cls()

        def on_visit(child):
            nonlocal streak
            if G.in_degree[child] <= 1 and G.out_degree[child] == 1 and G[child].get('freq', 0) == 0:
                streak.append(child)
            else:
                if G.in_degree[child] <= 1:
                    streak.append(child)
                if len(streak) > 1:
                    streaks.append(streak)
                streak = streak_cls()

        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    on_visit(child)
                    stack.append((child, iter(G[child])))
            except StopIteration:
                stack.pop()
        return streaks

    streaks = dfs_streaks(G, source)
    n_to_streak = {n: s for s in streaks for n in s}
    G2 = collapse(G, streaks)
    reverse_map = ub.invert_dict(G2.graph['mapping'], False)
    for n in G2.nodes():
        subs = list(reverse_map[n])
        if len(subs) == 1:
            G2.node[n].update(G.node[subs[0]])
        else:
            n2 = n_to_streak[subs[0]][-1]
            G2.node[n].update(G.node[n2])


    import plottool as pt
    G2.graph['rankdir'] = 'LR'
    pt.dump_nx_ondisk(G2, 'classes-freq.png')


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
                'pristipomoides sieboldii',     # Perciformes Lutjanidae pristipomoides sieboldii - (lavandar jobfish)
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
                'pristipomoides sieboldii',     # lavandar jobfish (round)
                'merluccius_productus',         # Gadiformes Merlucciidae Merluccius productus - north pacific hake (round)
                'lycodes_diapterus',            # Perciformes Zoarcidae Lycodes diapterus - black eelpout
            ],

            'rock_fish': [
                'sebastes_2species',       # Scorpaeniformes Sebastidae
                'sebastolobus_altivelis',  # Scorpaeniformes Sebastidae Sebastolobus altivelis - longspine thornyhead
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
                'unknown rockfish',
                'greenstriped rockfish',
                'yelloweye rockfish',   # Scorpaeniformes Sebastes ruberrimus
                'rosethorn rockfish',   # Sebastes helvomaculatus
                'stripetail rockfish',  # Sebastes saxicola
                'pygmy/puget sound rockfish',  # Sebastes emphaeus
                'redstripe rockfish',  # Sebastes proriger
                'wsr/sebastomus',
                'unknown sebastomus',

                'unknown sculpin',  # Scorpaeniformes Cottoidei Cottoidea
                'threadfin sculpin',

                'poacher/cottid',   # Scorpaeniformes Cottoidea Agonidae
                'unknown poacher',
            ],

            'round_fish': [
                'unknown roundfish',
                'spotted ratfish',  # Chimaeriformes Chimaeridae Hydrolagus colliei

                'unknown eelpout',     # Perciformes Zoarcidae Lycodes
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
                'Herring',   # Clupeiformes Clupeidea Clupea harengus
                'Pollock',   # Gadiformes Gadidae Pollachius

                'Smelt Unid.',  # Osmeriformes Osmeridae

                'Pacific Cod',   # Gadiformes Gadidae Gadus macrocephalus
                'Gadoid Unid.',  # Gadiformes Gadidae

                'Eelpout Unid.',  # Perciformes Zoarcidae
                'Prowfish',       # Perciformes Zaproridae Zaprora silenus
                'Prickleback',    # Perciformes Zoarcoidei Stichaeidae - eel-like
                'Stichaeidae',    # Perciformes Zoarcoidei Stichaeidae - a prickleback

                'Ronquil Unid.',  # Perciformes Bathymasteridae - related to eelpouts, but not the same
                'Searcher',       # Perciformes Bathymasteridae Bathymaster signatus

            ],

            'flat_fish': [
                'Flatfish Unid.',
                'Pacific Halibut',      # Pleuronectiformes Pleuronectidae Hippoglossus stenolepis
                'Arrowtooth Flounder',  # Pleuronectiformes Pleuronectidae Atheresthes stomias
                'Dover Sole',           # Pleuronectiformes Soleidae Solea solea
                'Rex Sole',
                'Rock Sole Unid.',      # Pleuronectiformes Pleuronectidae Lepidopsetta bilineata
                'Flathead Sole',
            ],

            'rock_fish': [

                'Snailfish Unid.',  # Scorpaeniformes Cyclopteroidea Liparidae

                'Shortspine Thornyhead',  # Scorpaeniformes Sebastidae Sebastolobus alascanus
                'Thornyhead Unid.',       # Scorpaeniformes Sebastidae Sebastolobus


                'Sablefish',  # Scorpaeniformes  Anoplopomatidae Anoplopoma fimbria

                'Rockfish Unid.',
                'Northern Rockfish',  # Scorpaeniformes Sebastes polyspinis
                'Dusky Rockfish',     # Scorpaeniformes Sebastes ciliatus
                'Harlequin Rockfish',  # Sebastes variegatus
                'Sharpchin Rockfish',  # Sebastes zacentrus
                'Blackspotted Rockfish',  # Sebastes melanostictus
                'Black Rockfish',  # Sebastes melanops
                'Redstripe Rockfish',  # Sebastes proriger
                'Shortraker Rockfish',  # Sebastes borealis
                'Silvergray Rockfish',  # Sebastes brevispinis
                'Rosethorn Rockfish',  # Sebastes helvomaculatus
                'Yelloweye Rockfish',   # Scorpaeniformes Sebastes ruberrimus
                'Quillback Rockfish',  # Sebastes maliger
                'Darkblotched Rockfish',  # Sebastes crameri


                'Sculpin Unid.',  # Scorpaeniformes Cottoidea
                'Poacher Unid.',  # Scorpaeniformes Cottoidea Agonidae
                'Irish Lord',     # Scorpaeniformes Cottoidea Cottidae Hemilepidotus hemilepidotus

                'Hexagrammidae sp.',  # Scorpaeniformes Hexagrammoidei Hexagrammidae
                'Greenling Unid.',    # Scorpaeniformes Hexagrammidae
                'Kelp Greenling',     # Scorpaeniformes Hexagrammidae decagrammus
                'Atka Mackerel',      # Scorpaeniformes Hexagrammidae Pleurogrammus Pleurogrammus monopterygius
                'Lingcod',            # Scorpaeniformes Hexagrammidae Ophiodon elongatus

                'Pacific Ocean Perch',  # Scorpaeniformes Sebastidae Sebastes alutus
            ],

            'other_fish': [
                'Fish Unid.',
            ],

            'shrimp': [
                # Shrimp can refer to a few things:
                # Euarthropoda Crustacea Malacostraca Decapoda Dendrobranchiata
                # Arthropoda Crustacea Malacostraca Pleocyemata Caridea
                'Shrimp',
                'Shrimp Unid.',
            ],

            'echinoderm': [
                'Starfish Unid.',
                'nudibranch',  # NOTE: not a echinoderm
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


def do_fine_graine_level_sets(self, mapping):

    inverted = ub.invert_dict(mapping, False)
    for sup, subs in inverted.items():
        print('sup = {!r}'.format(sup))
        for sub in subs:
            if sub in self.name_to_cat:
                cat = self.name_to_cat[sub]
                n = len(self.cid_to_aids[cat['id']])
                if n:
                    print('  * {} = {}'.format(sub, n))

    mapping = get_coarse_mapping()
    inverted = ub.invert_dict(mapping, False)

    fine_grained_map = {}

    custom_fine_grained_map = {v: k for k, vs in {
        'unidentified roundfish': [
            'unidentified roundfish',
            'unidentified roundfish (less than half)',
            'unknown roundfish'
            'Rockfish Unid.'
        ],

        'unidentified sebastomus': [
            'sebastes_2species',
            'unknown sebastomus',
            'unknown rockfish',
            'Thornyhead Unid.',
            'Hexagrammidae sp.',
        ],

        'prickleback': [
            'Prickleback',
            'Stichaeidae',
        ],

        'Flatfish Unid.': [
            'Flatfish Unid.',
            'unknown flatfish',
        ]
    }.items() for v in vs}

    catnames = [cat['name'] for cat in self.cats.values()]
    catnames = list(mapping.keys())

    for name in catnames:
        # normalize the name
        norm = normalize_name(name)
        fine_grained_map[name] = norm

    fine_grained_level_set = ub.invert_dict(fine_grained_map, False)
    print(ub.repr2(fine_grained_level_set))

    for sup, subs in inverted.items():
        print('* COARSE-CLASS = {!r}'.format(sup))
        for norm in sorted(set([normalize_name(sub) for sub in subs])):
            raws = fine_grained_level_set.get(norm, [])
            if raws:
                print('    * fine-class = {!r}'.format(norm))
                if len(raws) > 1:
                    # or list(raws)[0] != norm:
                    print(ub.indent('* raw-classes = {}'.format(ub.repr2(raws, nl=1)), ' ' * 8))

    import networkx as nx
    G = nx.DiGraph()
    for norm in fine_grained_map.values():
        G.add_node(norm)

    for sup, subs in inverted.items():
        G.add_node(sup)
        for norm in sorted(set([normalize_name(sub) for sub in subs])):
            G.add_edge(norm, sup)
    if False:
        import plottool as pt
        pt.show_nx(G, layoutkw=dict(prog='neato'), arrow_width=.1, sep=10)
