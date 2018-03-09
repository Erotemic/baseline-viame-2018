import ubelt as ub


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
                'Rock Sole Unid.',
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

    import re
    def normalize_name(name):
        norm = custom_fine_grained_map.get(name, name).lower()
        norm = norm.replace(' (less than half)', '')
        norm = norm.replace('probable', '')
        norm = norm.replace('width', '')
        norm = norm.replace('inexact', '')
        # norm = norm.replace('Unid.', 'unidentified')
        # norm = norm.replace('unknown', 'unidentified')
        norm = norm.replace('unid.', '')
        norm = norm.replace('unknown', '')
        norm = norm.replace('unidentified', '')
        norm = re.sub('  *', ' ', norm)
        norm = norm.strip()
        return norm

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
