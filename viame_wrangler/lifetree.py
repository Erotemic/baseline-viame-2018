from Bio import Entrez
import ubelt as ub
import time
import tqdm


class Entry(ub.NiceRepr):
    def __init__(entry, code, common_names=[], note=None):
        entry.code = code
        entry.common_names = common_names
        entry.note = note
        entry.lineage = None

    @property
    def parts(entry):
        return [p.split(':') if ':' in p else (None, p)
                for p in entry.code.split(' ')]

    def partial_path(entry):
        # Infer what we can about the ranks of each part in the code
        parts = list(entry.parts)
        # The very last part might be a species
        path = []
        for rank, value in parts[:-1]:
            path.append((rank, value))

        rank, value = parts[-1]
        if value.islower() or rank == 'species':
            assert value.islower(), 'species is lowercase'
            assert rank is None or rank == 'species'
            assert len(parts) > 1
            rank2, value2 = parts[-2]
            assert rank2 is None or rank2 == 'genus'
            assert value2
            path.append(('species', '{} {}'.format(value2, value)))
        else:
            assert value[0].isupper()
            path.append((rank, value))
        return path

    @property
    def id(entry):
        # Transform the code into a node-id
        return entry.partial_path()[-1][1]

    def __nice__(entry):
        # if entry.lineage:
        #     return ' '.split([v for k, v in entry.lineage.items()])
        nice = ' '.join([p.split(':')[-1] for p in entry.code.split(' ')])
        return nice
        # return entry.code


class Lineage(ub.NiceRepr):
    """
    https://en.wikipedia.org/wiki/Taxonomic_rank
    """
    PRIMARY_RANKS = [
        'domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus',
        'species',
    ]
    RANKS = [
        'life',
        'domain',
        'kingdom', 'infrakingdom',
        'phylum', 'subphylum', 'infraphylum', 'microphylum',
        'superclass', 'class', 'subclass', 'infraclass', 'parvclass',
        'superorder', 'order', 'suborder', 'infraorder',
        'superfamily', 'family', 'subfamily',
        'tribe', 'subtribe',
        'genus',
        'species',
    ]
    def __init__(self, d=None):
        self.data = ub.odict([(r, None) for r in self.RANKS])
        if d:
            self.update(d)

    def update(self, other):
        for k, v in other.items():
            assert k in self.data
            old = self.data[k]
            assert old is None or old == v
            self.data[k] = v
        if self.data['species'] and ' ' in self.data['species']:
            genus, species = self.data['species'].split(' ')
            if self.data['genus']:
                assert self.data['genus'] == genus
            self.data['genus'] = genus
            self.data['species'] = species

    def items(self):
        return self.data.items()

    def code(self):
        return ' '.join(['{}:{}'.format(k, v) for k, v in self.data.items() if v])

    def lineage_path(node, fix_species=True):
        path = []
        for rank in node.RANKS:
            v = None
            if rank == 'species' and fix_species:
                v = node.data[rank]
                if v is not None:
                    v = node.data['genus'] + ' ' + v
            else:
                v = node.data[rank]
            if v is not None:
                path.append(v)
        return path

    def linage_edges(node):
        edges = []
        for u, v in ub.iter_window(node.lineage_path(), 2):
            edges += [(u, v)]
        return edges

    @property
    def id(self):
        return list(self.lineage_path())[-1]

    def __nice__(self):
        return ' '.join(self.lineage_path(False))

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()


class LifeCatalog(object):
    """
    Keep records mapping scientific names to common names so we can build out a
    proper classification heirachy from lineages pulled from the NCBI database.

    domain

    kingdom (regnum)
        subregnum

    phylum or division (divisio, phylum)
        subdivisio or subphylum

    class (classis)
        subclassis

    order (ordo)
        subordo
    _________________

    family (familia)
            subfamilia
        tribe (tribus)
            subtribus

    genus (genus)
            subgenus
        section (sectio)
            subsection
        series (series)
            subseries

    species (species)
            subspecies
        variety (varietas)
            subvarietas
        form (forma)
            subforma

    Notes
    -----
    Clade is not a taxonomic ranking, it's a description of a group of
    organisms. A clade is all the descendants of a common ancestor


    References:
        http://ctdbase.org/detail.go?type=taxon&acc=860188

    """

    scientific_aliases = [
        ('phylum:Arthropoda', 'phylum:Euarthropoda')
    ]

    def __init__(self):
        self.entries = [
            # non-fish
            Entry('domain:Eukarya kingdom:Animalia phylum:Chordata', 'cordate'),
            Entry('domain:Eukarya kingdom:Animalia phylum:Chordata subphylum:Vertebrata', 'vertebrate'),
            Entry('domain:Eukarya kingdom:Animalia phylum:Arthropoda', ['arthropod', 'Euarthropoda']),
            Entry('domain:Eukarya kingdom:Animalia phylum:Mollusca', 'mollusc'),
            Entry('domain:Eukarya kingdom:Animalia phylum:Echinodermata', 'echinoderm'),

            Entry('phylum:Chordata class:Mammalia order:Primates family:Hominidae genus:Homo species:sapiens', 'human'),
            Entry('phylum:Chordata class:Ascidiacea order:Aplousobranchia family:Didemnidae genus:Didemnum', 'Didemnum'),

            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda infraorder:Brachyura', 'crab'),
            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda genus:Cancer borealis', 'jonah crab'),
            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda genus:Cancer irroratus', 'rock crab'),
            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda genus:Chionoecetes bairdi', ['tanner crab', 'bairdi crab']),
            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda family:Nephropidae', 'lobster'),
            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda family:Nephropidae genus:Homarus americanus', 'american lobster'),
            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda suborder:Dendrobranchiata', 'decapod shrimps'),
            Entry('phylum:Arthropoda subphylum:Crustacea class:Malacostraca order:Decapoda suborder:Pleocyemata infraorder:Caridea', 'caridean shrimp'),

            Entry('phylum:Mollusca class:Cephalopoda order:Octopoda', 'unclassified octopus'),
            Entry('phylum:Mollusca class:Gastropoda', 'unclassified snail'),
            Entry('phylum:Mollusca class:Gastropoda infraclass:Euthyneura superorder:Nudipleura order:Nudibranchia', ['Nudibranchs', 'opistobranch sea slug']),
            Entry('phylum:Mollusca class:Gastropoda family:Buccinidae genus:Buccinum undatum', 'waved whelk'),
            Entry('phylum:Mollusca class:Bivalivia order:Osteroida family:Pectinidae', 'unclassified scallop'),

            # echinoderms
            Entry('phylum:Echinodermata class:Holothuroidea genus:Psolus', 'sea cucumber'),
            Entry('Echinodermata Holothuroidea order:Dendrochirotida Psolus squamatus', 'squamatus sea cucumber'),
            Entry('Echinodermata superclass:Asterozoa class:Asteroidea', 'starfish'),
            Entry('Echinodermata superclass:Asterozoa class:Asteroidea genus:Rathbunaster californicus', 'californicus starfish'),

            # Flat fish
            Entry('phylum:Chordata class:Actinopterygii order:Pleuronectiformes', 'unclassified flat fish'),
            Entry('Pleuronectiformes family:Pleuronectidae genus:Glyptocephalus zachirus', 'rex sole'),
            Entry('Pleuronectiformes family:Pleuronectidae genus:Eopsetta jordani', 'petrale sole'),
            Entry('Pleuronectiformes family:Pleuronectidae genus:Parophrys vetulus', 'english sole'),
            Entry('Pleuronectiformes family:Pleuronectidae genus:Hippoglossus stenolepis', 'Pacific Halibut'),
            Entry('Pleuronectiformes family:Pleuronectidae genus:Atheresthes stomias', 'Arrowtooth Flounder'),
            Entry('Pleuronectiformes family:Pleuronectidae genus:Lepidopsetta bilineata', 'Rock Sole'),
            Entry('Pleuronectiformes family:Pleuronectidae genus:Hippoglossoides elassodon', 'Flathead Sole'),

            Entry('Pleuronectiformes family:Soleidae genus:Solea solea', ['dover sole', 'black sole', 'common sole']),

            # Round Fish - any fish that is not a flatfish
            Entry('phylum:Chordata class:Actinopterygii order:Anacanthini', note='ray-finned fish'),
            Entry('phylum:Chordata class:Actinopterygii order:Carcharhiniformes', 'Carcharhiniforme'),
            Entry('phylum:Chordata class:Actinopterygii order:Chimaeriformes', 'Chimaeriforme'),
            Entry('phylum:Chordata class:Actinopterygii order:Clupeiformes', 'Clupeiformes'),
            Entry('phylum:Chordata class:Actinopterygii order:Gadiformes', 'Gadiforme'),
            Entry('phylum:Chordata class:Actinopterygii order:Lophiiformes', 'Lophiiformes'),
            Entry('phylum:Chordata class:Actinopterygii order:Osmeriformes', 'Osmeriforme'),
            Entry('phylum:Chordata class:Actinopterygii order:Perciformes', 'Perciforme'),
            Entry('phylum:Chordata class:Actinopterygii order:Rajiformes', 'Rajiforme'),
            Entry('phylum:Chordata class:Actinopterygii order:Scorpaeniformes', 'Scorpaeniforme'),

            Entry('order:Lophiiformes family:Lophius', 'monkfish'),
            Entry('order:Rajiformes family:Rajidae', 'skate'),
            Entry('order:Rajiformes family:Rajidae genus:Dipturus species:oxyrinchus', 'longnose skate'),
            Entry('order:Carcharhiniformes family:Carcharhinidae', 'requiem shark'),
            Entry('order:Carcharhiniformes family:Carcharhinidae genus:Carcharhinus plumbeus', 'sandbar shark'),
            Entry('order:Osmeriformes family:Osmeridae', 'Smelt'),

            Entry('Perciformes family:Pholidichthyidae genus:Pholidichthys leucotaenia', 'convict worm'),
            Entry('Perciformes family:Lutjanidae genus:Pristipomoides filamentosus', 'crimson jobfish'),
            Entry('Perciformes family:Lutjanidae genus:Pristipomoides sieboldii', 'lavandar jobfish'),

            Entry('Perciformes family:Zoarcidae genus:Lycodes', 'eelpout'),
            Entry('Perciformes family:Zoarcidae Lycodes diapterus', 'black eelpout'),
            Entry('Perciformes family:Zoarcidae Lycodes pacificus', 'blackbelly eelpout'),
            Entry('Perciformes family:Zaproridae genus:Zaprora silenus', 'Prowfish'),
            Entry('Perciformes suborder:Zoarcoidei genus:Stichaeidae', ['Prickleback', 'Stichaeidae']),

            Entry('Perciformes family:Bathymasteridae', 'Ronquil'),
            Entry('Perciformes family:Bathymasteridae genus:Bathymaster signatus', 'Searcher'),

            Entry('Chimaeriformes family:Chimaeridae genus:Hydrolagus colliei', 'spotted ratfish'),
            Entry('Clupeiformes family:Clupeidae genus:Clupea harengus', 'Herring'),

            Entry('Gadiformes family:Merlucciidae genus:Merluccius productus', 'north pacific hake'),
            Entry('Gadiformes family:Gadidae genus:Pollachius', 'Pollock'),
            Entry('Gadiformes family:Gadidae', 'Gadoid'),
            Entry('Gadiformes family:Gadidae genus:Gadus macrocephalus', 'Pacific Cod'),

            # rockfish
            Entry('Scorpaeniformes family:Sebastidae genus:Sebastes', 'rockfish'),
            Entry('Scorpaeniformes Sebastes borealis', 'Shortraker Rockfish'),
            Entry('Scorpaeniformes Sebastes brevispinis', 'Silvergray Rockfish'),
            Entry('Scorpaeniformes Sebastes ciliatus', 'Dusky Rockfish'),
            Entry('Scorpaeniformes Sebastes crameri', 'Darkblotched Rockfish'),
            Entry('Scorpaeniformes Sebastes elongatus', 'greenstriped rockfish'),
            Entry('Scorpaeniformes Sebastes emphaeus', ['puget sound rockfish', 'pygmy rockfish']),
            Entry('Scorpaeniformes Sebastes helvomaculatus', 'rosethorn rockfish'),
            Entry('Scorpaeniformes Sebastes maliger', 'Quillback Rockfish'),
            Entry('Scorpaeniformes Sebastes melanops', 'Black Rockfish'),
            Entry('Scorpaeniformes Sebastes melanostictus', 'Blackspotted Rockfish'),
            Entry('Scorpaeniformes Sebastes polyspinis', 'Northern Rockfish'),
            Entry('Scorpaeniformes Sebastes proriger', 'redstripe rockfish'),
            Entry('Scorpaeniformes Sebastes ruberrimus', 'yelloweye rockfish'),
            Entry('Scorpaeniformes Sebastes saxicola', 'stripetail rockfish'),
            Entry('Scorpaeniformes Sebastes variegatus', 'Harlequin Rockfish'),
            Entry('Scorpaeniformes Sebastes zacentrus', 'Sharpchin Rockfish'),
            Entry('Scorpaeniformes Sebastes alutus', 'Pacific Ocean Perch'),

            Entry('Scorpaeniformes family:Sebastidae genus:Sebastolobus', 'Thornyhead'),
            Entry('Scorpaeniformes family:Sebastidae genus:Sebastolobus altivelis', 'Longspine thornyhead'),
            Entry('Scorpaeniformes family:Sebastidae genus:Sebastolobus alascanus', 'Shortspine thornyhead'),

            Entry('Scorpaeniformes suborder:Cottoidei superfamily:Cottoidea', 'sculpin'),
            Entry('Scorpaeniformes Cottoidea family:Cottidae genus:Icelinus filamentosus', 'threadfin sculpin'),
            Entry('Scorpaeniformes Cottoidea family:Cottidae genus:Hemilepidotus hemilepidotus', 'Irish Lord'),

            Entry('Scorpaeniformes family:Agonidae', 'poacher'),
            Entry('Scorpaeniformes Agonidae genus:Aspidophoroides monopterygius', 'alligatorfish'),

            Entry('Scorpaeniformes superfamily:Cyclopteroidea family:Liparidae', 'Snailfish'),

            Entry('Scorpaeniformes family:Anoplopomatidae genus:Anoplopoma fimbria', 'Sablefish'),

            Entry('Scorpaeniformes suborder:Hexagrammoidei family:Hexagrammidae', ['Hexagrammidae', 'greenling']),  # incorporates greenlings
            Entry('Scorpaeniformes Hexagrammidae', ['Greenling']),
            Entry('Scorpaeniformes Hexagrammidae genus:Hexagrammos decagrammus', 'Kelp Greenling'),
            Entry('Scorpaeniformes Hexagrammidae genus:Pleurogrammus monopterygius', 'Atka Mackerel'),
            Entry('Scorpaeniformes Hexagrammidae genus:Ophiodon elongatus', 'Lingcod'),
        ]

    def parse_entries(self):
        r"""
        CommandLine:
            python -m viame_wrangler.lifetree parse_entries

        References:
            https://github.com/HadrienG/taxadb

        Example:
            >>> from viame_wrangler.lifetree import *  # NOQA
            >>> self = LifeCatalog()
            >>> G = self.parse_entries()
        """
        import networkx as nx
        dag = nx.DiGraph()

        def _setrank(node_id, rank):
            """ checks if rank is consistent or sets it """
            node = dag.node[node_id]
            old = node.get('rank', None)
            if old is not None:
                assert old == rank, 'inconsistent rank labels'
            node['rank'] = rank

        def _setcommon(node_id, common_names):
            """ checks if common_names are consistent or adds to it """
            if common_names is not None:
                if not ub.iterable(common_names):
                    common_names = [common_names]
            else:
                common_names = []
            old =  dag.node[node_id].get('common_names', set())
            new = {c.lower() for c in common_names}
            dag.node[node_id]['common_names'] = old.union(new)

        for entry in tqdm.tqdm(self.entries):
            # For each entry, parse the parts of the coded scientific name to
            # build up the nodes in the tree structure.
            node_id = entry.id
            dag.add_node(node_id)
            _setcommon(node_id, entry.common_names)

            for pu, pv in ub.iter_window(entry.partial_path(), 2):
                urank, u = pu
                vrank, v = pv
                dag.add_edge(u, v)
                if urank:
                    _setrank(u, urank)
                if vrank:
                    _setrank(v, vrank)

        for node_id in dag.nodes():
            node = dag.node[node_id]
            common_names = list(node.get('common_names', []))
            rank = node['rank']
            node['label'] = rank + ':' + node_id
            if common_names:
                node['label'] = node['label'] + '\n' + ub.repr2(common_names, nl=0)

        # TODO: Try and use the NCBI database to find the rank and lineage of
        # items that were not given
        # db = MyTaxdb('taxadb.sqlite')

        assert nx.is_directed_acyclic_graph(dag)
        G = nx.algorithms.dag.transitive_reduction(dag)
        assert nx.is_tree(G), 'should reduce to a tree'

        # transfer node attributes
        for node_id, data in dag.nodes(data=True):
            G.nodes[node_id].update(data)

        # Find the full lineage of each entry
        Gr = G.reverse()
        def make_lineage(node_id):
            path = [node_id] + [e[1] for e in nx.bfs_edges(Gr, node_id)]
            lineage = ub.odict([(G.node[n]['rank'], n) for n in path][::-1])
            return Lineage(lineage)

        for entry in self.entries:
            node_id = entry.id
            entry.lineage = make_lineage(node_id)

        for entry in self.entries:
            print('entry.lineage = {}'.format(entry.lineage.code()))

        # The leafs are the finest-grained categories
        leafs = [n for n in G.nodes() if G.out_degree(n) == 0]
        fine_categories = []
        for node_id in leafs:
            lineage = make_lineage(node_id)
            fullname = ' '.join(list(lineage.lineage_path(False)))
            fine_categories.append(fullname)
        print(ub.repr2(sorted(fine_categories)))

        if False:
            import plottool as pt
            G.graph['rankdir'] = 'LR'
            pt.dump_nx_ondisk(G, 'classes.png')

            G = dag
            G.graph['rankdir'] = 'LR'
            pt.dump_nx_ondisk(G, 'dag-classes.png')
            # pt.qtensure()
            # pt.show_nx(G, layoutkw={'prog': 'dot'}, arrow_width=.0001)
        return G


class QueryGaurd(object):
    def __init__(self):
        self.timer = None

    def wait(self):
        if self.timer is None:
            self.timer = ub.Timer().tic()
        else:
            while self.timer.toc() < 3:
                time.sleep(.1)
            self.timer = None


GAURD = QueryGaurd()


def names():
    """
    pip install biopython

    https://github.com/HadrienG/taxadb

    https://stackoverflow.com/questions/16504238/attempting-to-obtain-taxonomic-information-from-biopython
    """
    # # Always tell NCBI who you are
    Entrez.email = ub.cmd('git config --global --get user.email')['out'].strip()

    def get_tax_info(species):
        """
        to get data from ncbi taxomomy, we need to have the taxid. we can
        get that by passing the species name to esearch, which will return
        the tax id.

        once we have the taxid, we can fetch the record
        """
        species = species.replace(' ', '+').strip()
        GAURD.wait()
        search = Entrez.esearch(term=species, db="taxonomy", retmode="xml")
        record = Entrez.read(search)
        ids = record['IdList']
        assert len(ids) == 1
        taxid = ids[0]
        search = Entrez.efetch(id=taxid, db="taxonomy", retmode="xml")
        taxdata = Entrez.read(search)
        return taxdata

    from viame_wranger.cats_2018 import get_coarse_mapping  # NOQA

    # from taxon_names_resolver import Resolver
    # from taxon_names_resolver import taxTree

    # terms = ['Homo sapiens', 'Gorilla gorilla', 'Pongo pongo', 'Macca mulatta',
    #          'Mus musculus', 'Ailuropoda melanoleuca', 'Ailurus fulgens',
    #          'Chlorotalpa tytonis', 'Arabidopsis thaliana',
    #          'Bacillus subtilus']

    # # RESOLVE
    # resolver = Resolver(terms=terms, datasource="NCBI")

    # resolver = Resolver(terms=['Pectinidae'], datasource="NCBI")
    # resolver.main()

    # ranks = resolver.retrieve('classification_path_ranks')
    # idents = resolver.retrieve('query_name')
    # lineages = resolver.retrieve('classification_path')
    # for rank, lineage in zip(ranks, lineages):

    #     print(' '.join(rank))
    #     print(' '.join(lineage))
    # treestring = taxTree(idents, ranks, lineages)


def build_database():
    names_file = 'taxadb/names.dmp'
    print("Parsing %s" % str(names_file))
    names_data = []
    with open(names_file, 'r') as f:
        for line in f:
            if 'scientific name' in line:
                line = line.replace('\n', '').replace('\t', '')
                parts = line.split('|')[0:-1]
                tax_id, name_txt, unique_name, name_class = parts
                if not unique_name:
                    unique_name = name_txt
                record = {
                    'ncbi_taxid': parts[0],
                    'tax_name': parts[1],
                    'unique_name': unique_name,
                    'name_class': name_class,
                }
                names_data.append(record)

    name_to_info = {}
    id_to_infos = ub.ddict(list)
    for info in names_data:
        name = info['unique_name']
        assert name not in name_to_info
        name_to_info[name] = info
        id_to_infos[info['ncbi_taxid']].append(info)

    for k, v in id_to_infos.items():
        if len(v) > 1:
            print(k)


class TaxRow(ub.NiceRepr):
    def __init__(self, info):
        self.__dict__.update(**info)

    def __nice__(self):
        return ub.repr2(self.__dict__, nl=0)

    def __hash__(self):
        return self.ncbi_taxid

    def __lt__(self, other):
        return self.ncbi_taxid < other.ncbi_taxid

    def __eq__(self, other):
        return self.ncbi_taxid == other.ncbi_taxid


class MyTaxdb(object):
    def __init__(self, fpath):
        from taxadb.taxid import TaxID
        self.taxid = TaxID(dbtype='sqlite', dbname=fpath)
        self.db = self.taxid.database
        self.cur = self.db.get_cursor()
        # db.get_tables()
        # db.get_columns('taxa')
        # db.get_columns('accession')

    def search(self, term):
        columns = ['parent_taxid', 'ncbi_taxid', 'tax_name', 'lineage_level']
        self.cur.execute(ub.codeblock(
            '''
            SELECT {}
            FROM taxa
            WHERE tax_name LIKE "%{}%"
            ''').format(','.join(columns), term))
        result = self.cur.fetchone()
        while result:
            if result:
                info = dict(zip(columns, result))
                yield TaxRow(info)
            result = self.cur.fetchone()

    def lookup(self, ncbi_taxid):
        columns = ['parent_taxid', 'ncbi_taxid', 'tax_name', 'lineage_level']
        self.cur.execute(ub.codeblock(
            '''
            SELECT {}
            FROM taxa
            WHERE ncbi_taxid={}
            ''').format(','.join(columns), ncbi_taxid))
        result = self.cur.fetchone()
        if not result:
            raise KeyError(ncbi_taxid)
        info = TaxRow(dict(zip(columns, result)))
        return info
