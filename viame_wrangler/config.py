from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import os
import ubelt as ub


class Config(ub.NiceRepr):

    def __init__(self, defaults, kwargs=None, argv=None):
        self._keys = []
        if kwargs is None:
            kwargs = {}
        for key, default in defaults.items():
            self._parse_value(key, default, kwargs, argv=argv)

    def __nice__(self):
        return ub.repr2(ub.odict([(k, self[k]) for k in self._keys]))

    def _parse_value(self, key, default, kwargs={}, argv=None):
        """ argv > kwargs > default """
        # constructor overrides default
        default = kwargs.get(key, default)
        # argv overrides constructor
        value = ub.argval('--' + key, default=default, argv=argv)
        setattr(self, key, value)
        self._keys.append(key)

    def __getitem__(self, key):
        assert key in self._keys
        return getattr(self, key)

    def __setitem__(self, key, value):
        assert key in self._keys
        return setattr(self, key, value)


class WrangleConfig(Config):
    """
    Example:
        >>> cfg = WrangleConfig()
    """
    def __init__(cfg, kw=None, argv=None):
        # cfg.phase = ub.argval('--phase', default='1', argv=argv)
        # cfg.data_dir = ub.truepath(ub.argval('--data', default='~/data', argv=argv))
        super().__init__({
            'workdir': '~/work/viame-challenge-2018',
            'img_root': '~/data/viame-challenge-2018/phase1-imagery',
            'annots': '~/data/viame-challenge-2018/phase1-annotations/*.json',
        }, kw, argv)
        for key in cfg._keys:
            cfg[key] = ub.truepath(cfg[key])


def _grabdata_girder(dpath, fname, hash, url, force=False):
    dest = os.path.join(dpath, fname)
    if not os.path.exists(dest) or force:
        from girder_client.cli import main  # NOQA
        command = 'girder-cli --api-url {} download {} {}'.format(url, hash, dpath)
        print(command)
        info = ub.cmd(command, verbout=1, verbose=1, shell=True)
        assert info['ret'] == 0
    else:
        print('already have {}'.format(dest))
    return dest


def download_phase1_annots():
    """

    References:
        http://www.viametoolkit.org/cvpr-2018-workshop-data-challenge/challenge-data-description/
        https://challenge.kitware.com/api/v1/item/5ac385f056357d4ff856e183/download
        https://challenge.kitware.com/girder#item/5ac385f056357d4ff856e183

    CommandLine:
        python ~/code/baseline-viame-2018/viame_wrangler/config.py download_phase0_annots --datadir=~/data
    """
    cfg = Config({'datadir': '~/data/viame-challenge-2018'})
    dpath = ub.truepath(cfg.datadir)
    fname = 'phase1-annotations.tar.gz'
    hash = '5ac385f056357d4ff856e183'
    url = 'https://challenge.kitware.com/api/v1'

    # FIXME: broken

    dest = _grabdata_girder(dpath, fname, hash, url, force=False)

    unpacked = join(dpath, fname.split('.')[0])
    if not os.path.exists(unpacked):
        info = ub.cmd('tar -xvzf "{}" -C "{}"'.format(dest, dpath), verbose=2, verbout=1)
        assert info['ret'] == 0


def download_phase0_annots():
    """
    CommandLine:
        python ~/code/baseline-viame-2018/viame_wrangler/config.py download_phase0_annots
    """
    cfg = Config({'datadir': ub.truepath('~/data/viame-challenge-2018')})
    dpath = cfg.datadir

    fname = 'phase0-annotations.tar.gz'
    hash = '5a9d839456357d0cb633d0e3'
    url = 'https://challenge.kitware.com/api/v1'

    dest = _grabdata_girder(dpath, fname, hash, url)

    unpacked = join(dpath, fname.split('.')[0])
    if not os.path.exists(unpacked):
        info = ub.cmd('tar -xvzf "{}" -C "{}"'.format(dest, dpath), verbose=2, verbout=1)
        assert info['ret'] == 0

    # fname = 'phase0-annotations.tar.gz'
    # dest = os.path.join(dpath, fname)
    # if not os.path.exists(dest):
    #     from girder_client.cli import main  # NOQA
    #     command = 'girder-cli --api-url https://challenge.kitware.com/api/v1 download 5a9d839456357d0cb633d0e3 {}'.format(dpath)
    #     info = ub.cmd(command, verbout=1, verbose=1, shell=True)
    #     assert info['ret'] == 0
    # unpacked = join(dpath, 'phase0-annotations')
    # if not os.path.exists(unpacked):
    #     info = ub.cmd('tar -xvzf "{}" -C "{}"'.format(dest, dpath), verbose=2, verbout=1)
    #     assert info['ret'] == 0
    # return dest

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m viame_wrangler.config
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
