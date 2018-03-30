from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import os
import ubelt as ub


class WrangleConfig(object):
    def __init__(cfg, argv=None):
        # cfg.phase = ub.argval('--phase', default='1', argv=argv)
        # cfg.data_dir = ub.truepath(ub.argval('--data', default='~/data', argv=argv))

        cfg.workdir = ub.truepath(ub.argval('--work', default='~/work/viame-challenge-2018', argv=argv))
        cfg.img_root = ub.truepath(ub.argval('--img_root', default='~/data/viame-challenge-2018/phase1-imagery', argv=argv))
        cfg.annots = ub.truepath(ub.argval('--annots', default='~/data/viame-challenge-2018/phase1-annotations', argv=argv))

        # cfg.challenge_data_dir = join(cfg.data_dir, 'viame-challenge-2018')
        ub.ensuredir(cfg.workdir)

        # if cfg.phase == '0':
        #     cfg.img_root = join(cfg.challenge_data_dir, 'phase0-imagery')
        #     cfg.annot_dir = join(cfg.challenge_data_dir, 'phase0-annotations')
        # elif cfg.phase == '1':
        #     cfg.img_root = join(cfg.challenge_data_dir, 'phase1-imagery')
        #     cfg.annot_dir = join(cfg.challenge_data_dir, 'phase1-annotations')
        # elif cfg.phase == 'full':
        #     cfg.img_root = join(cfg.challenge_data_dir, 'phase0-imagery')
        #     cfg.annot_dir = join(cfg.challenge_data_dir, 'full-datasets')
        # else:
        #     raise KeyError(cfg.phase)


def download_phase0_annots():
    """
    CommandLine:
        python ~/code/baseline-viame-2018/viame_wrangler/config.py download_phase0_annots
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

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m viame_wrangler.config
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
