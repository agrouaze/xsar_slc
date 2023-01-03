import os
import warnings
import zipfile
import yaml
from importlib_resources import files
from pathlib import Path
import fsspec
import aiohttp

def _load_config():
    """
    load config from default xsar/config.yml file or user ~/.xsarslc/config.yml
    Returns
    -------
    dict
    """
    user_config_file = Path('~/.xsarslc/config.yml').expanduser()
    default_config_file = files('xsarslc').joinpath('config.yml')

    if user_config_file.exists():
        config_file = user_config_file
    else:
        config_file = default_config_file

    config = yaml.load(
        config_file.open(),
        Loader=yaml.FullLoader)
    return config


config = _load_config()

def get_test_file(fname):
    """
    get test file from  https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/
    file is unzipped and extracted to `config['data_dir']`

    Parameters
    ----------
    fname: str
        file name to get (without '.zip' extension)

    Returns
    -------
    str
        path to file, relative to `config['data_dir']`

    """
    res_path = config['data_dir']
    base_url = 'https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata'
    file_url = '%s/%s.zip' % (base_url, fname)
    if not os.path.exists(os.path.join(res_path, fname)):
        warnings.warn("Downloading %s" % file_url)
        local_file = url_get(file_url)
        warnings.warn("Unzipping %s" % os.path.join(res_path, fname))
        with zipfile.ZipFile(local_file, 'r') as zip_ref:
            zip_ref.extractall(res_path)
    return os.path.join(res_path, fname)

def url_get(url, cache_dir=os.path.join(config['data_dir'], 'fsspec_cache')):
    """
    Get fil from url, using caching.

    Parameters
    ----------
    url: str
    cache_dir: str
        Cache dir to use. default to `os.path.join(config['data_dir'], 'fsspec_cache')`

    Raises
    ------
    FileNotFoundError

    Returns
    -------
    filename: str
        The local file name

    Notes
    -----
    Due to fsspec, the returned filename won't match the remote one.
    """

    if '://' in url:
        with fsspec.open(
                'filecache::%s' % url,
                https={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
                filecache={'cache_storage': os.path.join(os.path.join(config['data_dir'], 'fsspec_cache'))}
        ) as f:
            fname = f.name
    else:
        fname = url

    return fname