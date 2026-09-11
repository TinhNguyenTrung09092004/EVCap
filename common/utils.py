"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from urllib.parse import urlparse

from common.registry import registry


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_cache_path(rel_path):
    return os.path.expanduser(os.path.join(registry.get_path("cache_root"), rel_path))


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
