"""为用户自行下载的ARDY应用已验证补丁；先核对全部文件，再备份和写入。"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil

PATCH_ROOT = Path(__file__).resolve().parent


def apply(source, patch_root, check_only=False):
    source, patch_root = Path(source).resolve(), Path(patch_root).resolve()
    manifest = json.loads((patch_root / 'manifest.json').read_text(encoding='utf-8'))
    if not (source / 'setup.py').is_file():
        raise ValueError('请先下载ARDY源码，并指定包含setup.py的目录')
    pending = []
    # 文件内容匹配才应用，兼容官方归档下载；不依赖用户必须使用Git克隆。
    for row in manifest['files']:
        target, overlay = source / row['path'], patch_root / 'files' / row['path']
        if not target.resolve().is_relative_to(source) or not overlay.resolve().is_relative_to(patch_root):
            raise ValueError('补丁路径越界')
        content = overlay.read_bytes()
        if hashlib.sha256(content).hexdigest() != row['patched_sha256']:
            raise ValueError('补丁文件校验失败：' + row['path'])
        actual = hashlib.sha256(target.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if target.exists() else None
        if actual == row['patched_content_sha256']:
            continue
        if actual != row['base_sha256'] and actual not in row.get('previous_content_sha256', []):
            raise ValueError('源码版本或本地修改冲突，未写入任何文件：' + row['path'])
        pending.append((target, overlay, row['path']))
    if not check_only and pending:
        backup = source / '.ardy-integration-backups' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        backup.mkdir(parents=True)
        added = []
        for target, overlay, relative in pending:
            if target.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            else:
                added.append(relative)
        (backup / 'added-files.json').write_text(json.dumps(added, indent=2), encoding='utf-8')
        for target, overlay, relative in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(overlay, target)
    return {'base_commit': manifest['base_commit'], 'pending': len(pending), 'check_only': check_only}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--check', action='store_true', help='只校验，不修改源码')
    parser.add_argument('--patch-root', type=Path, default=PATCH_ROOT)
    args = parser.parse_args()
    print(json.dumps(apply(args.source, args.patch_root, args.check)))


if __name__ == '__main__':
    main()
