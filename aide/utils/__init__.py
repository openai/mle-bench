import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger("aide")


def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)


def clean_up_dataset(path: Path):
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


import shutil
import zipfile
from pathlib import Path

def extract_archives(path: Path):
    """
    递归解压所有 ZIP 文件，并处理单层目录嵌套问题。
    使用 list() 预读文件列表，避免在迭代时修改目录结构导致死循环。
    """
    # 1. 查找当前路径下所有的 zip
    # 注意：这里转成 list 是为了防止后续 unlink 或 move 干扰迭代器
    zip_files = list(path.rglob("*.zip"))
    
    if not zip_files:
        return

    for zip_f in zip_files:
        if not zip_f.exists(): continue
        
        # 定义解压的目标目录（去掉 .zip 后缀）
        target_dir = zip_f.with_suffix("")
        
        # 如果目录已存在，为了安全我们先加个后缀或跳过
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.warning(f"Target {target_dir} already exists and is not empty. Skipping.")
            zip_f.unlink() # 或者保留，取决于你的策略
            continue

        logger.info(f"Extracting {zip_f}...")
        try:
            # 执行解压
            with zipfile.ZipFile(zip_f, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # 解压成功后立即删除当前的 zip，防止重复处理
            zip_f.unlink()
            
            # 2. 处理“多余的一层目录” (Flatten)
            sub_items = list(target_dir.iterdir())
            if len(sub_items) == 1 and sub_items[0].is_dir():
                single_subdir = sub_items[0]
                # 将子目录里的所有内容移到上一层
                for item in single_subdir.iterdir():
                    # 使用 move 的路径覆盖逻辑要小心，这里建议直接 move 到 target_dir
                    shutil.move(str(item), str(target_dir))
                # 删除现在已经空的子目录
                single_subdir.rmdir()

            # 3. [关键] 递归调用：处理刚刚解压出来的嵌套 zip
            extract_archives(target_dir)

        except Exception as e:
            logger.error(f"Error extractin！！！g {zip_f}: {e}")


def preproc_data(path: Path):
    extract_archives(path)
    clean_up_dataset(path)
