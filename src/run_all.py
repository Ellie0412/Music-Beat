#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd

# 把 src 加入 PYTHONPATH，无需安装包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import load_clean as lc
from src import eda as eda
from src import composite as comp
from src import corr as corr
from src import model as model

# ---------- 路径常量 ----------
PROJ  = Path("/Users/ellie/Documents/Assignments/university-python/music_beat")
FIG   = PROJ / "figures"
LOG_F = PROJ / "run.log"

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[logging.FileHandler(LOG_F, mode="w"),
              logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ---------- 工具函数 ----------
def log_step(msg):
    """带颜色的步骤分隔"""
    log.info("\033[96m" + "=" * 60 + "\033[0m")
    log.info("\033[96m>>> %s\033[0m", msg)


def safe_run(func, *args, **kw):
    """捕获异常并继续"""
    try:
        return func(*args, **kw)
    except Exception as exc:
        log.error("步骤失败: %s", exc, exc_info=True)
        return None


# ---------- 主流程 ----------
def main(skip_train: bool = False, verbose: bool = False):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log_step("1. 加载与类型转换")
    train, test, test_id = lc.load()
    train, test = lc.coerce_numeric(train, test)

    log_step("2. EDA 概览 & BPM 分布图")
    safe_run(eda.quick_overview, train, test)
    safe_run(eda.bpm_dist, train)
    safe_run(eda.batch_plot_train_test, train, test)
    
    log_step("3. 生成组合特征 & 单特征大图")
    train_ext = comp.add_composite(train)
    test_ext  = comp.add_composite(test)
    safe_run(comp.batch_plot_composite, train_ext, test_ext)

    log_step("4. 相关性热力图")
    full = pd.concat([train_ext, test_ext], ignore_index=True)
    safe_run(corr.plot_lower_corr, full, "Merged_All_Features")
    safe_run(corr.plot_lower_corr, test_ext, "Test_All_Features")
    safe_run(corr.plot_joint_corr, train_ext, test_ext)

    if skip_train:
        log.info(">>> 训练已跳过，所有图片已保存至 %s", FIG)
        return

    log_step("5. 训练 CatBoost + SHAP")
    X = train_ext.drop(columns=["BeatsPerMinute"])
    y = train_ext["BeatsPerMinute"]
    safe_run(model.train_eval_save, X, y)

    log.info(">>> 全部完成！日志查看：%s", LOG_F)


# ---------- 命令行入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="music_beat 端到端流程")
    parser.add_argument("-s", "--skip-train", action="store_true", help="跳过训练与 SHAP")
    parser.add_argument("-v", "--verbose",    action="store_true", help="打印调试信息")
    args = parser.parse_args()
    main(skip_train=args.skip_train, verbose=args.verbose)
