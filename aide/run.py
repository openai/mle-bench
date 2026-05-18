import atexit
import os
import logging
import sys
import shutil
import time
import threading
import math
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed
from omegaconf import OmegaConf
from rich.status import Status
import torch
import random
from . import backend
from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .utils.data_preview import clean_task_desc
from openai import OpenAI
from pathlib import Path
from .utils.skill import SkillEvolver
logger = logging.getLogger("aide")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree
def run():
    cfg = load_cfg()
    logger.info(f'Starting run "{cfg.exp_name}"')
    task_desc = load_task_desc(cfg)
    task_desc = clean_task_desc(task_desc,cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)
    global_step = 0
    skill_evolver = SkillEvolver(cfg,task_desc_str)
    skill_tool = skill_evolver.get_skill_guidance(task_desc, global_step)
    logger.info(f"Preparing workspace for '{cfg.exp_name}'...")
    if cfg.workspace_dir.exists() and any(cfg.workspace_dir.iterdir()):
            logger.info(f"Workspace for '{cfg.exp_name}' already exists. Skipping workspace preparation.")
    else:
            logger.info(f"Preparing workspace for '{cfg.exp_name}'...")
            prep_agent_workspace(cfg)

    def execute_node_task(node,exec_callback):
        """
        这就是那个‘跑代码并写笔记’的厨师。
        """
        try:
            # 1. 运行代码 (调用解释器)
            # 注意：这里的 exec_callback 对应 interpreter.run
            exec_result = exec_callback(node.code, id=node.id)

            # 2. 解析结果并更新 MCTS 奖励 (把原先 step 里的逻辑搬过来)
            agent.parse_exec_result(node=node, exec_result=exec_result)
            
            metric_val = node.metric.value if node.metric.value is not None else 1e9
            if node.metric.maximize:
                reward = metric_val if metric_val != 1e9 else 0.0
            else:
                reward = math.exp(-0.5 * metric_val)

            node.update_mcts(reward)

            # 3. 线程安全地存入 Journal
            with lock:
                
                agent.journal.append(node)
            return node
        except Exception as e:
            logger.error(f"Execution task failed for node {node.id}: {e}")
            return None
    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        kaggle_tool=skill_tool,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.search.num_drafts
    logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

    lock = threading.Lock()
    completed = 0
    executor = ThreadPoolExecutor(max_workers=max_workers)

    pending_draft_nodes = []
    if initial_draft_count > 0 and total_steps > 0:
        logger.info(f"📝 Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

        def step_task_generate_only():
            logger.info(f"[step_task_generate_only] Generating draft from virtual root")
            node_list = agent.step(None , "draft")
            return node_list[0]

        draft_futures = []
        for draft_idx in range(min(initial_draft_count, total_steps)):
            logger.info(f"🔨 Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)")
            draft_futures.append(executor.submit(step_task_generate_only))
            # 等待所有 draft 并发生成完毕
        for idx, fut in enumerate(as_completed(draft_futures)):
            try:
                cur_node = fut.result()
                if cur_node:
                    pending_draft_nodes.append(cur_node)
                    logger.info(f"✅ Draft {idx + 1} code generated: node.id={cur_node.id}, added to virtual_root.children")
    
            except Exception as e:
                logger.exception(f"❌ Exception during draft {idx + 1} generation: {e}")

        logger.info(f"✅ Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")

    if pending_draft_nodes or completed < total_steps:
        logger.info(f"🚀 Phase 2: Pipelined parallel execution")
        logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
        logger.info(f"   - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            try:
                executed_node = execute_node_task(node, exec_callback)
                logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                return None
        interrupted = False
        try:
            def pipeline_step():
                """
                一个完整的流水线步骤：选择并写代码 -> 运行代码 -> 更新结果
                """
                # 1. 思考 (根据 search_policy 产生新代码节点)
                # 注意：这里需要加锁，防止多个线程同时调用 search_policy 选了同一个父节点
                with lock:
                    res = agent.search_policy()  # 只调用一次
                    current_progress = len(agent.journal)
                    if res:
                        new_nodes, policy = res  # 如果 res 有值，直接解包
                    else:
                        new_nodes = None
                        policy = "draft"
                    # 注意：这里要确保 new_nodes 能被 agent.step 处理
                skill_tool = skill_evolver.get_skill_guidance(task_desc, current_progress)
                agent.kaggle_tool = skill_tool
                new_nodes = agent.step(new_nodes, policy)

                if not new_nodes:
                    return None
                res =[]
                for node in new_nodes:
                    if node.node_type == "deadend":
                        # 直接跳过执行，但保留在结果列表中作为“墓碑”返回给主循环
                        logger.info(f"Skipping execution for deadend node: {node.id}")
                        res.append(node)
                    else:
                        # 正常节点，执行代码并更新节点状态
                        node = execute_node_task(node, exec_callback)
                        res.append(node)
                return res

            futures = set()
            for i, node in enumerate(pending_draft_nodes):
                futures.add(executor.submit(execute_draft_node, node))
                logger.info(f"📤 Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
                    time.sleep(10)
                    logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

            initial_step_tasks = min(max_workers, total_steps - completed) - len(pending_draft_nodes)
            if initial_step_tasks > 0:
                for _ in range(initial_step_tasks):
                    futures.add(executor.submit(pipeline_step))
                    logger.info(f"📤 Submitted initial step_task to fill thread pool")

            while completed < total_steps:
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                if not done:
                    #logger.info(f"⏳ Waiting... Futures: {len(futures)}, Completed: {completed}/{total_steps}")
                    continue  # timeout, no completed futures, retry (allows SIGINT handling)

                for fut in done:
                    futures.remove(fut)
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(f"✅ Task completed")
                        else:
                            logger.warning(f"⚠️  Task returned None (execution failed)")
                    except Exception as e:
                        logger.exception(f"❌ Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal)
                        global_step = len(journal)

                    if completed + len(futures) < total_steps:
                        futures.add(executor.submit(pipeline_step))

                        # --- 修复开始：兼容 List 和 单个 Node ---
                        node_str = "None"
                        if cur_node:
                            if isinstance(cur_node, list) and len(cur_node) > 0 and cur_node[-1]:
                                node_str = str(cur_node[-1].id)  # 如果是列表，取最后一个节点的 id
                            elif not isinstance(cur_node, list):
                                node_str = str(cur_node.id)
                        # --- 修复结束 ---

                        logger.info(f"📤 Submitted next task based on node {node_str}")
                    logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")

        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            interpreter.terminate_all_subprocesses()
            executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
            raise
        finally:
            if not interrupted:
                executor.shutdown(wait=True)
    else:
        logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

    interpreter.cleanup_session(-1)


if __name__ == "__main__":    
    run()
