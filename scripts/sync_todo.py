"""sync_todo.py —— 对比 Todo 清单和 VLA 实际文件,自动生成同步报告。

用法:
    cd VLA
    python scripts/sync_todo.py                                  # 默认对比 执行计划/落地方案v4-1的Todo清单.md
    python scripts/sync_todo.py --todo path/to/other.md          # 对比其它 todo
    python scripts/sync_todo.py --apply-stub --todo path/to/other.md  # 自动 stub 缺失文件(只创建空 __init__.py 等)

输出:
    === 同步报告 ===
    [OK] VLA 中存在: 60 个 Todo 计划文件中,有 60 个已存在
    [X] 缺失: 0
    [WARN]  VLA 实际存在但 Todo 没列: 5

退出码:
    0: 完美同步
    1: 有缺失文件(需要新建)
    2: 有孤儿子(可能是 Todo 漏了,也可能是有意为之)
"""
import argparse
import re
import sys
from pathlib import Path


def parse_todo_files(todo_path: Path, repo_root: Path) -> list:
    """从 Todo 清单 markdown 里抽出所有 `- [ ] **path/to/file.py` 形式。

    兼容写法:
        - `- [ ] **VLA/A_common/...`  ← 文件路径带 VLA/ 前缀
        - `- [ ] **A_common/...`     ← 相对 VLA/ 的路径
        - `- [ ] padp_policy.py`      ← 嵌套列表中的短路径(从最近的 ## 4. B_model/ 上下文推断)
    """
    files = set()
    current_dir = None  # 当前最近的 ## 4.x 标题

    with open(todo_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        # 跳过 strikethrough 行(废弃项)
        if "~~" in line:
            continue

        m = re.match(r"^#{2,4}\s+\S+.*[/:]([A-G_a-z_]+)/", line)
        if m:
            current_dir = m.group(1)
            continue

        # 匹配 - [ ] **path**
        m2 = re.match(r"^\s*-\s*\[\s*\]\s+\*?\*?([^\s*]+(?:\.[A-Za-z]+)?)\*?\*?", line)
        if not m2:
            continue
        raw = m2.group(1)

        # 路径清洗:去掉 md 标记
        raw = raw.strip("`*")

        # 如果已经有 VLA/ 前缀 → 直接用
        if raw.startswith("VLA/"):
            rel = raw[4:]
        elif "/" in raw or raw.endswith((".py", ".yaml", ".yml", ".md", ".txt", ".json")):
            rel = raw
        else:
            # 短路径(如 padp_policy.py)→ 用最近的 ## 标题推断父目录
            if current_dir:
                rel = f"{current_dir}/{raw}"
            else:
                continue

        files.add(rel)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--todo",
        type=Path,
        default=Path("执行计划/落地方案v4-1的Todo清单.md"),
        help="Todo 清单 markdown 路径(相对 VLA/ 仓库根)",
    )
    parser.add_argument(
        "--apply-stub",
        action="store_true",
        help="对所有 missing 文件创建空 stub(只创建空文件,不写代码)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 格式(便于 AI 解析)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    todo_path = repo_root / args.todo

    if not todo_path.exists():
        print(f"[X] Todo 文件不存在: {todo_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[FILE] 解析 Todo 文件: {todo_path.relative_to(repo_root)}")
    todo_files = parse_todo_files(todo_path, repo_root)
    print(f"   → 找到 {len(todo_files)} 个文件")

    # 扫描 VLA 实际存在的 .py / .yaml / .md / .json / .txt 文件
    actual_files = set()
    skip_dirs = {".git", "__pycache__", "执行计划"}
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if any(d in p.parts for d in skip_dirs):
            continue
        if p.suffix not in (".py", ".yaml", ".yml", ".md", ".json", ".txt", ".sh"):
            continue
        rel = p.relative_to(repo_root).as_posix()
        actual_files.add(rel)

    print(f"[DIR] VLA 实际文件: {len(actual_files)} 个\n")

    todo_set = set(todo_files)
    actual_set = actual_files

    # === 三向 diff ===
    missing_in_actual = sorted(todo_set - actual_set)   # Todo 写要,但 VLA 没有
    extra_in_actual = sorted(actual_set - todo_set)     # VLA 有,但 Todo 没列

    if args.json:
        import json
        print(json.dumps({
            "todo_file": str(args.todo),
            "todo_count": len(todo_set),
            "actual_count": len(actual_set),
            "missing": missing_in_actual,
            "extra": extra_in_actual,
        }, ensure_ascii=False, indent=2))
        sys.exit(0 if not missing_in_actual else 1)

    print("=" * 60)
    print("  同步报告")
    print("=" * 60)
    print(f"  Todo 计划: {len(todo_set)}")
    print(f"  VLA 实际: {len(actual_set)}")
    print()
    if missing_in_actual:
        print(f"[X] Todo 写要,但 VLA 缺({len(missing_in_actual)} 个):")
        for p in missing_in_actual:
            print(f"   - {p}")
        if args.apply_stub:
            print()
            print("[FIX] 正在创建 stub ...")
            for p in missing_in_actual:
                fp = repo_root / p
                fp.parent.mkdir(parents=True, exist_ok=True)
                if p.endswith(".py"):
                    fp.write_text('"""\nTODO: 由 sync_todo.py 自动生成的 stub。\n请按 Todo 清单与 AI 提示词填实。\n"""\n', encoding="utf-8")
                else:
                    fp.touch()
                print(f"   ✓ {p}")
    else:
        print("[OK] 所有 Todo 计划文件都已存在")

    print()
    if extra_in_actual:
        print(f"[WARN]  VLA 实际存在,但 Todo 没列({len(extra_in_actual)} 个):")
        for p in extra_in_actual:
            print(f"   - {p}")
    else:
        print("[OK] 没有 Todo 没列的多余文件")

    print()
    print("=" * 60)
    if missing_in_actual:
        print(f"[NOTE] 状态: [X] 缺 {len(missing_in_actual)} 个文件")
        sys.exit(1)
    elif extra_in_actual:
        print(f"[NOTE] 状态: [WARN] Todo 漏了 {len(extra_in_actual)} 个(可能是合理的)")
        sys.exit(2)
    else:
        print("[NOTE] 状态: [OK] 完美同步")
        sys.exit(0)


if __name__ == "__main__":
    main()
