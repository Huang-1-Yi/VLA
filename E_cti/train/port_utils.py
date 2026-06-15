# ============================================================
# PADP-VLA v1.3
# Server 端口自动避让 helper (4 路消融用)
# ============================================================

"""E_cti.train.port_utils —— 4 路消融的 server 端口自动避让工具。

设计:
  - 默认端口 8765 (PADP_v3 历史约定)
  - 启动 server 前, 从 preferred 端口往下找空闲端口
    (8765 → 8764 → 8763 → ...)
  - 找到空闲端口后返回, server 端用它监听

用法:
  from E_cti.train.port_utils import find_free_port
  port = find_free_port(8765)
  serve_forever(host, port, ...)

为什么是 -1 不是 +1?
  - 4 路消融 (1.1_10D, 1.1_7D, 1.2_10D, 1.2_7D) 在同一服务器上跑
  - 用户多 GPU, 每路一个 GPU, 但需要避免 server 端口冲突
  - 8765 是 PADP 黄金命令历史端口, 优先用
  - 如果被占用, 8764/8763/... 是自然延续, 不会跟其它服务撞 (其它服务通常 8000/9000 系列)
"""
import socket


def find_free_port(preferred: int = 8765, max_try: int = 100) -> int:
    """从 preferred 端口开始往下找空闲端口。

    Args:
        preferred: 首选端口 (默认 8765)
        max_try:   最大尝试次数 (默认 100, 覆盖 8765-8466)

    Returns:
        找到的空闲端口 (int)

    Raises:
        OSError: 范围内全部被占用
    """
    for offset in range(max_try):
        port = preferred - offset
        if port <= 0:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                # 立即释放, 让 server 端用它监听
                return port
            except OSError:
                # 端口被占用, 继续往下找
                continue
    raise OSError(
        f"No free port in range [{preferred - max_try + 1}, {preferred}]. "
        f"全部 {max_try} 个候选端口都被占用。"
    )


if __name__ == "__main__":
    # 单测
    port = find_free_port()
    print(f"✅ Found free port: {port}")
    # 验证它真的可用 (bind + listen)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", port))
        s.listen(1)
        print(f"✅ Verified: port {port} can listen")
