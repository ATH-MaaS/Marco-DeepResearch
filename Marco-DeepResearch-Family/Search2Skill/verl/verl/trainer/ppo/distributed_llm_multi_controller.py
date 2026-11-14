import requests
import logging
from verl.utils import ray_compat as ray
from concurrent.futures import ThreadPoolExecutor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DistLLMController")

class DistributedLLMController:
    def __init__(self, port=48899, sgl_port=30003):
        """
        初始化多机控制器
        :param port: Agent 监听端口 (请确保 llm_controller.py 也是这个端口)
        :param sgl_port: SGLang 服务端口
        """
        self.agent_port = port
        self.sgl_port = sgl_port
        
        # 1. 自动获取集群所有节点的 IP
        self.node_ips = self._get_node_ips()
        logger.info(f"[Controller] Detected {len(self.node_ips)} nodes: {self.node_ips}")

    def _get_node_ips(self):
        """利用 Ray API 获取集群内网 IP 列表"""
        if not ray.is_initialized():
            # 尝试连接现有的 Ray 集群
            ray.init(ignore_reinit_error=True, address='auto')
        
        ips = []
        for node in ray.nodes():
            if node["Alive"]:
                # NodeManagerAddress 通常是节点在局域网的 IP
                ips.append(node["NodeManagerAddress"])
        return list(set(ips))

    def wake_up(self, start_command_template=None):
        """
        唤醒集群内所有节点的 SGLang 服务
        """
        logger.info(">>> [Controller] Sending WAKE_UP signal to ALL nodes...")
        
        # 定义发送给单个节点的逻辑
        def _send_wake_to_node(ip):
            url = f"http://{ip}:{self.agent_port}/wake_up"
            try:
                logger.info(f"-> Contacting Agent at {ip}...")
                
                # [优化] 禁用代理，防止内网请求走代理
                # 设置超时 900s (15分钟)
                response = requests.post(
                    url, 
                    timeout=3500, 
                    proxies={"http": None, "https": None} 
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # 兼容 started 和 already_running 两种成功状态
                    if data.get("status") in ["started", "already_running"]:
                        logger.info(f"✅ Node {ip} is READY! (PID: {data.get('pid')})")
                        return True, None
                    else:
                        error_msg = f"Agent failure on {ip}: {data}"
                        logger.error(error_msg)
                        return False, error_msg
                else:
                    return False, f"HTTP {response.status_code}"
            except Exception as e:
                logger.error(f"❌ Failed to contact node {ip}: {e}")
                return False, str(e)

        # 2. 并发向所有节点发送请求
        success_count = 0
        errors = []
        
        with ThreadPoolExecutor(max_workers=len(self.node_ips)) as executor:
            # 提交任务
            futures = {executor.submit(_send_wake_to_node, ip): ip for ip in self.node_ips}
            
            # 获取结果
            for future in futures:
                ip = futures[future]
                try:
                    success, msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        errors.append(f"{ip}: {msg}")
                except Exception as e:
                    errors.append(f"{ip} Exception: {str(e)}")

        # 3. 判定结果：必须全部成功才算成功
        if success_count == len(self.node_ips):
            logger.info(">>> [Controller] 🎉 All nodes started successfully!")
        else:
            error_summary = "; ".join(errors)
            logger.error(f">>> [Controller] 💀 Cluster startup FAILED! Details: {error_summary}")
            # 这里抛出异常会中断训练，防止后续 Rollout 报错
            raise RuntimeError(f"Cluster startup failed: {error_summary}")

    def sleep(self):
        """
        休眠集群内所有节点的 SGLang 服务
        """
        logger.info(">>> [Controller] Sending SLEEP signal to ALL nodes...")
        
        def _send_sleep_to_node(ip):
            url = f"http://{ip}:{self.agent_port}/sleep"
            try:
                # 同样禁用代理
                requests.post(
                    url, 
                    timeout=30, 
                    proxies={"http": None, "https": None}
                )
            except Exception as e:
                logger.warning(f"Failed to send sleep to {ip}: {e}")

        # 并发发送停止指令 (Best Effort，不抛异常)
        with ThreadPoolExecutor(max_workers=len(self.node_ips)) as executor:
            executor.map(_send_sleep_to_node, self.node_ips)
            
        logger.info(">>> [Controller] Sleep signal sent to cluster.")