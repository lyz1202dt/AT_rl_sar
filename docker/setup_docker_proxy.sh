#!/bin/bash

# 为 Docker 守护进程配置代理的脚本

echo "正在配置 Docker 守护进程代理..."

# 创建 Docker 服务配置目录
sudo mkdir -p /etc/systemd/system/docker.service.d

# 创建代理配置文件
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
EOF

echo "代理配置已创建，正在重启 Docker 服务..."

# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 重启 Docker 服务
sudo systemctl restart docker

echo "Docker 服务已重启"

# 验证配置
echo "验证 Docker 代理配置："
sudo systemctl show --property=Environment docker

echo ""
echo "配置完成！现在可以重新运行构建命令："
echo "  docker/container.sh build"
