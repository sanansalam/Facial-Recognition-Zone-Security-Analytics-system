#!/bin/bash
# run_tunnel.sh — Quick Tunnel for Security App

cd "$(dirname "$0")"

if [ ! -f "./cloudflared" ]; then
    echo "❌ cloudflared binary not found. Downloading..."
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    chmod +x cloudflared
fi

echo "🚀 Starting Cloudflare Quick Tunnel for localhost:9000..."
./cloudflared tunnel --url http://localhost:9000
