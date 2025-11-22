#!/usr/bin/env bash
set -euo pipefail

curl -fL -o FlowStatsfile.csv "https://media.githubusercontent.com/media/lova-52/DDos-Attack-Detection-and-Mitigation-using-Machine-Learning-and-Web-Proxy/refs/heads/main/Dataset/FlowStatsfile.csv?download=true"
echo "Saved: FlowStatsfile.csv"
