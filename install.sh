#!/bin/bash

# Python 버전 확인
python_version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
echo "Installing packages for Python ${python_version}"

# 가상환경 재생성
rm -rf venv
python -m venv venv
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel

# requirements.txt 설치
pip install -r requirements.txt