#!/usr/bin/env bash
# Compila o relatório LaTeX preservando versões anteriores com timestamp UTC.
# Uso: bash scripts/compile_report.sh
set -e
cd "$(dirname "$0")/../report-latex"

ts=$(date -u +%Y%m%dT%H%M%S%NZ)
mkdir -p archive

# Arquiva a versão anterior antes de sobrescrever main.pdf / main.tex.
if [ -f main.pdf ]; then cp -n main.pdf "archive/main-${ts}-pre.pdf"; fi
if [ -f main.tex ]; then cp -n main.tex "archive/main-${ts}-pre.tex"; fi

pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

# Cópia atual com timestamp; main.pdf segue como "latest".
cp -n main.pdf "active-inheritance-report-${ts}.pdf"
cp -n main.pdf "relatorio-completo-${ts}.pdf"

echo "OK: main.pdf atualizado + active-inheritance-report-${ts}.pdf + relatorio-completo-${ts}.pdf"
echo "    versão anterior preservada em archive/main-${ts}-pre.{pdf,tex}"
