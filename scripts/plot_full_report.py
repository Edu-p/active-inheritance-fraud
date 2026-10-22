"""Figures for the complete E1-E10 + FakeWhatsApp report."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
FIG = Path("report-latex/images")
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"Real": "#5b5b5b", "Static": "#e69f00", "AI": "#0072b2"}


def runs(group, arm=None, n=None):
    out = []
    for f in (OUT / group).glob("*/results.json"):
        d = json.loads(f.read_text())
        if arm and d.get("arm") != arm: continue
        cn = d.get("config", {}).get("n_samples")
        if n is not None and str(cn) != str(n): continue
        out.append(d)
    return out


def mean_sd(ds, key="f1"):
    xs = [d["test"][key] for d in ds]
    return statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0


def bars(ax, title, groups):
    labels, means, sds, colors = [], [], [], []
    for label, ds, arm in groups:
        m, s = mean_sd(ds); labels.append(label); means.append(m); sds.append(s); colors.append(COLORS[arm])
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, capsize=3, color=colors, edgecolor="black", linewidth=.5)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1 (fake/misinformation)"); ax.set_title(title)
    ax.grid(axis="y", alpha=.2)
    for i, m in enumerate(means): ax.text(i, m + sds[i] + .025, f"{m:.3f}", ha="center", fontsize=7)


def performance():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    bars(axes[0], "Fake.br — 10% fake", [
        ("Real", runs("e1", "real"), "Real"),
        ("Static 158", runs("e2", "static", 158), "Static"),
        ("AI", runs("e3", "ai"), "AI")])
    bars(axes[1], "Fake.br — 5% fake", [
        ("Real", runs("e6-5pct", "real"), "Real"),
        ("Static 158", runs("e6-5pct", "static"), "Static"),
        ("AI", runs("e6-5pct", "ai"), "AI")])
    # grouped FakeWhatsApp
    scenarios = [("Full", "fakewhatsapp-full"), ("10%", "fakewhatsapp-10pct"), ("5%", "fakewhatsapp-5pct")]
    x = np.arange(3); width=.25
    for j,(arm,label) in enumerate((("real","Real"),("static","Static"),("ai","AI"))):
        vals=[mean_sd(runs(g,arm))[0] for _,g in scenarios]; err=[mean_sd(runs(g,arm))[1] for _,g in scenarios]
        axes[2].bar(x+(j-1)*width,vals,width,yerr=err,capsize=2,label=label,color=COLORS[label],edgecolor="black",linewidth=.4)
    axes[2].set_xticks(x,[s[0] for s in scenarios]);axes[2].set_ylim(0,1.02);axes[2].set_ylabel("F1 (misinformation)");axes[2].set_title("FakeWhatsApp.Br — temporal");axes[2].legend(fontsize=7);axes[2].grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(FIG/"full_performance.png",dpi=300,bbox_inches="tight");plt.close(fig)


def recall():
    fig, ax=plt.subplots(figsize=(7.2,3.5))
    scenarios=[("FB 10%","e1","e2","e3"),("FB 5%","e6-5pct","e6-5pct","e6-5pct"),("WApp full","fakewhatsapp-full","fakewhatsapp-full","fakewhatsapp-full"),("WApp 10%","fakewhatsapp-10pct","fakewhatsapp-10pct","fakewhatsapp-10pct"),("WApp 5%","fakewhatsapp-5pct","fakewhatsapp-5pct","fakewhatsapp-5pct")]
    x=np.arange(len(scenarios));width=.25
    for j,(arm,label) in enumerate((("real","Real"),("static","Static"),("ai","AI"))):
        vals=[];errs=[]
        for name,gr,gs,ga in scenarios:
            group={"real":gr,"static":gs,"ai":ga}[arm]
            n=158 if group=="e2" and arm=="static" else None
            ds=runs(group,arm,n);m,s=mean_sd(ds,"recall");vals.append(m);errs.append(s)
        ax.bar(x+(j-1)*width,vals,width,yerr=errs,capsize=2,label=label,color=COLORS[label],edgecolor="black",linewidth=.4)
    ax.set_xticks(x,[s[0] for s in scenarios]);ax.set_ylim(0,1.02);ax.set_ylabel("Recall da classe positiva");ax.set_title("Escassez afeta principalmente o recall");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(FIG/"full_recall.png",dpi=300,bbox_inches="tight");plt.close(fig)


def uncertainty():
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    series=[("Fake.br 10%","e3"),("Fake.br 5%","e6-5pct"),("WhatsApp full","fakewhatsapp-full"),("WhatsApp 10%","fakewhatsapp-10pct"),("WhatsApp 5%","fakewhatsapp-5pct")]
    for label,group in series:
        hs=[d.get("history",[]) for d in runs(group,"ai")]
        epochs=range(max(map(len,hs)))
        means=[]
        for e in epochs:
            vals=[h[e]["score_parts"]["uncertainty"] for h in hs if e<len(h)]
            means.append(statistics.mean(vals))
        ax.plot(list(epochs),means,marker="o",label=label)
    ax.set_xlabel("Ciclo");ax.set_ylabel("Incerteza média do batch");ax.set_ylim(0,1.02);ax.set_title("A incerteza satura após o primeiro ciclo");ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/"full_uncertainty.png",dpi=300,bbox_inches="tight");plt.close(fig)


def templates():
    sets=[]
    # Real corpus rates
    import pandas as pd
    real=pd.read_csv("dataset/fakewhatsapp/train-full.csv").preprocessed_news.astype(str).tolist()
    sets.append(("Real",real))
    for arm,label in (("static","Static"),("ai","AI herdado")):
        texts=[]
        for g in ("fakewhatsapp-full","fakewhatsapp-10pct","fakewhatsapp-5pct"):
            for d in runs(g,arm): texts.extend(d.get("synthetic",[]))
        sets.append((label,texts))
    urgent=[sum(bool(re.match(r"(?i)^\W*urgente",t)) for t in ts)/len(ts) for _,ts in sets]
    received=[sum("acabei de receber" in t.lower() for t in ts)/len(ts) for _,ts in sets]
    x=np.arange(3);w=.34
    fig,ax=plt.subplots(figsize=(6.6,3.5));ax.bar(x-w/2,urgent,w,label="Começa com URGENTE",color="#d55e00");ax.bar(x+w/2,received,w,label="Contém 'acabei de receber'",color="#cc79a7")
    ax.set_xticks(x,[x[0] for x in sets]);ax.set_ylabel("Proporção");ax.set_ylim(0,1);ax.set_title("Colapso de template no prompt WhatsApp");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(FIG/"full_templates.png",dpi=300,bbox_inches="tight");plt.close(fig)


if __name__=="__main__":
    performance();recall();uncertainty();templates();print("Full report figures generated")
