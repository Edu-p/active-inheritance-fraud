"""Central Active Inheritance loop."""
from __future__ import annotations

from .metrics import classification_metrics, diversity_metrics


class ActiveInheritancePipeline:
    def __init__(self, cfg, learner, embedder, generator, evaluator) -> None:
        self.cfg = cfg
        self.learner = learner
        self.embedder = embedder
        self.generator = generator
        self.evaluator = evaluator

    def run(self, train, val, test):
        cfg = self.cfg
        tr_texts, tr_y = train
        va_texts, va_y = val
        te_texts, te_y = test

        inherited: list[str] = []
        history = []

        # t = 0: warm-up on Data 0 only
        init_loss = self.learner.fit(tr_texts, tr_y, epochs=cfg.learner_epochs)
        print(f"[t=0] loss={init_loss:.4f} (Data 0)")

        for t in range(cfg.epochs):
            batch = [
                x for x in self.generator.generate_batch(inherited, cfg.batch_size)
                if x and x.strip()
            ]
            if not batch:
                print(f"[epoch {t}] empty batch — skipping")
                continue

            y_synth = [1] * len(batch)  # synthetic samples are always "fake"
            selected, mean_score, parts = self.evaluator.select(
                batch, y_synth, inherited, min(cfg.top_k, len(batch))
            )
            inherited.extend(selected)

            combined_texts = tr_texts + inherited
            combined_y = tr_y + [1] * len(inherited)
            loss = self.learner.fit(
                combined_texts, combined_y, epochs=cfg.learner_epochs
            )

            val_metrics = self._evaluate(va_texts, va_y)

            print(
                f"[epoch {t}] loss={loss:.4f} | batch: score={mean_score:.3f} "
                f"(err={parts['error']:.3f}, unc={parts['uncertainty']:.3f}, "
                f"div={parts['diversity']:.3f}) | inherited={len(inherited)} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            history.append({
                "epoch": t,
                "loss": float(loss),
                "val": val_metrics,
                "inherited_total": len(inherited),
                "pool_diversity": diversity_metrics(inherited, self.embedder),
                "tokens_used": int(self.generator.tokens_used),
                "mean_score_topk": mean_score,
                "score_parts": parts,
            })

        test_metrics = self._evaluate(te_texts, te_y)
        return history, test_metrics, inherited

    def _evaluate(self, texts, y):
        pred = self.learner.predict_class(texts)
        return classification_metrics(y, pred)
