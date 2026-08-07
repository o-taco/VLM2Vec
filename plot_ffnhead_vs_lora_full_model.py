"""FFN-head-only vs. LoRA, full (unpruned, 28-layer) Qwen3-VL-2B: train loss
curves plus best/latest in-batch accuracy. Extends the N=3-9 crossover sweep
(plot_2b_ffnhead_vs_lora.png) out to full depth.

Data provenance, all exact:
- LoRA loss: trainer_state.json log_history on o-taco/qwen3vl-aokvqa (steps
  10-440; this run's trainer_state was pushed before it reached step 500).
- LoRA best/latest accuracy (84.0/85.0): the only surviving record of this
  point is the original pruning_accuracy_v2.png's own printed data-labels at
  x=0% pruned (shared start point for all pruning-pattern curves in that
  figure) -- no raw train_acc log for the full/unpruned LoRA run survived on
  this (recycled) container. Re-verify against source if this matters later.
- FFN-head loss: logs/ffnhead_full_model/train_2b_full_ffnhead.log (this
  session's run, steps 10-500).
- FFN-head best/latest accuracy: logs/ffnhead_full_model/train_acc_scan*.log
  (this session's compute_train_accuracy.py --no_lora sweep over all 10
  saved checkpoints, 320-500).
"""
import matplotlib.pyplot as plt

lora_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
              200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
              370, 380, 390, 400, 410, 420, 430, 440]
lora_loss = [4.9871, 4.2031, 3.9844, 3.8689, 3.7229, 3.5887, 3.4769, 3.3582, 3.2865, 3.3174, 3.2919,
             3.1884, 2.961, 2.9534, 2.803, 2.7385, 2.6987, 2.577, 2.5421, 2.5622, 2.5086, 2.3459,
             2.4471, 2.367, 2.2668, 2.3305, 2.2252, 2.2333, 2.2436, 2.2355, 2.2665, 2.1506, 2.1351,
             2.1234, 2.0684, 2.2558, 2.1153, 2.1504, 2.1294, 1.9429, 1.9684, 2.1433, 2.1084, 2.1021]

ffn_steps = list(range(10, 501, 10))
ffn_loss = [5.7594, 4.8477, 4.4237, 4.2095, 4.1663, 4.0939, 4.0638, 4.0473, 3.9914, 3.9569, 3.8963,
            3.8485, 3.8512, 3.786, 3.7822, 3.6982, 3.6836, 3.6312, 3.604, 3.5195, 3.5731, 3.4687,
            3.4871, 3.5001, 3.4605, 3.4214, 3.3812, 3.3071, 3.368, 3.3752, 3.3473, 3.324, 3.2368,
            3.3567, 3.296, 3.3554, 3.3137, 3.2608, 3.2861, 3.1792, 3.2523, 3.2885, 3.229, 3.2904,
            3.2818, 3.189, 3.2556, 3.1722, 3.1754, 3.1874]

ffn_ckpt_steps = [320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
ffn_ckpt_acc = [63.5, 65.5, 66.0, 65.5, 65.0, 66.5, 68.5, 66.5, 67.0, 67.0]
ffn_best_step, ffn_best_acc = 440, 68.5
ffn_latest_acc = 67.0

lora_best_acc, lora_latest_acc = 84.0, 85.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})

ax1.plot(lora_steps, lora_loss, color="#9467bd", linewidth=2, label="LoRA (full 28L)")
ax1.plot(ffn_steps, ffn_loss, color="#ff7f0e", linewidth=2, label="FFN-head-only (full 28L, frozen backbone)")
ax1.scatter(ffn_ckpt_steps, [ffn_loss[s // 10 - 1] for s in ffn_ckpt_steps], color="#ff7f0e", s=18, zorder=3)
ax1.axhline(4.1589, color="0.6", linewidth=1, linestyle=":")
ax1.text(5, 4.22, "chance loss = ln(64) = 4.16", fontsize=8.5, color="0.4", va="bottom")
ax1.set_xlabel("Training step")
ax1.set_ylabel("Train loss")
ax1.set_title("Train loss: full-depth 2B, LoRA vs. FFN-head-only")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)

methods = ["LoRA", "FFN-head-only"]
best = [lora_best_acc, ffn_best_acc]
latest = [lora_latest_acc, ffn_latest_acc]
colors = ["#9467bd", "#ff7f0e"]
x = [0, 1]
width = 0.32
ax2.bar([xi - width / 2 for xi in x], best, width, color=colors, alpha=1.0, label="Best checkpoint")
ax2.bar([xi + width / 2 for xi in x], latest, width, color=colors, alpha=0.55, label="Latest (step 500)")
for xi, v in zip(x, best):
    ax2.annotate(f"{v:.1f}%", (xi - width / 2, v), textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=9, fontweight="bold")
for xi, v in zip(x, latest):
    ax2.annotate(f"{v:.1f}%", (xi + width / 2, v), textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=9, fontweight="bold")
ax2.axhline(25, color="0.5", linestyle=":", linewidth=1.2)
ax2.text(1.35, 27, "chance (25%)", fontsize=8, color="0.4", ha="right")
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_ylabel("In-batch accuracy (4-way, %)")
ax2.set_ylim(0, 100)
ax2.set_title("Full-depth accuracy")
ax2.legend(fontsize=8.5, loc="upper center", framealpha=0.9)
ax2.grid(True, axis="y", alpha=0.3)

fig.suptitle("Qwen3-VL-2B, full unpruned depth (28 layers): LoRA vs. FFN-head-only", fontsize=13, y=1.02)
fig.text(0.01, -0.05,
         "Note: LoRA's best/latest (84.0/85.0%) are read from pruning_accuracy_v2.png's own printed labels at x=0% pruned\n"
         "(no raw log survived for this point); everything else is exact from this session's logs. Same crossover trend as\n"
         "the N=3-9 sweep (plot_2b_ffnhead_vs_lora.png) but wider: LoRA's lead over FFN-head-only grows with more depth.",
         fontsize=8, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/depth_pruning/ffnhead_vs_lora_full_model.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/ffnhead_vs_lora_full_model.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/ffnhead_vs_lora_full_model.{png,pdf}")
