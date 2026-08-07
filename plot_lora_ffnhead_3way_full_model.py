"""3-way comparison, full (unpruned, 28-layer) Qwen3-VL-2B: LoRA-only vs.
FFN-head-only (frozen backbone) vs. LoRA+FFN-head combined. Extends
plot_ffnhead_vs_lora_full_model.py (the 2-way version) with the combined-run
data gathered this session.

Data provenance, all exact:
- LoRA loss: trainer_state.json log_history on o-taco/qwen3vl-aokvqa (steps
  10-440; this run's trainer_state was pushed before it reached step 500).
- LoRA best/latest accuracy (84.0/85.0): the only surviving record is the
  original pruning_accuracy_v2.png's own printed data-labels at x=0% pruned
  -- no raw train_acc log for the full/unpruned LoRA-only run survived on
  this (recycled) container.
- FFN-head-only loss: logs/ffnhead_full_model/train_2b_full_ffnhead.log
  (steps 10-500). Accuracy: logs/ffnhead_full_model/train_acc_scan*.log,
  compute_train_accuracy.py --no_lora over all 10 saved checkpoints.
- LoRA+FFN-head loss: logs/lora_ffnhead_full_model/train_2b_full_lora_ffnhead.log
  (this session's run, steps 10-500). Accuracy:
  logs/lora_ffnhead_full_model/acc_scan.log, compute_train_accuracy.py
  (lora=True, so both the LoRA adapter and the checkpoint's ffn_head.pt load)
  over all 10 saved checkpoints.

Palette: dataviz skill categorical slots 1-3 (blue/orange/aqua), validated
via scripts/validate_palette.js for all-pairs CVD/normal-vision separation
in light mode (worst adjacent CVD dE 9.2, normal-vision dE 27.6). The aqua
slot (LoRA+FFN-head) is below the 3:1 contrast floor against a white
surface, so per the palette's relief rule it always carries a direct value
label / marker, never color alone.
"""
import matplotlib.pyplot as plt

COLOR_LORA = "#2a78d6"       # slot 1, blue
COLOR_FFN = "#eb6834"        # slot 2, orange
COLOR_COMBINED = "#1baf7a"   # slot 3, aqua -- always direct-labeled (contrast WARN)

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

combined_steps = list(range(10, 501, 10))
combined_loss = [4.9864, 4.1877, 3.9453, 3.744, 3.5245, 3.3571, 3.1865, 2.9854, 2.7783, 2.7529,
                  2.6627, 2.6768, 2.4951, 2.4704, 2.3483, 2.3755, 2.3774, 2.2541, 2.2325, 2.2699,
                  2.249, 2.0921, 2.2063, 2.1297, 2.0642, 2.1197, 1.9903, 2.0835, 2.025, 2.0573,
                  2.0705, 1.9555, 1.9801, 1.9644, 1.9175, 2.0759, 1.9622, 2.0042, 1.9603, 1.8066,
                  1.817, 1.991, 1.9741, 1.952, 1.9207, 1.7903, 1.9368, 1.9008, 1.8051, 1.9312]

ffn_ckpt_steps = [320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
ffn_ckpt_acc = [63.5, 65.5, 66.0, 65.5, 65.0, 66.5, 68.5, 66.5, 67.0, 67.0]
ffn_best_acc, ffn_latest_acc = 68.5, 67.0

combined_ckpt_acc = [85.0, 87.0, 86.0, 88.0, 86.5, 84.5, 85.0, 86.0, 86.5, 84.5]
combined_best_acc, combined_latest_acc = 88.0, 84.5

lora_best_acc, lora_latest_acc = 84.0, 85.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})

ax1.plot(lora_steps, lora_loss, color=COLOR_LORA, linewidth=2, label="LoRA only")
ax1.plot(ffn_steps, ffn_loss, color=COLOR_FFN, linewidth=2, label="FFN-head-only (frozen backbone)")
ax1.plot(combined_steps, combined_loss, color=COLOR_COMBINED, linewidth=2.2, label="LoRA + FFN-head")
ax1.scatter(ffn_ckpt_steps, [ffn_loss[s // 10 - 1] for s in ffn_ckpt_steps], color=COLOR_FFN, s=16, zorder=3)
ax1.scatter(ffn_ckpt_steps, [combined_loss[s // 10 - 1] for s in ffn_ckpt_steps], color=COLOR_COMBINED, s=16, zorder=3)
ax1.axhline(4.1589, color="0.6", linewidth=1, linestyle=":")
ax1.text(5, 4.22, "chance loss = ln(64) = 4.16", fontsize=8.5, color="0.4", va="bottom")
ax1.set_xlabel("Training step")
ax1.set_ylabel("Train loss")
ax1.set_title("Train loss: full-depth 2B, all three methods")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)

methods = ["LoRA\nonly", "FFN-head\nonly", "LoRA +\nFFN-head"]
best = [lora_best_acc, ffn_best_acc, combined_best_acc]
latest = [lora_latest_acc, ffn_latest_acc, combined_latest_acc]
colors = [COLOR_LORA, COLOR_FFN, COLOR_COMBINED]
x = [0, 1, 2]
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
ax2.text(2.35, 27, "chance (25%)", fontsize=8, color="0.4", ha="right")
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylabel("In-batch accuracy (4-way, %)")
ax2.set_ylim(0, 100)
ax2.set_title("Full-depth accuracy")
ax2.legend(fontsize=8.5, loc="upper center", framealpha=0.9)
ax2.grid(True, axis="y", alpha=0.3)

fig.suptitle("Qwen3-VL-2B, full unpruned depth (28 layers): LoRA vs. FFN-head vs. combined", fontsize=13, y=1.02)
fig.text(0.01, -0.06,
         "Note: LoRA-only's best/latest (84.0/85.0%) are read from pruning_accuracy_v2.png's own printed labels at x=0% pruned\n"
         "(no raw log survived for this point); everything else is exact from this session's logs. Combining LoRA + FFN-head\n"
         "beats LoRA alone on best-checkpoint accuracy (+4.0pt) but shows more checkpoint-to-checkpoint variance (88.0->84.5->85.0\n"
         "across steps 380-440) and lands roughly even with LoRA-only at the final step (84.5% vs 85.0%).",
         fontsize=8, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/depth_pruning/lora_ffnhead_3way_full_model.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/lora_ffnhead_3way_full_model.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/lora_ffnhead_3way_full_model.{png,pdf}")
