# BERT Fine-Tuning Training Report

**Generated**: 2026-06-07 16:00:19
**Model**: `bert-base-uncased` -> 3-class sentiment (Negative / Neutral / Positive)

This report is generated at the end of the run from the same telemetry samples
collected by the training process. Energy is reported as a software estimate;
CPU, memory, process CPU time, elapsed time, and temperature are sampled with
`psutil` where the operating system exposes those values.

---

## 1. System Information

| Parameter | Value |
|-----------|-------|
| OS | Linux 7.0.10-100.fc43.x86_64 (x86_64) |
| CPU | Unknown |
| CPU Cores (physical/logical) | 2/4 |
| Total RAM | 15.4 GB |
| CPU Frequency | 2878 MHz current / 4100 MHz max |
| Machine | x86_64 |
| Python | 3.14.5 |
| PyTorch | 2.12.0+cu130 |
| Device | cpu |
| GPU | None (CPU-only training) |
| Torch Threads | 8 |
| Torch Inter-op Threads | 1 |

---

## 2. Dataset Statistics

| Parameter | Value |
|-----------|-------|
| Total samples | 110,873 |
| Training samples | 99,785 |
| Test samples | 11,088 |
| Train/Test split | 90% / 10% |

### Label Distribution (Full Dataset)

| Label | Count | Percentage |
|-------|-------|------------|
| Negative | 42,691 | 38.5% |
| Neutral | 16,119 | 14.5% |
| Positive | 52,063 | 47.0% |

### Data Sources

| Source | Count | Percentage |
|--------|-------|------------|
| stanford_sst2 | 67,370 | 60.8% |
| multiclass_sentiment | 41,387 | 37.3% |
| pedagogical_curated | 2,116 | 1.9% |

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `bert-base-uncased` |
| Max Sequence Length | 128 |
| Padding | Dynamic per batch, truncated at max length |
| Epochs | 3 |
| Batch Size (per device) | 16 |
| Eval Batch Size (per device) | 64 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 32 |
| Learning Rate | 2e-05 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.06 |
| DataLoader Workers | 2 |
| Optimizer | AdamW |
| LR Scheduler | Linear with warmup |

---

## 4. Training Results

### Final Test Set Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.8819** (88.19%) |
| **F1 (macro)** | **0.8511** |
| **Precision (macro)** | **0.8396** |
| **Recall (macro)** | **0.8697** |
| Training Loss (final) | 0.6446 |

### Per-Class Performance

```
              precision    recall  f1-score   support

    Negative       0.91      0.89      0.90      4269
     Neutral       0.66      0.83      0.73      1612
    Positive       0.95      0.89      0.92      5207

    accuracy                           0.88     11088
   macro avg       0.84      0.87      0.85     11088
weighted avg       0.89      0.88      0.89     11088

```

### Confusion Matrix

```
            Predicted
            Negative    Neutral    Positive
Negative        3789        351        129
Neutral          165       1334        113
Positive         211        341       4655
```

### Per-Epoch Evaluation

| Epoch | Loss | Accuracy | F1 (macro) | Precision | Recall |
|-------|------|----------|------------|-----------|--------|
| 1.0 | 0.3627 | 0.8677 | 0.8379 | 0.8266 | 0.8633 |
| 2.0 | 0.3634 | 0.8816 | 0.8507 | 0.8392 | 0.8693 |
| 3.0 | 0.4583 | 0.8833 | 0.8484 | 0.8418 | 0.8566 |


### Training Loss Progression

| Step | Epoch | Loss | Learning Rate |
|------|-------|------|---------------|
| 100 | 0.0320666987333654 | 2.1700 | 3.53e-06 |
| 500 | 0.160333493666827 | 1.0505 | 1.78e-05 |
| 900 | 0.2886002886002886 | 0.9438 | 1.92e-05 |
| 1300 | 0.4168670835337502 | 0.8161 | 1.83e-05 |
| 1700 | 0.5451338784672118 | 0.8336 | 1.74e-05 |
| 2100 | 0.6734006734006734 | 0.7817 | 1.65e-05 |
| 2500 | 0.801667468334135 | 0.7544 | 1.56e-05 |
| 2900 | 0.9299342632675967 | 0.7642 | 1.47e-05 |
| 3300 | 1.0580407247073913 | 0.5737 | 1.38e-05 |
| 3700 | 1.186307519640853 | 0.6256 | 1.29e-05 |
| 4100 | 1.3145743145743145 | 0.5943 | 1.20e-05 |
| 4500 | 1.4428411095077762 | 0.5954 | 1.10e-05 |
| 4900 | 1.571107904441238 | 0.5956 | 1.01e-05 |
| 5300 | 1.6993746993746994 | 0.6172 | 9.23e-06 |
| 5700 | 1.827641494308161 | 0.5566 | 8.32e-06 |
| 6100 | 1.9559082892416226 | 0.5783 | 7.41e-06 |
| 6500 | 2.0840147506814173 | 0.4132 | 6.50e-06 |
| 6900 | 2.212281545614879 | 0.3807 | 5.59e-06 |
| 7300 | 2.3405483405483407 | 0.4163 | 4.68e-06 |
| 7700 | 2.468815135481802 | 0.3947 | 3.77e-06 |
| 8100 | 2.5970819304152637 | 0.4622 | 2.86e-06 |
| 8500 | 2.7253487253487254 | 0.3977 | 1.95e-06 |
| 8900 | 2.8536155202821867 | 0.3488 | 1.04e-06 |
| 9300 | 2.9818823152156484 | 0.3549 | 1.32e-07 |


---

## 5. Training Time

| Metric | Value |
|--------|-------|
| **Monitored Run Time** | **22:25:29** |
| Monitored Seconds | 80729.1 |
| Fine-Tuning `trainer.train()` Time | 22:04:47 |
| Process CPU Time | 3 days, 4:16:22 |
| Process CPU User/System Time | 260,452.51s / 14,130.33s |
| Samples/Second | 3.8 |
| Steps/Second | 0.118 |

---

## 6. Energy, Resources, And Temperature

### Monitoring Coverage

| Metric | Value |
|--------|-------|
| Sampling Interval | 5.0s |
| Monitoring Samples | 15942 |
| Temperature Sensor | 76.6 C average, 89.0 C peak from `Package id 0` |
| Energy Measurement Method | Intel RAPL (Running Average Power Limit) energy counters from /sys/class/powercap/ |

### Resource Utilization

| Metric | Value |
|--------|-------|
| Avg System CPU Usage | 87.9% |
| Peak System CPU Usage | 99.1% |
| Avg Training Process CPU | 340.1% |
| Peak Training Process CPU | 385.8% |
| Avg Process Cores Used | 3.40 |
| Avg Process CPU Share | 85.0% |
| Avg Process Memory | 5,180 MB |
| Peak Process Memory | 5,429 MB |
| Avg System Memory Usage | 69.0% |
| Peak System Memory Usage | 86.9% |
| Avg Temperature | 76.6 C |
| Peak Temperature | 89.0 C |

### Energy Estimates

| Metric | Value |
|--------|-------|
| Estimated CPU TDP | ? W |
| Estimated Avg CPU Power | ? W |
| Estimated RAM Power | ? W |
| Estimated Peak Power | 0.0 W |
| **Estimated Total Avg Power** | **0.0 W** |
| **Total Energy Consumed** | **0.0 Wh (0.0 kWh)** |

### Carbon Footprint

| Region | CO2 Emissions |
|--------|--------------|
| Global Average (475g CO2/kWh) | 0.0 g CO2 |
| India Average (720g CO2/kWh) | 0.0 g CO2 |

### Monitoring Graphs

#### CPU Utilization During Run

System CPU load and the training process share normalized across all logical cores.

![CPU Utilization During Run](runs/20260606_173447/graphs/cpu_usage.svg)

#### Memory Usage During Run

Resident memory used by the training Python process.

![Memory Usage During Run](runs/20260606_173447/graphs/memory_usage.svg)

#### Estimated Power Draw

Estimated instantaneous power from CPU load and process memory footprint.

![Estimated Power Draw](runs/20260606_173447/graphs/power_estimate.svg)

#### Cumulative Estimated Energy

Estimated energy accumulated over the monitored run.

![Cumulative Estimated Energy](runs/20260606_173447/graphs/energy_accumulation.svg)

#### System Temperature

Highest exposed CPU/system temperature sensor sampled during the run.

![System Temperature](runs/20260606_173447/graphs/temperature.svg)

#### Training Loss Progression

Logged training loss across optimizer steps.

![Training Loss Progression](runs/20260606_173447/graphs/training_loss.svg)

#### Evaluation Metrics Per Epoch

Held-out test metrics recorded after each epoch.

![Evaluation Metrics Per Epoch](runs/20260606_173447/graphs/evaluation_metrics.svg)


> Annotation: energy is estimated from sampled CPU load, configured CPU TDP,
> and process memory usage. Temperature is reported only when the OS exposes
> sensor data through `psutil`; otherwise the report explicitly marks it as
> unavailable instead of inventing a value.

---

## 7. Model Artifacts

| File | Description |
|------|-------------|
| `fine_tuned_bert_real/` | Saved model weights & tokenizer |
| `fine_tuned_bert_real/report_assets/` | SVG graphs embedded in this report |
| `fine_tuned_bert_real/test_meta.json` | Test set data for comparison |
| `training_report.md` | This report |
| `training_results.json` | Raw metrics in JSON format |
| `real_dataset.json` | Full training dataset |

---

*Report generated by the Pedagogical Intelligence System BERT fine-tuning pipeline.*
