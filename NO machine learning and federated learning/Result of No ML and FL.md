APTOS Diabetic Retinopathy - Rule-Based Classifier Evaluation
============================================================
Total samples: 3662
Testing samples: 3662
Initializing rule-based classifier...

Starting image processing and classification...
Processing images:   3%|▉                                 | 98/3662 [00:11<06:09,  9.64it/s]Processed 100/3662 images
Processing images:   5%|█▊                               | 199/3662 [00:24<08:39,  6.67it/s]Processed 200/3662 images
Processing images:   8%|██▋                              | 298/3662 [00:35<06:26,  8.70it/s]Processed 300/3662 images
Processing images:  11%|███▌                             | 398/3662 [00:47<04:24, 12.35it/s]Processed 400/3662 images
Processing images:  14%|████▍                            | 499/3662 [00:59<07:27,  7.07it/s]Processed 500/3662 images
Processing images:  16%|█████▍                           | 598/3662 [01:12<05:49,  8.77it/s]Processed 600/3662 images
Processing images:  19%|██████▎                          | 699/3662 [01:24<04:32, 10.87it/s]Processed 700/3662 images
Processing images:  22%|███████▏                         | 799/3662 [01:35<04:58,  9.60it/s]Processed 800/3662 images
Processing images:  25%|████████                         | 898/3662 [01:45<05:11,  8.88it/s]Processed 900/3662 images
Processing images:  27%|█████████                        | 999/3662 [01:57<05:43,  7.75it/s]Processed 1000/3662 images
Processing images:  30%|█████████▌                      | 1098/3662 [02:09<07:34,  5.64it/s]Processed 1100/3662 images
Processing images:  33%|██████████▍                     | 1199/3662 [02:19<03:53, 10.55it/s]Processed 1200/3662 images
Processing images:  35%|███████████▎                    | 1297/3662 [02:30<05:47,  6.81it/s]Processed 1300/3662 images
Processing images:  38%|████████████▏                   | 1399/3662 [02:40<04:06,  9.16it/s]Processed 1400/3662 images
Processing images:  41%|█████████████                   | 1499/3662 [02:52<03:23, 10.63it/s]Processed 1500/3662 images
Processing images:  44%|█████████████▉                  | 1599/3662 [03:03<03:27,  9.95it/s]Processed 1600/3662 images
Processing images:  46%|██████████████▊                 | 1699/3662 [03:14<03:07, 10.45it/s]Processed 1700/3662 images
Processing images:  49%|███████████████▋                | 1798/3662 [03:25<02:34, 12.10it/s]Processed 1800/3662 images
Processing images:  52%|████████████████▌               | 1899/3662 [03:36<04:21,  6.75it/s]Processed 1900/3662 images
Processing images:  55%|█████████████████▍              | 1999/3662 [03:47<02:45, 10.03it/s]Processed 2000/3662 images
Processing images:  57%|██████████████████▎             | 2098/3662 [03:58<02:48,  9.28it/s]Processed 2100/3662 images
Processing images:  60%|███████████████████▏            | 2199/3662 [04:08<03:10,  7.68it/s]Processed 2200/3662 images
Processing images:  63%|████████████████████            | 2298/3662 [04:18<02:29,  9.11it/s]Processed 2300/3662 images
Processing images:  66%|████████████████████▉           | 2399/3662 [04:29<02:13,  9.46it/s]Processed 2400/3662 images
Processing images:  68%|█████████████████████▊          | 2497/3662 [04:39<01:46, 10.90it/s]Processed 2500/3662 images
Processing images:  71%|██████████████████████▋         | 2598/3662 [04:49<02:17,  7.73it/s]Processed 2600/3662 images
Processing images:  74%|███████████████████████▌        | 2699/3662 [05:01<01:51,  8.66it/s]Processed 2700/3662 images
Processing images:  76%|████████████████████████▍       | 2798/3662 [05:12<02:31,  5.71it/s]Processed 2800/3662 images
Processing images:  79%|█████████████████████████▎      | 2898/3662 [05:24<01:10, 10.84it/s]Processed 2900/3662 images
Processing images:  82%|██████████████████████████▏     | 2999/3662 [05:34<00:55, 12.04it/s]Processed 3000/3662 images
Processing images:  85%|███████████████████████████     | 3099/3662 [05:44<00:47, 11.92it/s]Processed 3100/3662 images
Processing images:  87%|███████████████████████████▉    | 3199/3662 [05:55<00:46,  9.94it/s]Processed 3200/3662 images
Processing images:  90%|████████████████████████████▊   | 3299/3662 [06:06<00:33, 10.99it/s]Processed 3300/3662 images
Processing images:  93%|█████████████████████████████▋  | 3398/3662 [06:17<00:36,  7.18it/s]Processed 3400/3662 images
Processing images:  96%|██████████████████████████████▌ | 3499/3662 [06:27<00:12, 13.36it/s]Processed 3500/3662 images
Processing images:  98%|███████████████████████████████▍| 3598/3662 [06:38<00:07,  8.97it/s]Processed 3600/3662 images
Processing images: 100%|████████████████████████████████| 3662/3662 [06:44<00:00,  9.04it/s]

============================================================
Evaluation Results

Processed images: 3662
Successfully processed: 3662/3662 images
Rule-based classification accuracy: 0.2712 (27.12%)

Detailed Classification Report:
                    precision    recall  f1-score   support

           0-No DR       0.00      0.00      0.00      1805
         1-Mild DR       0.00      0.00      0.00       370
     2-Moderate DR       0.28      0.99      0.44       999
       3-Severe DR       0.03      0.02      0.03       193
4-Proliferative DR       0.00      0.00      0.00       295

          accuracy                           0.27      3662
         macro avg       0.06      0.20      0.09      3662
      weighted avg       0.08      0.27      0.12      3662

Confusion Matrix:
[[   0    0 1707   98    0]
 [   0    0  364    6    0]
 [   0    1  989    9    0]
 [   0    0  189    4    0]
 [   0    1  290    4    0]]

============================================================
Benchmark Results Summary
Rule-based classification accuracy: 27.12%
Method: Traditional image processing + Rule-based judgment
Compliant with 'no ML' requirement: No machine learning algorithms used
============================================================
中文:
准确率: 27.12%
处理时间: 6分44秒
主要问题: 过度预测"中度"类别
传统规则方法准确率较低，证明了机器学习方法的必要性。
============================================================
Русский:
Точность: 27.12%
Время обработки: 6 мин 44 сек
Основная проблема: Избыточное предсказание класса "средняя степень"
Традиционный метод на основе правил имеет низкую точность, что подтверждает необходимость методов машинного обучения.
