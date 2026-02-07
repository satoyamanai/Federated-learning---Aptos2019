##Federated-learning--Aptos2019##
A project of federated learning in medicine from the RUDN University.

项目描述：
本系统旨在通过计算机视觉和机器学习技术实现糖尿病视网膜病变（Diabetic Retinopathy, DR）的自动分级诊断。
我们探索并实现了三种不同的技术路径，以全面评估不同方法在医疗图像分析任务中的适用性和效果。
1）无机器学习方法-基于传统图像处理和特征工程的方法。
2）深度学习方法-使用预训练的DenseNet121进行端到端的图像分类。
3）联邦学习方法-在隐私保护框架下进行分布式模型训练。

数据集
- 来源: APTOS 2019 Blindness Detection Challenge
- 类别: 0-4级共5个等级（无病变至增殖性病变）
- 图像数量: 训练集3,662张，尺寸可变
- 数据分布: 类别不均衡（大多数样本为无病变或轻度病变）

----------------------------------------------------------

Project Description: 
This system aims to achieve automated grading and diagnosis of diabetic retinopathy (DR) using computer vision and machine learning techniques.
We explored and implemented three different technical approaches to comprehensively evaluate the applicability and effectiveness of different methods in medical image analysis tasks.
1) No machine learning method – based on traditional image processing and feature engineering.
2) Deep learning method – using pre-trained DenseNet121 for end-to-end image classification.
3) Federated learning method – distributed model training within a privacy-preserving framework.

Dataset
- Source: APTOS 2019 Blindness Detection Challenge
- Classes: 0-4 (5 levels, from no lesion to proliferative lesions)
- Number of images: 3,662 images in the training set, variable size
- Data distribution: Imbalanced class distribution (most samples are no lesions or mild lesions)

----------------------------------------------------------

Описание проекта: 
Данная система направлена ​​на автоматическую оценку степени тяжести и диагностику диабетической ретинопатии (ДР) с использованием методов компьютерного зрения и машинного обучения.
Мы исследовали и реализовали три различных технических подхода для всесторонней оценки применимости и эффективности различных методов в задачах анализа медицинских изображений.
1) Метод без машинного обучения – основан на традиционной обработке изображений и инженерии признаков.
2) Метод глубокого обучения – с использованием предварительно обученной сети DenseNet121 для сквозной классификации изображений.
3) Метод федеративного обучения – распределенное обучение модели в рамках системы, обеспечивающей конфиденциальность.

Набор данных
- Источник: Конкурс APTOS 2019 по обнаружению слепоты
- Классы: 0-4 (5 уровней, от отсутствия поражений до пролиферативных поражений)
- Количество изображений: 3662 изображения в обучающем наборе, переменный размер
- Распределение данных: Несбалансированное распределение классов (большинство образцов – отсутствие поражений или легкие поражения)
