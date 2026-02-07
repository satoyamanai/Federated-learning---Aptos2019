import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

#设置正确的数据路径
data_base_path = "Aptos2019"
train_csv_path = os.path.join(data_base_path, "train.csv")
train_images_dir = os.path.join(data_base_path, "train_images")

print(f"Training CSV path: {train_csv_path}")
print(f"Training Image Catalog: {train_images_dir}")

# QWK回调类
class QWKCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X = validation_data[0]
        self.Y = validation_data[1]
        self.history = []
        self.best_score = 0
    
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X, verbose=0)
        score = cohen_kappa_score(
            np.argmax(self.Y, axis=1),
            np.argmax(pred, axis=1),
            labels=[0, 1, 2, 3, 4],
            weights='quadratic'
        )
        print(f"Epoch {epoch+1}: QWK: {score:.4f}")
        self.history.append(score)
        
        if score > self.best_score:
            self.best_score = score
            print(f'Saving checkpoint: {score:.4f}')
            self.model.save(os.path.join(data_base_path, 'best_model.h5'))

class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)
                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []
            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix'
    
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# 加载图像数据
def load_raw_images_df(data_frame, filenamecol, labelcol, img_size, n_classes):
    n_images = len(data_frame)
    X = np.empty((n_images, img_size, img_size, 3))
    Y = np.zeros((n_images, n_classes))
    
    print("Loading images...")
    for index, entry in data_frame.iterrows():
        if index % 100 == 0:
            print(f"Processed {index}/{n_images} images")
            
        Y[index, entry[labelcol]] = 1
        
        # 加载图像
        img_path = entry[filenamecol]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            img = np.zeros((img_size, img_size, 3))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
        
        X[index, :] = img / 255.0
    
    print("Image loading completed!")
    return X, Y

# 主程序
def main():
   
    batch_size = 32
    img_size = 224
    
    # 加载数据
    print("Loading data...")
    train_raw_data = pd.read_csv(train_csv_path)
    train_raw_data["filename"] = train_raw_data["id_code"].map(
        lambda x: os.path.join(train_images_dir, x + ".png")
    )
    
    print("Diagnosis:")
    print(train_raw_data.diagnosis.value_counts().sort_index())
    
    # 类别标签
    class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    # 分割数据
    train_df, val_df = train_test_split(train_raw_data, random_state=42, shuffle=True, test_size=0.2)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # 加载图像数据
    X_train, Y_train = load_raw_images_df(train_df, "filename", "diagnosis", img_size, 5)
    X_val, Y_val = load_raw_images_df(val_df, "filename", "diagnosis", img_size, 5)
    
    # 数据增强
    datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    training_generator = MixupGenerator(X_train, Y_train, batch_size=batch_size, alpha=0.2, datagen=datagen)()
    
    # 构建模型
    def build_model():
        # 使用预训练的DenseNet121
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(img_size, img_size, 3)
        )
        
        # 冻结基础模型的前面层
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # 添加自定义层
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        outputs = layers.Dense(5, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=outputs)
        
        # 编译模型 - 使用Focal Loss
        optimizer = optimizers.Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(),  # 使用Focal Loss处理类别不平衡
            metrics=['accuracy']
        )
        
        print(model.summary())
        return model
    
    # 创建模型
    model = build_model()
    
    # 回调函数
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            os.path.join(data_base_path, 'checkpoint.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        QWKCallback((X_val, Y_val))
    ]
    
    # 训练模型
    print("Starting training...")
    
    # 第一阶段训练
    print("First stage training...")
    history1 = model.fit(
        training_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=20,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 第二阶段训练（解冻更多层）
    print("Second stage training...")
    for layer in model.layers[-30:]:
        layer.trainable = True
    
    # 重新编译模型以应用更改
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001),
        loss=focal_loss(),  # 继续使用Focal Loss
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        training_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=30,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 合并历史记录
    history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
    }
    
    # 评估模型
    print("Evaluating model...")
    Y_val_pred = model.predict(X_val)
    Y_val_pred_labels = np.argmax(Y_val_pred, axis=1)
    Y_val_true_labels = np.argmax(Y_val, axis=1)
    
    # 计算最终指标
    final_accuracy = np.mean(Y_val_pred_labels == Y_val_true_labels)
    qwk_score = cohen_kappa_score(Y_val_true_labels, Y_val_pred_labels, weights='quadratic')
    
    print(f"\nFinal accuracy:{final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"QWK score:{qwk_score:.4f}")
    
    # 详细分类报告
    print("\nDetailed classfication report:")
    print(classification_report(Y_val_true_labels, Y_val_pred_labels, target_names=class_labels))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(Y_val_true_labels, Y_val_pred_labels, np.array(class_labels), normalize=True)
    plt.show()
    
    # 绘制训练历史
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 保存最终模型
    model.save(os.path.join(data_base_path, 'final_model.h5'))
    print("Model saved to:best_model.h5")
    print("Trian completed.")

if __name__ == "__main__":
    main()
