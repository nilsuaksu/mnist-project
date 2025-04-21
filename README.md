{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9bc3c49f-a702-456b-9880-75dbe5469979",
   "metadata": {},
   "source": [
    "<font size=\"10\">**MNIST ile El Yazısı Rakam Tanıma**</font>\n",
    "\n",
    "Bu proje, MNIST veri setini kullanarak el yazısı rakamlarını tanımak amacıyla bir Convolutional Neural Network (CNN) modeli geliştirmeyi amaçlamaktadır. Model, MNIST veri setindeki 28x28 boyutundaki siyah-beyaz görüntüleri alır ve doğru rakamı tahmin eder. Bu proje, derin öğrenme ve görüntü işleme konularında deneyim kazanmayı hedeflemektedir.\n",
    "\n",
    "<font size=\"8\">**🚀 Projenin Amacı**</font>\n",
    "\n",
    "Bu projenin amacı, derin öğrenme tekniklerini kullanarak el yazısı rakamlarını doğru bir şekilde tanıyabilen bir model geliştirmektir. Modelin amacı, eğitim verileri üzerinde yüksek doğrulukla çalışarak, test verileri üzerinde doğru tahminler yapabilmektir.\n",
    "\n",
    "<font size=\"6\">**Kullanılan Teknolojiler**</font>\n",
    "\n",
    "Python: Python programlama dili\n",
    "\n",
    "TensorFlow: Derin öğrenme modelini oluşturmak için\n",
    "\n",
    "Keras: TensorFlow üzerine kurulmuş bir yüksek seviyeli derin öğrenme API'si\n",
    "\n",
    "NumPy: Sayısal hesaplamalar için\n",
    "\n",
    "Matplotlib & Seaborn: Veri görselleştirme için\n",
    "\n",
    "scikit-learn: Model değerlendirme ve metrikler için"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ef45b25-a74f-4e72-bb8f-7fb5f9c3a7b7",
   "metadata": {},
   "source": [
    "<font size=\"6\">**🧑‍💻 Proje Adımları**</font>\n",
    "\n",
    "<font size=\"6\">**1. Veri Setinin Yüklenmesi ve Hazırlanması**</font>\n",
    "\n",
    "İlk adım olarak, MNIST veri seti keras.datasets modülünden yüklenir. Veri seti, 60.000 eğitim ve 10.000 test örneğinden oluşur. Veriler normalize edilir ve etiketler, modelin doğru tahmin yapabilmesi için one-hot encoding yöntemiyle dönüştürülür."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "483213ce-1083-4a43-96b3-150681b66969",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.datasets import mnist\n",
    "\n",
    "# MNIST veri setini yükleyin\n",
    "(x_train, y_train), (x_test, y_test) = mnist.load_data()\n",
    "\n",
    "# Verileri normalize et\n",
    "x_train, x_test = x_train / 255.0, x_test / 255.0\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba857afd-5bcc-4b27-96c8-82a7c3f5025a",
   "metadata": {},
   "source": [
    "<font size=\"6\">**2. Veri Analizi ve Görselleştirme**</font>\n",
    "\n",
    "Verilerin dağılımını ve örneklerini görselleştirerek veri hakkında bir anlayış edinmek önemlidir. İlk 9 resim ve sınıf dağılımı görselleştirilebilir."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e490ac2-3c83-4a91-9282-b1955aa6b900",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# İlk 9 görüntüyü görselleştirme\n",
    "for i in range(9):\n",
    "    plt.subplot(3, 3, i+1)\n",
    "    plt.imshow(x_train[i], cmap='gray')\n",
    "    plt.title(f\"Label: {y_train[i]}\")\n",
    "    plt.axis('off')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9becc74e-0757-4365-b42a-b57b589b5c8d",
   "metadata": {},
   "source": [
    "<font size=\"6\">**3. Modelin Oluşturulması**</font>\n",
    "\n",
    "Bu projede, Convolutional Neural Network (CNN) tabanlı bir model kullanacağız. Modelde, konvolüsyonel katmanlar, havuzlama katmanları ve tam bağlantılı katmanlar yer alacak."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19ad9ebc-378f-4702-8c3d-6244806d2e81",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras import models, layers\n",
    "\n",
    "# CNN modeli oluşturma\n",
    "model = models.Sequential([\n",
    "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Conv2D(64, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(64, activation='relu'),\n",
    "    layers.Dense(10, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ecceb0fc-f4f8-4230-ad53-99fbbebd9691",
   "metadata": {},
   "source": [
    "<font size=\"6\">**4. Modelin Eğitilmesi**</font>\n",
    "\n",
    "Modeli eğitirken EarlyStopping, ModelCheckpoint ve ReduceLROnPlateau gibi callback'leri kullanarak modelin erken durmasını ve en iyi şekilde eğitilmesini sağlarız."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e9f4e9d-9c5f-4951-9bb2-0e56fcf5d9c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "\n",
    "early_stopping = EarlyStopping(monitor='val_loss', patience=3)\n",
    "checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)\n",
    "\n",
    "history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), \n",
    "                    callbacks=[early_stopping, checkpoint])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "244b2236-46ec-4ef1-b03b-31b1527afeb9",
   "metadata": {},
   "source": [
    "<font size=\"6\">**5. Modelin Değerlendirilmesi**</font>\n",
    "\n",
    "Modelin eğitim ve doğrulama kaybı ile doğruluğu, görselleştirilerek incelenir. Ayrıca Confusion Matrix ve Classification Report ile modelin başarısı daha detaylı bir şekilde değerlendirilir."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2689ced-a8d1-4b26-8d09-88507fb24519",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "# Test seti üzerinde model değerlendirmesi\n",
    "test_loss, test_acc = model.evaluate(x_test, y_test)\n",
    "print(f\"Test accuracy: {test_acc}\")\n",
    "\n",
    "# Confusion matrix\n",
    "y_pred = model.predict(x_test)\n",
    "y_pred = y_pred.argmax(axis=1)\n",
    "print(confusion_matrix(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "119c2c4f-3b05-4aa8-a285-95a592688859",
   "metadata": {},
   "source": [
    "<font size=\"6\">**6. Tek Görüntü Tahmini**</font>\n",
    "\n",
    "Model, tek bir görüntü üzerinde tahmin yaparak doğru sonucu ve tahmin olasılıklarını gösterir."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af13df79-524d-4b0e-a986-c8daefae23b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Tek bir görüntü üzerinde tahmin yapma\n",
    "image = x_test[0].reshape(1, 28, 28, 1)\n",
    "prediction = model.predict(image)\n",
    "predicted_label = np.argmax(prediction)\n",
    "\n",
    "print(f\"Predicted Label: {predicted_label}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30c1cd56-4690-483e-a5ea-44392d37d471",
   "metadata": {},
   "source": [
    "<font size=\"6\">**🛠️ Kurulum ve Çalıştırma**</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15ff1904-9196-4ac5-b60d-341e908c1f92",
   "metadata": {},
   "source": [
    "<font size=\"5\">**1. Gerekli Kütüphanelerin Yüklenmesi**</font>\n",
    "\n",
    "Python ortamınızı hazırladıktan sonra, aşağıdaki komutla gerekli kütüphaneleri yükleyebilirsiniz:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "320b4a18-ff20-47c7-9997-4d405d642486",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install tensorflow matplotlib seaborn numpy scikit-learn"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7771cc20-d0a7-44b3-8505-d2b2ad2f2423",
   "metadata": {},
   "source": [
    "<font size=\"5\">**2. Projenin Çalıştırılması**</font>\n",
    "\n",
    "Projeyi çalıştırmak için, aşağıdaki komutu kullanarak Jupyter Notebook veya bir Python dosyasını çalıştırabilirsiniz:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ea27462-dad9-4cd9-98e8-b375ef13f318",
   "metadata": {},
   "outputs": [],
   "source": [
    "python mnist_project.py"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b453f1a-10d5-4a1d-869e-9c1ec887e843",
   "metadata": {},
   "source": [
    "Ya da Jupyter Notebook kullanıyorsanız:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4430893b-83ad-42c0-84f0-3508aaa8bf3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "jupyter notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "91f49dbb-fb67-4a70-b20a-59b033da1c80",
   "metadata": {},
   "source": [
    "<font size=\"5\">**3. Modelin Kaydedilmesi ve Yüklenmesi**</font>\n",
    "\n",
    "Eğitilen model, bir dosyaya kaydedilebilir ve daha sonra tekrar yüklenip tahmin yapmak için kullanılabilir:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c91eb7ef-440d-4a24-ae86-9d1a9552fd85",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('mnist_model.h5')\n",
    "\n",
    "# Modeli yükleme\n",
    "from tensorflow.keras.models import load_model\n",
    "loaded_model = load_model('mnist_model.h5')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a5d56ad-2a62-4da1-8520-04932c319c0a",
   "metadata": {},
   "source": [
    "<font size=\"6\">**📈 Sonuçlar**</font>\n",
    "\n",
    "Model, MNIST veri setindeki el yazısı rakamları tahmin etme konusunda yüksek doğruluk sağlar. Eğitim süreci ve test sonuçları görselleştirilebilir ve modelin başarımı değerlendirilebilir."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c52d522b-e5f9-42d6-a0e3-b04a1e94a912",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
