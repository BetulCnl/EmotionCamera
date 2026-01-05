# EmotionCamera

Bu proje, gerçek zamanlı kamera görüntüsü üzerinden duygu ve el hareketi tanıma işlemleri gerçekleştirerek, kullanıcıya uygun çıkartmalar (sticker) gösteren yapay zeka tabanlı bir uygulamadır.

## Özellikler

*   **Duygu Tanıma:** Kullanıcının yüz ifadelerinden anlık duygu durumunu (Mutlu, Üzgün, Kızgın, Korkmuş, İğrenmiş vb.) tespit eder.
*   **El Hareketi Tanıma:** "Tamam" (OK), "Başparmak Yukarı" (Thumbs Up) ve "Düşünme" gibi el hareketlerini algılar.
*   **Özel Efektler:** Dil çıkarma gibi özel mimikleri algılayarak "çılgın kedi" gibi eğlenceli çıkartmalar ekler.
*   **Çift Arayüz:** İster masaüstü uygulaması olarak (OpenCV penceresi), ister tarayıcı üzerinden (Streamlit) çalıştırılabilir.

## Kurulum

1.  Bu projeyi bilgisayarınıza indirin.
2.  Sanal ortam (virtual environment) oluşturun ve aktif edin:

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  Gerekli Python kütüphanelerini yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Model Dosyasını İndirin:**
    *   `assets` klasörü içine `models` adında yeni bir klasör oluşturun.
    *   Google MediaPipe `hand_landmarker.task` dosyasını indirin ve `assets/models/` içine kaydedin.
    *   *(Bu dosya proje boyutunu artırmamak için burada eklenmemiştir.)*

## Kullanım

### Masaüstü Uygulaması (Hızlı Mod)

Bilgisayarınızda basit bir pencere olarak çalıştırmak için:

```bash
python source/main.py
```

### Web Arayüzü (Görsel Mod)

Daha şık bir arayüz ile tarayıcı üzerinden kullanmak için:

```bash
run_web.bat
```

veya terminalden:

```bash
streamlit run source/streamlit_app.py
```

## Teknolojiler

*   Python
*   OpenCV (Görüntü İşleme)
*   MediaPipe (El ve Yüz İşaretçileri)
*   DeepFace (Duygu Analizi)
*   Streamlit (Web Arayüzü)

---

**Not:** Uygulamanın düzgün çalışması için kamera erişimine izin vermeniz gerekmektedir.
