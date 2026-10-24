# EmotionCamera

Webcam görüntüsünden yüz ifadesini ve bazı el hareketlerini tanıyıp duruma uygun bir kedi/maymun çıkartması gösteren uygulama. Masaüstünde OpenCV penceresiyle ya da tarayıcıda Streamlit ile çalışıyor.

## Nasıl çalışıyor

- **Duygu:** DeepFace her 5 karede bir, ayrı bir thread'de çalışır. Sonuç son 7 tahminin çoğunluğuna göre yumuşatılır.
- **El hareketi:** MediaPipe Hand Landmarker ile parmak noktaları bulunur, kurallarla "OK", "başparmak yukarı" ve "düşünme" (işaret parmağı çeneye yakın) ayırt edilir.
- **Dil çıkarma:** Ağız bölgesindeki kırmızı ve koyu piksel oranına bakan basit bir sezgisel kontrol.

Öncelik sırası: el hareketi > dil çıkarma > duygu.

| Algılanan | Çıkartma |
|---|---|
| OK veya başparmak yukarı | `okay_monkey` |
| Düşünme | `thinking_monkey` |
| Dil çıkarma / iğrenme | `crazy_cat` |
| Mutlu | `happy_cat` |
| Üzgün | `sad_cat` |
| Kızgın | `angry_monkey` |
| Korku / şaşkınlık | `scared_monkey` |

## Kurulum

Proje Windows'ta geliştirildi: kamera `CAP_DSHOW` ile açılıyor, `requirements.txt` içinde `tensorflow-intel` var ve `.bat` dosyaları Windows için. Başka işletim sisteminde bu noktaların değiştirilmesi gerekir.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

El modeli repoda yok. [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) sayfasından `hand_landmarker.task` dosyasını indirip şu yola koy:

```
source/assets/models/hand_landmarker.task
```

DeepFace ilk çalıştırmada kendi duygu modelini internetten indirir.

## Çalıştırma

Masaüstü penceresi (`q` veya `Esc` ile çıkılır):

```bash
python source/main.py
```

Tarayıcı arayüzü:

```bash
streamlit run source/streamlit_app.py
```

Aynı işi `run.bat` ve `run_web.bat` de yapıyor (önce `venv`'i etkinleştiriyorlar).

## Dosyalar

```
source/processor.py      tüm algılama mantığı (EmotionProcessor)
source/main.py           OpenCV penceresi
source/streamlit_app.py  Streamlit arayüzü
source/assets/           kodun kullandığı çıkartma görselleri
assets/                  aynı görsellerin kopyası (kod burayı kullanmıyor)
```
