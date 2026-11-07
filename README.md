# MobileNet GPU Benchmark - Android App

Benchmark TensorFlow Lite MobileNet inference on **CPU (XNNPACK)** vs **GPU Delegate (OpenGL ES)**.

## 📱 Compilar el APK

### Opción A: Android Studio (recomendado, 2 min)
1. Abre Android Studio → **Open** → selecciona la carpeta `MobileNetGPUBenchmark`
2. Espera a que Gradle sincronice
3. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
4. APK en `app/build/outputs/apk/debug/app-debug.apk`

### Opción B: Línea de comandos (requiere Android SDK)
```bash
export ANDROID_HOME=/ruta/a/android-sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools
./gradlew assembleDebug
```

### Opción C: En esta tablet (proot-distro Ubuntu)
```bash
pkg install proot-distro
proot-distro install ubuntu
proot-distro login ubuntu -- bash -c "
  apt update && apt install -y openjdk-17-jdk wget unzip
  cd /home/$USER
  wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
  unzip commandlinetools-linux-11076708_latest.zip -d android-sdk
  export ANDROID_HOME=/home/$USER/android-sdk
  export PATH=\$PATH:\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools
  yes | sdkmanager --licenses
  sdkmanager 'platforms;android-34' 'build-tools;34.0.0' 'platform-tools'
  cd MobileNetGPUBenchmark
  ./gradlew assembleDebug
"
```

## 📊 Qué mide
- **CPU**: TensorFlow Lite con delegado XNNPACK (4 hilos)
- **GPU**: TensorFlow Lite GPU Delegate (OpenGL ES 3.1)
- Modelo: MobileNet v1 1.0 224x224 cuantizado (4.3 MB)
- 20 iteraciones (descartando 2 warmup)

## 📥 Modelo
Se descarga automáticamente al iniciar desde:
`https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant.tflite`

## 📄 Resultados esperados en Shield Tablet (Tegra K1)
| Configuración | Tiempo aprox |
|---------------|--------------|
| CPU (XNNPACK) | ~100-150 ms |
| GPU Delegate  | ~30-60 ms (si funciona) |

## 🔧 Requisitos
- Android 7.0+ (API 24)
- OpenGL ES 3.1+ para GPU delegate
- Internet para descargar modelo

## 📝 Licencia
Apache 2.0 - Código de ejemplo educativo