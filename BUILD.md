# Building MobileNetGPUBenchmark

## Requirements

- **Android Studio** (recommended) or **JDK 17+** + **Android SDK**
- **Android SDK** with:
  - Platform SDK 34 (compileSdk)
  - Build Tools 34+
  - Android NDK (optional, for native libs)
- **Gradle 8.6+** (included via wrapper)

## Quick Start (Android Studio)

1. Clone the repo:
   ```bash
   git clone https://github.com/manuti/MobileNetGPUBenchmark.git
   cd MobileNetGPUBenchmark
   ```

2. Open in Android Studio: `File > Open > Select project folder`

3. Let Gradle sync complete

4. Connect device (USB debugging) or create emulator

5. Run: `Run > Run 'app'` or `./gradlew installDebug`

## Command Line Build

### Linux/macOS/Windows (with Android SDK configured)

```bash
# Set ANDROID_HOME (adjust path to your SDK location)
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin

# Or create local.properties in project root:
echo "sdk.dir=$HOME/Android/Sdk" > local.properties

# Build debug APK
./gradlew assembleDebug

# Install on connected device
./gradlew installDebug

# Run tests
./gradlew test
```

### Windows (PowerShell)
```powershell
$env:ANDROID_HOME = "$env:USERPROFILE\AppData\Local\Android\Sdk"
$env:PATH += ";$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\cmdline-tools\latest\bin"
.\gradlew.bat assembleDebug
```

### CI/CD (GitHub Actions)
The repository includes a workflow at `.github/workflows/android.yml` for automated builds.

## Project Structure

```
MobileNetGPUBenchmark/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/mobilenetgpubenchmark/
│   │   │   ├── MainActivity.kt      # Main benchmark logic
│   │   │   └── BenchmarkApplication.kt
│   │   ├── res/
│   │   │   ├── layout/activity_main.xml
│   │   │   ├── values/strings.xml, themes.xml, colors.xml
│   │   │   └── xml/file_paths.xml, data_extraction_rules.xml, backup_rules.xml
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── gradle/wrapper/                  # Gradle 8.6 wrapper
├── build.gradle.kts                 # Root build config
├── settings.gradle.kts
├── gradle.properties                # JVM args: -Xmx512m
└── local.properties (gitignored)    # SDK path - create this!
```

## Key Configuration

### gradle.properties
```properties
org.gradle.jvmargs=-Xmx512m -Dfile.encoding=UTF-8
android.useAndroidX=true
android.enableJetifier=true
kotlin.code.style=official
```
Reduced heap (512MB) for CI/low-memory environments.

### app/build.gradle.kts
- `minSdk 24` (Android 7.0)
- `targetSdk 34` (Android 14)
- `compileSdk 34`
- MultiDex enabled

### Dependencies
- TensorFlow Lite 2.16.1 (CPU + GPU delegate)
- OkHttp 4.12.0 (model download)
- Kotlin Coroutines 1.7.3
- AndroidX Core, AppCompat, Material, ConstraintLayout

## Permissions

The app requests at runtime:
- **Android 13+ (API 33+)**: `READ_MEDIA_IMAGES`, `READ_MEDIA_VIDEO`, `READ_MEDIA_AUDIO`
- **Android 10-12 (API 29-32)**: `READ_EXTERNAL_STORAGE`
- **Android 9 and below**: `WRITE_EXTERNAL_STORAGE`, `READ_EXTERNAL_STORAGE`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `SDK location not found` | Create `local.properties` with `sdk.dir=/path/to/android/sdk` |
| `OutOfMemoryError` | Increase `-Xmx` in `gradle.properties` or use `--no-daemon` |
| `Gradle daemon failed` | Run with `--no-daemon` or increase system memory |
| Model download fails | Check internet; app falls back to cached model in internal storage |
| GPU delegate unavailable | Normal on some devices; CPU benchmark still runs |

## Artifacts

- **Debug APK**: `app/build/outputs/apk/debug/app-debug.apk`
- **Release APK**: `app/build/outputs/apk/release/app-release-unsigned.apk` (needs signing)

## Signing Release Build

Create `keystore.properties` (gitignored):
```properties
storeFile=../release.keystore
storePassword=****
keyAlias=****
keyPassword=****
```

Add to `app/build.gradle.kts`:
```kotlin
android {
    signingConfigs {
        create("release") {
            val props = Properties().apply { load(FileInputStream(rootProject.file("keystore.properties"))) }
            storeFile = file(props["storeFile"] as String)
            storePassword = props["storePassword"] as String
            keyAlias = props["keyAlias"] as String
            keyPassword = props["keyPassword"] as String
        }
    }
    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = true
            proguardFiles(...)
        }
    }
}
```

## License

MIT License - see LICENSE file
