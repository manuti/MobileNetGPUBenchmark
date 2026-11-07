package com.example.mobilenetgpubenchmark

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.ProgressBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.BufferedWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvResults: TextView
    private lateinit var btnRunBenchmark: Button
    private lateinit var btnViewLogs: Button
    private lateinit var progressBar: ProgressBar

    private val MODEL_URL = "https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant.tflite"
    private val MODEL_FILENAME = "mobilenet_v1_1.0_224_quant.tflite"
    private var modelFile: File? = null
    private lateinit var fileLogger: FileLogger

    // Permission request codes
    private val PERMISSION_REQUEST_STORAGE = 1001
    private val PERMISSION_REQUEST_MEDIA = 1002

    // Track if permissions granted
    private var storagePermissionGranted = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize file logger FIRST (before any other operations)
        fileLogger = FileLogger(this)
        fileLogger.log("=== App started ===")
        fileLogger.log("Android SDK: ${Build.VERSION.SDK_INT}, Device: ${Build.MODEL}, Manufacturer: ${Build.MANUFACTURER}")

        // Global exception handler to prevent silent crashes
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            Log.e("Benchmark", "UNCAUGHT EXCEPTION in ${thread.name}", throwable)
            fileLogger.log("UNCAUGHT EXCEPTION in ${thread.name}: ${throwable.message}")
            fileLogger.log(Log.getStackTraceString(throwable))
            runOnUiThread {
                tvStatus?.text = "CRASH: ${throwable.javaClass.simpleName}: ${throwable.message}"
                tvResults?.text = Log.getStackTraceString(throwable)
            }
        }

        try {
            setContentView(R.layout.activity_main)

            tvStatus = findViewById(R.id.tvStatus)
            tvResults = findViewById(R.id.tvResults)
            btnRunBenchmark = findViewById(R.id.btnRunBenchmark)
            btnViewLogs = findViewById(R.id.btnViewLogs)
            progressBar = findViewById(R.id.progressBar)

            btnRunBenchmark.setOnClickListener { checkPermissionsAndRunBenchmark() }
            btnViewLogs.setOnClickListener { openLogFile() }
            tvResults.text = ""
            tvStatus.text = "Verificando permisos..."

            // Check permissions on startup
            checkStoragePermission()

        } catch (e: Exception) {
            Log.e("Benchmark", "onCreate error", e)
            fileLogger.log("onCreate error: ${e.message}")
            fileLogger.log(Log.getStackTraceString(e))
            tvStatus.text = "onCreate CRASH: ${e.message}"
            tvResults.text = Log.getStackTraceString(e)
        }
    }

    private fun checkStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // Android 13+
            // On Android 13+, we need READ_MEDIA_IMAGES/VIDEO/AUDIO
            val hasMediaPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
            if (hasMediaPermission) {
                storagePermissionGranted = true
                fileLogger.log("Media permission already granted (Android 13+)")
                initAfterPermissions()
            } else {
                fileLogger.log("Requesting media permissions (Android 13+)")
                requestMediaPermissions()
            }
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) { // Android 10-12
            // On Android 10+, we can use MediaStore without special permission for own files
            // But for broader access, request READ_EXTERNAL_STORAGE
            val hasStoragePermission = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
            if (hasStoragePermission) {
                storagePermissionGranted = true
                fileLogger.log("Storage permission already granted (Android 10-12)")
                initAfterPermissions()
            } else {
                fileLogger.log("Requesting storage permission (Android 10-12)")
                requestStoragePermissionLegacy()
            }
        } else { // Android 9 and below
            val hasStoragePermission = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
            if (hasStoragePermission) {
                storagePermissionGranted = true
                fileLogger.log("Storage permission already granted (Android 9-)")
                initAfterPermissions()
            } else {
                fileLogger.log("Requesting storage permission (Android 9-)")
                requestStoragePermissionLegacy()
            }
        }
    }

    private fun requestMediaPermissions() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.READ_MEDIA_IMAGES) ||
            shouldShowRequestPermissionRationale(Manifest.permission.READ_MEDIA_VIDEO) ||
            shouldShowRequestPermissionRationale(Manifest.permission.READ_MEDIA_AUDIO)) {
            showPermissionRationaleDialog(
                "Permiso de almacenamiento necesario",
                "La app necesita acceso a archivos multimedia para guardar los logs y resultados del benchmark en la carpeta Descargas.",
                { requestPermissions(arrayOf(
                    Manifest.permission.READ_MEDIA_IMAGES,
                    Manifest.permission.READ_MEDIA_VIDEO,
                    Manifest.permission.READ_MEDIA_AUDIO
                ), PERMISSION_REQUEST_MEDIA) }
            )
        } else {
            requestPermissions(arrayOf(
                Manifest.permission.READ_MEDIA_IMAGES,
                Manifest.permission.READ_MEDIA_VIDEO,
                Manifest.permission.READ_MEDIA_AUDIO
            ), PERMISSION_REQUEST_MEDIA)
        }
    }

    private fun requestStoragePermissionLegacy() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.READ_EXTERNAL_STORAGE) ||
            (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P && shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE))) {
            showPermissionRationaleDialog(
                "Permiso de almacenamiento necesario",
                "La app necesita permiso para escribir en el almacenamiento externo y guardar los logs y resultados del benchmark.",
                { requestPermissions(arrayOf(
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                ), PERMISSION_REQUEST_STORAGE) }
            )
        } else {
            requestPermissions(arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ), PERMISSION_REQUEST_STORAGE)
        }
    }

    private fun showPermissionRationaleDialog(title: String, message: String, onPositive: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("Conceder") { _, _ -> onPositive() }
            .setNegativeButton("Cancelar") { dialog, _ ->
                dialog.dismiss()
                fileLogger.log("Permission denied by user")
                runOnUiThread {
                    tvStatus.text = "Permiso denegado. La app usará almacenamiento interno."
                    btnRunBenchmark.isEnabled = true
                    btnRunBenchmark.text = "INICIAR (solo interno)"
                }
            }
            .setCancelable(false)
            .show()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            PERMISSION_REQUEST_STORAGE, PERMISSION_REQUEST_MEDIA -> {
                val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
                storagePermissionGranted = allGranted
                fileLogger.log("Permission result: requestCode=$requestCode, granted=$allGranted")
                if (allGranted) {
                    fileLogger.onPermissionGranted()
                    initAfterPermissions()
                } else {
                    runOnUiThread {
                        tvStatus.text = "Permiso denegado. Usando almacenamiento interno."
                        btnRunBenchmark.isEnabled = true
                        btnRunBenchmark.text = "INICIAR (solo interno)"
                        Toast.makeText(this, "Permiso denegado. Los logs se guardarán solo en almacenamiento interno.", Toast.LENGTH_LONG).show()
                    }
                    initAfterPermissions() // Continue with internal storage
                }
            }
        }
    }

    private fun initAfterPermissions() {
        runOnUiThread {
            if (storagePermissionGranted) {
                tvStatus.text = "Permisos OK. Descargando modelo..."
            } else {
                tvStatus.text = "Sin permisos externos. Descargando modelo..."
            }
            btnRunBenchmark.isEnabled = false
        }
        downloadModel()
    }

    private fun checkPermissionsAndRunBenchmark() {
        // Re-check permissions at runtime
        val hasPermission = when {
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU ->
                ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ->
                ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
            else ->
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        }

        storagePermissionGranted = hasPermission
        fileLogger.log("Runtime permission check: granted=$hasPermission")

        if (!hasPermission) {
            // Re-request
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                requestMediaPermissions()
            } else {
                requestStoragePermissionLegacy()
            }
            // Still allow running with internal storage
        }

        runBenchmark()
    }

    private fun downloadModel() {
        tvStatus.text = "Descargando modelo..."
        fileLogger.log("Starting model download from $MODEL_URL")
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val client = OkHttpClient.Builder()
                    .followRedirects(true)
                    .followSslRedirects(true)
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(60, TimeUnit.SECONDS)
                    .writeTimeout(30, TimeUnit.SECONDS)
                    .build()

                val request = Request.Builder().url(MODEL_URL).build()
                val response = client.newCall(request).execute()

                fileLogger.log("Download response code: ${response.code}")
                if (!response.isSuccessful) throw Exception("HTTP ${response.code}: ${response.message}")

                val bytes = response.body?.bytes() ?: throw Exception("Empty response body")
                fileLogger.log("Downloaded ${bytes.size} bytes")

                modelFile = File(filesDir, MODEL_FILENAME)
                FileOutputStream(modelFile).use { it.write(bytes) }
                fileLogger.log("Model saved to internal storage: ${modelFile?.absolutePath}")

                // Verify model file
                if (modelFile?.exists() != true) throw Exception("Model file not created")
                if (modelFile?.length() == 0L) throw Exception("Model file is empty")

                runOnUiThread {
                    updateStatus("Modelo listo (${formatBytes(bytes.size.toLong())}). Presiona INICIAR.")
                    btnRunBenchmark.isEnabled = true
                    if (storagePermissionGranted) {
                        btnRunBenchmark.text = "INICIAR BENCHMARK"
                    } else {
                        btnRunBenchmark.text = "INICIAR (solo interno)"
                    }
                }
            } catch (e: Exception) {
                Log.e("Benchmark", "Download error", e)
                fileLogger.log("Download error: ${e.message}")
                fileLogger.log(Log.getStackTraceString(e))
                runOnUiThread {
                    updateStatus("Error descargando: ${e.message}")
                    tvResults.text = "ERROR DE DESCARGA:\n${Log.getStackTraceString(e)}"
                    // Try to use bundled model or cached model
                    tryUseCachedModel()
                }
            }
        }
    }

    private fun tryUseCachedModel() {
        // Check if model already exists in internal storage
        val cachedModel = File(filesDir, MODEL_FILENAME)
        if (cachedModel.exists() && cachedModel.length() > 0) {
            modelFile = cachedModel
            fileLogger.log("Using cached model: ${cachedModel.absolutePath}")
            runOnUiThread {
                updateStatus("Usando modelo cacheado (${formatBytes(cachedModel.length())}). Presiona INICIAR.")
                btnRunBenchmark.isEnabled = true
            }
        } else {
            fileLogger.log("No cached model available")
            runOnUiThread {
                btnRunBenchmark.isEnabled = false
            }
        }
    }

    private fun runBenchmark() {
        tvStatus.text = "Ejecutando benchmark..."
        fileLogger.log("Starting benchmark (storagePermissionGranted=$storagePermissionGranted)")
        btnRunBenchmark.isEnabled = false
        progressBar.visibility = View.VISIBLE
        tvResults.text = ""

        lifecycleScope.launch(Dispatchers.IO) {
            val results = mutableListOf<String>()
            var cpuTime = -1.0
            var gpuTime = -1.0

            try {
                // Test 1: CPU (XNNPACK)
                results.add("=== CPU (XNNPACK) ===")
                fileLogger.log("=== CPU (XNNPACK) ===")
                cpuTime = runInference(useGpu = false)
                if (cpuTime >= 0) {
                    results.add("Tiempo promedio: ${String.format("%.1f", cpuTime)} ms")
                    fileLogger.log("CPU time: ${String.format("%.1f", cpuTime)} ms")
                } else {
                    results.add("FALLÓ: Ver logs")
                    fileLogger.log("CPU inference FAILED")
                }
                results.add("")

                // Test 2: GPU Delegate
                results.add("=== GPU Delegate (OpenGL ES) ===")
                fileLogger.log("=== GPU Delegate (OpenGL ES) ===")
                gpuTime = runInference(useGpu = true)
                if (gpuTime >= 0) {
                    results.add("Tiempo promedio: ${String.format("%.1f", gpuTime)} ms")
                    val speedup = cpuTime / gpuTime
                    results.add("Speedup: ${String.format("%.2f", speedup)}x")
                    fileLogger.log("GPU time: ${String.format("%.1f", gpuTime)} ms, Speedup: ${String.format("%.2f", cpuTime / gpuTime)}x")
                } else {
                    results.add("GPU delegate NO disponible en este dispositivo")
                    fileLogger.log("GPU delegate NOT available")
                }

            } catch (e: Exception) {
                Log.e("Benchmark", "Benchmark error", e)
                fileLogger.log("Benchmark error: ${e.message}")
                fileLogger.log(Log.getStackTraceString(e))
                results.add("ERROR: ${e.message}")
                results.add(Log.getStackTraceString(e))
            } finally {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunBenchmark.isEnabled = true
                    tvResults.text = results.joinToString("\n")
                    tvStatus.text = "Benchmark completado"
                    fileLogger.log("Benchmark completed")

                    // Save results to external storage if permission granted
                    if (storagePermissionGranted) {
                        saveResultsToExternalStorage(results.joinToString("\n"))
                    }
                }
            }
        }
    }

    private fun runInference(useGpu: Boolean): Double {
        val modelPath = modelFile?.absolutePath
        if (modelPath == null) {
            Log.e("Benchmark", "No model file available")
            fileLogger.log("No model file available")
            return -1.0
        }

        var interpreter: Interpreter? = null
        var gpuDelegate: GpuDelegate? = null

        try {
            Log.d("Benchmark", "Creating interpreter for ${if (useGpu) "GPU" else "CPU"}")
            fileLogger.log("Creating interpreter for ${if (useGpu) "GPU" else "CPU"}")
            val options = Interpreter.Options().setNumThreads(4)

            if (useGpu) {
                try {
                    Log.d("Benchmark", "Creating GPU delegate...")
                    fileLogger.log("Creating GPU delegate...")
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    Log.d("Benchmark", "GPU delegate created successfully")
                    fileLogger.log("GPU delegate created successfully")
                } catch (e: Exception) {
                    Log.w("Benchmark", "GPU delegate creation failed: ${e.message}", e)
                    fileLogger.log("GPU delegate creation failed: ${e.message}")
                    fileLogger.log(Log.getStackTraceString(e))
                    return -1.0
                }
            }

            Log.d("Benchmark", "Creating interpreter with model: $modelPath")
            fileLogger.log("Creating interpreter with model: $modelPath")
            interpreter = Interpreter(File(modelPath), options)
            interpreter.allocateTensors()
            Log.d("Benchmark", "Interpreter created and tensors allocated")
            fileLogger.log("Interpreter created and tensors allocated")

            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            val inputShape = inputTensor.shape()
            val outputShape = outputTensor.shape()
            val inputType = inputTensor.dataType()
            val outputType = outputTensor.dataType()

            Log.d("Benchmark", "Input: shape=$inputShape type=$inputType")
            Log.d("Benchmark", "Output: shape=$outputShape type=$outputType")
            fileLogger.log("Input: shape=$inputShape type=$inputType")
            fileLogger.log("Output: shape=$outputShape type=$outputType")

            // Create dummy input
            val inputBuffer = TensorBuffer.createFixedSize(inputShape, inputType)
            val flatSize = inputShape.reduce { a, b -> a * b }

            when (inputType) {
                org.tensorflow.lite.DataType.UINT8 -> {
                    val intValues = IntArray(flatSize)
                    for (i in intValues.indices) intValues[i] = (Math.random() * 255).toInt()
                    inputBuffer.loadArray(intValues)
                }
                org.tensorflow.lite.DataType.FLOAT32 -> {
                    val floatValues = FloatArray(flatSize)
                    for (i in floatValues.indices) floatValues[i] = Math.random().toFloat() * 255f
                    inputBuffer.loadArray(floatValues)
                }
                else -> {
                    Log.e("Benchmark", "Unsupported input type: $inputType")
                    fileLogger.log("Unsupported input type: $inputType")
                    return -1.0
                }
            }

            val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType)

            // Warmup
            Log.d("Benchmark", "Warmup...")
            fileLogger.log("Warmup...")
            repeat(3) { interpreter.run(inputBuffer.buffer, outputBuffer.buffer) }

            // Benchmark
            val iterations = 10
            val times = LongArray(iterations)
            Log.d("Benchmark", "Running $iterations iterations...")
            fileLogger.log("Running $iterations iterations...")
            for (i in 0 until iterations) {
                val start = System.nanoTime()
                interpreter.run(inputBuffer.buffer, outputBuffer.buffer)
                times[i] = System.nanoTime() - start
            }

            val avgMs: Double = times.drop(2).average().toDouble() / 1_000_000.0
            Log.d("Benchmark", "Average time: $avgMs ms")
            fileLogger.log("Average time: $avgMs ms")
            return avgMs

        } catch (e: Exception) {
            Log.e("Benchmark", "Inference error for ${if (useGpu) "GPU" else "CPU"}", e)
            fileLogger.log("Inference error for ${if (useGpu) "GPU" else "CPU"}: ${e.message}")
            fileLogger.log(Log.getStackTraceString(e))
            return -1.0
        } finally {
            try { interpreter?.close() } catch (e: Exception) { Log.e("Benchmark", "Error closing interpreter", e); fileLogger.log("Error closing interpreter: ${e.message}") }
            try { gpuDelegate?.close() } catch (e: Exception) { Log.e("Benchmark", "Error closing GPU delegate", e); fileLogger.log("Error closing GPU delegate: ${e.message}") }
        }
    }

    private fun saveResultsToExternalStorage(results: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                val fileName = "benchmark_result_$timestamp.txt"

                // Use MediaStore to save to Downloads folder (Android 10+)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    val resolver = contentResolver
                    val contentValues = android.content.ContentValues().apply {
                        put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                        put(MediaStore.MediaColumns.MIME_TYPE, "text/plain")
                        put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                    }
                    val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)
                    uri?.let { outputUri ->
                        resolver.openOutputStream(outputUri)?.use { outputStream ->
                            outputStream.write(results.toByteArray())
                            fileLogger.log("Results saved to external storage: $outputUri")
                            runOnUiThread {
                                Toast.makeText(this@MainActivity, "Resultados guardados en Descargas/$fileName", Toast.LENGTH_LONG).show()
                            }
                        }
                    }
                } else {
                    // Legacy: use Environment.getExternalStoragePublicDirectory
                    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    val file = File(downloadsDir, fileName)
                    FileOutputStream(file).use { it.write(results.toByteArray()) }
                    fileLogger.log("Results saved to external storage (legacy): ${file.absolutePath}")
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Resultados guardados en Descargas/$fileName", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                Log.e("Benchmark", "Failed to save results to external storage", e)
                fileLogger.log("Failed to save results to external storage: ${e.message}")
                fileLogger.log(Log.getStackTraceString(e))
            }
        }
    }

    private fun openLogFile() {
        fileLogger.log("User requested to open log file")
        val logFile = fileLogger.getLogFile()

        if (!logFile.exists()) {
            Toast.makeText(this, "Archivo de log no encontrado", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val uri: Uri
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", logFile)
            } else {
                uri = Uri.fromFile(logFile)
            }

            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "text/plain")
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }

            if (intent.resolveActivity(packageManager) != null) {
                startActivity(intent)
                fileLogger.log("Opened log file with external viewer")
            } else {
                // Fallback: show log content in app
                showLogContentInApp()
            }
        } catch (e: Exception) {
            Log.e("Benchmark", "Error opening log file", e)
            fileLogger.log("Error opening log file: ${e.message}")
            showLogContentInApp()
        }
    }

    private fun showLogContentInApp() {
        val logFile = fileLogger.getLogFile()
        val content = logFile.readText()
        AlertDialog.Builder(this)
            .setTitle("Log de la aplicación")
            .setMessage(content)
            .setPositiveButton("Copiar al portapapeles") { _, _ ->
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                val clip = android.content.ClipData.newPlainText("Benchmark Log", content)
                clipboard.primaryClip = clip
                Toast.makeText(this, "Log copiado al portapapeles", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cerrar", null)
            .show()
    }

    private fun updateStatus(msg: String) {
        tvStatus.text = msg
        Log.d("Benchmark", msg)
        fileLogger.log(msg)
    }

    private fun formatBytes(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> "${bytes / 1024} KB"
            else -> "${bytes / (1024 * 1024)} MB"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        fileLogger.log("=== App destroyed ===")
    }
}

class FileLogger(private val context: Context) {
    private val internalLogFile: File
    private var externalLogFile: File? = null
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault())
    private val logLock = Any()

    init {
        // Internal storage (always available, no permissions needed)
        internalLogFile = File(context.filesDir, "benchmark_log.txt")
        logInternal("=== FileLogger initialized ===")
        logInternal("Internal log: ${internalLogFile.absolutePath}")

        // Try to initialize external log file if permissions allow
        tryInitExternalLog()
    }

    private fun tryInitExternalLog() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // On Android 10+, we can try to create a file in Downloads via MediaStore
            // But we'll defer actual creation until first write with permission
            externalLogFile = null // Will be created on first write with permission
        } else {
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (downloadsDir.exists() && downloadsDir.canWrite()) {
                externalLogFile = File(downloadsDir, "benchmark_log.txt")
                logInternal("External log (legacy): ${externalLogFile?.absolutePath}")
            }
        }
    }

    fun log(message: String) {
        val timestamp = dateFormat.format(Date())
        val logLine = "[$timestamp] $message"

        synchronized(logLock) {
            // Always write to internal storage
            writeToFile(internalLogFile, logLine)

            // Also write to external storage if available
            externalLogFile?.let { writeToFile(it, logLine) }
        }

        // Also log to Android logcat
        Log.d("Benchmark", message)
    }

    private fun writeToFile(file: File, line: String) {
        try {
            BufferedWriter(FileWriter(file, true)).use { writer ->
                writer.write(line)
                writer.newLine()
            }
        } catch (e: Exception) {
            Log.e("FileLogger", "Failed to write log to ${file.absolutePath}", e)
        }
    }

    /** Call this when storage permission is granted to create/write external log */
    fun onPermissionGranted() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // On Android 10+, we'll create the external log file on first write via MediaStore
            // For now, just mark that we have permission
            logInternal("Storage permission granted, external logging enabled for Android 10+")
        } else {
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (downloadsDir.exists() && downloadsDir.canWrite()) {
                externalLogFile = File(downloadsDir, "benchmark_log.txt")
                // Copy existing internal log to external
                if (internalLogFile.exists()) {
                    internalLogFile.copyTo(externalLogFile!!, true)
                }
                logInternal("External log enabled: ${externalLogFile?.absolutePath}")
            }
        }
    }

    fun getLogFile(): File {
        return internalLogFile
    }

    private fun logInternal(message: String) {
        val timestamp = dateFormat.format(Date())
        val logLine = "[$timestamp] $message"
        writeToFile(internalLogFile, logLine)
        Log.d("Benchmark", message)
    }
}

// Extension function for file copy
private fun File.copyTo(target: File, overwrite: Boolean) {
    if (target.exists() && !overwrite) return
    target.parentFile?.mkdirs()
    copyTo(target, 8192)
}