package com.example.mobilenetgpubenchmark

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.ProgressBar
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvResults: TextView
    private lateinit var btnRunBenchmark: Button
    private lateinit var progressBar: ProgressBar

    private val MODEL_URL = "https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant.tflite"
    private val MODEL_FILENAME = "mobilenet_v1_1.0_224_quant.tflite"
    private var modelFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Global exception handler to prevent silent crashes
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            Log.e("Benchmark", "UNCAUGHT EXCEPTION in ${thread.name}", throwable)
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
            progressBar = findViewById(R.id.progressBar)

            btnRunBenchmark.setOnClickListener { runBenchmark() }
            tvResults.text = ""
            tvStatus.text = "Iniciando..."

            // Download model on startup
            downloadModel()
        } catch (e: Exception) {
            Log.e("Benchmark", "onCreate error", e)
            tvStatus.text = "onCreate CRASH: ${e.message}"
            tvResults.text = Log.getStackTraceString(e)
        }
    }

    private fun downloadModel() {
        tvStatus.text = "Descargando modelo..."
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d("Benchmark", "Starting download from $MODEL_URL")
                val client = OkHttpClient.Builder()
                    .followRedirects(true)
                    .followSslRedirects(true)
                    .build()
                val request = Request.Builder().url(MODEL_URL).build()
                val response = client.newCall(request).execute()

                Log.d("Benchmark", "Response code: ${response.code}")
                if (!response.isSuccessful) throw Exception("HTTP ${response.code}: ${response.message}")

                val bytes = response.body?.bytes() ?: throw Exception("Empty response body")
                Log.d("Benchmark", "Downloaded ${bytes.size} bytes")

                modelFile = File(filesDir, MODEL_FILENAME)
                FileOutputStream(modelFile).use { it.write(bytes) }
                Log.d("Benchmark", "Model saved to ${modelFile?.absolutePath}")

                // Verify model file
                if (modelFile?.exists() != true) throw Exception("Model file not created")
                if (modelFile?.length() == 0L) throw Exception("Model file is empty")

                runOnUiThread {
                    updateStatus("Modelo listo (${formatBytes(bytes.size.toLong())}). Presiona INICIAR.")
                    btnRunBenchmark.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e("Benchmark", "Download error", e)
                runOnUiThread {
                    updateStatus("Error descargando: ${e.message}")
                    tvResults.text = Log.getStackTraceString(e)
                }
            }
        }
    }

    private fun runBenchmark() {
        tvStatus.text = "Ejecutando benchmark..."
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
                cpuTime = runInference(useGpu = false)
                if (cpuTime >= 0) {
                    results.add("Tiempo promedio: ${String.format("%.1f", cpuTime)} ms")
                } else {
                    results.add("FALLÓ: Ver logs")
                }
                results.add("")

                // Test 2: GPU Delegate
                results.add("=== GPU Delegate (OpenGL ES) ===")
                gpuTime = runInference(useGpu = true)
                if (gpuTime >= 0) {
                    results.add("Tiempo promedio: ${String.format("%.1f", gpuTime)} ms")
                    val speedup = cpuTime / gpuTime
                    results.add("Speedup: ${String.format("%.2f", speedup)}x")
                } else {
                    results.add("GPU delegate NO disponible en este dispositivo")
                }

            } catch (e: Exception) {
                Log.e("Benchmark", "Benchmark error", e)
                results.add("ERROR: ${e.message}")
                results.add(Log.getStackTraceString(e))
            } finally {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunBenchmark.isEnabled = true
                    tvResults.text = results.joinToString("\n")
                    tvStatus.text = "Benchmark completado"
                }
            }
        }
    }

    private fun runInference(useGpu: Boolean): Double {
        val modelPath = modelFile?.absolutePath
        if (modelPath == null) {
            Log.e("Benchmark", "No model file available")
            return -1.0
        }

        var interpreter: Interpreter? = null
        var gpuDelegate: GpuDelegate? = null

        try {
            Log.d("Benchmark", "Creating interpreter for ${if (useGpu) "GPU" else "CPU"}")
            val options = Interpreter.Options().setNumThreads(4)

            if (useGpu) {
                try {
                    Log.d("Benchmark", "Creating GPU delegate...")
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    Log.d("Benchmark", "GPU delegate created successfully")
                } catch (e: Exception) {
                    Log.w("Benchmark", "GPU delegate creation failed: ${e.message}", e)
                    return -1.0
                }
            }

            Log.d("Benchmark", "Creating interpreter with model: $modelPath")
            interpreter = Interpreter(File(modelPath), options)
            interpreter.allocateTensors()
            Log.d("Benchmark", "Interpreter created and tensors allocated")

            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            val inputShape = inputTensor.shape()
            val outputShape = outputTensor.shape()
            val inputType = inputTensor.dataType()
            val outputType = outputTensor.dataType()

            Log.d("Benchmark", "Input: shape=$inputShape type=$inputType")
            Log.d("Benchmark", "Output: shape=$outputShape type=$outputType")

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
                    return -1.0
                }
            }

            val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType)

            // Warmup
            Log.d("Benchmark", "Warmup...")
            repeat(3) { interpreter.run(inputBuffer.buffer, outputBuffer.buffer) }

            // Benchmark
            val iterations = 10
            val times = LongArray(iterations)
            Log.d("Benchmark", "Running $iterations iterations...")
            for (i in 0 until iterations) {
                val start = System.nanoTime()
                interpreter.run(inputBuffer.buffer, outputBuffer.buffer)
                times[i] = System.nanoTime() - start
            }

            val avgMs: Double = times.drop(2).average().toDouble() / 1_000_000.0
            Log.d("Benchmark", "Average time: $avgMs ms")
            return avgMs

        } catch (e: Exception) {
            Log.e("Benchmark", "Inference error for ${if (useGpu) "GPU" else "CPU"}", e)
            return -1.0
        } finally {
            try { interpreter?.close() } catch (e: Exception) { Log.e("Benchmark", "Error closing interpreter", e) }
            try { gpuDelegate?.close() } catch (e: Exception) { Log.e("Benchmark", "Error closing GPU delegate", e) }
        }
    }

    private fun updateStatus(msg: String) {
        tvStatus.text = msg
        Log.d("Benchmark", msg)
    }

    private fun formatBytes(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> "${bytes / 1024} KB"
            else -> "${bytes / (1024 * 1024)} MB"
        }
    }
}