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
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
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
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvResults = findViewById(R.id.tvResults)
        btnRunBenchmark = findViewById(R.id.btnRunBenchmark)
        progressBar = findViewById(R.id.progressBar)

        btnRunBenchmark.setOnClickListener { runBenchmark() }
        tvResults.text = ""

        // Download model on startup
        downloadModel()
    }

    private fun downloadModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            updateStatus("Descargando modelo MobileNet...")
            try {
                val client = OkHttpClient()
                val request = Request.Builder().url(MODEL_URL).build()
                val response = client.newCall(request).execute()

                if (!response.isSuccessful) throw Exception("HTTP ${response.code}")

                val bytes = response.body?.bytes() ?: throw Exception("Empty response")
                modelFile = File(filesDir, MODEL_FILENAME)
                FileOutputStream(modelFile).use { it.write(bytes) }

                runOnUiThread {
                    updateStatus("Modelo listo (${formatBytes(bytes.size)}). Presiona INICIAR.")
                    btnRunBenchmark.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    updateStatus("Error descargando: ${e.message}")
                    Log.e("Benchmark", "Download error", e)
                }
            }
        }
    }

    private fun runBenchmark() {
        btnRunBenchmark.isEnabled = false
        progressBar.visibility = View.VISIBLE
        tvResults.text = ""

        lifecycleScope.launch(Dispatchers.IO) {
            val results = mutableListOf<String>()

            // Test 1: CPU (XNNPACK)
            results.add("=== CPU (XNNPACK) ===")
            val cpuTime = runInference(useGpu = false)
            results.add("Tiempo promedio: ${String.format("%.1f", cpuTime)} ms")
            results.add("")

            // Test 2: GPU Delegate
            results.add("=== GPU Delegate (OpenGL ES) ===")
            val gpuTime = runInference(useGpu = true)
            if (gpuTime >= 0) {
                results.add("Tiempo promedio: ${String.format("%.1f", gpuTime)} ms")
                val speedup = cpuTime / gpuTime
                results.add("Speedup: ${String.format("%.2f", speedup)}x")
            } else {
                results.add("GPU delegate NO disponible en este dispositivo")
            }

            runOnUiThread {
                progressBar.visibility = View.GONE
                btnRunBenchmark.isEnabled = true
                tvResults.text = results.joinToString("\n")
                updateStatus("Benchmark completado")
            }
        }
    }

    private fun runInference(useGpu: Boolean): Double {
        val modelPath = modelFile?.absolutePath ?: return -1.0
        var interpreter: Interpreter? = null
        var gpuDelegate: GpuDelegate? = null

        try {
            val options = Interpreter.Options().setNumThreads(4)

            if (useGpu) {
                try {
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    Log.d("Benchmark", "GPU delegate creado exitosamente")
                } catch (e: Exception) {
                    Log.w("Benchmark", "GPU delegate falló: ${e.message}")
                    return -1.0
                }
            }

            interpreter = Interpreter(File(modelPath), options)
            interpreter.allocateTensors()

            val inputShape = interpreter.getInputTensor(0).shape()
            val outputShape = interpreter.getOutputTensor(0).shape()

            // Create dummy input (random noise)
            val inputBuffer = TensorBuffer.createFixedSize(inputShape, interpreter.getInputTensor(0).dataType())
            val floatValues = FloatArray(inputShape.reduce { a, b -> a * b })
            for (i in floatValues.indices) floatValues[i] = (Math.random() * 255).toFloat()
            inputBuffer.loadArray(floatValues)

            val outputBuffer = TensorBuffer.createFixedSize(outputShape, interpreter.getOutputTensor(0).dataType())

            // Warmup
            repeat(3) { interpreter.run(inputBuffer.buffer, outputBuffer.buffer) }

            // Benchmark
            val iterations = 20
            val times = LongArray(iterations)
            for (i in 0 until iterations) {
                val start = System.nanoTime()
                interpreter.run(inputBuffer.buffer, outputBuffer.buffer)
                times[i] = System.nanoTime() - start
            }

            val avgMs: Double = times.drop(2).average().toDouble() / 1_000_000.0
            return avgMs

        } catch (e: Exception) {
            Log.e("Benchmark", "Inference error", e)
            return -1.0
        } finally {
            interpreter?.close()
            gpuDelegate?.close()
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