package com.example.mobilenetgpubenchmark

import android.app.Application
import android.content.Context
import androidx.multidex.MultiDex

class BenchmarkApplication : Application() {
    override fun attachBaseContext(base: Context) {
        super.attachBaseContext(base)
        MultiDex.install(this)
    }
}