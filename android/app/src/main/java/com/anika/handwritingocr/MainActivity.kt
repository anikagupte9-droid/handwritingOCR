package com.anika.handwritingocr

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.ScrollView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


class MainActivity : ComponentActivity() {

    private lateinit var root: LinearLayout
    private lateinit var screenScrollView: ScrollView

    private lateinit var title: TextView
    private lateinit var subtitle: TextView

    private lateinit var imageContainer: CropImageView

    private lateinit var useCropButton: Button
    private lateinit var recognizeButton: Button
    private lateinit var benchmarkButton: Button
    private lateinit var cameraButton: Button
    private lateinit var galleryButton: Button

    private lateinit var statusText: TextView
    private lateinit var resultText: TextView
    private lateinit var progressBar: ProgressBar

    private var originalBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null

    private val galleryLauncher =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri: Uri? ->

            if (uri != null) {

                try {

                    val bitmap =
                        contentResolver
                            .openInputStream(uri)
                            ?.use {
                                BitmapFactory.decodeStream(it)
                            }

                    if (bitmap != null) {
                        showImage(bitmap)
                    }

                } catch (e: Exception) {

                    statusText.text =
                        "Could not open image."
                }
            }
        }


    private val cameraLauncher =
        registerForActivityResult(
            ActivityResultContracts.TakePicturePreview()
        ) { bitmap: Bitmap? ->

            if (bitmap != null) {
                showImage(bitmap)
            }
        }


    private val cameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (granted) {
                cameraLauncher.launch(null)
            } else {
                statusText.text =
                    "Camera permission is required."
            }
        }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        buildUI()
    }


    private fun buildUI() {

        root = LinearLayout(this)

        root.orientation =
            LinearLayout.VERTICAL

        root.gravity =
            Gravity.CENTER_HORIZONTAL

        root.setPadding(
            30,
            30,
            30,
            30
        )

        root.setBackgroundColor(
            android.graphics.Color.WHITE
        )


        title = TextView(this)

        title.text =
            "Offline Handwriting OCR"

        title.textSize = 30f

        title.setTextColor(
            android.graphics.Color.BLACK
        )

        title.gravity =
            Gravity.CENTER

        root.addView(
            title,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        )


        subtitle = TextView(this)

        subtitle.text =
            "Upload an image or take a photo."

        subtitle.textSize = 17f

        subtitle.setTextColor(
            android.graphics.Color.DKGRAY
        )

        subtitle.gravity =
            Gravity.CENTER

        subtitle.setPadding(
            0,
            15,
            0,
            25
        )

        root.addView(
            subtitle,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        )


        imageContainer =
            CropImageView(this)

        imageContainer.visibility =
            View.GONE

        val imageHeight = (280 * resources.displayMetrics.density).toInt()

        root.addView(
            imageContainer,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                imageHeight
            )
        )


        statusText = TextView(this)

        statusText.textSize = 16f

        statusText.setTextColor(
            android.graphics.Color.DKGRAY
        )

        statusText.gravity =
            Gravity.CENTER

        statusText.setPadding(
            0,
            12,
            0,
            12
        )

        root.addView(
            statusText,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        )


        useCropButton =
            createButton("Use This Line")

        useCropButton.visibility =
            View.GONE

        useCropButton.setOnClickListener {
            createCrop()
        }

        root.addView(useCropButton)


        recognizeButton =
            createButton("Recognize Text")

        recognizeButton.visibility =
            View.GONE

        recognizeButton.setOnClickListener {
            recognize()
        }

        root.addView(recognizeButton)

        benchmarkButton =
            createButton("Compare FP32 / FP16 / INT8")

        benchmarkButton.visibility = View.GONE

        benchmarkButton.setOnClickListener {
            runAllModels()
        }

        root.addView(benchmarkButton)


        cameraButton =
            createButton("Take Photo")

        cameraButton.setOnClickListener {
            openCamera()
        }

        root.addView(cameraButton)


        galleryButton =
            createButton("Upload Image")

        galleryButton.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        root.addView(galleryButton)


        progressBar =
            ProgressBar(this)

        progressBar.visibility =
            View.GONE

        root.addView(
            progressBar,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        )


        resultText = TextView(this)

        resultText.textSize = 17f

        resultText.setTextColor(
            android.graphics.Color.BLACK
        )

        resultText.setPadding(
            0,
            20,
            0,
            10
        )

        root.addView(
            resultText,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        )


        screenScrollView = ScrollView(this).apply {
            isFillViewport = true
            isVerticalScrollBarEnabled = true
            overScrollMode = View.OVER_SCROLL_IF_CONTENT_SCROLLS
            addView(
                root,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
            )
        }

        setContentView(screenScrollView)
    }


    private fun createButton(
        text: String
    ): Button {

        val button =
            Button(this)

        button.text = text

        button.textSize = 16f

        val params =
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )

        params.setMargins(
            0,
            6,
            0,
            6
        )

        button.layoutParams = params

        return button
    }


    private fun showImage(
        bitmap: Bitmap
    ) {

        originalBitmap = bitmap
        croppedBitmap = null

        resultText.text = ""

        imageContainer.setBitmap(bitmap)

        imageContainer.visibility =
            View.VISIBLE

        useCropButton.visibility =
            View.VISIBLE

        recognizeButton.visibility =
            View.GONE

        benchmarkButton.visibility =
            View.GONE

        statusText.text =
            "Move the white box over ONE complete handwriting line."

        title.text =
            "Crop one handwriting line"

        subtitle.text =
            "Drag the box anywhere over the handwriting."
    }


    private fun createCrop() {

        val bitmap =
            originalBitmap

        if (bitmap == null) {
            return
        }

        val crop =
            imageContainer.getCropBitmap()

        if (crop == null) {

            statusText.text =
                "Please select a valid crop."

            return
        }

        /*
         * TrOCR expects 384 × 384.
         */
        croppedBitmap =
            Bitmap.createScaledBitmap(
                crop,
                384,
                384,
                true
            )

        statusText.text =
            "Ready: 384 × 384"

        recognizeButton.visibility =
            View.VISIBLE

        benchmarkButton.visibility =
            View.VISIBLE

        resultText.text = ""
    }


    private fun recognize() {

        val bitmap = croppedBitmap

        if (bitmap == null) {
            statusText.text = "Please press Use This Line first."
            return
        }

        recognizeButton.isEnabled = false
        progressBar.visibility = View.VISIBLE
        statusText.text = "Running offline OCR..."

        lifecycleScope.launch(Dispatchers.Default) {

            var engine: TrocrEngine? = null

            try {
                engine = TrocrEngine(this@MainActivity)

                val output = engine.recognize(bitmap)
                val latency = engine.inferenceTimeMs

                engine.close()
                engine = null

                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    recognizeButton.isEnabled = true
                    statusText.text = "Recognition complete"

                    resultText.text = """
                        Recognized Text:

                        $output


                        MODEL PERFORMANCE

                        Model: ONNX TrOCR
                        Precision: INT8
                        Model Size: 60.23 MB

                        Phone Inference Time:
                        ${"%.2f".format(latency)} ms

                        Character Error Rate:
                        Not measured on phone yet
                    """.trimIndent()
                }

            } catch (e: Exception) {
                engine?.close()

                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    recognizeButton.isEnabled = true
                    statusText.text = "OCR failed"
                    resultText.text = e.stackTraceToString()
                }
            }
        }
    }



    private fun runAllModels() {

        val bitmap = croppedBitmap

        if (bitmap == null) {
            statusText.text = "Please press Use This Line first."
            return
        }

        recognizeButton.isEnabled = false
        benchmarkButton.isEnabled = false
        progressBar.visibility = View.VISIBLE
        statusText.text = "Benchmarking FP32, FP16 and INT8..."

        lifecycleScope.launch(Dispatchers.Default) {

            data class ModelResult(
                val name: String,
                val precision: String,
                val sizeMb: String,
                val latencyMs: Double,
                val text: String
            )

            val results = mutableListOf<ModelResult>()
            val models = listOf(
                Triple("onnx_fp32", "FP32", "246 MB"),
                Triple("onnx_fp16", "FP16", "117.78 MB"),
                Triple("onnx_int8", "INT8", "60.23 MB")
            )

            var failureMessage: String? = null

            for ((folder, precision, size) in models) {
                var engine: TrocrEngine? = null

                try {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Running $precision model..."
                    }

                    engine = TrocrEngine(
                        this@MainActivity,
                        folder
                    )

                    val output = engine.recognize(bitmap)
                    val latency = engine.inferenceTimeMs

                    results.add(
                        ModelResult(
                            name = "ONNX TrOCR",
                            precision = precision,
                            sizeMb = size,
                            latencyMs = latency,
                            text = output
                        )
                    )

                } catch (e: Exception) {
                    failureMessage =
                        "$precision failed: ${e.message}"
                    break
                } finally {
                    engine?.close()
                    engine = null
                    System.gc()
                }
            }

            withContext(Dispatchers.Main) {
                progressBar.visibility = View.GONE
                recognizeButton.isEnabled = true
                benchmarkButton.isEnabled = true

                if (results.isEmpty()) {
                    statusText.text = "Benchmark failed"
                    resultText.text = failureMessage ?: "No results."
                    return@withContext
                }

                statusText.text =
                    if (failureMessage == null) {
                        "Model comparison complete"
                    } else {
                        "Benchmark partially complete"
                    }

                val report = buildString {
                    appendLine("MODEL COMPARISON")
                    appendLine("Same cropped image used for all models")
                    appendLine()

                    for (result in results) {
                        appendLine("${result.name} — ${result.precision}")
                        appendLine("Model size: ${result.sizeMb}")
                        appendLine(
                            "Phone inference: ${"%.2f".format(result.latencyMs)} ms"
                        )
                        appendLine("Recognized text:")
                        appendLine(result.text)
                        appendLine()
                        appendLine("--------------------------------")
                        appendLine()
                    }

                    appendLine("CER")
                    appendLine(
                        "Not calculated in this phone benchmark yet."
                    )

                    if (failureMessage != null) {
                        appendLine()
                        appendLine("ERROR")
                        appendLine(failureMessage)
                    }
                }

                resultText.text = report

                screenScrollView.post {
                    screenScrollView.smoothScrollTo(
                        0,
                        resultText.bottom
                    )
                }
            }
        }
    }

    private fun openCamera() {

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) ==
            PackageManager.PERMISSION_GRANTED
        ) {

            cameraLauncher.launch(null)

        } else {

            cameraPermissionLauncher.launch(
                Manifest.permission.CAMERA
            )
        }
    }
}


/*
 * Custom image view with a movable crop box.
 *
 * The crop box can move all the way to the
 * left/right/top/bottom edges of the image.
 */
private class CropImageView(
    context: android.content.Context
) : View(context) {

    private var bitmap: Bitmap? = null

    private val imageRect =
        RectFCompat()

    private var scale = 1f

    private var cropLeft = 0f
    private var cropTop = 0f

    private var cropWidth = 0f
    private var cropHeight = 0f

    private var dragging = false

    private var lastTouchX = 0f
    private var lastTouchY = 0f

    private val imagePaint =
        Paint(
            Paint.ANTI_ALIAS_FLAG or
                    Paint.FILTER_BITMAP_FLAG
        )

    private val borderPaint =
        Paint(
            Paint.ANTI_ALIAS_FLAG
        )

    init {

        borderPaint.style =
            Paint.Style.STROKE

        borderPaint.strokeWidth =
            5f

        borderPaint.color =
            android.graphics.Color.WHITE
    }


    fun setBitmap(
        newBitmap: Bitmap
    ) {

        bitmap = newBitmap

        cropLeft = 0f
        cropTop = 0f

        invalidate()
    }


    override fun onDraw(
        canvas: Canvas
    ) {

        super.onDraw(canvas)

        val source =
            bitmap ?: return

        val viewWidth =
            width.toFloat()

        val viewHeight =
            height.toFloat()

        if (
            viewWidth <= 0 ||
            viewHeight <= 0
        ) {
            return
        }

        scale =
            min(
                viewWidth / source.width,
                viewHeight / source.height
            )

        val displayedWidth =
            source.width * scale

        val displayedHeight =
            source.height * scale

        val left =
            (viewWidth - displayedWidth) / 2f

        val top =
            (viewHeight - displayedHeight) / 2f

        imageRect.set(
            left,
            top,
            left + displayedWidth,
            top + displayedHeight
        )

        canvas.drawBitmap(
            source,
            null,
            imageRect.toAndroidRectF(),
            imagePaint
        )

        /*
         * Wide crop box for a handwriting line.
         */
        cropWidth =
            displayedWidth * 0.90f

        cropHeight =
            displayedHeight * 0.16f

        if (cropHeight < 80f) {
            cropHeight = 80f
        }

        cropHeight =
            min(
                cropHeight,
                displayedHeight
            )

        /*
         * Keep crop inside image.
         */
        cropLeft =
            cropLeft.coerceIn(
                0f,
                displayedWidth - cropWidth
            )

        cropTop =
            cropTop.coerceIn(
                0f,
                displayedHeight - cropHeight
            )

        val cropRect =
            android.graphics.RectF(
                left + cropLeft,
                top + cropTop,
                left + cropLeft + cropWidth,
                top + cropTop + cropHeight
            )

        canvas.drawRect(
            cropRect,
            borderPaint
        )
    }


    override fun onTouchEvent(
        event: MotionEvent
    ): Boolean {

        when (event.actionMasked) {

            MotionEvent.ACTION_DOWN -> {

                val source =
                    bitmap ?: return false

                val viewWidth =
                    width.toFloat()

                val viewHeight =
                    height.toFloat()

                val displayedWidth =
                    source.width * scale

                val displayedHeight =
                    source.height * scale

                val imageLeft =
                    (viewWidth -
                            displayedWidth) / 2f

                val imageTop =
                    (viewHeight -
                            displayedHeight) / 2f

                val x =
                    event.x - imageLeft

                val y =
                    event.y - imageTop

                if (
                    x >= cropLeft &&
                    x <= cropLeft + cropWidth &&
                    y >= cropTop &&
                    y <= cropTop + cropHeight
                ) {

                    dragging = true

                    lastTouchX =
                        event.x

                    lastTouchY =
                        event.y

                    return true
                }
            }


            MotionEvent.ACTION_MOVE -> {

                if (!dragging) {
                    return true
                }

                val dx =
                    event.x - lastTouchX

                val dy =
                    event.y - lastTouchY

                cropLeft =
                    (
                            cropLeft + dx
                            ).coerceIn(
                            0f,
                            max(
                                0f,
                                imageRect.width() -
                                        cropWidth
                            )
                        )

                cropTop =
                    (
                            cropTop + dy
                            ).coerceIn(
                            0f,
                            max(
                                0f,
                                imageRect.height() -
                                        cropHeight
                            )
                        )

                lastTouchX =
                    event.x

                lastTouchY =
                    event.y

                invalidate()

                return true
            }


            MotionEvent.ACTION_UP,
            MotionEvent.ACTION_CANCEL -> {

                dragging = false

                return true
            }
        }

        return true
    }


    fun getCropBitmap(): Bitmap? {

        val source =
            bitmap ?: return null

        if (
            scale <= 0 ||
            cropWidth <= 0 ||
            cropHeight <= 0
        ) {
            return null
        }

        val sourceX =
            (
                    cropLeft / scale
                    ).roundToInt()
                .coerceIn(
                    0,
                    source.width - 1
                )

        val sourceY =
            (
                    cropTop / scale
                    ).roundToInt()
                .coerceIn(
                    0,
                    source.height - 1
                )

        val sourceWidth =
            (
                    cropWidth / scale
                    ).roundToInt()
                .coerceAtMost(
                    source.width - sourceX
                )

        val sourceHeight =
            (
                    cropHeight / scale
                    ).roundToInt()
                .coerceAtMost(
                    source.height - sourceY
                )

        if (
            sourceWidth <= 0 ||
            sourceHeight <= 0
        ) {
            return null
        }

        return Bitmap.createBitmap(
            source,
            sourceX,
            sourceY,
            sourceWidth,
            sourceHeight
        )
    }
}


/*
 * Tiny RectF wrapper so the custom View stays
 * independent of Compose.
 */
private class RectFCompat {

    private var leftValue = 0f
    private var topValue = 0f
    private var rightValue = 0f
    private var bottomValue = 0f

    fun set(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float
    ) {

        leftValue = left
        topValue = top
        rightValue = right
        bottomValue = bottom
    }

    fun width(): Float =
        rightValue - leftValue

    fun height(): Float =
        bottomValue - topValue

    fun toAndroidRectF():
            android.graphics.RectF {

        return android.graphics.RectF(
            leftValue,
            topValue,
            rightValue,
            bottomValue
        )
    }
}S