package com.anika.handwritingocr

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.os.SystemClock
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.min

class TrocrEngine(
    private val context: Context,
    private val modelFolder: String = "onnx_int8"
) {

    companion object {
        private const val TAG = "TrOCR"
        private const val IMAGE_SIZE = 384

        private const val START_TOKEN = 2L
        private const val EOS_TOKEN = 2L
        private const val PAD_TOKEN = 1L

        private const val TOKENIZER_VOCAB_SIZE = 64002
        private const val MAX_TOKENS = 32
    }

    private val ortEnvironment =
        OrtEnvironment.getEnvironment()

    private val sessionOptions =
        OrtSession.SessionOptions()

    private val encoderSession: OrtSession
    private val decoderSession: OrtSession

    private val vocabulary: Map<Int, String>

    var inferenceTimeMs: Double = 0.0
        private set

    init {
        Log.d(TAG, "Loading model: $modelFolder")

        val encoderFile =
            extractAssetToFile("$modelFolder/encoder_model.onnx")

        val decoderFile =
            extractAssetToFile("$modelFolder/decoder_model.onnx")

        Log.d(TAG, "Loading encoder...")
        encoderSession = ortEnvironment.createSession(
            encoderFile.absolutePath,
            sessionOptions
        )

        Log.d(TAG, "Loading decoder...")
        decoderSession = ortEnvironment.createSession(
            decoderFile.absolutePath,
            sessionOptions
        )

        Log.d(TAG, "Loading vocabulary...")
        vocabulary = loadVocabulary()

        Log.d(TAG, "Vocabulary loaded: ${vocabulary.size} tokens")
    }

    // =========================================================
    // MODEL EXTRACTION
    // =========================================================

    private fun extractAssetToFile(assetPath: String): File {

        val modelDirectory = File(
            context.filesDir,
            "trocr_models/$modelFolder"
        )

        if (!modelDirectory.exists()) {
            modelDirectory.mkdirs()
        }

        val filename = assetPath.substringAfterLast("/")
        val outputFile = File(modelDirectory, filename)

        if (outputFile.exists() && outputFile.length() > 0L) {
            return outputFile
        }

        Log.d(TAG, "Extracting $assetPath...")

        val temporaryFile = File(
            modelDirectory,
            "$filename.tmp"
        )

        try {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(temporaryFile).use { output ->
                    input.copyTo(output, bufferSize = 64 * 1024)
                }
            }

            if (!temporaryFile.renameTo(outputFile)) {
                temporaryFile.copyTo(outputFile, overwrite = true)
                temporaryFile.delete()
            }

        } catch (e: Exception) {
            temporaryFile.delete()
            throw e
        }

        Log.d(
            TAG,
            "Extracted ${outputFile.length() / (1024 * 1024)} MB"
        )

        return outputFile
    }

    // =========================================================
    // VOCABULARY
    // =========================================================

    private fun loadVocabulary(): Map<Int, String> {

        val jsonText =
            context.assets.open("vocab.json")
                .bufferedReader()
                .use { it.readText() }

        val json = JSONObject(jsonText)
        val result = HashMap<Int, String>()
        val keys = json.keys()

        while (keys.hasNext()) {

            val key = keys.next()

            try {
                val value = json.get(key)
                val id = key.toIntOrNull()

                if (id != null && value is String) {
                    result[id] = value
                }

            } catch (e: Exception) {
                Log.w(TAG, "Skipping vocabulary entry: $key", e)
            }
        }

        result[0] = result[0] ?: "<s>"
        result[1] = result[1] ?: "<pad>"
        result[2] = result[2] ?: "</s>"

        return result
    }

    // =========================================================
    // IMAGE PREPROCESSING
    // =========================================================

    private fun preprocess(bitmap: Bitmap): FloatArray {

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            IMAGE_SIZE,
            IMAGE_SIZE,
            true
        )

        val planeSize = IMAGE_SIZE * IMAGE_SIZE
        val input = FloatArray(3 * planeSize)

        var index = 0

        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {

                val pixel = resized.getPixel(x, y)

                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                input[index] = (r - 0.5f) / 0.5f
                input[index + planeSize] = (g - 0.5f) / 0.5f
                input[index + 2 * planeSize] = (b - 0.5f) / 0.5f

                index++
            }
        }

        Log.d(
            TAG,
            "Input statistics: min=${input.minOrNull()} " +
                    "max=${input.maxOrNull()} " +
                    "mean=${input.average()}"
        )

        return input
    }

    // =========================================================
    // ENCODER
    // =========================================================

    private fun runEncoder(
        bitmap: Bitmap
    ): Array<Array<FloatArray>> {

        val input = preprocess(bitmap)

        val pixelTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(input),
            longArrayOf(
                1,
                3,
                IMAGE_SIZE.toLong(),
                IMAGE_SIZE.toLong()
            )
        )

        try {
            encoderSession.run(
                mapOf("pixel_values" to pixelTensor)
            ).use { result ->

                @Suppress("UNCHECKED_CAST")
                return result[0].value as Array<Array<FloatArray>>
            }
        } finally {
            pixelTensor.close()
        }
    }

    // =========================================================
    // DECODER
    // =========================================================

    private fun runDecoder(
        tokenIds: List<Long>,
        encoderHiddenStates: Array<Array<FloatArray>>
    ): FloatArray {

        val ids = LongArray(tokenIds.size) {
            tokenIds[it]
        }

        val inputIds = OnnxTensor.createTensor(
            ortEnvironment,
            LongBuffer.wrap(ids),
            longArrayOf(1, ids.size.toLong())
        )

        val encoderTensor = OnnxTensor.createTensor(
            ortEnvironment,
            encoderHiddenStates
        )

        try {
            decoderSession.run(
                mapOf(
                    "input_ids" to inputIds,
                    "encoder_hidden_states" to encoderTensor
                )
            ).use { result ->

                @Suppress("UNCHECKED_CAST")
                val logits =
                    result[0].value as Array<Array<FloatArray>>

                return logits[0][logits[0].size - 1]
            }

        } finally {
            inputIds.close()
            encoderTensor.close()
        }
    }

    // =========================================================
    // GREEDY DECODING
    // =========================================================

    private fun greedyDecode(
        encoderHiddenStates: Array<Array<FloatArray>>
    ): List<Long> {

        val tokens = mutableListOf<Long>()
        tokens.add(START_TOKEN)

        for (step in 0 until MAX_TOKENS) {

            val logits = runDecoder(
                tokens,
                encoderHiddenStates
            )

            var bestToken = START_TOKEN.toInt()
            var bestScore = Float.NEGATIVE_INFINITY

            val limit = min(
                logits.size,
                TOKENIZER_VOCAB_SIZE
            )

            for (i in 0 until limit) {

                val score = logits[i]

                if (score > bestScore) {
                    bestScore = score
                    bestToken = i
                }
            }

            val nextToken = bestToken.toLong()

            Log.d(
                TAG,
                "step=$step token=$nextToken score=$bestScore"
            )

            if (nextToken == EOS_TOKEN) break
            if (nextToken == PAD_TOKEN) break

            tokens.add(nextToken)
        }

        return tokens
    }

    // =========================================================
    // TOKEN DECODING
    // =========================================================

    private fun tokenToString(tokenId: Long): String {

        val id = tokenId.toInt()

        if (id < 0 || id >= TOKENIZER_VOCAB_SIZE) {
            return ""
        }

        return vocabulary[id] ?: ""
    }

    private fun decodeTokens(tokens: List<Long>): String {

        val pieces = StringBuilder()

        for (tokenId in tokens) {

            if (
                tokenId == START_TOKEN ||
                tokenId == EOS_TOKEN ||
                tokenId == PAD_TOKEN
            ) {
                continue
            }

            if (
                tokenId < 0 ||
                tokenId >= TOKENIZER_VOCAB_SIZE
            ) {
                continue
            }

            val piece = tokenToString(tokenId)

            if (piece.isEmpty()) continue

            pieces.append(piece)
        }

        var text = pieces.toString()

        text = text.replace("▁", " ")
        text = text.replace("Ġ", " ")

        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")
        text = text.replace("<unk>", "")

        text = text.replace(
            Regex("\\s+([,.!?;:%])"),
            "$1"
        )

        text = text.replace(
            Regex("\\(\\s+"),
            "("
        )

        text = text.replace(
            Regex("\\s+\\)"),
            ")"
        )

        text = text.replace(
            Regex("\\s+"),
            " "
        )

        return text.trim()
    }

    // =========================================================
    // PUBLIC OCR
    // =========================================================

    fun recognize(bitmap: Bitmap): String {

        val startTimeNs =
            SystemClock.elapsedRealtimeNanos()

        try {

            Log.d(TAG, "Starting OCR...")

            val encoderOutput = runEncoder(bitmap)

            Log.d(TAG, "Encoder complete")

            val tokenIds = greedyDecode(encoderOutput)

            Log.d(TAG, "Generated IDs: $tokenIds")

            val text = decodeTokens(tokenIds)

            Log.d(TAG, "Decoded text: $text")

            inferenceTimeMs =
                (SystemClock.elapsedRealtimeNanos() - startTimeNs) /
                        1_000_000.0

            Log.d(
                TAG,
                "OCR runtime: %.2f ms".format(inferenceTimeMs)
            )

            return text

        } catch (e: Exception) {

            Log.e(TAG, "OCR failed", e)
            throw e
        }
    }

    // =========================================================
    // CLEANUP
    // =========================================================

    fun close() {

        try {
            encoderSession.close()
        } catch (_: Exception) {
        }

        try {
            decoderSession.close()
        } catch (_: Exception) {
        }

        try {
            sessionOptions.close()
        } catch (_: Exception) {
        }
    }
}