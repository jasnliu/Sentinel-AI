package com.example.camera2;

import android.graphics.Bitmap;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public final class AlertReporter {
    private static final String TAG = "AlertReporter";
    private static final MediaType MEDIA_TYPE_JPEG = MediaType.get("image/jpeg");

    private final OkHttpClient httpClient;
    private final String baseUrl;
    private final String apiKey;

    public AlertReporter(String baseUrl, String apiKey) {
        this.baseUrl = normalizeBaseUrl(baseUrl);
        this.apiKey = apiKey == null ? "" : apiKey;
        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(20, TimeUnit.SECONDS)
                .writeTimeout(20, TimeUnit.SECONDS)
                .build();
    }

    public void reportAlert(
            String deviceId,
            long timestampMs,
            float confidence,
            int consecutiveHits,
            Bitmap image
    ) {
        if (baseUrl.isEmpty()) {
            Log.w(TAG, "SERVER_BASE_URL is empty; skipping upload.");
            return;
        }
        if (deviceId == null || deviceId.trim().isEmpty()) {
            Log.w(TAG, "deviceId is empty; skipping upload.");
            return;
        }
        if (image == null) {
            Log.w(TAG, "image is null; skipping upload.");
            return;
        }

        byte[] jpegBytes;
        try {
            jpegBytes = bitmapToJpeg(image, 80);
        } catch (IOException e) {
            Log.w(TAG, "Failed to encode JPEG", e);
            return;
        }

        MultipartBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("device_id", deviceId)
                .addFormDataPart("timestamp_ms", Long.toString(timestampMs))
                .addFormDataPart("confidence", Float.toString(confidence))
                .addFormDataPart("consecutive_hits", Integer.toString(consecutiveHits))
                .addFormDataPart(
                        "image",
                        timestampMs + ".jpg",
                        RequestBody.create(jpegBytes, MEDIA_TYPE_JPEG)
                )
                .build();

        Request.Builder req = new Request.Builder()
                .url(baseUrl + "/api/v1/alerts")
                .post(requestBody);
        if (!apiKey.isEmpty()) {
            req.header("X-API-Key", apiKey);
        }

        httpClient.newCall(req.build()).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.w(TAG, "Alert upload failed", e);
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                try (Response r = response) {
                    Log.i(TAG, "Alert upload response: " + r.code() + " " + (r.body() != null ? r.body().string() : ""));
                }
            }
        });
    }

    private static byte[] bitmapToJpeg(Bitmap bitmap, int quality) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        boolean ok = bitmap.compress(Bitmap.CompressFormat.JPEG, quality, baos);
        if (!ok) {
            throw new IOException("Bitmap.compress returned false");
        }
        return baos.toByteArray();
    }

    private static String normalizeBaseUrl(String raw) {
        if (raw == null) {
            return "";
        }
        String trimmed = raw.trim();
        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        return trimmed;
    }
}

