
package com.example.camera2;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.Settings;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;
    private static final String MODEL_ASSET_NAME = "forest_fire_classifier_mobilenetv3_small.ptl";
    private static final int FIRE_CLASS_INDEX = 0;
    private TextureView textureView;
    private TextView tvResult;
    private Button btnProcess;
    private CameraDevice cameraDevice;
    private CameraCaptureSession cameraCaptureSession;
    private CaptureRequest.Builder captureRequestBuilder;
    private Module module;
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private final Size imageSize = new Size(224, 224);

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private boolean hasPromptedForAllFilesAccess = false;

    // Define the SurfaceTextureListener
    private final TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            // When the TextureView is available, open the camera
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Handle changes to the size of the SurfaceTexture, if necessary
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            // Handle updates to the SurfaceTexture
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textureView = findViewById(R.id.texture_view);
        tvResult = findViewById(R.id.tv_result);
        btnProcess = findViewById(R.id.btn_process);

        // Request camera permission if not already granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST_CODE);
        } else {
            if (textureView.isAvailable()) {
                openCamera();
            } else {
                textureView.setSurfaceTextureListener(textureListener);
            }
        }

        btnProcess.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureAndProcessImage();
            }
        });

        try {
            module = LiteModuleLoader.load(assetFilePath(MODEL_ASSET_NAME, true));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
            String cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size previewSize = map.getOutputSizes(SurfaceTexture.class)[0];
            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    cameraDevice = camera;
                    startCameraPreview();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                    cameraDevice.close();
                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {
                    cameraDevice.close();
                    cameraDevice = null;
                }
            }, backgroundHandler); // Use backgroundHandler for camera operations
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void startCameraPreview() {
        SurfaceTexture texture = textureView.getSurfaceTexture();
        texture.setDefaultBufferSize(imageSize.getWidth(), imageSize.getHeight());
        Surface surface = new Surface(texture);

        try {
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);

            cameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession session) {
                    cameraCaptureSession = session;
                    updatePreview();
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                }
            }, backgroundHandler); // Use backgroundHandler for camera operations
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void updatePreview() {
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSession.setRepeatingRequest(captureRequestBuilder.build(), null, backgroundHandler); // Use backgroundHandler instead of null
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void captureAndProcessImage() {
        Bitmap bitmap = textureView.getBitmap();
        if (bitmap == null) {
            return;
        }

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);
        Bitmap centerCroppedBitmap = Bitmap.createBitmap(resizedBitmap, 16, 16, 224, 224);
        dumpPreprocessedImage(centerCroppedBitmap);

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                centerCroppedBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );

        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        Log.d(TAG, "scores=" + Arrays.toString(scores));

        boolean isFire = scores[FIRE_CLASS_INDEX] > scores[1];
        if (isFire) {
            tvResult.setText("FIRE");
            tvResult.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
        } else {
            tvResult.setText("NO FIRE");
            tvResult.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
        }
    }

    private void dumpPreprocessedImage(Bitmap bitmap) {
        long timestampMs = System.currentTimeMillis();
        String filename = timestampMs + ".png";

        File primaryDir = new File("/sdcard/tmp");
        File primaryFile = new File(primaryDir, filename);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            Log.w(TAG, "No all-files access; /sdcard/tmp write may fail. Falling back if needed.");
            promptForAllFilesAccessOnce();
        }

        if (tryWritePng(bitmap, primaryFile)) {
            Log.d(TAG, "Wrote preprocessed image: " + primaryFile.getAbsolutePath());
            return;
        }

        File fallbackDir = new File(getExternalFilesDir(null), "tmp");
        File fallbackFile = new File(fallbackDir, filename);
        if (tryWritePng(bitmap, fallbackFile)) {
            Log.w(TAG, "Wrote preprocessed image to app storage instead: " + fallbackFile.getAbsolutePath());
        } else {
            Log.e(TAG, "Failed to write preprocessed image to disk.");
        }
    }

    private void promptForAllFilesAccessOnce() {
        if (hasPromptedForAllFilesAccess) {
            return;
        }
        hasPromptedForAllFilesAccess = true;

        Toast.makeText(
                this,
                "To write /sdcard/tmp, enable 'All files access' for this app in Settings.",
                Toast.LENGTH_LONG
        ).show();

        try {
            Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
            intent.setData(Uri.parse("package:" + getPackageName()));
            startActivity(intent);
        } catch (ActivityNotFoundException e) {
            startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
        }
    }

    private boolean tryWritePng(Bitmap bitmap, File outFile) {
        File parent = outFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            Log.w(TAG, "Failed to create directory: " + parent.getAbsolutePath());
            return false;
        }

        try (FileOutputStream os = new FileOutputStream(outFile)) {
            boolean ok = bitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
            os.flush();
            if (!ok) {
                Log.w(TAG, "Bitmap.compress returned false for: " + outFile.getAbsolutePath());
            }
            return ok;
        } catch (Exception e) {
            Log.w(TAG, "Failed to write PNG: " + outFile.getAbsolutePath(), e);
            return false;
        }
    }

    private String assetFilePath(String assetName, boolean forceOverwrite) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (!forceOverwrite && file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
        return file.getAbsolutePath();
    }

    // Start the background thread
    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    // Stop the background thread
    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }

    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void closeCamera() {
        if (cameraCaptureSession != null) {
            cameraCaptureSession.close();
            cameraCaptureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }
}
