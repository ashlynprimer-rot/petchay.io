package com.example.vegdetector;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import org.tensorflow.*;
import org.tensorflow.ndarray.*;
import org.tensorflow.types.TFloat32;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

@RestController
public class PredictController {

    // Path to SavedModel directory (exported from TF or converted)
    private static final String SAVED_MODEL_DIR = "models/saved_model";
    // Path to labels file (one label per line)
    private static final String LABELS_FILE = "models/labels.txt";

    private final List<String> labels;
    private final Map<String, Map<String, String>> nutrientDB;

    public PredictController() throws IOException {
        // load labels
        labels = Files.readAllLines(Paths.get(LABELS_FILE)).stream()
                .map(String::trim).filter(s->!s.isEmpty()).collect(Collectors.toList());

        // simple nutrient DB — extend as needed
        nutrientDB = new HashMap<>();
        nutrientDB.put("pechay", Map.of(
                "calories","13 kcal", "protein","1.5 g","carbohydrates","2.2 g",
                "fiber","1.0 g","vitaminA","223% DV", "vitaminC","45% DV", "calcium","105 mg","iron","0.8 mg"
        ));
        nutrientDB.put("carrot", Map.of(
                "calories","41 kcal","protein","0.9 g","carbohydrates","9.6 g","fiber","2.8 g",
                "vitaminA","334% DV","vitaminC","9% DV","calcium","33 mg","iron","0.3 mg"
        ));
        nutrientDB.put("tomato", Map.of(
                "calories","18 kcal","protein","0.9 g","carbohydrates","3.9 g","fiber","1.2 g",
                "vitaminA","17% DV","vitaminC","21% DV","calcium","10 mg","iron","0.3 mg"
        ));
        // add more entries as needed...
    }

    @PostMapping(path="/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> predict(@RequestPart("file") MultipartFile file) {
        try {
            // Read image bytes -> BufferedImage
            BufferedImage img = ImageIO.read(file.getInputStream());
            if (img == null) return ResponseEntity.badRequest().body(Map.of("error","Invalid image"));

            // Preprocess image into float tensor the model expects
            // This example assumes model expects [1, height, width, 3] float32 normalized to [0,1],
            // and that the model input size is 224x224. Adjust to your model.
            final int modelSize = 224;
            BufferedImage resized = resizeImage(img, modelSize, modelSize);
            float[] floatData = bufferedImageToFloatArray(resized, modelSize, modelSize);

            // Create Tensor from float array
            try (SavedModelBundle b = SavedModelBundle.load(SAVED_MODEL_DIR, "serve")) {
                try (TFloat32 input = TFloat32.scalarOf(0f)) {
                    // Build NDArray for the model input
                    Shape shape = Shape.of(1, modelSize, modelSize, 3);
                    try (Tensor<TFloat32> t = TFloat32.tensorOf(NdArrays.ofFloats(shape))) {
                        // fill tensor
                        FloatDataBuffer db = t.data();
                        db.write(floatData, 0, floatData.length);
                        // Run session: adjust input/output names to match your model signature
                        List<Tensor> outputs = b.session().runner()
                                .feed("serving_default_input_1:0", t) // common name for Keras saved models; adjust if different
                                .fetch("StatefulPartitionedCall:0")   // common output name, but change if different
                                .run();
                        try (Tensor<?> out = outputs.get(0)) {
                            float[] scores = new float[(int) out.shape().size(1)];
                            out.rawData().asFloats().read(scores);

                            // find best index
                            int best = 0;
                            for (int i = 1; i < scores.length; i++) {
                                if (scores[i] > scores[best]) best = i;
                            }
                            String label = (best < labels.size()) ? labels.get(best) : ("class_"+best);
                            float confidence = scores[best];

                            // Lookup nutrients (case-insensitive contains match)
                            String matchedKey = nutrientDB.keySet().stream()
                                    .filter(k -> label.toLowerCase().contains(k.toLowerCase()))
                                    .findFirst().orElse(null);

                            Map<String,Object> resp = new LinkedHashMap<>();
                            resp.put("detected", label);
                            resp.put("confidence", confidence);
                            if (matchedKey != null) resp.put("nutrients", nutrientDB.get(matchedKey));
                            else resp.put("nutrients", Map.of("note","No nutrient data available for "+label));

                            return ResponseEntity.ok(resp);
                        }
                    }
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            return ResponseEntity.status(500).body(Map.of("error", ex.getMessage()));
        }
    }

    // --- Helper methods ---
    private static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        g.dispose();
        return resizedImage;
    }

    private static float[] bufferedImageToFloatArray(BufferedImage img, int w, int h) {
        float[] data = new float[w*h*3];
        int idx = 0;
