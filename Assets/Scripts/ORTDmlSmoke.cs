using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.IO;
using System.Linq;
using UnityEngine;

public class ORTDmlSmoke : MonoBehaviour 
{
    void Start()
    {
        var modelPath = Path.Combine(Application.streamingAssetsPath, "pose_landmark_full.onnx");
        if (!File.Exists(modelPath))
        {
            Debug.LogError($"[SMOKE] Model not found: {modelPath}");
            return;
        }

        try
        {
            using var opts = new SessionOptions();
            opts.AppendExecutionProvider_DML();
            using var session = new InferenceSession(modelPath, opts);

            Debug.Log("[SMOKE] ONNX Runtime initialized with DirectML (GPU).");

            var meta = session.InputMetadata.First();
            string inputName = meta.Key;
            var dims = meta.Value.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
            int len = Math.Max(1, dims.Aggregate(1, (a, b) => a * (b > 0 ? b : 1)));
            var data = new float[len];
            var tensor = new DenseTensor<float>(data, dims);

            using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });
            var firstOut = results.First();
            Debug.Log($"[SMOKE] Inference OK-> in:{inputName} dim:[{string.Join(",", dims)}] out:{firstOut.Name}");

            bool dmlLoaded = System.Diagnostics.Process.GetCurrentProcess().Modules
                .Cast<System.Diagnostics.ProcessModule>()
                .Any(m => m.ModuleName.Equals("onnxruntime_providers_dml.dll", StringComparison.OrdinalIgnoreCase));
            Debug.Log(dmlLoaded ? "DML provider loaded." : "DML NOT detected(may be CPU fallback).");

        }

        catch (DllNotFoundException e)
        {
            Debug.LogError("[SMOKE] Missing native DLL: " + e.Message);
        }

        catch (OnnxRuntimeException e)
        {
            Debug.LogError("[SMOKE] ORT error: " + e.Message);
        }

        catch (BadImageFormatException e)
        {
            Debug.LogError("[SMOKE] Arch mismatch (need x86_64): " + e.Message);
        }

        catch (Exception e)
        {
            Debug.LogError("[SMOKE] Unexpected failure: " + e);
        }

    }
} 
