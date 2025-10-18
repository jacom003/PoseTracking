using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class BlazePoseRunner : MonoBehaviour
{
    [Header("UI")]
    public RawImage webCamView;           // assign in Inspector
    public RectTransform overlayRoot;     // same rect as webCamView
    public Image dotPrefab;               // small UI Image prefab (circle)

    [Header("Model")]
    public string modelFile = "pose_landmark_full.onnx"; // in StreamingAssets

    WebCamTexture cam;
    InferenceSession session;
    string inputName;
    int inW = 256, inH = 256; // from your log
    Texture2D readTex;
    RenderTexture rt;

    List<Image> dots = new List<Image>(); // 33 dots

    void Start()
    {
        // 1) Webcam
        cam = new WebCamTexture();
        cam.Play();
        if (webCamView) webCamView.texture = cam;

        // 2) ONNX session (DML GPU)
        var path = Path.Combine(Application.streamingAssetsPath, modelFile);
        Debug.Log("[POSE] Loading: " + path);
        if (!File.Exists(path)) { Debug.LogError("[POSE] Model not found."); enabled = false; return; }

        var opts = new SessionOptions();
        opts.AppendExecutionProvider_DML();
        session = new InferenceSession(path, opts);

        var inMeta = session.InputMetadata.First();
        inputName = inMeta.Key;
        var dims = inMeta.Value.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
        // Expect [1,256,256,3]:
        inH = dims[1]; inW = dims[2];
        Debug.Log($"[POSE] Input shape: [{string.Join(",", dims)}]");

        // 3) GPU resize target + readback tex
        rt = new RenderTexture(inW, inH, 0, RenderTextureFormat.ARGB32);
        readTex = new Texture2D(inW, inH, TextureFormat.RGBA32, false);

        // 4) Make 33 dots
        for (int i = 0; i < 33; i++)
        {
            var d = Instantiate(dotPrefab, overlayRoot);
            d.rectTransform.sizeDelta = new Vector2(10, 10);
            d.enabled = false; // hide until we get data
            dots.Add(d);
        }
    }

    void OnDestroy()
    {
        cam?.Stop();
        session?.Dispose();
        if (rt) rt.Release();
    }

    void Update()
    {
        if (cam == null || !cam.isPlaying || !cam.didUpdateThisFrame) return;

        // Run every 2 frames to save time
        if ((Time.frameCount & 1) != 0) return;

        // 1) Resize to 256x256 via GPU blit, then read back
        Graphics.Blit(cam, rt);
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        readTex.ReadPixels(new Rect(0, 0, inW, inH), 0, 0, false);
        readTex.Apply(false);
        RenderTexture.active = prev;

        // 2) Pack NHWC float32 0..1
        var px = readTex.GetPixels32();
        var input = new float[inW * inH * 3];
        int k = 0;
        for (int y = 0; y < inH; y++)
        {
            for (int x = 0; x < inW; x++)
            {
                var c = px[y * inW + x];
                // NHWC: R,G,B
                input[k++] = c.r / 255f;
                input[k++] = c.g / 255f;
                input[k++] = c.b / 255f;
            }
        }

        // 3) Run inference
        var tensor = new DenseTensor<float>(input, new int[] { 1, inH, inW, 3 });
        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });

        // 4) Inspect outputs
        var first = results.First();
        var asTensor = first.AsTensor<float>();
        var arr = asTensor.ToArray();
        // Try to interpret as 33 landmarks (x,y,z,[visibility?]):
        // common encodings: 33*4=132 or 33*3=99, sometimes more with aux outputs.
        int num = 33;
        int stride = (arr.Length >= num * 4) ? 4 : 3;
        if (arr.Length < num * stride)
        {
            Debug.Log($"[POSE] Unexpected output len={arr.Length}. Showing raw dims only.");
            return;
        }

        // 5) Draw landmarks (assumes x,y in [0..1], y down)
        // Map to overlay rect
        var rect = overlayRoot.rect;
        for (int i = 0; i < num; i++)
        {
            float x = arr[i * stride + 0];
            float y = arr[i * stride + 1];
            // Some exports provide y up; if points look flipped, use: y = 1f - y;
            y = 1f - y;

            var dot = dots[i];
            dot.enabled = true;
            var rtDot = dot.rectTransform;
            rtDot.anchoredPosition = new Vector2((x - 0.5f) * rect.width, (y - 0.5f) * rect.height);
        }
    }
}
