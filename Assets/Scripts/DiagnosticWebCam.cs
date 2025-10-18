using UnityEngine;
using UnityEngine.UI;

public class DiagnosticWebCam : MonoBehaviour
{
    public RawImage target;
    public int requestedWidth = 640;
    public int requestedHeight = 480;
    public int requestedFps = 30;
    public bool useFrontFacing = false;

    WebCamTexture cam;

    void Start()
    {
        if (!target) { Debug.LogError("[CAM] No RawImage target assigned."); enabled = false; return; }

        var devices = WebCamTexture.devices;
        Debug.Log($"[CAM] Devices found: {devices.Length}");
        for (int i = 0; i < devices.Length; i++)
            Debug.Log($"[CAM] {i}: {devices[i].name} (frontFacing={devices[i].isFrontFacing})");

        if (devices.Length == 0) { Debug.LogError("[CAM] No camera devices found."); return; }

        int index = 0;
        if (useFrontFacing)
            for (int i = 0; i < devices.Length; i++) if (devices[i].isFrontFacing) { index = i; break; }

        cam = new WebCamTexture(devices[index].name, requestedWidth, requestedHeight, requestedFps);
        Debug.Log($"[CAM] Starting {devices[index].name} @ {requestedWidth}x{requestedHeight} {requestedFps}fps");
        target.texture = cam;
        target.material = null;
        cam.Play();
    }

    void OnDestroy()
    {
        if (cam != null && cam.isPlaying) cam.Stop();
    }
}
