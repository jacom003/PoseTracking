using UnityEngine;
using UnityEngine.UI;

public class UnityWebCamSource : MonoBehaviour
{
    public RawImage target;
    public int requestedWidth = 640, requestedHeight = 480, requestedFps = 30;

    WebCamTexture _tex;

    public Texture SourceTexture => _tex;
    public bool HasFrame => _tex != null && _tex.didUpdateThisFrame;

    void Start()
    {
        _tex = new WebCamTexture(requestedWidth, requestedHeight, requestedFps);
        target.texture = _tex;
        target.material.mainTexture = _tex;
        _tex.Play();
    }

    void OnDestroy()
    {
        if (_tex != null && _tex.isPlaying) _tex.Stop();
    }

}
