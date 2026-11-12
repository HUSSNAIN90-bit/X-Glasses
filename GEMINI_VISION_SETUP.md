# Gemini Vision setup

The backend uses Gemini for image-based scene understanding. Configure the API key in the backend environment (do not commit secrets):

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_VISION_MODEL=gemini-2.5-flash
```

The existing Groq configuration remains responsible for text chat and intent parsing. YOLO26 and face recognition continue to run in the backend; Gemini receives only frames submitted by the mobile Live Vision sampler.

The mobile Live Vision sampler captures one reduced-quality frame approximately every 10 seconds while enabled. This is a networked vision feature: Gemini calls require internet access. Local YOLO26/face-recognition inference can continue on the backend host without sending every camera frame to Gemini.
