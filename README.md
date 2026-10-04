# 🎃 Halloween AI

**AI-powered trick-or-treat automation that detects visitors, remembers who has already been served, dispenses sweets and triggers an interactive scare mirror.**

> Built for a real Halloween installation — not just a demo.

## How it works

```text
Visitor
   │
   ▼
Tracking Camera
   │
   ├── YOLO + ByteTrack ─────────────┐
   │                                 │
   └── Costume / motion fallback     │
                                     ▼
                                ID Camera
                                     │
                       ┌─────────────┴─────────────┐
                       │                           │
                  Previous visitor?           New visitor
                       │                           │
                     BLOCK                         ▼
                                           Sweet Dispenser
                                                 │
                                                 ▼
                                            Scare Mirror
```

## ✨ Features

- Ultralytics YOLO person detection
- ByteTrack visitor tracking
- Second **ID Camera** for appearance matching
- Session-based repeat-serving protection
- Costume/presence fallback when YOLO misses unusual outfits
- Basic low/wide animal-motion rejection
- Raspberry Pi HTTP sweet-dispenser integration
- Interactive Scare Mirror HTTP integration
- Fail-open behaviour for uncertain genuine visitors
- Flask control/status portal
- Two live MJPEG camera previews
- Simulation mode for safe testing
- Web-preview resolution/FPS independent of AI processing

## 📸 Show the build

A repository like this benefits enormously from real photos and short GIFs. Add:

```text
docs/tracking-demo.gif
docs/id-camera-demo.gif
docs/dispenser-demo.gif
docs/full-build.jpg
```

A 10–20 second clip showing **approach → identify → dispense → scare** is ideal.

## Hardware

The original build uses a Windows AI host with two USB cameras, a Raspberry Pi-controlled stepper sweet dispenser and a separate Scare Mirror application.

Default camera roles:

```text
0 = Tracking camera
1 = ID Camera
2 = Optional spare camera
```

The software is intentionally adaptable to other cameras and dispenser mechanisms.

## Quick start

```powershell
git clone https://github.com/YOUR-USERNAME/HalloweenAI.git
cd HalloweenAI
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configure the integration addresses as environment variables:

```powershell
$env:HALLOWEEN_AI_LIVE="false"
$env:DISPENSER_URL="http://YOUR-PI:5000/dispense"
$env:DISPENSER_STATUS_URL="http://YOUR-PI:5000/status"
$env:MIRROR_ENABLED="true"
$env:MIRROR_BASE_URL="http://YOUR-MIRROR:5050"
python person_detector.py
```

Then open:

```text
http://YOUR-AI-PC:8989
```

The public repository defaults to **simulation mode**. Set `HALLOWEEN_AI_LIVE=true` only after testing your hardware.

> `.env.example` is a configuration reference. The current application reads environment variables directly; it does not automatically load `.env`.

## Visitor matching

The ID Camera captures several body crops and creates HSV colour histograms. New visitors are compared with people already served during the current session.

This is lightweight **appearance matching, not facial recognition**. Similar clothing, lighting, occlusion and camera position can affect it.

A dedicated person-ReID implementation is one of the project's best opportunities for contributors.

## Halloween costume fallback

Halloween costumes are difficult for conventional person detectors. Inflatable suits, cloaks, wings and props may cause YOLO to miss a genuine visitor.

The fallback watches persistent directional motion when no person track exists. V2.3 also filters some low/wide animal-like movement and asks the ID Camera for an appearance match when possible.

The design principle is:

> **Use AI to reduce repeat servings — don't require perfect AI recognition before somebody can get sweets.**

## Key tuning

```python
YOLO_IMAGE_SIZE = 320
TRACKING_INFERENCE_INTERVAL = 3
APPEARANCE_MATCH_THRESHOLD = 0.82

WEB_STREAM_WIDTH = 640
WEB_STREAM_HEIGHT = 360
WEB_STREAM_FPS = 8
```

Every camera position is different, so expect to tune motion, geometry and appearance thresholds for your installation.

## 🚀 Roadmap / contributors wanted

Ideas that would make meaningful contributions:

- dedicated person-ReID embeddings
- stronger animal rejection
- costume-aware detection
- multiple-person/queue handling
- cross-camera association
- sweet-drop sensor verification and automatic retry
- MQTT / Home Assistant integration
- alternative dispenser hardware
- Linux/Raspberry Pi AI-host support
- Docker packaging
- dashboard improvements
- statistics/event history
- more Halloween props and effects

See `CONTRIBUTING.md`.

## Safety

This project can control physical machinery. Keep fingers, hair and clothing away from moving mechanisms, provide a physical power disconnect, test mechanisms unloaded, and design the hardware so software failure cannot create a trapping/crushing hazard.

Computer vision is probabilistic. Do not use this project for safety-critical access control or consequential decisions about people.

## Privacy

Current appearance profiles are held in memory for the running session. If you add persistent images, video, identifiers or appearance profiles, consider the privacy/data-protection requirements that apply where you operate the system.

## License

MIT. See `LICENSE`.

## ⭐ Build one?

Fork it, improve it and share what you built. Camera compatibility reports, hardware designs, Halloween effects and pull requests are welcome.
