# PyTorchVideo (action recognition) dataset format

Standard "video folder" layout — one subdirectory per action class:

```
data/video/
├── train/
│   ├── idle/
│   │   ├── clip001.mp4
│   │   ├── clip002.mp4
│   │   └── ...
│   ├── walking/
│   ├── running/
│   └── waving/
└── val/
    ├── idle/
    ├── walking/
    ├── running/
    └── waving/
```

Class indices are assigned **alphabetically** by folder name when the
dataset loads. The order of `configs/video_labels.txt` must match.
For the layout above, alphabetical order is `idle, running, walking, waving`
— update `video_labels.txt` accordingly.

## Clip requirements

- **Format:** `.mp4`, `.avi`, `.mov`, or `.mkv` (anything PyAV can read).
- **Length:** at least `num_frames * sampling_rate / fps` seconds. With
  the default config (8 frames × stride 8 @ 30fps ≈ 2.1s), clips of
  **3+ seconds** are safe.
- **Resolution:** any — frames get resized to 224×224. Higher resolution
  doesn't help; it just slows down loading.
- **Counts:** aim for **50+ clips per class** for a usable model.
  Pretrained Kinetics weights help a lot if your classes resemble
  Kinetics-400 actions.

## Tips

- **Trim aggressively.** A 30-second clip with 2 seconds of the action
  hurts; cut to just the action.
- **Vary subjects, angles, lighting.** Otherwise the model memorizes
  backgrounds.
- **Keep val separate.** Don't put different clips of the *same recording*
  in train and val — split by recording session.
