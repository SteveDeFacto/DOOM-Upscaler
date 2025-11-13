# DOOM-Upscaler
Upscale any DOOM WAD or PK3

[![Demo Video](https://img.youtube.com/vi/3fHyq4_hprc/hqdefault.jpg)](https://www.youtube.com/watch?v=3fHyq4_hprc)

## Character modeling with Pix2Vox+

The repository now includes `character_modeler.py`, a helper script that converts the
multi-angle sprite folders from your WAD/PK3 assets into fully modeled 3D meshes using
Pix2Vox+. The script also exports JSON keyframes for each animation (walking,
attacking, idle, etc.) so you can remap the original sprite timing to the generated 3D
assets.

```
python character_modeler.py \
  --sprite-root path/to/sprites \
  --output-dir characters \
  --pix2vox-weights /path/to/pix2vox++/weights.pth \
  --pix2vox-command "python ~/Pix2VoxPlusPlus/test.py --weights {weights} --input {input} --output {output} --format {format} --device {device}"
```

`--pix2vox-command` is a template for the Pix2Vox+ CLI you already use. The script
replaces `{input}`, `{output}`, `{weights}`, `{format}`, `{device}`, and `{name}` for
each character before invoking the command. Meshes plus the animation metadata are
written to the `characters/` folder.
