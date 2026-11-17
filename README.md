# VLA-Duo

## Requirements

- Linux x86-64 host with Docker, git, Python 3.11+ and `adb`
- OnePlus 13 (Snapdragon 8 Elite) or OnePlus 15 (Snapdragon 8 Elite Gen 5), USB debugging on, charger connected

## Quick start

```sh
./vladuo.sh build            # build in Docker -> out/
./vladuo.sh models           # download and convert the public checkpoints -> models/
./vladuo.sh push             # copy binaries and models to the phone
./vladuo.sh burst npu        # NPU-only
./vladuo.sh burst duo        # VLA-Duo
./vladuo.sh summary table2   # -> results/<soc>/summary/
./vladuo.sh sustain --mode duo --period <T ms>   # fixed-period run (5 min)
./vladuo.sh summary sustain
```


## Notes

For sustain runs, close other apps and attach a cooler to the phone. `--gpu-cap <MHz>` caps the GPU clock and needs root.

On the Snapdragon 8 Elite Gen 5 (OnePlus 15), `--cfg xmem` runs the GPU part with a faster ViT kernel.
Run `./vladuo.sh profile quick --cfg xmem` and `./vladuo.sh promote <profile.json>` once before using it.

`./vladuo.sh --help` lists all commands.
