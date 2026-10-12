"""List input audio devices so we can pick the right microphone."""
from __future__ import annotations

import sounddevice as sd


def main() -> None:
    apis = sd.query_hostapis()
    print("--- HOST APIS ---")
    for i, h in enumerate(apis):
        print(f"  [{i}] {h['name']:<24} default_input={h.get('default_input_device')}")

    print("\n--- INPUT DEVICES ---")
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        api = apis[d["hostapi"]]["name"]
        marker = "  <== system default" if i == default_in else ""
        print(
            f"  [{i:>2}] {d['name'][:52]:<52} ch={d['max_input_channels']:<3} "
            f"sr={int(d['default_samplerate']):<6} api={api:<18}{marker}"
        )

    print("\nUse:  .\\run.bat --device <index>")


if __name__ == "__main__":
    main()
