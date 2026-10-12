# Generate benchmark speech clips with the built-in Windows SAPI voices.
param(
  [string]$OutDir = "bench\audio"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$voice = New-Object -ComObject SAPI.SpVoice
$voices = @($voice.GetVoices())
Write-Host "Available SAPI voices:"
foreach ($v in $voices) { Write-Host ("  - " + $v.GetDescription()) }
if ($voices.Count -gt 0) { $voice.Voice = $voices.Item(0) }
$voice.Rate = 1

# ~15 words per clip, the user's stated budget unit.
$clips = [ordered]@{
  "clip15a" = "the quick brown fox jumps over the lazy dog while the crowd watches in total silence and waits"
  "clip15b" = "please write these words into the text file so the speech system can read them back out loud"
  "clip15c" = "machine learning models running on a graphics card can transcribe speech much faster than real time"
}

foreach ($name in $clips.Keys) {
  $path = Join-Path $OutDir "$name.wav"
  # 22.05 kHz 16-bit mono is plenty for ASR and keeps files small.
  $fmt = New-Object -ComObject SAPI.SpAudioFormat
  $fmt.Type = 22
  $stream = New-Object -ComObject SAPI.SpFileStream
  $stream.Format = $fmt
  $stream.Open($path, 3, $false)   # 3 = SSFMCreateForWrite
  $voice.AudioOutputStream = $stream
  $voice.Speak($clips[$name]) | Out-Null
  $stream.Close()
  Write-Host "wrote $path"
}

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($voice) | Out-Null
