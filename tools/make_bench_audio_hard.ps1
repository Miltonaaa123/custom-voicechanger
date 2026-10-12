# Harder / longer benchmark speech: connected speech, no clear pauses, longer clauses.
param([string]$OutDir = "bench\audio_hard")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$voice = New-Object -ComObject SAPI.SpVoice
$voice.Rate = 2   # faster speech = harder for the model, closer to natural dictation

$clips = [ordered]@{
  "hard1" = "so what I want to happen is whenever you reboot the program the text file gets cleared and it updates to whatever you are saying right now"
  "hard2" = "I need the transcriber to be lightweight because I am going to feed the output straight into a speech synthesis model afterwards"
  "hard3" = "honestly it should probably run on the graphics card since the central processing unit would struggle to keep up with continuous dictation"
}

foreach ($name in $clips.Keys) {
  $path = Join-Path $OutDir "$name.wav"
  $fmt = New-Object -ComObject SAPI.SpAudioFormat
  $fmt.Type = 22
  $stream = New-Object -ComObject SAPI.SpFileStream
  $stream.Format = $fmt
  $stream.Open($path, 3, $false)
  $voice.AudioOutputStream = $stream
  $voice.Speak($clips[$name]) | Out-Null
  $stream.Close()
  Write-Host "wrote $path`n    EXPECTED: $($clips[$name])"
}
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($voice) | Out-Null
