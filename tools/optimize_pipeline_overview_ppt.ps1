$ErrorActionPreference = "Stop"

$srcPath = 'D:\github\3DRR_low_light\report\Pipeline_Overview\Pipeline.pptx'
$dstPath = 'D:\github\3DRR_low_light\report\Pipeline_Overview\Pipeline_optimized.pptx'
$assetRoot = 'D:\github\3DRR_low_light\report\Pipeline_Overview'
$outPng = 'D:\github\3DRR_low_light\report\Pipeline_Overview\_after_preview.png'

function Rgb([int]$r,[int]$g,[int]$b){ return ($r -bor ($g -shl 8) -bor ($b -shl 16)) }

function Get-ShapeByTextContains($slide, [string]$needle){
  for($i=1; $i -le $slide.Shapes.Count; $i++){
    $s = $slide.Shapes.Item($i)
    try{
      if($s.HasTextFrame -and $s.TextFrame.HasText){
        $t = [string]$s.TextFrame.TextRange.Text
        if($t -like "*${needle}*"){ return $s }
      }
    } catch {}
  }
  return $null
}

function Set-TextContains($slide, [string]$needle, [string]$newText){
  $s = Get-ShapeByTextContains $slide $needle
  if($null -ne $s){
    try { $s.TextFrame.TextRange.Text = $newText } catch {}
  }
}

function Get-PictureNear($slide, [double]$x, [double]$y){
  $best = $null; $bestDist = 1.0e18
  for($i=1; $i -le $slide.Shapes.Count; $i++){
    $s = $slide.Shapes.Item($i)
    if($s.Type -ne 13){ continue }
    $dx = [double]$s.Left - $x; $dy = [double]$s.Top - $y
    $d2 = $dx*$dx + $dy*$dy
    if($d2 -lt $bestDist){ $bestDist = $d2; $best = $s }
  }
  return $best
}

function Replace-PictureNear($slide, [double]$x, [double]$y, [string]$path){
  if(!(Test-Path $path)){ return }
  $pic = Get-PictureNear $slide $x $y
  if($null -eq $pic){ return }
  $l = [double]$pic.Left; $t = [double]$pic.Top; $w = [double]$pic.Width; $h = [double]$pic.Height
  try { $pic.Delete() } catch {}
  $slide.Shapes.AddPicture($path, 0, -1, $l, $t, $w, $h) | Out-Null
}

function Add-StepBadge($slide, [double]$x,[double]$y,[double]$w,[double]$h,[string]$text,[int]$lineRgb){
  $b = $slide.Shapes.AddShape(5,$x,$y,$w,$h)
  $b.Fill.ForeColor.RGB = (Rgb 255 255 255)
  $b.Fill.Transparency = 0.04
  $b.Line.ForeColor.RGB = $lineRgb
  $b.Line.Weight = 1.0
  $b.TextFrame.MarginLeft = 2
  $b.TextFrame.MarginRight = 2
  $b.TextFrame.MarginTop = 0
  $b.TextFrame.MarginBottom = 0
  $tr = $b.TextFrame.TextRange
  $tr.Text = $text
  $tr.Font.Name = 'Cambria'
  $tr.Font.Size = 9.0
  $tr.Font.Bold = -1
  $tr.Font.Color.RGB = (Rgb 52 50 46)
  $tr.ParagraphFormat.Alignment = 2
}

function Add-ProxyThumb($slide,[double]$x,[double]$y,[double]$w,[double]$h,[string]$imgPath,[int]$lineRgb){
  if(!(Test-Path $imgPath)){ return }
  $frame = $slide.Shapes.AddShape(5,$x,$y,$w,$h)
  $frame.Fill.ForeColor.RGB = (Rgb 255 255 255)
  $frame.Line.ForeColor.RGB = $lineRgb
  $frame.Line.Weight = 0.9
  $m = 2.0
  $slide.Shapes.AddPicture($imgPath,0,-1,$x+$m,$y+$m,$w-2*$m,$h-2*$m) | Out-Null
}

$ppt = New-Object -ComObject PowerPoint.Application
# Open as untitled editable copy even when source is locked
$pres = $ppt.Presentations.Open($srcPath,$true,$true,$false)
$slide = $pres.Slides.Item(1)

# Stage2 / Stage3 text updates (few formulas)
Set-TextContains $slide 'no-densify refinement with persistent sparse support' 'mid-hard mining + periodic topology refresh (spawn/prune)'
Set-TextContains $slide 'weighted support' 'mid-hard mining'
Set-TextContains $slide 'Sparse-Guided Anchor' 'Sparse-Guided Geometry'
Set-TextContains $slide 'Sparse Loss' 'Sparse Geo Loss'
Set-TextContains $slide 'freeze geometry; optimize opacity, SH, illum, and chroma' 'freeze geometry; optimize opacity, SH, illum, chroma only'
Set-TextContains $slide "Illum Head" "Illum Head`r`n+`r`nY Shadow-Lift"
Set-TextContains $slide "（C' = C + ΔC）" 'bounded additive residual'
Set-TextContains $slide "（Y' = g_Y · Y）" 'shadow-lift Y branch'
Set-TextContains $slide 'Weighted Shadow' 'Shadow Blend + Route'
Set-TextContains $slide 'W_C stays Conservative in deep shadows' 'W_C conservative in deep shadows'
Set-TextContains $slide 'W_Y boosts informative dark regions' 'W_Y boosts informative dark regions'

# Relabel stage2 mechanism pills
$wd = Get-ShapeByTextContains $slide 'Weak Depth'
if($wd -and $wd.Left -gt 700 -and $wd.Top -lt 340){ try { $wd.TextFrame.TextRange.Text = 'Topology x250' } catch {} }
$st = Get-ShapeByTextContains $slide 'Structure'
if($st -and $st.Left -gt 800 -and $st.Top -lt 340){ try { $st.TextFrame.TextRange.Text = 'mid-hard x400' } catch {} }

# Additional stage2 mechanism pill
$tmpl = Get-ShapeByTextContains $slide 'Sparse Geo Loss'
if($tmpl){
  $dup = $tmpl.Duplicate()
  $dup.Left = 606; $dup.Top = 276; $dup.Width = 124; $dup.Height = 24
  $dup.TextFrame.TextRange.Text = 'hardest_global_mixed'
  $dup.TextFrame.TextRange.Font.Size = 8.2
}

# Asset replacements in stage3
Replace-PictureNear $slide 1010 263 (Join-Path $assetRoot 'stage6\_stage3_revised_assets\supervision_br.png')
Replace-PictureNear $slide 1010 356 (Join-Path $assetRoot 'stage6\_stage3_revised_assets\proxy_target_br.png')
Replace-PictureNear $slide 1111 457 (Join-Path $assetRoot 'stage6\_stage3_revised_assets\proxy_shadow_weight_br.png')
Replace-PictureNear $slide 1053 558 (Join-Path $assetRoot 'stage6\_stage3_revised_assets\W_C_concept.png')
Replace-PictureNear $slide 1223 558 (Join-Path $assetRoot 'stage6\_stage3_revised_assets\W_Y_concept.png')

# Add proxy global/shadow thumbnails and route note
Add-ProxyThumb $slide 1344 454 54 24 (Join-Path $assetRoot 'stage6\_stage3_revised_assets\proxy_global_br.png') (Rgb 183 177 170)
Add-ProxyThumb $slide 1344 490 54 24 (Join-Path $assetRoot 'stage6\_stage3_revised_assets\proxy_shadow_br.png') (Rgb 183 177 170)
$routeTxt = $slide.Shapes.AddTextbox(1, 1236, 522, 170, 16)
$routeTxt.Fill.Visible = 0; $routeTxt.Line.Visible = 0
$routeTxt.TextFrame.MarginLeft = 0; $routeTxt.TextFrame.MarginRight = 0; $routeTxt.TextFrame.MarginTop = 0; $routeTxt.TextFrame.MarginBottom = 0
$routeR = $routeTxt.TextFrame.TextRange
$routeR.Text = 'route switch: global / local'
$routeR.Font.Name = 'Calibri'; $routeR.Font.Size = 8.0; $routeR.Font.Color.RGB = (Rgb 112 109 103)
$routeR.ParagraphFormat.Alignment = 2

# Step badges 15k/5k/5k
Add-StepBadge $slide 486 132 42 18 '15k' (Rgb 131 152 161)
Add-StepBadge $slide 877 132 34 18 '5k' (Rgb 171 144 132)
Add-StepBadge $slide 1364 96 34 18 '5k' (Rgb 123 143 164)

# Legend compact + topology event item
$legendTitle = Get-ShapeByTextContains $slide 'Legend'
if($legendTitle){ $legendTitle.Left = 408; $legendTitle.Top = 718 }
$dataLabel = Get-ShapeByTextContains $slide 'Data Flow'
if($dataLabel){ $dataLabel.TextFrame.TextRange.Text = 'Data'; $dataLabel.Left = 540; $dataLabel.Top = 721; $dataLabel.Width = 44 }
$supLabel = Get-ShapeByTextContains $slide 'Supervision Flow'
if($supLabel){ $supLabel.TextFrame.TextRange.Text = 'Supervision'; $supLabel.Left = 690; $supLabel.Top = 721; $supLabel.Width = 84 }
$gradLabel = Get-ShapeByTextContains $slide 'Gradient Flow'
if($gradLabel){ $gradLabel.TextFrame.TextRange.Text = 'Gradient'; $gradLabel.Left = 830; $gradLabel.Top = 721; $gradLabel.Width = 66 }

$topoLine = $slide.Shapes.AddLine(900,730,965,730)
$topoLine.Line.ForeColor.RGB = (Rgb 214 144 76)
$topoLine.Line.Weight = 1.7
$topoLine.Line.EndArrowheadStyle = 3
$topoTxt = $slide.Shapes.AddTextbox(1, 970, 721, 70, 17)
$topoTxt.Fill.Visible = 0; $topoTxt.Line.Visible = 0
$topoR = $topoTxt.TextFrame.TextRange
$topoR.Text = 'Topology'
$topoR.Font.Name = 'Calibri'; $topoR.Font.Size = 8.8; $topoR.Font.Color.RGB = (Rgb 42 41 38)
$topoR.ParagraphFormat.Alignment = 1

$slide.Export($outPng,'PNG',3600,2025)
$pres.SaveAs($dstPath)

$pres.Close(); $ppt.Quit()
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($slide)
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres)
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
[GC]::Collect(); [GC]::WaitForPendingFinalizers()

Write-Output "Saved optimized copy: $dstPath"
Write-Output "Exported preview: $outPng"

# Try overwrite original if lock has been released
try {
  Copy-Item -LiteralPath $dstPath -Destination $srcPath -Force -ErrorAction Stop
  Write-Output "Overwrote original: $srcPath"
} catch {
  Write-Output "Original still locked, kept optimized copy only."
}
