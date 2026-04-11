$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$assetRoot = Join-Path $root "report\Pipeline_Overview"
$generatedRoot = Join-Path $assetRoot "generated"
$pptxOut = Join-Path $assetRoot "Pipeline_Overview.pptx"
$pngOut = Join-Path $assetRoot "Pipeline_Overview.png"

python (Join-Path $PSScriptRoot "prepare_pipeline_overview_assets.py")

Add-Type -AssemblyName System.Drawing

function Rgb([int]$r, [int]$g, [int]$b) {
    return ($r -bor ($g -shl 8) -bor ($b -shl 16))
}

function Add-TextBox(
    $slide,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [string]$text,
    [double]$fontSize,
    [int]$fontRgb,
    [bool]$bold = $false,
    [bool]$italic = $false,
    [int]$align = 1,
    [string]$fontName = "Calibri"
) {
    $box = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $box.Fill.Visible = 0
    $box.Line.Visible = 0
    $box.TextFrame.MarginLeft = 0
    $box.TextFrame.MarginRight = 0
    $box.TextFrame.MarginTop = 0
    $box.TextFrame.MarginBottom = 0
    $box.TextFrame.WordWrap = -1
    $range = $box.TextFrame.TextRange
    $range.Text = $text
    $range.Font.Name = $fontName
    $range.Font.Size = $fontSize
    $range.Font.Color.RGB = $fontRgb
    $range.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $range.Font.Italic = $(if ($italic) { -1 } else { 0 })
    $range.ParagraphFormat.Alignment = $align
    return $box
}

function Add-RoundedModule(
    $slide,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [int]$fillRgb,
    [int]$lineRgb
) {
    $shape = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fillRgb
    $shape.Fill.Transparency = 0.06
    $shape.Line.ForeColor.RGB = $lineRgb
    $shape.Line.Weight = 2.0
    $shape.Line.DashStyle = 7
    return $shape
}

function Add-RoundedCard(
    $slide,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [int]$fillRgb,
    [int]$lineRgb,
    [double]$weight = 1.15
) {
    $shape = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fillRgb
    $shape.Line.ForeColor.RGB = $lineRgb
    $shape.Line.Weight = $weight
    return $shape
}

function Add-Badge(
    $slide,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [string]$text,
    [int]$lineRgb,
    [double]$fontSize = 11.5
) {
    $shape = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = (Rgb 255 255 255)
    $shape.Line.ForeColor.RGB = $lineRgb
    $shape.Line.Weight = 1.1
    $shape.TextFrame.MarginLeft = 6
    $shape.TextFrame.MarginRight = 6
    $shape.TextFrame.MarginTop = 1
    $shape.TextFrame.MarginBottom = 1
    $range = $shape.TextFrame.TextRange
    $range.Text = $text
    $range.Font.Name = "Cambria"
    $range.Font.Size = $fontSize
    $range.Font.Bold = -1
    $range.Font.Italic = -1
    $range.Font.Color.RGB = (Rgb 40 40 37)
    $range.ParagraphFormat.Alignment = 2
    return $shape
}

function Add-Pill(
    $slide,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [string]$text,
    [int]$fillRgb,
    [int]$lineRgb,
    [double]$fontSize = 8.6
) {
    $shape = $slide.Shapes.AddShape(5, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fillRgb
    $shape.Line.ForeColor.RGB = $lineRgb
    $shape.Line.Weight = 0.95
    $shape.Adjustments.Item(1) = 0.5
    $shape.TextFrame.MarginLeft = 5
    $shape.TextFrame.MarginRight = 5
    $shape.TextFrame.MarginTop = 1
    $shape.TextFrame.MarginBottom = 1
    $range = $shape.TextFrame.TextRange
    $range.Text = $text
    $range.Font.Name = "Calibri"
    $range.Font.Size = $fontSize
    $range.Font.Color.RGB = (Rgb 56 55 51)
    $range.ParagraphFormat.Alignment = 2
    return $shape
}

function Add-Oval(
    $slide,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [int]$fillRgb,
    [int]$lineRgb,
    [double]$lineWeight = 0.8,
    [double]$fillTransparency = 0.0
) {
    $shape = $slide.Shapes.AddShape(9, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fillRgb
    $shape.Fill.Transparency = $fillTransparency
    $shape.Line.ForeColor.RGB = $lineRgb
    $shape.Line.Weight = $lineWeight
    return $shape
}

function Get-ImageFitRect(
    [string]$path,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [double]$margin = 6.0
) {
    $img = [System.Drawing.Image]::FromFile($path)
    try {
        $iw = [double]$img.Width
        $ih = [double]$img.Height
    }
    finally {
        $img.Dispose()
    }
    $innerW = $w - 2 * $margin
    $innerH = $h - 2 * $margin
    $scale = [Math]::Min($innerW / $iw, $innerH / $ih)
    $pw = $iw * $scale
    $ph = $ih * $scale
    $px = $x + $margin + (($innerW - $pw) / 2)
    $py = $y + $margin + (($innerH - $ph) / 2)
    return @($px, $py, $pw, $ph)
}

function Add-ImageCard(
    $slide,
    [string]$path,
    [double]$x,
    [double]$y,
    [double]$w,
    [double]$h,
    [int]$fillRgb,
    [int]$lineRgb
) {
    $frame = Add-RoundedCard $slide $x $y $w $h $fillRgb $lineRgb
    $rect = Get-ImageFitRect $path $x $y $w $h
    $pic = $slide.Shapes.AddPicture($path, 0, -1, $rect[0], $rect[1], $rect[2], $rect[3])
    return @{ Frame = $frame; Picture = $pic }
}

function Add-Line(
    $slide,
    [double]$x1,
    [double]$y1,
    [double]$x2,
    [double]$y2,
    [int]$rgb,
    [double]$weight = 1.6,
    [bool]$arrowEnd = $false
) {
    $line = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    try {
        $line.Line.ForeColor.RGB = $rgb
        $line.Line.Weight = $weight
        if ($arrowEnd) {
            $line.Line.EndArrowheadStyle = 3
        }
    }
    catch {
    }
    return $line
}

function Add-PolylineArrow($slide, $points, [int]$rgb, [double]$weight = 1.6) {
    for ($i = 0; $i -lt ($points.Count - 1); $i++) {
        $p1 = $points[$i]
        $p2 = $points[$i + 1]
        $last = ($i -eq ($points.Count - 2))
        Add-Line $slide $p1[0] $p1[1] $p2[0] $p2[1] $rgb $weight $last | Out-Null
    }
}

$msoFalse = 0
$ppLayoutBlank = 12

$slideW = 1440
$slideH = 810

$canvasRgb = Rgb 249 246 242
$whiteRgb = Rgb 255 255 255
$ink = Rgb 42 41 38
$muted = Rgb 114 111 106
$data = Rgb 32 32 32
$grad = Rgb 192 65 66

$inputFill = Rgb 237 231 225
$inputLine = Rgb 152 143 136
$stage1Fill = Rgb 224 232 236
$stage1Line = Rgb 131 152 161
$stage2Fill = Rgb 238 225 216
$stage2Line = Rgb 171 144 132
$priorFill = Rgb 225 232 216
$priorLine = Rgb 134 149 123
$stage3Fill = Rgb 219 227 235
$stage3Line = Rgb 123 143 164
$cardLine = Rgb 183 177 170
$innerFill = Rgb 252 250 247

if (Test-Path $pptxOut) {
    try {
        Remove-Item $pptxOut -Force
    }
    catch {
        $pptxOut = Join-Path $assetRoot "Pipeline_Overview_v2.pptx"
        $pngOut = Join-Path $assetRoot "Pipeline_Overview_v2.png"
    }
}
if (Test-Path $pngOut) {
    Remove-Item $pngOut -Force -ErrorAction SilentlyContinue
}

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Add()
$pres.PageSetup.SlideWidth = $slideW
$pres.PageSetup.SlideHeight = $slideH
$slide = $pres.Slides.Add(1, $ppLayoutBlank)
$slide.FollowMasterBackground = $msoFalse
$slide.Background.Fill.ForeColor.RGB = $canvasRgb

# Header
Add-TextBox $slide 270 20 900 36 "Three-stage low-light 3DGS pipeline" 23 $ink $true $false 2 "Cambria" | Out-Null
Add-TextBox $slide 280 52 880 18 "fixed-pose sparse bootstrap  ->  sparse-guided geometry refinement  ->  appearance-only decoupling" 10.5 $muted $false $true 2 "Cambria" | Out-Null

# Main module boxes
$inputBox = Add-RoundedModule $slide 28 118 190 520 $inputFill $inputLine
$stage1Box = Add-RoundedModule $slide 255 105 280 245 $stage1Fill $stage1Line
$stage2Box = Add-RoundedModule $slide 565 105 360 245 $stage2Fill $stage2Line
$priorBox = Add-RoundedModule $slide 255 388 670 230 $priorFill $priorLine
$stage3Box = Add-RoundedModule $slide 955 72 455 590 $stage3Fill $stage3Line
$stage3Inner = Add-RoundedModule $slide 973 150 419 252 $innerFill $stage3Line
$stage3Inner.Line.DashStyle = 4
$stage3Inner.Fill.Transparency = 0.02
$legendBox = Add-RoundedModule $slide 392 730 656 48 (Rgb 244 241 237) $cardLine

# Module badges
Add-Badge $slide 52 104 140 30 "Low-light Inputs" $inputLine 11.8 | Out-Null
Add-Badge $slide 296 91 198 30 "Stage 1  Geometry Bootstrap" $stage1Line 11.4 | Out-Null
Add-Badge $slide 611 91 270 30 "Stage 2  Sparse-guided Refinement" $stage2Line 11.2 | Out-Null
Add-Badge $slide 486 374 208 30 "Prior Preparation" $priorLine 11.8 | Out-Null
Add-Badge $slide 1032 58 300 30 "Stage 3  Appearance-only Decoupling" $stage3Line 11.2 | Out-Null

# Subtitles
Add-TextBox $slide 48 142 150 16 "training views and shared RGB supervision" 8.6 $muted $false $false 1 | Out-Null
Add-TextBox $slide 276 130 230 16 "stable sparse scaffold with early densification" 8.7 $muted $false $false 1 | Out-Null
Add-TextBox $slide 585 130 300 16 "no-densify refinement with persistent sparse support" 8.7 $muted $false $false 1 | Out-Null
Add-TextBox $slide 280 414 300 16 "fixed-pose sparse and conservative weak cues" 8.7 $muted $false $false 1 | Out-Null
Add-TextBox $slide 977 98 360 16 "freeze geometry; optimize opacity, SH, illum, and chroma" 8.7 $muted $false $false 1 | Out-Null

# Input content
Add-RoundedCard $slide 52 176 112 78 $whiteRgb $cardLine | Out-Null
Add-RoundedCard $slide 60 166 112 78 $whiteRgb $cardLine | Out-Null
$front = Add-RoundedCard $slide 68 156 112 78 $whiteRgb $grad
$front.Line.Weight = 1.45
Add-ImageCard $slide (Join-Path $assetRoot "Low_Light_3.JPG") 52 176 112 78 $whiteRgb $cardLine | Out-Null
Add-ImageCard $slide (Join-Path $assetRoot "Low_Light_2.JPG") 60 166 112 78 $whiteRgb $cardLine | Out-Null
Add-ImageCard $slide (Join-Path $assetRoot "Low_Light_1.JPG") 68 156 112 78 $whiteRgb $grad | Out-Null
Add-TextBox $slide 40 248 164 18 "Low-light training views  I_in" 10.4 $ink $true $false 2 | Out-Null

Add-ImageCard $slide (Join-Path $generatedRoot "train_aug_tile.png") 58 310 126 92 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 58 408 126 18 "Enhanced supervision  I_sup" 10.2 $ink $true $false 2 | Out-Null
Add-Pill $slide 56 440 130 24 "gamma + exposure matched" $whiteRgb $inputLine 8.2 | Out-Null
Add-Pill $slide 66 470 110 24 "shared RGB target" $whiteRgb $inputLine 8.2 | Out-Null

# Stage 1 content
$stage1Init = Add-RoundedCard $slide 274 164 110 80 $whiteRgb $cardLine
Add-TextBox $slide 286 172 86 12 "hybrid init" 8.4 $ink $true $false 2 | Out-Null
Add-Oval $slide 292 204 5.5 5.5 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 304 198 5.5 5.5 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 316 206 5.5 5.5 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 328 196 5.5 5.5 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 340 208 5.5 5.5 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 352 200 5.5 5.5 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 314 184 9 9 (Rgb 231 171 103) (Rgb 184 132 74) 0.8 0.08 | Out-Null
Add-Oval $slide 338 182 9 9 (Rgb 231 171 103) (Rgb 184 132 74) 0.8 0.08 | Out-Null
Add-Oval $slide 326 218 9 9 (Rgb 231 171 103) (Rgb 184 132 74) 0.8 0.08 | Out-Null
Add-TextBox $slide 284 228 90 11 "filtered sparse  +  5K random GS" 6.9 $muted $false $false 2 | Out-Null
Add-ImageCard $slide (Join-Path $assetRoot "stage4\rendering.JPG") 403 156 114 86 $whiteRgb $cardLine | Out-Null
Add-Line $slide 387 200 397 200 $data 1.8 $true | Out-Null
Add-TextBox $slide 274 250 110 16 "bootstrap seeds" 8.5 $ink $false $false 2 | Out-Null
Add-TextBox $slide 402 248 116 16 "geometry base" 8.5 $ink $false $false 2 | Out-Null
Add-Pill $slide 270 274 118 24 "all sparse + 5K random GS" $whiteRgb $stage1Line 7.8 | Out-Null
Add-Pill $slide 276 306 84 24 "only densify" $whiteRgb $stage1Line 8.0 | Out-Null
Add-Pill $slide 366 306 72 24 "RGB + exp" $whiteRgb $stage1Line 8.0 | Out-Null
Add-Pill $slide 444 306 76 24 "depth + mv" $whiteRgb $stage1Line 8.0 | Out-Null

# Stage 2 content
$stage2Guide = Add-RoundedCard $slide 586 160 118 82 $whiteRgb $cardLine
Add-TextBox $slide 600 168 90 12 "weighted support" 8.4 $ink $true $false 2 | Out-Null
Add-Oval $slide 634 198 10 10 (Rgb 214 122 96) (Rgb 179 92 67) 0.9 0.02 | Out-Null
Add-Oval $slide 608 192 6 6 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 658 186 6 6 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 614 220 6 6 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Oval $slide 662 218 6 6 (Rgb 182 178 172) (Rgb 182 178 172) 0.5 | Out-Null
Add-Line $slide 613 195 639 203 (Rgb 150 146 140) 0.8 $false | Out-Null
Add-Line $slide 661 189 643 203 (Rgb 150 146 140) 0.8 $false | Out-Null
Add-Line $slide 617 223 639 206 (Rgb 150 146 140) 0.8 $false | Out-Null
Add-Line $slide 665 221 644 206 (Rgb 150 146 140) 0.8 $false | Out-Null
Add-TextBox $slide 596 228 98 11 "active Gaussian + KNN sparse neighbors" 6.7 $muted $false $false 2 | Out-Null
Add-ImageCard $slide (Join-Path $assetRoot "stage5\rendering.JPG") 787 156 118 86 $whiteRgb $cardLine | Out-Null
Add-Line $slide 708 200 779 200 $data 1.8 $true | Out-Null
Add-TextBox $slide 585 248 122 16 "sparse-guided anchor" 8.4 $ink $false $false 2 | Out-Null
Add-TextBox $slide 788 248 116 16 "refined geometry" 8.4 $ink $false $false 2 | Out-Null
Add-RoundedCard $slide 586 274 164 54 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 598 284 140 14 "pbar_i = sum_j w_ij p_ij" 10.4 $ink $false $false 2 "Cambria" | Out-Null
Add-TextBox $slide 598 302 140 14 "L_sparse = mean rho(||mu_i - pbar_i||_2)" 8.7 $ink $false $false 2 "Cambria" | Out-Null
Add-RoundedCard $slide 770 272 136 58 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 778 282 120 14 "w_ij: quality + local density + proximity" 8.8 $ink $false $false 2 | Out-Null
Add-TextBox $slide 780 301 116 12 "quality from track length and reprojection error" 7.1 $muted $false $false 2 | Out-Null
Add-Pill $slide 586 334 84 24 "no densify" $whiteRgb $stage2Line 8.0 | Out-Null
Add-Pill $slide 676 334 84 24 "weak depth" $whiteRgb $stage2Line 8.0 | Out-Null
Add-Pill $slide 766 334 76 24 "structure" $whiteRgb $stage2Line 8.0 | Out-Null
Add-Pill $slide 848 334 58 24 "active GS" $whiteRgb $stage2Line 8.0 | Out-Null

# Prior content
Add-TextBox $slide 294 438 162 18 "fixed-pose sparse" 10.6 $ink $true $false 2 | Out-Null
Add-TextBox $slide 514 438 152 18 "weak depth" 10.6 $ink $true $false 2 | Out-Null
Add-TextBox $slide 735 438 150 18 "structure" 10.6 $ink $true $false 2 | Out-Null
Add-ImageCard $slide (Join-Path $assetRoot "stage5\cupcake_sparse_base.png") 286 462 178 112 $whiteRgb $cardLine | Out-Null
Add-ImageCard $slide (Join-Path $assetRoot "Depth.JPG") 506 462 168 112 $whiteRgb $cardLine | Out-Null
Add-ImageCard $slide (Join-Path $assetRoot "Structure.JPG") 726 462 168 112 $whiteRgb $cardLine | Out-Null
Add-Pill $slide 312 582 126 24 "voxel dedup + filtering" $whiteRgb $priorLine 8.0 | Out-Null
Add-Pill $slide 542 582 96 24 "Marigold prior" $whiteRgb $priorLine 8.0 | Out-Null
Add-Pill $slide 748 582 124 24 "illumination-invariant" $whiteRgb $priorLine 8.0 | Out-Null

# Stage 3 content
Add-Pill $slide 981 116 152 24 "geometry frozen: mu, q, s" $whiteRgb $stage3Line 8.3 | Out-Null

Add-ImageCard $slide (Join-Path $generatedRoot "train_aug_tile.png") 986 176 78 58 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 986 239 78 16 "I_sup" 10.2 $ink $true $false 2 | Out-Null
Add-TextBox $slide 978 256 94 18 "global RGB appearance ref" 7.5 $muted $false $false 2 | Out-Null

Add-ImageCard $slide (Join-Path $generatedRoot "proxy_target_tile.png") 986 286 78 58 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 986 349 78 16 "I_proxy" 10.2 $ink $true $false 2 | Out-Null
Add-TextBox $slide 976 366 98 22 "shadow-aware`r`nluminance ref" 7.5 $muted $false $false 2 | Out-Null

Add-ImageCard $slide (Join-Path $assetRoot "stage6\base.JPG") 1082 206 88 68 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 1082 279 88 16 "Base RGB  I_base" 8.8 $ink $false $false 2 | Out-Null

Add-RoundedCard $slide 1180 222 50 34 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 1186 228 38 10 "YCbCr" 8.2 $ink $true $false 2 | Out-Null
Add-TextBox $slide 1184 240 42 10 "split" 8.0 $ink $false $false 2 | Out-Null

Add-RoundedCard $slide 1240 176 88 52 $whiteRgb $stage3Line | Out-Null
Add-TextBox $slide 1248 184 72 12 "Illum head  A_Y" 8.8 $ink $true $false 2 | Out-Null
Add-TextBox $slide 1248 200 72 10 "Y-only gain" 8.0 $muted $false $false 2 | Out-Null
Add-TextBox $slide 1245 212 78 10 "Y' = clip(2 sigma(A_Y) * Y)" 6.2 $ink $false $false 2 "Cambria" | Out-Null

Add-RoundedCard $slide 1240 252 88 60 $whiteRgb $stage3Line | Out-Null
Add-TextBox $slide 1246 262 76 12 "Chroma head  A_C" 8.8 $ink $true $false 2 | Out-Null
Add-TextBox $slide 1246 278 76 10 "bounded additive" 8.0 $muted $false $false 2 | Out-Null
Add-TextBox $slide 1245 290 78 10 "Delta_C = a_c tanh(A_C)" 6.2 $ink $false $false 2 "Cambria" | Out-Null

$mergeNode = Add-Oval $slide 1340 238 18 18 $whiteRgb $cardLine 1.0 0.0
Add-TextBox $slide 1343 239 12 12 "+" 10.2 $ink $true $false 2 | Out-Null
Add-TextBox $slide 1328 260 42 12 "recombine" 7.0 $muted $false $false 2 | Out-Null

Add-ImageCard $slide (Join-Path $assetRoot "stage6\rendering.JPG") 1362 206 44 68 $whiteRgb $cardLine | Out-Null
Add-TextBox $slide 1358 279 52 16 "I_rec" 8.8 $ink $false $false 2 | Out-Null

Add-Line $slide 1170 240 1178 240 $data 1.8 $true | Out-Null
Add-PolylineArrow $slide @(@(1230, 239), @(1238, 239), @(1238, 202), @(1240, 202)) $data 1.5
Add-PolylineArrow $slide @(@(1230, 239), @(1238, 239), @(1238, 282), @(1240, 282)) $data 1.5
Add-PolylineArrow $slide @(@(1328, 202), @(1338, 202), @(1338, 244), @(1340, 244)) $data 1.5
Add-PolylineArrow $slide @(@(1328, 282), @(1338, 282), @(1338, 250), @(1340, 250)) $data 1.5
Add-Line $slide 1354 247 1360 247 $data 1.8 $true | Out-Null

Add-RoundedCard $slide 984 432 180 102 $whiteRgb $stage3Line | Out-Null
Add-RoundedCard $slide 1178 432 212 102 $whiteRgb $stage3Line | Out-Null
Add-TextBox $slide 996 444 156 16 "Target Decoupling" 10.2 $ink $true | Out-Null
Add-TextBox $slide 1190 444 188 16 "Spatial Confidence" 10.2 $ink $true | Out-Null
Add-TextBox $slide 998 470 150 54 "RGB(I_base)  ->  I_sup`r`nY(I_rec)  ->  I_proxy`r`nCb/Cr(I_rec)  ->  I_sup" 8.9 $muted | Out-Null
Add-TextBox $slide 1190 470 188 54 "W_Y emphasizes informative dark regions for relighting.`r`nW_C down-weights severe shadows for chroma correction." 8.6 $muted | Out-Null
Add-TextBox $slide 990 552 392 18 "L = L_rgb + lambda_Y L_Y + lambda_c L_CbCr + lambda_cr ||Delta_C||_1" 8.6 $muted $false $false 2 "Cambria" | Out-Null

# Data flow arrows
Add-Line $slide 218 245 247 245 $data 1.9 $true | Out-Null
Add-Line $slide 535 228 557 228 $data 1.9 $true | Out-Null
Add-Line $slide 925 228 947 228 $data 1.9 $true | Out-Null
Add-PolylineArrow $slide @(@(132, 232), @(132, 388), @(286, 388)) $data 1.6
Add-PolylineArrow $slide @(@(375, 462), @(375, 350), @(318, 350)) $data 1.5
Add-PolylineArrow $slide @(@(375, 462), @(375, 350), @(642, 350), @(642, 244)) $data 1.5

# Gradient / supervision arrows
Add-PolylineArrow $slide @(@(184, 356), @(230, 356), @(230, 132), @(395, 132)) $grad 1.5
Add-PolylineArrow $slide @(@(184, 356), @(240, 356), @(240, 132), @(736, 132)) $grad 1.5
Add-TextBox $slide 392 116 50 14 "L_rgb" 7.6 $grad | Out-Null
Add-TextBox $slide 736 116 50 14 "L_rgb" 7.6 $grad | Out-Null

Add-PolylineArrow $slide @(@(590, 462), @(590, 364), @(430, 364), @(430, 350)) $grad 1.4
Add-PolylineArrow $slide @(@(590, 462), @(590, 364), @(720, 364), @(720, 350)) $grad 1.4
Add-TextBox $slide 422 352 58 14 "L_depth" 7.4 $grad | Out-Null
Add-TextBox $slide 708 352 58 14 "L_depth" 7.4 $grad | Out-Null

Add-PolylineArrow $slide @(@(810, 462), @(810, 350), @(852, 350)) $grad 1.4
Add-TextBox $slide 846 352 58 14 "L_struct" 7.4 $grad | Out-Null

Add-TextBox $slide 635 252 64 14 "L_sparse" 7.4 $grad | Out-Null

Add-PolylineArrow $slide @(@(1064, 205), @(1074, 205), @(1074, 222), @(1082, 222)) $grad 1.4
Add-PolylineArrow $slide @(@(1064, 315), @(1140, 315), @(1140, 202), @(1240, 202)) $grad 1.4
Add-PolylineArrow $slide @(@(1064, 205), @(1128, 205), @(1128, 282), @(1240, 282)) $grad 1.4
Add-TextBox $slide 1070 192 42 14 "L_rgb" 7.4 $grad | Out-Null
Add-TextBox $slide 1144 318 52 14 "L_Y, W_Y" 7.2 $grad | Out-Null
Add-TextBox $slide 1130 286 72 14 "L_CbCr, W_C" 7.0 $grad | Out-Null

# Legend
Add-TextBox $slide 410 745 56 16 "Legend" 10.4 $ink $true | Out-Null
Add-Line $slide 508 754 600 754 $data 1.7 $true | Out-Null
Add-TextBox $slide 612 744 74 16 "data flow" 8.8 $ink | Out-Null
Add-Line $slide 748 754 840 754 $grad 1.7 $true | Out-Null
Add-TextBox $slide 852 744 160 16 "gradient / supervision" 8.8 $ink | Out-Null

$pres.SaveAs($pptxOut)
$slide.Export($pngOut, "PNG", 3600, 2025)
try {
    $pres.Close()
}
catch {
}
try {
    $ppt.Quit()
}
catch {
}
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($slide)
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres)
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
[GC]::Collect()
[GC]::WaitForPendingFinalizers()

Write-Output "[PipelineOverviewPPT] saved $pptxOut"
Write-Output "[PipelineOverviewPPT] exported $pngOut"
