#!/bin/bash
set -e

OUTDIR=/workspace/videos
mkdir -p "$OUTDIR"
cd "$OUTDIR"

VIDEOS=(
Cm73ma6Ibcs q01CUy_gwdU 28CIeC8cZks TJR1oYDDTwg HfEVEGf1A8Q
2sriHX3PbXw _zrgbA3FMVE TiQBTesZUJQ 5dZ_lvDgevk GcRKREorGSc
xi6r3hZe5Tg Z4HGQL_McDQ aJI8XTa_DII IsdbCjlZ5cQ t-RtDI2RWQs
8QdE--Y-x7U idZkam9zqAs xECIRjlxM3U o-gLbgpzCc8 QWXlvx1GoTY
-hgaSElC3wU hROKtPqktO8 tH_5YbklevQ USfvmoalqsw evYm0cELO3I
RbpKkvlHYTw Mcggugol2ts JPPMz8fEml0 uusf1qG_uZ4 O14bbpvy2x0
3_upA09AntU gbDR39yIs3Y rSE2YPcv89U T1yhBv1ytzw Xjf5N9S3jAA
tKIFQI9cH2c VTCDQYYKA9o KLIa2UaE2KE CgvJqGxzRfE EwskdNETNx8
vZV2WCKMsKs sk00epALZps AeEYQ62t8hA NzCO0G8AGLU Vk_Af0htZGU
moALQl818ZY qAIRFyR6NyQ Za2Z_JRxCuk qYMnM5blZIE gXnhqF0TqqI
SRq0weUKskM 81SbCR6p3Z0 S8vPx-u9p_A QgWRyDV9Ozs pXD3txG2bVQ
RjDrZkBwzho XNtNNplAwiI EpMLAQbSYAw KktLi3UifPY wNCPgIVz15c
uW9mcG0rdLY TZ0j6kr4ZJ0 ihfjEFGdZdc vaL_vSdZKZo -WnyRMZqV1U
7HjNIPIgUg4 8NHmfpkxTSw YlQugR7KSKg lDlA7cfNk8A f8DKD78BrQA
FaV0tIaWWEg rk24OUu_kJQ 16Z-XQh9jhk QB7FoIpx8os KbahC-QCKU8
_T2Avd3tFHc RCAqKnvu_P0 iA_69g87Ilw wgBlACG927Y 2zkJFv-ro4A
LLSJrEgOOtw Va_9Q6ekm60 1FsiZgGZU70 jp2M1hIEtsk oZEVgDXJwCc
Hf-n1yfd8II JlrzSvCsIjE cXDT44zT8JY cWEnogdsW78 rp4NKWb7dXk
Aiem1w_TvaA Z86xysw5Ncc hjoDzK0siaM 2LH3JCGkEBU 9-gOCOu_KGU
9tBsMSDoDqk k2FIFQIYBvA W-BnDvXXfOs 4LA_tH-VSnQ vHlSoxg8WHo
pe_LddfHAUU JTa_Ue2MSwc 20lTg3yUrO4
)

TOTAL=${#VIDEOS[@]}
COUNT=0
SKIP=0
FAIL=0

for vid in "${VIDEOS[@]}"; do
    COUNT=$((COUNT + 1))
    if [ -f "${vid}.mp4" ]; then
        echo "[$COUNT/$TOTAL] SKIP (exists): $vid"
        SKIP=$((SKIP + 1))
        continue
    fi
    echo "[$COUNT/$TOTAL] Downloading: $vid"
    yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
        --merge-output-format mp4 \
        --no-playlist \
        -o "${vid}.mp4" \
        "https://www.youtube.com/watch?v=${vid}" 2>&1 || {
            echo "FAILED: $vid"
            FAIL=$((FAIL + 1))
        }
done

echo ""
echo "=== DONE ==="
echo "Total: $TOTAL | Downloaded: $((COUNT - SKIP - FAIL)) | Skipped: $SKIP | Failed: $FAIL"
echo "Disk usage:"
du -sh "$OUTDIR"
