#!/bin/bash

## Libraries/Input
shopt -s nullglob #allows create an empty array
homePath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" #cur. script locat.
curScrName=${0##*/} #delete all before last backSlash
funcList="$homePath/funcList.sh"
source "$funcList"

EchoLineSh
lenStr=${#curScrName}
lenStr=$((25 + lenStr))
printf "%-${lenStr}s %s\n"\
        "The location of $curScrName:"\
        "$homePath"
printf "%-${lenStr}s %s\n"\
        "The $curScrName is executed from:"\
        "$PWD"
EchoLineSh

## Input arguments
fileInp=${1:-"/u/n/a/nazarovs/private/enhancer/data/upp.bed"}
chrmsSize=${2:-"/u/n/a/nazarovs/private/enhancer/data/hg19.chrom.sizes"}
genome=${3:-"/u/n/a/nazarovs/private/enhancer/data/hg19.fa"}
seqLen=${4:-60} #length of sequenec to cut if possible

fileInp=$(readlink -m "$fileInp")
chrmsSize=$(readlink -m "$chrmsSize")
for file in "$fileInp" "$chrmsSize"; do
  ChkExist f "$file" "Input file: $file\n"
done
dataDir="$(dirname "$fileInp")/dataBanan"
mkdir -p "$dataDir"


## Splitting data according to unique id to generate positive samples
posRangeDir="$dataDir/posRange"
mkdir -p "$posRangeDir"
echo "Splitting data on unique Id ..."
readarray -t uniqId <<< "$(cut -f 1 "$fileInp" | sort | uniq)"
numFiles=0
for id in "${uniqId[@]}"; do
  fileOut="$posRangeDir/$id.$(basename "$fileInp")"
  awk -v id="^$id\$"\
      '{if ($1 ~ id) print($2 "\t" $3 "\t" $3-$2 )}'\
      "$fileInp"  > "$fileOut"

  if [[ $? -ne 0 ]]; then
      echo "Not able to proceed id: $id"
  else
    ((numFiles++))
  fi
done
echo "Done: $numFiles files -> $posRangeDir"
EchoLineSh


## Generate negative samples with specific sequence length
rscr="$homePath/detectComplementRange.R"
negRangeDir="$dataDir/negRange${seqLen}"
mkdir -p "$negRangeDir"
echo "Generating negative samples ..."
numFiles=0
for file in "$posRangeDir"/*; do
  id="$(basename "$file")"
  id="${id%%.*}"
  chrmSize=$(grep -w "$id" "$chrmsSize" | cut -f 2)
  Rscript "$rscr" "$chrmSize" "$file" "$negRangeDir" "$seqLen" &>/dev/null
  if [[ $? -ne 0 ]]; then
      echo "Not able to proceed file: $file"
  else
    ((numFiles++))
  fi
done
echo "Done: $numFiles files -> $negRangeDir"
EchoLineSh


## Generate nucleotide sequence and combine together
for dir in "$posRangeDir" "$negRangeDir"; do
  seqDir="${dir%Range*}Seq${seqLen}"
  mkdir -p "$seqDir"
  echo "Generating sequence for files in $dir ..."
 
  numFiles=0
  for file in "$dir"/*; do
    printf "$((numFiles+1)). $file ... "
    chr="$(basename "$file")"
    chr="${chr%%.*}"
    seqFile="$seqDir/$(basename "$file")"
    echo > "$seqFile"
    exFl=0
    while read -r start restCols; do
      end=$((start + seqLen - 1)) #truncated file
      eval "samtools faidx $genome $chr:${start}-${end}" | tail -n +2 |\
          tr '[:lower:]' '[:upper:]' >> "$seqFile"
      if [[ $? -ne 0 ]]; then
          exFl=1
          continue
      fi
    done < "$file"
    
    if [[ "$exFl" -ne 0 ]]; then
        printf "Error!\n"
    else
      printf "Done!\n"
      ((numFiles++))
    fi
  done
  echo "Done: $numFiles files -> $dataDir"
  EchoLineSh


  ## Combine all files in one set
  outDir="${dir%Range*}WholeSeq${seqLen}"
  mkdir -p "$outDir"
  outFile="$outDir/$(basename $fileInp)"
  echo "Combining all files in $dir ..."
  tmpFile=$(mktemp -uq tmp.XXXX) 
  cat "$seqDir"/* > "$tmpFile"
  grep -v N "$tmpFile" > "$outFile"
  
  echo "Done: 1 file in $outDir"
  EchoLineSh
done
