require("GenomicRanges")
args <- commandArgs(trailingOnly = TRUE)

chrSize <- strtoi(args[1])
fileInp <- args[2]
dirOut <- args[3]
minLen <- strtoi(args[4])

## chrSize <- 249250621
## fileInp <- "/u/n/a/nazarovs/private/enhancer/data/posRange/chr10.hg19.enhancers.bed"
## dirOut <- "/u/n/a/nazarovs/private/enhancer/data/negRange20"
## minLen <- 20
## Create GRange according to the ChrSize
chrRange <- IRanges(start = 1, end = chrSize)

## Read and combine ranges from the inpFile
dataInp <- read.table(file = fileInp, sep = '\t', check.names = F, header = F)

dataInpRange <- IRanges(start = dataInp[, 1], end = dataInp[, 2])

## Generate The Complement range according to the chromosome
dataOutRange <- setdiff(chrRange, dataInpRange)
dataOutRange <- dataOutRange[width(dataOutRange) >= minLen, ]
for (i in 1:length(dataOutRange)){
    start <- sample(end(dataOutRange[i]) - minLen + 1, 1)
    dataOutRange[i] <- IRanges(start = start, width = minLen)
}
fileOut <- unlist(strsplit(fileInp, "/"))
fileOut <-  paste0(dirOut, "/",  fileOut[length(fileOut)])
write.table(x = cbind(start(dataOutRange), end(dataOutRange)), file = fileOut, row.names = F, col.names = F, append = F)
