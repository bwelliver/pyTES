// Macro to convert average pulses from time domain to frequency domain.

// Average pulses are stored in root files per tower and are in a Global Directory as QAverageVector
// They have the following format, as an example:
// KEY: QAverageVector	AveragePulses@Average_ds3054_chan0001;1

// The frequency format is to store it all in a flat file with the following format:
// KEY: TGraph	APfft1;1	Graph

// We will need to do the following:
// 1. Loop over files and read in the QAverageVector in time format.
// 2. FFT it accurately, convert to a NPS, and store it in a TGraph
// That's it I guess...


TFile *f = TFile::Open("average_pulses_ds3054_tower001.root");
TDirectory *global = f->GetDirectory("Global");
QAverageVector *av = global->Get("AveragePulses@Average_ds3054_chan0001");

// Create FFT
int numFreqs = 1000;
QRealComplexFFT transformer(numFreqs);
transformer.SetWindowType(QFFT::StrToWindowType("Hann"), 2);
QVectorC transformedPulse(numFreqs);

int err = transformer.TransformToFreq(*av, transformedPulse);

QVector realPart = transformedPulse.Re();
QVector imagPart = transformedPulse.Im();
QVector npsVec(numFreqs);
npsVec.Initialize(0);
npsVec[0] = mean*mean;
for (int k = 1; k < numFreqs; k++) {
    npsVec[k] = realPart[k]*realPart[k] + imagPart[k]*imagPart[k];
}
// At this point npsVec should be a NPS vector
TGraph *g = new TGraph("")